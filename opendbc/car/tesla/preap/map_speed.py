"""Map speed limit from the MCU (CAN 760 UI_gpsVehicleSpeed.UI_mppSpeedLimit).

Pre-AP broadcasts the navigation database's limit for the road it thinks the car
is on. Nothing in openpilot consumed it. This turns it into a set speed the
driver can ask for, and never into one that is applied on its own.

Two measurements over nine logged drives (13.7k GPS samples at 2 Hz) shape the
validity rules below:

  * The field reports 25 when it has no limit for the road, not when the limit
    is 25. Those stretches ran a median 34 mph with a p90 of 45, and the
    transition counts give it away -- 45<->25 flipped 44 times and 35<->25 21
    times, against 1-6 for every genuine road-to-road change. So 25 is never
    usable, which does cost a real 25 mph zone; on this car that trade is
    already paid for by how often 25 means nothing at all.

  * Real values still flicker at road boundaries and in town, which is exactly
    where following them automatically would produce unrequested decel. A value
    therefore has to hold steady for STABLE_MS before it can be applied.

Neither rule can make a wrong limit safe on its own. What does is that the
driver asks for it by pulling the stalk (see PreAPEngagement), so a stale or
wrong value costs a re-pull rather than a deceleration.

A readout answers a different question and needs its own view of the same
signal. Nothing acts on what the driver reads, so a value that is merely old is
worth more than a blank sign -- but only for as long as it is plausibly still
the same road. Measured across those drives: a third of the time there is no
usable value at all, and the gaps sort cleanly by **distance**, not by duration.
Gaps the limit came back unchanged from covered at most 0.48 miles; gaps the
road had changed across ran a median 0.53. The longest gap in the set lasted
761 seconds and covered 0.28 miles -- sitting at a light, same road throughout,
which any time-based hold would have thrown away. So the readout holds by
distance travelled, and says so on the face of the sign while it is holding.
"""

from opendbc.car.common.conversions import Conversions as CV

# The "no limit for this road" placeholder, in the cluster's display units.
SENTINEL = 25.0

# How long one value must hold before it may be applied.
STABLE_MS = 2000

# The message runs near 10 Hz. Past this with no fresh frame, the CAN parser is
# repeating its last payload and the limit is not evidence of anything.
STALE_MS = 3000

# Readout only. A shorter hold than STABLE_MS: nothing acts on this, so the
# only cost of adopting a value early is a number that flickers.
DISPLAY_STABLE_MS = 1000

# How far the readout may carry a value the MCU has stopped reporting. Chosen
# off the gap measurement in the module docstring: 0.4 miles covers 19 of the
# 21 gaps the road did not change across, and stops short of the median gap
# that it did. Metres, so it does not depend on the cluster's display units.
DISPLAY_HOLD_M = 643.7  # 0.4 mi

# Ceiling on one frame's contribution to the hold, in seconds. A stalled parser
# or a clock step must not spend the whole budget in a single update.
DISPLAY_MAX_STEP_S = 0.5


class MapSpeedLimit:
  """Debounced view of the MCU's map speed limit."""

  def __init__(self):
    self.raw = 0.0                # last value seen, display units
    self.stable = 0.0             # value that has held for STABLE_MS, else 0
    self.display = 0.0            # value for the readout, held across gaps
    self.display_held = False     # the readout is showing a value the MCU dropped
    self._candidate = 0.0
    self._candidate_since_ms = 0
    self._display_candidate = 0.0
    self._display_candidate_since_ms = 0
    self._hold_distance_m = 0.0
    self._last_update_ms = 0
    self._last_ts_nanos = 0
    self._last_rx_ms = 0

  def update(self, raw_limit, ts_nanos, curr_time_ms, v_ego=0.0):
    """Feed one frame. raw_limit is already scaled by the DBC (5 per count)."""
    if ts_nanos != self._last_ts_nanos:
      self._last_ts_nanos = ts_nanos
      self._last_rx_ms = curr_time_ms

    value = float(raw_limit or 0.0)
    self.raw = value
    usable = value > 0.0 and value != SENTINEL
    self._update_display(value if usable else 0.0, curr_time_ms, v_ego)

    if not usable:
      self._candidate = 0.0
      self._candidate_since_ms = 0
      self.stable = 0.0
      return

    if value != self._candidate:
      # A new value restarts the hold and withdraws the old one rather than
      # leaving it standing: once the road has changed, the previous limit is
      # not a safer answer than no answer.
      self._candidate = value
      self._candidate_since_ms = curr_time_ms
      self.stable = 0.0
    elif curr_time_ms - self._candidate_since_ms >= STABLE_MS:
      self.stable = value

  def _update_display(self, usable_value, curr_time_ms, v_ego):
    """Advance the readout's own view: debounce, then hold across a gap."""
    elapsed_s = 0.0
    if self._last_update_ms:
      elapsed_s = min(max(curr_time_ms - self._last_update_ms, 0) / 1000.0, DISPLAY_MAX_STEP_S)
    self._last_update_ms = curr_time_ms

    if usable_value > 0.0:
      if usable_value != self._display_candidate:
        self._display_candidate = usable_value
        self._display_candidate_since_ms = curr_time_ms
      if curr_time_ms - self._display_candidate_since_ms >= DISPLAY_STABLE_MS:
        self.display = usable_value
        self.display_held = False
        self._hold_distance_m = 0.0
      return

    self._display_candidate = 0.0
    self._display_candidate_since_ms = curr_time_ms
    if self.display <= 0.0:
      return

    # Distance, not time. A long wait at a light is still the same road; half a
    # mile of driving is a claim about one the car has not seen a limit for.
    self.display_held = True
    self._hold_distance_m += max(float(v_ego), 0.0) * elapsed_s
    if self._hold_distance_m > DISPLAY_HOLD_M:
      self._clear_display()

  def _clear_display(self):
    self.display = 0.0
    self.display_held = False
    self._hold_distance_m = 0.0

  def _fresh(self, curr_time_ms):
    return self._last_rx_ms > 0 and (curr_time_ms - self._last_rx_ms) <= STALE_MS

  def limit(self, curr_time_ms):
    """Applicable limit in display units, or 0.0 if there is none."""
    return self.stable if self._fresh(curr_time_ms) else 0.0

  def raw_ms(self, speed_units):
    """The MCU's unfiltered value in m/s, placeholder and all.

    Published so a wrong number on the sign can be attributed without the CAN
    log: a reading the MCU itself got wrong looks different from one this
    module carried across a gap, and the two want different fixes.
    """
    if self.raw <= 0.0:
      return 0.0
    return self.raw * (CV.MPH_TO_MS if speed_units == "MPH" else CV.KPH_TO_MS)

  def display_limit_ms(self, speed_units, curr_time_ms):
    """Readout limit in m/s, or 0.0 when the sign should be blank."""
    if not self._fresh(curr_time_ms):
      self._clear_display()
      return 0.0
    if self.display <= 0.0:
      return 0.0
    return self.display * (CV.MPH_TO_MS if speed_units == "MPH" else CV.KPH_TO_MS)

  def limit_ms(self, speed_units, curr_time_ms):
    """Applicable limit in m/s, for publishing on carState."""
    limit = self.limit(curr_time_ms)
    if limit <= 0.0:
      return 0.0
    return limit * (CV.MPH_TO_MS if speed_units == "MPH" else CV.KPH_TO_MS)

  def target_kph(self, speed_units, offset, curr_time_ms):
    """Set speed for limit + offset, in kph. 0.0 when no limit is applicable.

    The offset is in the cluster's display units, so a driver reading MPH gets
    the mph they asked for and the number that lands in the MAX box is whole.
    """
    limit = self.limit(curr_time_ms)
    if limit <= 0.0:
      return 0.0
    unit_kph = CV.MPH_TO_KPH if speed_units == "MPH" else 1.0
    return max((limit + offset) * unit_kph, 0.0)
