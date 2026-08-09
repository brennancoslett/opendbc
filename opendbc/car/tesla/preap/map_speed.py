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
"""

from opendbc.car.common.conversions import Conversions as CV

# The "no limit for this road" placeholder, in the cluster's display units.
SENTINEL = 25.0

# How long one value must hold before it may be applied.
STABLE_MS = 2000

# The message runs near 10 Hz. Past this with no fresh frame, the CAN parser is
# repeating its last payload and the limit is not evidence of anything.
STALE_MS = 3000


class MapSpeedLimit:
  """Debounced view of the MCU's map speed limit."""

  def __init__(self):
    self.raw = 0.0                # last value seen, display units
    self.stable = 0.0             # value that has held for STABLE_MS, else 0
    self._candidate = 0.0
    self._candidate_since_ms = 0
    self._last_ts_nanos = 0
    self._last_rx_ms = 0

  def update(self, raw_limit, ts_nanos, curr_time_ms):
    """Feed one frame. raw_limit is already scaled by the DBC (5 per count)."""
    if ts_nanos != self._last_ts_nanos:
      self._last_ts_nanos = ts_nanos
      self._last_rx_ms = curr_time_ms

    value = float(raw_limit or 0.0)
    self.raw = value

    if value <= 0.0 or value == SENTINEL:
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

  def _fresh(self, curr_time_ms):
    return self._last_rx_ms > 0 and (curr_time_ms - self._last_rx_ms) <= STALE_MS

  def limit(self, curr_time_ms):
    """Applicable limit in display units, or 0.0 if there is none."""
    return self.stable if self._fresh(curr_time_ms) else 0.0

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
