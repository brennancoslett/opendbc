"""Vision-based ACC (no pedal): stock-CC set-speed modulation.

Ports the Tinkla tesla-unity ACCController concept onto NAP's architecture.
openpilot's longitudinal planner runs normally (vision leads via the model
when no radar is fitted); with the standard zero-gain tuning, CC.actuators.accel
is the planner's target accel passed through as pure feedforward. This module
translates that request into spoofed cruise-stalk speed presses that walk the
DI's stock cruise set speed toward the planned speed, capped at the
driver-owned ceiling (engagement.pedal_speed_kph).

Decel authority is whatever the DI does when its set speed drops — motor
regen only, no friction brakes. Large decel requests CANCEL the stock CC
instead (coasting regen beats CC's shallow slew). The driver is the brakes.

Decisions are always computed and logged via carlog. Whether they are
actually transmitted is controlled by nap_conf.vision_acc_live_tx
(NAPVisionACCLiveTX param, see carcontroller.py) — off by default so a
fresh install is a dry run, and toggleable from the NAP settings panel
without a reboot or redeploy so a dry-run drive's logged decisions can be
reviewed before flipping it live. The tesla_preap.h safety-mode button
gating (PREAP_FLAG_VISION_ACC) enforces the button values and TX rate
regardless of this setting.

A/B TUNING TELEMETRY
--------------------
Every log line is prefixed "VisionACC.tlm" (see _log_telemetry). One line at
5 Hz while modulating, 1 Hz otherwise, self-labeled with the projection value
so runs at different gains are directly comparable. To pull a drive's data:

    grep -h "VisionACC.tlm" /data/log/swaglog.* | sed 's/.*VisionACC.tlm //'

Fields (all key=value):
  f       carcontroller frame (100 Hz clock)
  proj    ACCEL_PROJECTION_S in effect this run (the A/B variable)
  ltx     nap_conf.vision_acc_live_tx (1 = presses transmitted → closed loop)
  region  1 = vision ACC operating (op-long + longActive + no gas + di ENABLED)
  reason  why not modulating when region=0 (not_op_long / not_long_active /
          gas_pressed / di_standby / di_off), or the active sub-state
          (active / hard_brake_cancel / human_holdoff / auto_spacing)
  di      DI cruise state, long, gas, units  — engagement context
  vEgo    m/s   — offline dv/dt of this is the fallback achieved-accel signal
  aEgo    m/s^2 — ACHIEVED longitudinal accel (the closed-loop response)
  aReq    m/s^2 — CC.actuators.accel, what the planner WANTED (the target)
  ccSet   kph   — current DI set speed (v_cruise_actual_kph)
  desired kph   — target set speed = vEgo + aReq*proj (capped at ceiling)
  ceil    kph   — driver ceiling (pedal_speed_kph)
  off     kph   — desired - ccSet (the gap the button logic acts on)
  btn     the button decided this frame (none if in deadband)

The core A/B question — does raising proj make the DI accelerate more? — is
answered by aReq vs aEgo tracking (and off, the set-speed gap it produced).
"""
import time

from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import CruiseButtons

# nap_conf is imported for live_tx visibility in the telemetry line. No cycle:
# nap_conf does not import this module.
try:
  from opendbc.car.tesla.preap.nap_conf import nap_conf
except Exception:  # pragma: no cover - defensive; telemetry falls back to ltx=-1
  nap_conf = None

# Stock CC on Pre-AP Model S only operates above ~17 mph (tesla-unity value)
MIN_CRUISE_SPEED_MS = 17.1 * CV.MPH_TO_MS

# Planner accel is projected this far ahead to form a target speed.
# Kept at 1.5 (tesla-unity value). A 2.0 bump was tried to make accel more
# perceptible, then reverted: drive 00000001--05edcaba20 (LiveTX on, closed
# loop at 1.5) shows the DI does NOT respond to a larger set-speed gap. When
# the planner wanted accel>0.2 m/s^2, achieved-vs-gap was non-monotonic — the
# bulk (3322 frames at a 3-5 kph gap) delivered only ~0.05 m/s^2, while smaller
# and larger gaps scattered high (reverse causality: grade/lead already driving
# accel). Projection only enlarges the gap, and a bigger gap doesn't buy more
# accel here — the limiter is the DI's inherently soft response to set-speed
# steps, which no feedforward horizon fixes. Any change to this needs an on-road
# A/B, not a value derived from a log recorded at the old gain. The
# VisionACC.tlm telemetry (see module docstring) is the instrument for that A/B.
ACCEL_PROJECTION_S = 1.5

# Don't fight the driver: no automated press within this window of a human
# stalk action (tesla-unity used 3s)
HUMAN_ACTION_HOLDOFF_MS = 3000
# Spacing between automated presses — the DI needs time to act on each step
# (tesla-unity value). Drive log 00000009--2dd6a2315a showed presses landing
# at a rigid ~500ms cadence through every accel/decel ramp (e.g. 7 straight
# DN_1ST steps at 1784069147-149), reported back as "stuttery"
# acceleration/deceleration — tune this if 400ms doesn't help.
AUTO_ACTION_SPACING_MS = 400

# Below this requested accel, stepped set-speed nudges can't track the
# request in time: cc_set_kph gets dragged down by each auto-press, which
# closes calc_button()'s offset_kph gap before it ever crosses the CANCEL
# threshold. Drive log 00000009--2dd6a2315a showed a -1.50 m/s^2 event
# (well under ACCEL_MIN=-3.48) still resolve to a half-step DN_1ST because
# of this self-correction. Bypass the offset math and holdoff/spacing gates
# entirely and drop CC now — the driver is the brakes.
ACCEL_CANCEL_THRESHOLD = -1.3

# A/B telemetry cadence at the 100 Hz carcontroller clock.
TLM_PERIOD_IN_FRAMES = 20    # 5 Hz while modulating (fine enough for accel dynamics)
TLM_PERIOD_OUT_FRAMES = 100  # 1 Hz otherwise, so non-engagement stays visible cheaply

# reason values that count as "vision ACC is operating"
_ACTIVE_REASONS = ("active", "hard_brake_cancel", "human_holdoff", "auto_spacing")


def _current_time_millis():
  return int(round(time.time() * 1000))


def get_cc_step_kph(is_mph):
  """Stock CC set-speed steps in kph: (half press, full press).

  Imperial cars step 1/5 mph, metric cars 1/5 kph.
  """
  if is_mph:
    return 1 * CV.MPH_TO_KPH, 5 * CV.MPH_TO_KPH
  return 1.0, 5.0


class VisionACCController:
  """Per-frame stock-CC button decisions. TX ownership stays with StockCCSpoofer."""

  def __init__(self):
    self.last_auto_action_ms = 0
    self.last_logged_button = None
    # telemetry scratch, refreshed every _decide()
    self._reason = "init"
    self._desired_kph = None
    self._offset_kph = None
    carlog.info(
      "VisionACC config: proj=%.2f spacing=%dms holdoff=%dms cancel_thr=%+.2f min_cruise=%.1fmph",
      ACCEL_PROJECTION_S, AUTO_ACTION_SPACING_MS, HUMAN_ACTION_HOLDOFF_MS,
      ACCEL_CANCEL_THRESHOLD, MIN_CRUISE_SPEED_MS * CV.MS_TO_MPH)

  def update(self, CC, CS, frame):
    """Return the CruiseButtons value to spoof this frame, or None."""
    button = self._decide(CC, CS)
    self._log_decision(button, CC, CS)
    self._log_telemetry(button, CC, CS, frame)
    if button is not None:
      self.last_auto_action_ms = _current_time_millis()
    return button

  def _decide(self, CC, CS):
    self._reason = "active"
    self._desired_kph = None
    self._offset_kph = None

    # Engagement FSM must be in long mode (double-pull), openpilot long active
    if not (getattr(CS, "cruiseEnabled", False) and getattr(CS, "enableLongControl", False)):
      self._reason = "not_op_long"
      return None
    if not CC.longActive:
      self._reason = "not_long_active"
      return None
    # Driver on the accelerator: the DI holds CC through it; stay out
    if CS.out.gasPressed:
      self._reason = "gas_pressed"
      return None
    # Only modulate a running stock CC — engaging it is StockCCSpoofer's job,
    # and after a CANCEL the driver must double-pull to rearm (no autoresume)
    di_state = getattr(CS, "di_cruise_state", "OFF")
    if di_state != "ENABLED":
      self._reason = "di_" + str(di_state).lower()
      return None

    # Hard braking ahead: bypass the holdoff/spacing gates below (this is a
    # safety cutoff, not a set-speed nudge) and skip calc_button()'s
    # offset-based CANCEL path, which self-corrects too fast to ever trigger.
    if CC.actuators.accel < ACCEL_CANCEL_THRESHOLD:
      self._reason = "hard_brake_cancel"
      return CruiseButtons.CANCEL

    now = _current_time_millis()
    engagement = getattr(CS, "engagement", None)
    last_human_ms = getattr(engagement, "last_stalk_non_cancel_ms", -10000) if engagement else -10000
    if now - last_human_ms < HUMAN_ACTION_HOLDOFF_MS:
      self._reason = "human_holdoff"
      return None
    if now - self.last_auto_action_ms < AUTO_ACTION_SPACING_MS:
      self._reason = "auto_spacing"
      return None

    cc_set_kph = getattr(CS, "v_cruise_actual_kph", 0.0)
    ceiling_kph = getattr(CS, "pedal_speed_kph", 0.0)
    desired_kph = CS.out.vEgo * CV.MS_TO_KPH + float(CC.actuators.accel) * ACCEL_PROJECTION_S * CV.MS_TO_KPH
    self._desired_kph = min(desired_kph, ceiling_kph)
    self._offset_kph = self._desired_kph - cc_set_kph
    return self.calc_button(
      desired_kph=desired_kph,
      cc_set_kph=cc_set_kph,
      ceiling_kph=ceiling_kph,
      v_ego=CS.out.vEgo,
      is_mph=CS.speed_units == "MPH",
    )

  @staticmethod
  def calc_button(desired_kph, cc_set_kph, ceiling_kph, v_ego, is_mph):
    """Pure decision table (tesla-unity _calc_button, modernized)."""
    half_kph, full_kph = get_cc_step_kph(is_mph)
    min_cruise_kph = MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH

    desired_kph = min(desired_kph, ceiling_kph)

    # Below the stock CC floor there is nothing to modulate — drop CC, coast
    if desired_kph < min_cruise_kph:
      return CruiseButtons.CANCEL

    offset_kph = desired_kph - cc_set_kph

    button = None
    if offset_kph < -2 * full_kph and cc_set_kph > 0:
      # Way over: CC off gives the strongest available decel (coast/regen)
      button = CruiseButtons.CANCEL
    elif offset_kph < -0.6 * full_kph:
      button = CruiseButtons.DECEL_2ND
    elif offset_kph < -0.9 * half_kph:
      button = CruiseButtons.DECEL_SET
    elif v_ego > MIN_CRUISE_SPEED_MS:
      # Speed up only with headroom under the driver's ceiling
      if offset_kph >= full_kph and cc_set_kph + full_kph <= ceiling_kph + 0.1:
        button = CruiseButtons.RES_ACCEL_2ND
      elif offset_kph >= half_kph and cc_set_kph + half_kph <= ceiling_kph + 0.1:
        button = CruiseButtons.RES_ACCEL

    # SCCM crash guard (tesla-unity): repeated decel presses at the min cruise
    # speed crash the stalk module — CANCEL instead of stepping below it.
    if CruiseButtons.is_decel(button) and cc_set_kph - half_kph < min_cruise_kph:
      button = CruiseButtons.CANCEL

    return button

  def _log_decision(self, button, CC, CS):
    """Dry-run visibility: log decision changes, not every frame."""
    if button == self.last_logged_button:
      return
    self.last_logged_button = button
    if button is not None:
      carlog.warning(
        "VisionACC would press %s | vEgo=%.1f accel=%+.2f ccSet=%.1fkph ceiling=%.1fkph di=%s longActive=%s",
        _BUTTON_NAMES.get(button, button), CS.out.vEgo, float(CC.actuators.accel),
        getattr(CS, "v_cruise_actual_kph", 0.0), getattr(CS, "pedal_speed_kph", 0.0),
        getattr(CS, "di_cruise_state", "OFF"), CC.longActive)

  def _log_telemetry(self, button, CC, CS, frame):
    """Continuous A/B tuning telemetry — see the module docstring for the schema.

    Throttled: 5 Hz while modulating, 1 Hz otherwise. The pair (aReq, aEgo) is
    the closed-loop tracking record; proj/ltx label the run; reason exposes the
    engagement gating so 'why isn't it engaging' is answerable from the log.
    """
    region = self._reason in _ACTIVE_REASONS
    period = TLM_PERIOD_IN_FRAMES if region else TLM_PERIOD_OUT_FRAMES
    if frame % period != 0:
      return

    try:
      live_tx = int(nap_conf.vision_acc_live_tx) if nap_conf is not None else -1
    except Exception:
      live_tx = -1
    a_ego = float(getattr(CS.out, "aEgo", 0.0))
    desired = self._desired_kph if self._desired_kph is not None else -1.0
    offset = self._offset_kph if self._offset_kph is not None else 0.0

    carlog.info(
      "VisionACC.tlm f=%d proj=%.2f ltx=%d region=%d reason=%s di=%s long=%d gas=%d units=%s "
      "vEgo=%.2f aEgo=%+.2f aReq=%+.2f ccSet=%.1f desired=%.1f ceil=%.1f off=%+.1f btn=%s",
      frame, ACCEL_PROJECTION_S, live_tx, int(region), self._reason,
      getattr(CS, "di_cruise_state", "OFF"), int(CC.longActive), int(CS.out.gasPressed),
      getattr(CS, "speed_units", "?"),
      CS.out.vEgo, a_ego, float(CC.actuators.accel),
      getattr(CS, "v_cruise_actual_kph", 0.0), desired,
      getattr(CS, "pedal_speed_kph", 0.0), offset,
      _BUTTON_NAMES.get(button, "none" if button is None else str(button)))


_BUTTON_NAMES = {
  CruiseButtons.CANCEL: "CANCEL",
  CruiseButtons.RES_ACCEL: "UP_1ST",
  CruiseButtons.RES_ACCEL_2ND: "UP_2ND",
  CruiseButtons.DECEL_SET: "DN_1ST",
  CruiseButtons.DECEL_2ND: "DN_2ND",
}
