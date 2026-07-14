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
"""
import time

from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import CruiseButtons

# Stock CC on Pre-AP Model S only operates above ~17 mph (tesla-unity value)
MIN_CRUISE_SPEED_MS = 17.1 * CV.MPH_TO_MS

# Planner accel is projected this far ahead to form a target speed
ACCEL_PROJECTION_S = 1.5

# Don't fight the driver: no automated press within this window of a human
# stalk action (tesla-unity used 3s)
HUMAN_ACTION_HOLDOFF_MS = 3000
# Spacing between automated presses — the DI needs time to act on each step
# (tesla-unity used 400ms; spoofer TX slots are 100ms apart)
AUTO_ACTION_SPACING_MS = 500


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

  def update(self, CC, CS, frame):
    """Return the CruiseButtons value to spoof this frame, or None."""
    button = self._decide(CC, CS)
    self._log_decision(button, CC, CS)
    if button is not None:
      self.last_auto_action_ms = _current_time_millis()
    return button

  def _decide(self, CC, CS):
    # Engagement FSM must be in long mode (double-pull), openpilot long active
    if not (getattr(CS, "cruiseEnabled", False) and getattr(CS, "enableLongControl", False)):
      return None
    if not CC.longActive:
      return None
    # Driver on the accelerator: the DI holds CC through it; stay out
    if CS.out.gasPressed:
      return None
    # Only modulate a running stock CC — engaging it is StockCCSpoofer's job,
    # and after a CANCEL the driver must double-pull to rearm (no autoresume)
    if getattr(CS, "di_cruise_state", "OFF") != "ENABLED":
      return None

    now = _current_time_millis()
    engagement = getattr(CS, "engagement", None)
    last_human_ms = getattr(engagement, "last_stalk_non_cancel_ms", -10000) if engagement else -10000
    if now - last_human_ms < HUMAN_ACTION_HOLDOFF_MS:
      return None
    if now - self.last_auto_action_ms < AUTO_ACTION_SPACING_MS:
      return None

    desired_kph = CS.out.vEgo * CV.MS_TO_KPH + float(CC.actuators.accel) * ACCEL_PROJECTION_S * CV.MS_TO_KPH
    return self.calc_button(
      desired_kph=desired_kph,
      cc_set_kph=getattr(CS, "v_cruise_actual_kph", 0.0),
      ceiling_kph=getattr(CS, "pedal_speed_kph", 0.0),
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


_BUTTON_NAMES = {
  CruiseButtons.CANCEL: "CANCEL",
  CruiseButtons.RES_ACCEL: "UP_1ST",
  CruiseButtons.RES_ACCEL_2ND: "UP_2ND",
  CruiseButtons.DECEL_SET: "DN_1ST",
  CruiseButtons.DECEL_2ND: "DN_2ND",
}
