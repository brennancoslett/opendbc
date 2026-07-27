"""Stock-CC stalk spoofer.

Translates engagement-FSM stalk intent (CS.preap_cc_cancel_needed,
CS.preap_cc_engage_needed) into 0x45 STW_ACTN_RQ CAN frames the Tesla DI obeys.
Owns the cancel-pending and ENGAGING state machines, the DI-cruiseState edge
events for teslaCCEngaged/Disengaged, and the cadence/timing of spoof TX.

Independent of pedal mode: runs every carcontroller tick. The pedal-control
side and this module communicate exclusively through CarState flags so
neither has to reach into the other's state.
"""
import time

from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import CruiseButtons


def _current_time_millis():
  return int(round(time.time() * 1000))


# Phases
_PHASE_IDLE = 0
_PHASE_ENGAGING = 1
_PHASE_RENORM = 2

# Delay between cancel-pending becoming true and CANCEL hitting the bus.
# Lets pedal mode (when engaging) establish control before the DI drops.
CANCEL_DELAY_FRAMES = 10  # 100 ms at 100 Hz

# Max time to keep retrying SET_ACCEL before giving up. The DI may stay in
# STANDBY (e.g. below min cruise speed) and never accept the engage.
CC_ENGAGE_TIMEOUT_FRAMES = 50  # 500 ms at 100 Hz

# Stale-resume normalization. The physical stalk pull is a RESUME gesture on
# the pre-AP DI: it re-enables the CC at the STORED set speed before any spoof
# lands, so a double-pull after an earlier faster cruise leaves the DI chasing
# the old number (observed 2026-07-14 drive 3: pull at 25 mph resumed a stale
# 48 mph set toward a lead). When ENGAGING finds the DI ENABLED with a set
# speed far from the current speed, cancel and re-set with a down-detent
# (up/down from STANDBY sets at current speed per the owner's manual; the
# pull-toward gesture is the only resume). Fails safe: if the DN press is
# ignored the overall timeout fires and the CC stays off.
STALE_SET_TOLERANCE_KPH = 8.1   # one full stalk step (5 mph); a fresh engage lands within a half step
RENORM_MAX_ROUNDS = 2
CC_RENORM_TIMEOUT_FRAMES = 150  # 1.5 s overall once renormalization starts


class StockCCSpoofer:
  """Stalk-spoof state machine. One TX slot per `frame % 10 == 0`."""

  def __init__(self):
    self.cc_engage_phase = _PHASE_IDLE
    self.cc_engage_start_frame = 0
    self.cancel_pending = False
    self.cancel_frame = -1_000_000
    self.prev_di_cc_engaged = False
    self.pcc_event = None
    self.renorm_rounds = 0
    self.renorm_seen_standby = False
    # No-pedal ACC speed-press request; one-shot, consumed at the next TX slot
    self.requested_button = None

  def request_button(self, button):
    """Queue a no-pedal ACC speed press (UP/DN 1ST/2ND) for the next TX slot.

    CANCEL must not come through here — route it via CS.preap_cc_cancel_needed
    so it gets the cancel-pending machinery and echo filtering.
    """
    if button == CruiseButtons.CANCEL:
      carlog.error("StockCC: CANCEL routed via request_button — dropped; use preap_cc_cancel_needed")
      return
    self.requested_button = button

  def update(self, CS, frame, tesla_can, can_bus_party):
    can_sends = []

    # --- Bridge engagement-FSM intent to internal state ---
    if getattr(CS, "preap_cc_cancel_needed", False):
      # Latch cancel_frame on the RISING edge only. No-pedal ACC asserts
      # preap_cc_cancel_needed every frame through a sustained decel CANCEL;
      # re-stamping cancel_frame each frame kept (frame - cancel_frame) at 0, so
      # cancel_ready (>= CANCEL_DELAY_FRAMES) never became true and the CANCEL
      # was never sent — drive 00000003--a061fe2143 (f5540+) commanded CANCEL
      # for ~1.8 s with the DI still ENABLED until the driver intervened. The
      # old one-shot engagement-FSM cancel set the flag for a single frame and
      # so never hit this. Latching from the first request keeps the ~100 ms
      # pedal-handoff delay while still firing; after each send cancel_pending
      # clears and a still-asserted request re-arms, re-sending every ~100 ms
      # until the DI drops.
      if not self.cancel_pending:
        self.cancel_frame = frame
      self.cancel_pending = True
      # Cancel beats engage: abort any in-flight ENGAGING.
      self.cc_engage_phase = _PHASE_IDLE
      CS.preap_cc_cancel_needed = False

    if getattr(CS, "preap_cc_engage_needed", False) and self.cc_engage_phase == _PHASE_IDLE:
      self.cc_engage_phase = _PHASE_ENGAGING
      self.cc_engage_start_frame = frame
      self.renorm_rounds = 0
      carlog.debug("StockCC: ENGAGING (di=%s)", getattr(CS, "di_cruise_state", "OFF"))
      CS.preap_cc_engage_needed = False

    # --- TX one frame per 10ms slot (frame % 10 == 0) ---
    cancel_ready = (frame - self.cancel_frame) >= CANCEL_DELAY_FRAMES
    timeout_limit = CC_ENGAGE_TIMEOUT_FRAMES if self.renorm_rounds == 0 else CC_RENORM_TIMEOUT_FRAMES
    if self.cancel_pending and cancel_ready and frame % 10 == 0:
      sent = self._send(CS, tesla_can, can_bus_party, CruiseButtons.CANCEL)
      if sent is not None:
        can_sends.append(sent)
        self.cancel_pending = False
    elif self.cc_engage_phase == _PHASE_ENGAGING and frame % 10 == 0:
      di_state = getattr(CS, "di_cruise_state", "OFF")
      timed_out = (frame - self.cc_engage_start_frame) >= timeout_limit
      if timed_out:
        carlog.warning("StockCC: engage timeout after %d frames",
                       frame - self.cc_engage_start_frame)
        self.cc_engage_phase = _PHASE_IDLE
      elif di_state == "ENABLED":
        if self._set_speed_stale(CS):
          if self.renorm_rounds >= RENORM_MAX_ROUNDS:
            carlog.warning("StockCC: stale set persists after %d renorm rounds — giving up",
                           self.renorm_rounds)
            self.cc_engage_phase = _PHASE_IDLE
          else:
            self.renorm_rounds += 1
            carlog.warning("StockCC: stale resume (set=%.1fkph vEgo=%.1fkph) — cancel + re-set, round %d",
                           getattr(CS, "v_cruise_actual_kph", 0.0), CS.out.vEgo * CV.MS_TO_KPH,
                           self.renorm_rounds)
            sent = self._send(CS, tesla_can, can_bus_party, CruiseButtons.CANCEL)
            if sent is not None:
              can_sends.append(sent)
              self._stamp_spoof(CS)
              self.renorm_seen_standby = False
              self.cc_engage_phase = _PHASE_RENORM
        else:
          carlog.debug("StockCC: ENABLED — exiting ENGAGING")
          self.cc_engage_phase = _PHASE_IDLE
      else:
        sent = self._send(CS, tesla_can, can_bus_party, CruiseButtons.SET_ACCEL)
        if sent is not None:
          can_sends.append(sent)
    elif self.cc_engage_phase == _PHASE_RENORM and frame % 10 == 0:
      di_state = getattr(CS, "di_cruise_state", "OFF")
      if (frame - self.cc_engage_start_frame) >= timeout_limit:
        carlog.warning("StockCC: renorm timeout after %d frames",
                       frame - self.cc_engage_start_frame)
        self.cc_engage_phase = _PHASE_IDLE
      elif not self.renorm_seen_standby:
        # Our cancel is in flight; ENABLED here just means the DI hasn't
        # processed it yet (~100 ms CAN latency) — wait for STANDBY before
        # trusting any ENABLED as the DN landing.
        if di_state == "STANDBY":
          self.renorm_seen_standby = True
          # Down-detent from STANDBY sets the CC at the current speed
          sent = self._send(CS, tesla_can, can_bus_party, CruiseButtons.DECEL_SET)
          if sent is not None:
            can_sends.append(sent)
            self._stamp_spoof(CS, speed=True)
      elif di_state == "ENABLED":
        # DN landed (or the DI resumed again) — ENGAGING re-checks staleness
        self.cc_engage_phase = _PHASE_ENGAGING
      elif di_state == "STANDBY":
        # DN not accepted yet — retry each slot until the timeout caps it
        sent = self._send(CS, tesla_can, can_bus_party, CruiseButtons.DECEL_SET)
        if sent is not None:
          can_sends.append(sent)
          self._stamp_spoof(CS, speed=True)
    elif self.requested_button is not None and not self.cancel_pending and frame % 10 == 0:
      # No-pedal ACC speed press — lowest priority: cancel and engage always win
      sent = self._send(CS, tesla_can, can_bus_party, self.requested_button)
      if sent is not None:
        can_sends.append(sent)
        # Stamp the FSM so the RX echo of this frame isn't read as a human press
        engagement = getattr(CS, "engagement", None)
        if engagement is not None:
          engagement.preap_last_speed_spoof_ms = _current_time_millis()
      self.requested_button = None

    # A queued speed press is stale after its slot chance passes with a
    # cancel/engage in flight — drop it rather than fire it late.
    if self.cancel_pending or self.cc_engage_phase != _PHASE_IDLE:
      self.requested_button = None

    # --- Edge events for teslaCCEngaged / teslaCCDisengaged ---
    di_cc_engaged = getattr(CS, "di_cruise_state", "OFF") == "ENABLED"
    if di_cc_engaged and not self.prev_di_cc_engaged:
      self.pcc_event = "teslaCCEngaged"
    elif not di_cc_engaged and self.prev_di_cc_engaged:
      self.pcc_event = "teslaCCDisengaged"
    else:
      self.pcc_event = None
    self.prev_di_cc_engaged = di_cc_engaged

    return can_sends

  def _set_speed_stale(self, CS):
    """DI ENABLED at a set speed far from the current speed = stale resume."""
    v_set = getattr(CS, "v_cruise_actual_kph", 0.0)
    if v_set <= 0:
      return False
    return abs(v_set - CS.out.vEgo * CV.MS_TO_KPH) > STALE_SET_TOLERANCE_KPH

  def _stamp_spoof(self, CS, speed=False):
    """Stamp the FSM echo windows so our own frames aren't read as human presses."""
    engagement = getattr(CS, "engagement", None)
    if engagement is None:
      return
    now = _current_time_millis()
    engagement.preap_last_cc_spoof_ms = now
    if speed:
      engagement.preap_last_speed_spoof_ms = now

  def _send(self, CS, tesla_can, can_bus_party, button):
    msg_stw = getattr(CS, "msg_stw_actn_req", None)
    if msg_stw is None:
      return None
    counter = (int(msg_stw.get("MC_STW_ACTN_RQ", 0)) + 1) % 16
    return tesla_can.create_action_request(button, can_bus_party, counter, msg_stw)
