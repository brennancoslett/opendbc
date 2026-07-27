"""Tests for StockCCSpoofer — the stalk-spoof state machine that translates
engagement-FSM intent into CAN frames the DI obeys.

The spoofer:
  - Runs every carcontroller tick (independent of openpilotLongitudinalControl).
  - Reads CS.preap_cc_cancel_needed / preap_cc_engage_needed (set by the FSM).
  - Owns cancel-pending and ENGAGING state machines.
  - TXes one frame per `frame % 10 == 0` slot when active.
  - Exits ENGAGING on DI_cruiseState == 'ENABLED' or after timeout.
  - Emits teslaCC{Engaged,Disengaged} edge events on di_cruise_state changes.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from opendbc.car.tesla.preap.stock_cc_spoofer import (
    StockCCSpoofer, CC_ENGAGE_TIMEOUT_FRAMES, CANCEL_DELAY_FRAMES,
)
from opendbc.car.tesla.values import CruiseButtons


CAN_BUS = 2  # AP party bus, doesn't matter for these tests


def make_cs(*, di_cruise_state="OFF", cancel_needed=False, engage_needed=False,
            msg_stw_present=True):
  msg = {"MC_STW_ACTN_RQ": 3} if msg_stw_present else None
  return SimpleNamespace(
    preap_cc_cancel_needed=cancel_needed,
    preap_cc_engage_needed=engage_needed,
    di_cruise_state=di_cruise_state,
    msg_stw_actn_req=msg,
  )


def make_can():
  m = MagicMock()
  m.create_action_request.return_value = ("STW_ACTN_RQ_FRAME",)
  return m


def tick_until(spoofer, cs, can, can_bus, start_frame, n_frames):
  """Tick n_frames consecutive frames, return list of (frame, sends)."""
  history = []
  for i in range(n_frames):
    frame = start_frame + i
    sends = spoofer.update(cs, frame, can, can_bus)
    history.append((frame, sends))
  return history


# ---- Cancel path ----

class TestCancelPath:

  def test_cancel_pending_set_on_flag(self):
    s = StockCCSpoofer()
    cs = make_cs(cancel_needed=True)
    s.update(cs, frame=0, tesla_can=make_can(), can_bus_party=CAN_BUS)
    assert s.cancel_pending is True
    # Flag consumed, not lingering
    assert cs.preap_cc_cancel_needed is False

  def test_cancel_respects_delay_then_fires(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(cancel_needed=True)
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_cancel_needed = False
    # Strictly before CANCEL_DELAY_FRAMES: no TX even on a frame%10==0 slot
    history = tick_until(s, cs, can, CAN_BUS, start_frame=1,
                         n_frames=CANCEL_DELAY_FRAMES - 1)
    sends_before = sum(len(sends) for _, sends in history)
    assert sends_before == 0, "Should not TX before CANCEL_DELAY_FRAMES"
    # At/past delay AND on a frame%10==0 slot: TX
    history = tick_until(s, cs, can, CAN_BUS,
                         start_frame=CANCEL_DELAY_FRAMES, n_frames=15)
    fired_frames = [f for f, sends in history if sends]
    assert fired_frames, "Cancel should fire at/after delay on a frame%10==0 slot"
    assert fired_frames[0] >= CANCEL_DELAY_FRAMES
    can.create_action_request.assert_called_with(
      CruiseButtons.CANCEL, CAN_BUS, 4, cs.msg_stw_actn_req)
    assert s.cancel_pending is False

  def test_cancel_no_msg_stw_no_tx(self):
    s = StockCCSpoofer()
    cs = make_cs(cancel_needed=True, msg_stw_present=False)
    can = make_can()
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_cancel_needed = False
    history = tick_until(s, cs, can, CAN_BUS, start_frame=11, n_frames=15)
    sends_total = sum(len(s) for _, s in history)
    assert sends_total == 0, "No msg_stw → no TX, no crash"

  def test_continuous_cancel_request_still_fires(self):
    # No-pedal ACC asserts preap_cc_cancel_needed EVERY frame through a sustained
    # decel CANCEL. cancel_frame must latch on the rising edge, not re-stamp each
    # frame — otherwise (frame - cancel_frame) stays 0, cancel_ready never trips,
    # and the CANCEL is never transmitted (regression: drive
    # 00000003--a061fe2143, ~1.8 s of CANCEL commanded, DI never dropped).
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(cancel_needed=True, di_cruise_state="ENABLED")
    fired = []
    for frame in range(40):
      cs.preap_cc_cancel_needed = True  # re-asserted every frame
      if s.update(cs, frame, can, CAN_BUS):
        fired.append(frame)
    assert fired, "Continuous cancel request must still transmit CANCEL"
    assert CANCEL_DELAY_FRAMES <= fired[0] < 20, "CANCEL must fire promptly, not defer forever"
    can.create_action_request.assert_called_with(
      CruiseButtons.CANCEL, CAN_BUS, 4, cs.msg_stw_actn_req)


# ---- Engage path ----

class TestEngagePath:

  def test_engage_enters_engaging_phase(self):
    s = StockCCSpoofer()
    cs = make_cs(engage_needed=True, di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=make_can(), can_bus_party=CAN_BUS)
    assert s.cc_engage_phase != 0  # not IDLE
    assert cs.preap_cc_engage_needed is False

  def test_engage_sends_set_accel_until_di_enabled(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(engage_needed=True, di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_engage_needed = False
    # Tick frames; SET_ACCEL fires on frame%10==0 while STANDBY
    history = tick_until(s, cs, can, CAN_BUS, start_frame=1, n_frames=20)
    fired = [f for f, sends in history if sends]
    assert len(fired) >= 1, "Should send SET_ACCEL while ENGAGING"
    can.create_action_request.assert_called_with(
      CruiseButtons.SET_ACCEL, CAN_BUS, 4, cs.msg_stw_actn_req)

  def test_engage_exits_on_di_enabled(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(engage_needed=True, di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_engage_needed = False
    # DI transitions to ENABLED
    cs.di_cruise_state = "ENABLED"
    s.update(cs, frame=10, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cc_engage_phase == 0  # IDLE
    # No further TX
    can.reset_mock()
    tick_until(s, cs, can, CAN_BUS, start_frame=20, n_frames=30)
    assert can.create_action_request.call_count == 0

  def test_engage_timeout(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(engage_needed=True, di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_engage_needed = False
    # Tick past the timeout; DI never reaches ENABLED
    tick_until(s, cs, can, CAN_BUS, start_frame=1, n_frames=CC_ENGAGE_TIMEOUT_FRAMES + 5)
    assert s.cc_engage_phase == 0  # IDLE after timeout


# ---- Cancel-vs-engage precedence ----

class TestPrecedence:

  def test_cancel_overrides_engaging(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(engage_needed=True, di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_engage_needed = False
    assert s.cc_engage_phase != 0
    # Now cancel arrives mid-engage
    cs.preap_cc_cancel_needed = True
    s.update(cs, frame=5, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cancel_pending is True
    assert s.cc_engage_phase == 0  # ENGAGING aborted


# ---- pcc_event edges ----

class TestPccEvent:

  def test_engaged_edge(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(di_cruise_state="STANDBY")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event is None, "no edge while DI stays in STANDBY"
    cs.di_cruise_state = "ENABLED"
    s.update(cs, frame=1, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event == "teslaCCEngaged"
    # Same state next frame: edge cleared
    s.update(cs, frame=2, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event is None

  def test_disengaged_edge(self):
    s = StockCCSpoofer()
    can = make_can()
    # Bring spoofer up to "engaged" steady state first (mirrors the boot
    # sequence: car starts, observes ENABLED, fires the rising edge once).
    cs = make_cs(di_cruise_state="ENABLED")
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event == "teslaCCEngaged"  # rising edge from initial False
    s.update(cs, frame=1, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event is None
    # Now the actual transition under test
    cs.di_cruise_state = "STANDBY"
    s.update(cs, frame=2, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.pcc_event == "teslaCCDisengaged"


# ---- Counter math ----

class TestCounterMath:

  def test_counter_increments_modulo_16(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs(cancel_needed=True)
    cs.msg_stw_actn_req = {"MC_STW_ACTN_RQ": 15}
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    cs.preap_cc_cancel_needed = False
    tick_until(s, cs, can, CAN_BUS, start_frame=11, n_frames=15)
    # Counter = (15+1) % 16 = 0
    last_call = can.create_action_request.call_args_list[-1]
    assert last_call.args[2] == 0


# ---- Stale-resume normalization ----

def make_cs_renorm(*, di_cruise_state, v_set_kph, v_ego_ms=11.2):
  """CS for renorm tests: 11.2 m/s ~= 40.3 kph ~= 25 mph."""
  cs = make_cs(di_cruise_state=di_cruise_state)
  cs.v_cruise_actual_kph = v_set_kph
  cs.out = SimpleNamespace(vEgo=v_ego_ms)
  cs.engagement = SimpleNamespace(preap_last_cc_spoof_ms=0, preap_last_speed_spoof_ms=-10000)
  return cs


def buttons_sent(can):
  return [c.args[0] for c in can.create_action_request.call_args_list]


class TestStaleResumeRenorm:
  """The physical stalk pull RESUMES the stored set speed on the pre-AP DI,
  so a double-pull after an earlier faster cruise leaves the CC chasing a
  stale number (2026-07-14 drive 3: pull at 25 mph resumed 48 mph). ENGAGING
  must cancel and re-set with a down-detent instead of accepting it."""

  def test_fresh_set_exits_engaging_without_renorm(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs_renorm(di_cruise_state="ENABLED", v_set_kph=40.0)  # ~vEgo, fresh
    cs.preap_cc_engage_needed = True
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    s.update(cs, frame=10, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cc_engage_phase == 0  # IDLE
    assert buttons_sent(can) == []

  def test_stale_resume_cancels_then_sets_with_down_detent(self):
    s = StockCCSpoofer()
    can = make_can()
    # DI resumed a stale 77 kph (48 mph) while doing 40 kph
    cs = make_cs_renorm(di_cruise_state="ENABLED", v_set_kph=77.0)
    cs.preap_cc_engage_needed = True
    # Frame 0 is a TX slot: ENGAGING sees ENABLED+stale immediately -> CANCEL
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)
    assert buttons_sent(can) == [CruiseButtons.CANCEL]
    assert cs.engagement.preap_last_cc_spoof_ms > 0, "cancel echo must be stamped"
    # CAN latency: DI still reads ENABLED — dwell, no TX, no phase change
    s.update(cs, frame=10, tesla_can=can, can_bus_party=CAN_BUS)
    assert buttons_sent(can) == [CruiseButtons.CANCEL]
    # DI drops to STANDBY: DN (set at current speed), speed echo stamped
    cs.di_cruise_state = "STANDBY"
    s.update(cs, frame=20, tesla_can=can, can_bus_party=CAN_BUS)
    assert buttons_sent(can) == [CruiseButtons.CANCEL, CruiseButtons.DECEL_SET]
    assert cs.engagement.preap_last_speed_spoof_ms > 0, "DN echo must be stamped"
    # DI enables at the fresh speed: done
    cs.di_cruise_state = "ENABLED"
    cs.v_cruise_actual_kph = 40.0
    s.update(cs, frame=30, tesla_can=can, can_bus_party=CAN_BUS)  # RENORM -> ENGAGING
    s.update(cs, frame=40, tesla_can=can, can_bus_party=CAN_BUS)  # fresh -> IDLE
    assert s.cc_engage_phase == 0
    assert buttons_sent(can) == [CruiseButtons.CANCEL, CruiseButtons.DECEL_SET]

  def test_persistent_stale_set_gives_up_after_max_rounds(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs_renorm(di_cruise_state="ENABLED", v_set_kph=77.0)
    cs.preap_cc_engage_needed = True
    # Round 1: cancel at frame 0 (ENABLED+stale on the entry slot)
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)    # CANCEL
    cs.di_cruise_state = "STANDBY"
    s.update(cs, frame=10, tesla_can=can, can_bus_party=CAN_BUS)   # DECEL_SET
    cs.di_cruise_state = "ENABLED"                                  # resumed stale again
    s.update(cs, frame=20, tesla_can=can, can_bus_party=CAN_BUS)   # RENORM -> ENGAGING
    # Round 2
    s.update(cs, frame=30, tesla_can=can, can_bus_party=CAN_BUS)   # CANCEL
    cs.di_cruise_state = "STANDBY"
    s.update(cs, frame=40, tesla_can=can, can_bus_party=CAN_BUS)   # DECEL_SET
    cs.di_cruise_state = "ENABLED"
    s.update(cs, frame=50, tesla_can=can, can_bus_party=CAN_BUS)   # RENORM -> ENGAGING
    # Rounds exhausted: give up, no further presses
    s.update(cs, frame=60, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cc_engage_phase == 0
    assert buttons_sent(can) == [CruiseButtons.CANCEL, CruiseButtons.DECEL_SET,
                                 CruiseButtons.CANCEL, CruiseButtons.DECEL_SET]

  def test_renorm_timeout_fires(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs_renorm(di_cruise_state="ENABLED", v_set_kph=77.0)
    cs.preap_cc_engage_needed = True
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)    # CANCEL -> RENORM
    # DI reaches STANDBY but ignores every DN press: overall timeout ends it
    cs.di_cruise_state = "STANDBY"
    for f in range(10, 200, 10):
      s.update(cs, frame=f, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cc_engage_phase == 0

  def test_fsm_cancel_aborts_renorm(self):
    s = StockCCSpoofer()
    can = make_can()
    cs = make_cs_renorm(di_cruise_state="ENABLED", v_set_kph=77.0)
    cs.preap_cc_engage_needed = True
    s.update(cs, frame=0, tesla_can=can, can_bus_party=CAN_BUS)    # CANCEL -> RENORM
    cs.preap_cc_cancel_needed = True
    s.update(cs, frame=1, tesla_can=can, can_bus_party=CAN_BUS)
    assert s.cc_engage_phase == 0
    assert s.cancel_pending
