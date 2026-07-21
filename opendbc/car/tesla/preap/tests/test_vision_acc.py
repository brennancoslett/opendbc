"""Tests for vision ACC — stock-CC set-speed modulation (no pedal).

Covers:
  - VisionACCController.calc_button decision table (hysteresis, ceiling,
    min-cruise-speed CANCEL, SCCM decel guard, MPH/KPH step units)
  - VisionACCController.update gating (engagement, DI state, gas override,
    human-action holdoff, automated-press spacing)
  - PreAPEngagement vision-ACC mode (ceiling capture on double pull, stalk
    adjustments, spoof echo suppression, brake drop)
  - StockCCSpoofer.request_button (slot cadence, priority, echo stamping)
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from opendbc.car import structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.preap import vision_acc as vacc
from opendbc.car.tesla.preap.engagement import PreAPEngagement, SPOOF_ECHO_WINDOW_MS
from opendbc.car.tesla.preap.stock_cc_spoofer import StockCCSpoofer
from opendbc.car.tesla.preap.vision_acc import (
  VisionACCController, MIN_CRUISE_SPEED_MS, HUMAN_ACTION_HOLDOFF_MS, AUTO_ACTION_SPACING_MS,
  DECEL_DEMAND_THRESHOLD, DECEL_CANCEL_SUSTAIN_S,
)
from opendbc.car.tesla.values import CruiseButtons

MIN_KPH = MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH  # ~27.5 kph
HALF_MPH_KPH = 1 * CV.MPH_TO_KPH   # ~1.6
FULL_MPH_KPH = 5 * CV.MPH_TO_KPH   # ~8.0


def calc(desired, cc_set, ceiling=200.0, v_ego=30.0, is_mph=False):
  return VisionACCController.calc_button(
    desired_kph=desired, cc_set_kph=cc_set, ceiling_kph=ceiling, v_ego=v_ego, is_mph=is_mph)


class TestCalcButtonDecel:

  def test_desired_below_min_cruise_cancels(self):
    assert calc(desired=MIN_KPH - 1, cc_set=50) == CruiseButtons.CANCEL

  def test_large_overspeed_cancels(self):
    # metric full press = 5 kph; offset < -10 → CANCEL
    assert calc(desired=50, cc_set=61) == CruiseButtons.CANCEL

  def test_moderate_overspeed_full_decel(self):
    # offset in (-10, -3): DN_2ND
    assert calc(desired=50, cc_set=55) == CruiseButtons.DECEL_2ND

  def test_small_overspeed_half_decel(self):
    # offset in (-3, -0.9): DN_1ST
    assert calc(desired=50, cc_set=51.5) == CruiseButtons.DECEL_SET

  def test_deadband_no_action(self):
    assert calc(desired=50, cc_set=50) is None
    assert calc(desired=50, cc_set=50.5) is None

  def test_sccm_guard_no_decel_below_min_cruise(self):
    # Set speed sits at the CC floor: a DN press would step below it and
    # can crash the SCCM — must CANCEL instead.
    cc_set = MIN_KPH + 0.5
    assert calc(desired=cc_set - 2, cc_set=cc_set) == CruiseButtons.CANCEL


class TestCalcButtonAccel:

  def test_full_press_speedup_with_headroom(self):
    assert calc(desired=60, cc_set=50) == CruiseButtons.RES_ACCEL_2ND

  def test_half_press_speedup(self):
    assert calc(desired=52, cc_set=50) == CruiseButtons.RES_ACCEL

  def test_no_speedup_at_ceiling(self):
    # desired is capped to the ceiling; cc_set already there → nothing to do
    assert calc(desired=80, cc_set=60, ceiling=60) is None

  def test_full_press_downgraded_when_ceiling_close(self):
    # Wants +6 but only +2 of ceiling headroom: full press would overshoot,
    # half press still fits
    assert calc(desired=56, cc_set=50, ceiling=52) == CruiseButtons.RES_ACCEL

  def test_no_speedup_below_min_cruise_vego(self):
    v_slow = MIN_CRUISE_SPEED_MS - 1
    assert calc(desired=60, cc_set=50, v_ego=v_slow) is None

  def test_mph_units_scale_steps(self):
    # 3 kph offset: for metric that's ≥ half(1)+... → full press territory
    # check imperial: half=1.6, full=8.0 → 3 kph is a half press
    assert calc(desired=53, cc_set=50, is_mph=True) == CruiseButtons.RES_ACCEL
    assert calc(desired=59, cc_set=50, is_mph=True) == CruiseButtons.RES_ACCEL_2ND


def make_cc(*, long_active=True, accel=0.0):
  return SimpleNamespace(longActive=long_active, actuators=SimpleNamespace(accel=accel))


def make_cs(*, cruise_enabled=True, enable_long=True, di_state="ENABLED",
            v_ego=30.0, gas_pressed=False, cc_set_kph=100.0, ceiling_kph=120.0,
            speed_units="KPH", last_human_ms=-100000, a_ego=0.0):
  return SimpleNamespace(
    cruiseEnabled=cruise_enabled,
    enableLongControl=enable_long,
    di_cruise_state=di_state,
    speed_units=speed_units,
    v_cruise_actual_kph=cc_set_kph,
    pedal_speed_kph=ceiling_kph,
    out=SimpleNamespace(vEgo=v_ego, gasPressed=gas_pressed, aEgo=a_ego),
    engagement=SimpleNamespace(last_stalk_non_cancel_ms=last_human_ms),
  )


class TestControllerGating:

  def setup_method(self):
    self.ctrl = VisionACCController()
    self.t_ms = 1_000_000
    vacc._current_time_millis = lambda: self.t_ms

  def teardown_method(self):
    import time
    vacc._current_time_millis = lambda: int(round(time.time() * 1000))

  def test_happy_path_decel(self):
    # 30 m/s = 108 kph, planner wants -1.0 m/s² (above ACCEL_CANCEL_THRESHOLD,
    # so this exercises the normal offset table) → desired ~102.6 (offset
    # -5.4, inside the -3..-10 kph band) → full-detent decel
    cs = make_cs(cc_set_kph=108.0)
    btn = self.ctrl.update(make_cc(accel=-1.0), cs, frame=0)
    assert btn == CruiseButtons.DECEL_2ND

  def test_inactive_fsm_returns_none(self):
    cs = make_cs(enable_long=False)
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=0) is None

  def test_long_not_active_returns_none(self):
    cs = make_cs(cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(long_active=False, accel=-1.0), cs, frame=0) is None

  def test_gas_pressed_returns_none(self):
    cs = make_cs(gas_pressed=True, cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=0) is None

  def test_di_not_enabled_returns_none(self):
    cs = make_cs(di_state="STANDBY", cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=0) is None

  def test_human_action_holdoff(self):
    # accel=-0.5 is a DN_1ST step, above the sustained-decel CANCEL threshold,
    # so this exercises the holdoff→press path (not the cancel path).
    cs = make_cs(cc_set_kph=108.0, last_human_ms=self.t_ms - HUMAN_ACTION_HOLDOFF_MS + 100)
    assert self.ctrl.update(make_cc(accel=-0.5), cs, frame=0) is None
    # Holdoff expired → decision resumes
    self.t_ms += 200
    assert self.ctrl.update(make_cc(accel=-0.5), cs, frame=1) is not None

  def test_automated_press_spacing(self):
    cs = make_cs(cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(accel=-0.5), cs, frame=0) is not None
    # Immediately after: spaced out
    self.t_ms += 100
    assert self.ctrl.update(make_cc(accel=-0.5), cs, frame=1) is None
    self.t_ms += AUTO_ACTION_SPACING_MS
    assert self.ctrl.update(make_cc(accel=-0.5), cs, frame=2) is not None


class TestHardBrakingBypass:
  """ACCEL_CANCEL_THRESHOLD: drive 00000009--2dd6a2315a showed a -1.50 m/s^2
  event still resolve to a half-step DN_1ST because calc_button()'s
  offset_kph self-corrects (each auto-press drags cc_set_kph toward the
  target) before the -2*full_kph CANCEL gap ever opens. These tests cover
  the direct accel bypass added to fix that.
  """

  def setup_method(self):
    self.ctrl = VisionACCController()
    self.t_ms = 1_000_000
    vacc._current_time_millis = lambda: self.t_ms

  def teardown_method(self):
    import time
    vacc._current_time_millis = lambda: int(round(time.time() * 1000))

  def test_hard_braking_cancels_instead_of_stepping(self):
    # Same setup as test_happy_path_decel (offset would be a DECEL_2ND
    # step), but accel is below threshold → CANCEL instead
    cs = make_cs(cc_set_kph=108.0)
    btn = self.ctrl.update(make_cc(accel=-1.5), cs, frame=0)
    assert btn == CruiseButtons.CANCEL

  def test_hard_braking_bypasses_human_holdoff(self):
    cs = make_cs(cc_set_kph=108.0, last_human_ms=self.t_ms - HUMAN_ACTION_HOLDOFF_MS + 100)
    assert self.ctrl.update(make_cc(accel=-1.5), cs, frame=0) == CruiseButtons.CANCEL

  def test_hard_braking_bypasses_action_spacing(self):
    cs = make_cs(cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(accel=-1.5), cs, frame=0) == CruiseButtons.CANCEL
    self.t_ms += 100
    assert self.ctrl.update(make_cc(accel=-1.5), cs, frame=1) == CruiseButtons.CANCEL

  def test_just_above_threshold_does_not_cancel(self):
    cs = make_cs(cc_set_kph=108.0)
    btn = self.ctrl.update(make_cc(accel=-1.29), cs, frame=0)
    assert btn != CruiseButtons.CANCEL


class TestSustainedDecelCancel:
  """Unmet decel: the planner wants to slow but the car isn't (stepped presses
  can't). Sustained past the window it drops CC to regen coast + driver. Gated
  on the aReq-vs-aEgo gap so it stays quiet when the car is actually slowing.
  """

  def setup_method(self):
    self.ctrl = VisionACCController()
    self.t_ms = 1_000_000
    vacc._current_time_millis = lambda: self.t_ms

  def teardown_method(self):
    import time
    vacc._current_time_millis = lambda: int(round(time.time() * 1000))

  def test_sustained_decel_cancels(self):
    cs = make_cs(cc_set_kph=108.0)
    # First frame arms the sustain timer but does not cancel yet (still steps)
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=0) != CruiseButtons.CANCEL
    # Same demand past the sustain window → CANCEL to regen coast
    self.t_ms += int(DECEL_CANCEL_SUSTAIN_S * 1000) + 20
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=1) == CruiseButtons.CANCEL

  def test_brief_decel_dip_does_not_cancel(self):
    cs = make_cs(cc_set_kph=108.0)
    self.ctrl.update(make_cc(accel=-1.0), cs, frame=0)  # arm timer
    self.t_ms += int(DECEL_CANCEL_SUSTAIN_S * 1000) // 2  # well under the window
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=1) != CruiseButtons.CANCEL

  def test_recovery_resets_sustain(self):
    cs = make_cs(cc_set_kph=108.0)
    self.ctrl.update(make_cc(accel=-1.0), cs, frame=0)  # arm timer
    self.t_ms += int(DECEL_CANCEL_SUSTAIN_S * 1000) - 100
    self.ctrl.update(make_cc(accel=-0.2), cs, frame=1)  # demand eases → timer resets
    # New demand's sustain clock restarted at frame 1, so a fresh frame must not cancel
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=2) != CruiseButtons.CANCEL

  def test_mild_decel_never_cancels(self):
    cs = make_cs(cc_set_kph=108.0)
    # Demand above (less negative than) the demand threshold never arms the timer
    for i in range(6):
      assert self.ctrl.update(make_cc(accel=DECEL_DEMAND_THRESHOLD + 0.1), cs, frame=i) != CruiseButtons.CANCEL
      self.t_ms += 300

  def test_decel_tracked_no_cancel(self):
    # Planner wants -1.0 and the car IS decelerating at -1.0 (aEgo tracks aReq):
    # the gap is ~0, so this is not a "steps failing" case — never cancel.
    cs = make_cs(cc_set_kph=108.0, a_ego=-1.0)
    for i in range(6):
      assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=i) != CruiseButtons.CANCEL
      self.t_ms += 300


class TestBrakeHandoffChime:
  """brake_handoff_edge one-shots on the first frame of a braking CANCEL —
  drives the distinct 'you take the brakes' chime.
  """

  def setup_method(self):
    self.ctrl = VisionACCController()
    self.t_ms = 1_000_000
    vacc._current_time_millis = lambda: self.t_ms

  def teardown_method(self):
    import time
    vacc._current_time_millis = lambda: int(round(time.time() * 1000))

  def test_hard_brake_raises_edge(self):
    cs = make_cs(cc_set_kph=108.0)
    assert self.ctrl.update(make_cc(accel=-1.5), cs, frame=0) == CruiseButtons.CANCEL
    assert self.ctrl.brake_handoff_edge is True

  def test_unmet_decel_raises_edge_once(self):
    cs = make_cs(cc_set_kph=108.0)  # aEgo=0 default → unmet
    self.ctrl.update(make_cc(accel=-1.0), cs, frame=0)  # arm, no cancel yet
    assert self.ctrl.brake_handoff_edge is False
    self.t_ms += int(DECEL_CANCEL_SUSTAIN_S * 1000) + 20
    assert self.ctrl.update(make_cc(accel=-1.0), cs, frame=1) == CruiseButtons.CANCEL
    assert self.ctrl.brake_handoff_edge is True   # rising edge
    # still cancelling → no repeat edge
    self.ctrl.update(make_cc(accel=-1.0), cs, frame=2)
    assert self.ctrl.brake_handoff_edge is False

  def test_no_edge_when_not_cancelling(self):
    cs = make_cs(cc_set_kph=108.0, a_ego=-1.0)  # decel tracked → no cancel
    self.ctrl.update(make_cc(accel=-1.0), cs, frame=0)
    assert self.ctrl.brake_handoff_edge is False


def pull_main(eng, t_ms, *, vision_acc=True, v_ego=25.0, di_state="STANDBY"):
  """One MAIN pull (press + release) at t_ms."""
  common = dict(v_ego=v_ego, speed_units="KPH", use_pedal=False,
                pedal_long_allowed=False, long_control_allowed=True,
                real_brake_pressed=False, di_cruise_state=di_state,
                vision_acc=vision_acc)
  eng.process_buttons(CruiseButtons.MAIN, CruiseButtons.IDLE, t_ms, **common)
  eng.process_buttons(CruiseButtons.IDLE, CruiseButtons.MAIN, t_ms + 50, **common)


class TestEngagementVisionACC:

  def make_engagement(self):
    return PreAPEngagement(double_pull_enabled=True, double_pull_window_ms=400)

  def test_double_pull_arms_and_captures_ceiling(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    assert eng.cruiseEnabled and not eng.enableLongControl
    pull_main(eng, 10200)
    assert eng.enableLongControl
    # Ceiling captured from current speed: 25 m/s = 90 kph
    assert eng.pedal_speed_kph == 90.0

  def test_double_pull_fires_cc_engage(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    common = dict(v_ego=25.0, speed_units="KPH", use_pedal=False,
                  pedal_long_allowed=False, long_control_allowed=True,
                  real_brake_pressed=False, di_cruise_state="STANDBY",
                  vision_acc=True)
    eng.process_buttons(CruiseButtons.MAIN, CruiseButtons.IDLE, 10200, **common)
    assert eng.preap_cc_engage_needed

  def test_single_pull_drops_to_lateral_only(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    pull_main(eng, 10200)
    assert eng.enableLongControl
    # A lone pull outside the window: back to lateral-only (ACC off)
    pull_main(eng, 20000)
    assert eng.cruiseEnabled and not eng.enableLongControl and eng.enableJustCC

  def test_stalk_up_press_raises_ceiling(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    pull_main(eng, 10200)
    assert eng.pedal_speed_kph == 90.0
    common = dict(v_ego=25.0, speed_units="KPH", use_pedal=False,
                  pedal_long_allowed=False, long_control_allowed=True,
                  real_brake_pressed=False, di_cruise_state="ENABLED",
                  vision_acc=True)
    eng.process_buttons(CruiseButtons.RES_ACCEL, CruiseButtons.IDLE, 20000, **common)
    assert eng.pedal_speed_kph == 91.0

  def test_spoof_echo_does_not_move_ceiling_or_human_clock(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    pull_main(eng, 10200)
    human_ms_before = eng.last_stalk_non_cancel_ms
    eng.preap_last_speed_spoof_ms = 20000 - (SPOOF_ECHO_WINDOW_MS - 50)
    common = dict(v_ego=25.0, speed_units="KPH", use_pedal=False,
                  pedal_long_allowed=False, long_control_allowed=True,
                  real_brake_pressed=False, di_cruise_state="ENABLED",
                  vision_acc=True)
    events = eng.process_buttons(CruiseButtons.RES_ACCEL, CruiseButtons.IDLE, 20000, **common)
    assert eng.pedal_speed_kph == 90.0
    assert eng.last_stalk_non_cancel_ms == human_ms_before
    assert all(be.type != structs.CarState.ButtonEvent.Type.accelCruise for be in events)

  def test_brake_drops_long_keeps_lateral(self):
    eng = self.make_engagement()
    pull_main(eng, 10000)
    pull_main(eng, 10200)
    assert eng.enableLongControl
    common = dict(v_ego=25.0, speed_units="KPH", use_pedal=False,
                  pedal_long_allowed=False, long_control_allowed=True,
                  real_brake_pressed=True, di_cruise_state="ENABLED",
                  vision_acc=True)
    eng.process_buttons(CruiseButtons.IDLE, CruiseButtons.IDLE, 20000, **common)
    assert eng.cruiseEnabled and not eng.enableLongControl and eng.enableJustCC


class TestSpooferRequestButton:

  def make_can(self):
    m = MagicMock()
    m.create_action_request.return_value = ("STW_ACTN_RQ_FRAME",)
    return m

  def make_cs(self):
    return SimpleNamespace(
      preap_cc_cancel_needed=False,
      preap_cc_engage_needed=False,
      di_cruise_state="ENABLED",
      msg_stw_actn_req={"MC_STW_ACTN_RQ": 3},
      engagement=SimpleNamespace(preap_last_speed_spoof_ms=-10000),
    )

  def test_cancel_rejected(self):
    s = StockCCSpoofer()
    s.request_button(CruiseButtons.CANCEL)
    assert s.requested_button is None

  def test_sends_on_slot_and_stamps_echo(self):
    s = StockCCSpoofer()
    cs = self.make_cs()
    can = self.make_can()
    s.request_button(CruiseButtons.DECEL_SET)
    # off-slot frame: nothing
    assert s.update(cs, 11, can, 2) == []
    assert s.requested_button == CruiseButtons.DECEL_SET
    # slot frame: TX + consume + echo stamp
    sends = s.update(cs, 20, can, 2)
    assert len(sends) == 1
    assert s.requested_button is None
    assert cs.engagement.preap_last_speed_spoof_ms > 0
    can.create_action_request.assert_called_with(CruiseButtons.DECEL_SET, 2, 4, cs.msg_stw_actn_req)

  def test_cancel_pending_beats_and_drops_request(self):
    s = StockCCSpoofer()
    cs = self.make_cs()
    cs.preap_cc_cancel_needed = True
    can = self.make_can()
    s.request_button(CruiseButtons.RES_ACCEL)
    s.update(cs, 0, can, 2)
    # queued speed press dropped while cancel is in flight
    assert s.requested_button is None
    assert s.cancel_pending
