"""Tests for no-pedal ACC — stock-CC set-speed modulation.

Covers:
  - NoPedalACCController.calc_button decision table (hysteresis, ceiling,
    min-cruise-speed CANCEL, SCCM decel guard, MPH/KPH step units)
  - NoPedalACCController.update gating (engagement, DI state, gas override,
    human-action holdoff, automated-press spacing)
  - PreAPEngagement no-pedal ACC mode (ceiling capture on double pull, stalk
    adjustments, spoof echo suppression, brake drop)
  - StockCCSpoofer.request_button (slot cadence, priority, echo stamping)
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from opendbc.car import structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.preap import no_pedal_acc as vacc
from opendbc.car.tesla.preap.engagement import PreAPEngagement, SPOOF_ECHO_WINDOW_MS
from opendbc.car.tesla.preap.stock_cc_spoofer import StockCCSpoofer
from opendbc.car.tesla.preap.no_pedal_acc import (
  NoPedalACCController, MIN_CRUISE_SPEED_MS, HUMAN_ACTION_HOLDOFF_MS, AUTO_ACTION_SPACING_MS,
  DECEL_CANCEL_HUMAN_SETTLE_MS, FLOOR_CLIMB_MAX_INTERVAL_MS,
  DECEL_DEMAND_THRESHOLD, DECEL_CANCEL_SUSTAIN_S,
)
from opendbc.car.tesla.values import CruiseButtons

MIN_KPH = MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH  # ~27.5 kph
HALF_MPH_KPH = 1 * CV.MPH_TO_KPH   # ~1.6
FULL_MPH_KPH = 5 * CV.MPH_TO_KPH   # ~8.0


def calc(desired, cc_set, ceiling=200.0, v_ego=30.0, is_mph=False):
  return NoPedalACCController.calc_button(
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


def make_cc(*, long_active=True, accel=0.0, plan_speed=0.0):
  # plan_speed is CC.planSpeedTarget in m/s; capnp default 0.0 = no plan
  return SimpleNamespace(longActive=long_active, actuators=SimpleNamespace(accel=accel),
                         planSpeedTarget=plan_speed)


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
    self.ctrl = NoPedalACCController()
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

  def test_mild_decel_never_presses_up(self):
    # Regression, drive 00000018: gating the ceiling-climb floor at the decel
    # demand threshold let a planner asking for mild decel push the set speed
    # UP, cancelling out the down-presses while closing on a lead. Any decel
    # demand at all must leave the target where the projection put it.
    cs = make_cs(v_ego=19.71, cc_set_kph=70.8, ceiling_kph=74.0, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=-0.12), cs, frame=0)
    assert not CruiseButtons.is_accel(btn)

  def test_set_speed_above_actual_walks_down_on_mild_decel(self):
    # That drive's approach shape: 82.1 kph set against 78.5 kph actual, planner
    # asking -0.16, DI still pulling toward the set speed. The projection is
    # anchored on vEgo, so the set speed comes down — the only mechanism here
    # that lowers it.
    cs = make_cs(v_ego=21.8, cc_set_kph=82.1, ceiling_kph=82.1, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=-0.16), cs, frame=0)
    assert btn == CruiseButtons.DECEL_SET

  def test_real_decel_demand_still_steps_down(self):
    # The other side of that gate: once the planner is meaningfully braking the
    # floor must not apply, or it would fight a genuine slowdown.
    cs = make_cs(v_ego=19.71, cc_set_kph=70.8, ceiling_kph=74.0, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=-0.5), cs, frame=0)
    assert btn == CruiseButtons.DECEL_SET

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

  def test_reaches_ceiling_on_mild_accel(self):
    # Near the max: set speed ~2 mph below the ceiling, planner wants a mild
    # positive accel. The projected target alone undershoots (offset < 1 step),
    # so without the ceiling floor it would stall — must produce an UP press.
    cs = make_cs(cc_set_kph=66.0, ceiling_kph=69.2, v_ego=18.1, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=0.22), cs, frame=0)
    assert btn in (CruiseButtons.RES_ACCEL, CruiseButtons.RES_ACCEL_2ND)

  def test_settled_behind_lead_does_not_ratchet_up(self):
    # Regression, drive 0000002e 18:14:52.1 (bookmarked): settled behind a lead
    # after a correct 11-press slowdown, vEgo 16.66 m/s (60.0 kph), set 61.2,
    # ceiling 80.5, planner at +0.01 — content, not asking to accelerate. The
    # projection lands at 60.0, BELOW the set, so the floor must not fire.
    # It used to, and the resulting ratchet ran the set to 72.4 in 3.6 s.
    cs = make_cs(v_ego=16.66, cc_set_kph=61.2, ceiling_kph=80.5, speed_units="MPH")
    assert self.ctrl.update(make_cc(accel=0.01), cs, frame=0) is None

  def test_set_above_actual_with_positive_demand_does_not_slam_down(self):
    # Regression, drive 0000002e 18:39:06.4: still catching up to the set speed
    # (vEgo 63.7 kph, set 72.4, ceiling 72.4) with the planner asking +0.30.
    # The projection lands at 65.3, so the raw offset is -7.1 kph and the
    # decision table answered DECEL_2ND — an unrequested full 5 mph down-step
    # while the planner wanted to accelerate. Holding the target at the set
    # speed must produce no press at all.
    cs = make_cs(v_ego=17.7, a_ego=0.3, cc_set_kph=72.4, ceiling_kph=72.4, speed_units="MPH")
    assert self.ctrl.update(make_cc(accel=0.30), cs, frame=0) is None

  def test_hold_deadlock_escape_climbs_the_last_step(self):
    # Regression, drive 0000004f (2026-08-04): 922 tlm frames parked 1-2 mph
    # under the ceiling, longest run 66 s. The DI's bang-bang dead zone parks
    # vEgo ~1.5 kph under the set (median +1.52 that day), from where the
    # projection needs aReq >= ~+0.28 to reach the set while the planner asks a
    # median +0.16 — so the hold branch held forever, one step under MAX. The
    # planner's terminal speed (median vEgo +2.7 kph in the stall windows) says
    # it wants more, and must drive one rate-limited climb.
    cs = make_cs(v_ego=17.0, cc_set_kph=62.7, ceiling_kph=64.4, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=0.16, plan_speed=17.75), cs, frame=0)
    assert btn == CruiseButtons.RES_ACCEL

  def test_hold_deadlock_escape_is_rate_limited(self):
    # Same frame again inside the floor interval (~2.8 s at aReq +0.16): the
    # escape shares the floor's rate limiter, so no second press yet.
    kw = dict(v_ego=17.0, cc_set_kph=62.7, ceiling_kph=64.4, speed_units="MPH")
    cc = make_cc(accel=0.16, plan_speed=17.75)
    assert self.ctrl.update(cc, make_cs(**kw), frame=0) == CruiseButtons.RES_ACCEL
    self.t_ms += AUTO_ACTION_SPACING_MS + 400
    assert self.ctrl.update(cc, make_cs(**kw), frame=1) is None
    self.t_ms += 2500
    assert self.ctrl.update(cc, make_cs(**kw), frame=2) == CruiseButtons.RES_ACCEL

  def test_hold_without_plan_speed_stays_a_hold(self):
    # planSpeedTarget unset (0.0, the capnp default): the escape must not fire
    # and the branch behaves exactly as before — old replays and any caller
    # that doesn't populate the field keep today's behavior.
    cs = make_cs(v_ego=17.0, cc_set_kph=62.7, ceiling_kph=64.4, speed_units="MPH")
    assert self.ctrl.update(make_cc(accel=0.16), cs, frame=0) is None

  def test_settled_behind_lead_does_not_escape_the_hold(self):
    # The 0000002e 18:14:52 ratchet frame with a realistic plan: settled behind
    # a lead the MPC's terminal speed sits AT vEgo (measured lead-source median
    # -1.2 kph on 0000004f), so the escape's wants-more gate must block and the
    # hold must hold — this is the failure mode the hold branch exists for.
    cs = make_cs(v_ego=16.66, cc_set_kph=61.2, ceiling_kph=80.5, speed_units="MPH")
    assert self.ctrl.update(make_cc(accel=0.01, plan_speed=16.7), cs, frame=0) is None

  def test_catching_up_does_not_escape_the_hold(self):
    # The 0000002e 18:39:06 catching-up frame with the plan wanting more: the
    # set already sits 8.7 kph over vEgo (far outside the DI's dead zone), the
    # car is still accelerating toward it, and climbing would pile on. The
    # DI-idle gate must block the escape.
    cs = make_cs(v_ego=17.7, a_ego=0.3, cc_set_kph=72.4, ceiling_kph=72.4, speed_units="MPH")
    assert self.ctrl.update(make_cc(accel=0.30, plan_speed=18.45), cs, frame=0) is None

  def test_settled_behind_lead_still_walks_down_when_planner_brakes(self):
    # The approach into that same event (18:14:43.2), where the controller was
    # doing the right thing: planner at -0.65 with the car tracking it, set 72.4
    # against 74.2 kph actual. The projection lands 1.75 kph under the set, so
    # the down-step must still fire — the new floor gate must not suppress it.
    cs = make_cs(v_ego=20.6, a_ego=-0.6, cc_set_kph=72.4, ceiling_kph=80.5, speed_units="MPH")
    btn = self.ctrl.update(make_cc(accel=-0.65), cs, frame=0)
    assert btn == CruiseButtons.DECEL_SET

  def test_no_unmet_decel_cancel_right_after_driver_lowers_set_speed(self):
    # Regression, drive 0000002e 18:39:12: the driver dropped the ceiling a full
    # detent (72.4 -> 64.4 kph) at 43.7 mph. The planner's target fell with it,
    # so a decel demand existed before the DI had been given any down-press, and
    # the unmet-decel arm CANCELed 1.8 s later — handing the brakes over for a
    # slowdown the driver had just asked for. Regen coast then took the car from
    # 43.7 mph toward 25.
    unmet = dict(v_ego=19.5, a_ego=0.06, cc_set_kph=62.8, ceiling_kph=64.4, speed_units="MPH")
    for elapsed in (0, 1000, 1800, DECEL_CANCEL_HUMAN_SETTLE_MS - 100):
      self.ctrl = NoPedalACCController()
      cs = make_cs(last_human_ms=self.t_ms - elapsed, **unmet)
      # hold the demand well past DECEL_CANCEL_SUSTAIN_S
      for frame in range(3):
        btn = self.ctrl.update(make_cc(accel=-0.44), cs, frame=frame)
        self.t_ms += 400
        cs = make_cs(last_human_ms=self.t_ms - elapsed - 400 * (frame + 1), **unmet)
      assert btn != CruiseButtons.CANCEL, f"cancelled {elapsed} ms after driver input"

  def test_unmet_decel_cancel_still_fires_once_settled(self):
    # The other side: with no recent driver input the hand-off must still work,
    # or the settle window would have disabled it outright.
    cs = make_cs(v_ego=19.5, a_ego=0.06, cc_set_kph=62.8, ceiling_kph=64.4,
                 speed_units="MPH", last_human_ms=-100000)
    btn = None
    for frame in range(3):
      btn = self.ctrl.update(make_cc(accel=-0.44), cs, frame=frame)
      self.t_ms += 400
    assert btn == CruiseButtons.CANCEL

  def test_hard_braking_ignores_the_settle_window(self):
    # A real emergency must hand off instantly even mid-settle: the
    # ACCEL_CANCEL_THRESHOLD path returns before the unmet-decel arm.
    cs = make_cs(cc_set_kph=108.0, last_human_ms=self.t_ms - 100)
    assert self.ctrl.update(make_cc(accel=-2.0), cs, frame=0) == CruiseButtons.CANCEL

  def test_floor_climb_is_rate_limited_to_the_requested_accel(self):
    # Regression, drive 0000002e 18:44:52 (bookmarked): behind a lead with the
    # ceiling untouched, the floor pressed every AUTO_ACTION_SPACING_MS and ran
    # the set speed up at 2.7 kph/s while aReq +0.12 asked for 0.43 kph/s. The
    # gap crossed the DI's ~2 kph bang-bang threshold, the car surged at +0.6,
    # overshot, and the whole thing limit-cycled. A floor-driven climb must now
    # wait half_kph / (3.6 * a_req) ~= 3.7 s at this demand.
    kw = dict(v_ego=18.35, cc_set_kph=66.0, ceiling_kph=72.4, speed_units="MPH")
    assert CruiseButtons.is_accel(self.ctrl.update(make_cc(accel=0.12), make_cs(**kw), frame=0))
    # past AUTO_ACTION_SPACING_MS but well inside the floor's own interval
    self.t_ms += 1000
    assert self.ctrl.update(make_cc(accel=0.12), make_cs(**kw), frame=1) is None
    self.t_ms += 2900
    assert CruiseButtons.is_accel(self.ctrl.update(make_cc(accel=0.12), make_cs(**kw), frame=2))

  def test_real_accel_demand_is_not_rate_limited(self):
    # Open road: the projection alone clears the floor, so the climb is not
    # floor-driven and normal spacing governs. The rate limit must not slow
    # down genuine acceleration.
    kw = dict(v_ego=18.35, cc_set_kph=66.0, ceiling_kph=90.0, speed_units="MPH")
    assert CruiseButtons.is_accel(self.ctrl.update(make_cc(accel=0.90), make_cs(**kw), frame=0))
    self.t_ms += AUTO_ACTION_SPACING_MS + 50
    assert CruiseButtons.is_accel(self.ctrl.update(make_cc(accel=0.90), make_cs(**kw), frame=1))

  def test_floor_climb_interval_bounds(self):
    half = 1 * CV.MPH_TO_KPH
    # tiny demand saturates at the cap rather than diverging
    assert NoPedalACCController._floor_climb_interval_ms(0.0, half) == FLOOR_CLIMB_MAX_INTERVAL_MS
    # strong demand never goes below the normal press spacing
    assert NoPedalACCController._floor_climb_interval_ms(5.0, half) == AUTO_ACTION_SPACING_MS
    # the typical logged climb demand lands near 3.7 s
    assert 3500 < NoPedalACCController._floor_climb_interval_ms(0.12, half) < 3900

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
    self.ctrl = NoPedalACCController()
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
    self.ctrl = NoPedalACCController()
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
    self.ctrl = NoPedalACCController()
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


def pull_main(eng, t_ms, *, no_pedal_acc=True, v_ego=25.0, di_state="STANDBY"):
  """One MAIN pull (press + release) at t_ms."""
  common = dict(v_ego=v_ego, speed_units="KPH", use_pedal=False,
                pedal_long_allowed=False, long_control_allowed=True,
                real_brake_pressed=False, di_cruise_state=di_state,
                no_pedal_acc=no_pedal_acc)
  eng.process_buttons(CruiseButtons.MAIN, CruiseButtons.IDLE, t_ms, **common)
  eng.process_buttons(CruiseButtons.IDLE, CruiseButtons.MAIN, t_ms + 50, **common)


class TestEngagementNoPedalACC:

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
                  no_pedal_acc=True)
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
                  no_pedal_acc=True)
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
                  no_pedal_acc=True)
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
                  no_pedal_acc=True)
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
