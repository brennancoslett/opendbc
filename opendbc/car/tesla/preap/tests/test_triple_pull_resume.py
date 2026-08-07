#!/usr/bin/env python3
"""Triple-pull resume: return to the set speed, not the speed you slowed to.

A double-pull captures whatever the car happens to be doing, which is the
wrong target after braking for traffic or coming off an exit. A third pull
inside the same burst retargets to the last speed that was actually set.
"""
import unittest

from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.preap.engagement import PreAPEngagement

WINDOW_MS = 400
MAIN = 2
IDLE = 0


class TestTriplePullResume(unittest.TestCase):

  def _make_engagement(self):
    return PreAPEngagement(double_pull_enabled=True, double_pull_window_ms=WINDOW_MS)

  def _pull(self, eng, t, v_ego, use_pedal=True):
    """One full stalk pull: MAIN rising edge then release."""
    for buttons, prev in ((MAIN, IDLE), (IDLE, MAIN)):
      eng.process_buttons(
        cruise_buttons=buttons, prev_cruise_buttons=prev,
        curr_time_ms=t if buttons == MAIN else t + 20,
        v_ego=v_ego, speed_units="KPH",
        use_pedal=use_pedal, pedal_long_allowed=use_pedal,
        long_control_allowed=True, real_brake_pressed=False)

  def _engage_at(self, eng, v_ego, t0=1000):
    """Double-pull engage, leaving the set speed at v_ego."""
    self._pull(eng, t0, v_ego)
    self._pull(eng, t0 + 200, v_ego)
    self.assertTrue(eng.enableLongControl)
    return t0 + 200

  def test_third_pull_resumes_the_set_speed_instead_of_the_current_one(self):
    eng = self._make_engagement()
    fast = 100.0 / CV.MS_TO_KPH
    self._engage_at(eng, fast)
    self.assertAlmostEqual(eng.pedal_speed_kph, 100.0, places=0)

    # Slowed for traffic, then dropped out.
    eng.process_buttons(
      cruise_buttons=IDLE, prev_cruise_buttons=IDLE, curr_time_ms=5000,
      v_ego=40.0 / CV.MS_TO_KPH, speed_units="KPH", use_pedal=True,
      pedal_long_allowed=True, long_control_allowed=True, real_brake_pressed=True)
    self.assertFalse(eng.enableLongControl)
    self.assertEqual(eng.pedal_speed_kph, 0.0)

    # Re-engage at 40: a double-pull alone would leave the target at 40.
    slow = 40.0 / CV.MS_TO_KPH
    t = self._engage_at(eng, slow, t0=9000)
    self.assertAlmostEqual(eng.pedal_speed_kph, 40.0, places=0)

    self._pull(eng, t + 200, slow)
    self.assertAlmostEqual(eng.pedal_speed_kph, 100.0, places=0)
    self.assertTrue(eng.enableLongControl, "resume must not drop longitudinal")

  def test_resume_survives_a_cancel(self):
    """Cancel clears the active target; the memory it resumes to outlives it."""
    eng = self._make_engagement()
    fast = 90.0 / CV.MS_TO_KPH
    self._engage_at(eng, fast)

    eng.process_buttons(
      cruise_buttons=1, prev_cruise_buttons=IDLE, curr_time_ms=20000,
      v_ego=fast, speed_units="KPH", use_pedal=True, pedal_long_allowed=True,
      long_control_allowed=True, real_brake_pressed=False)
    self.assertFalse(eng.cruiseEnabled)
    self.assertEqual(eng.pedal_speed_kph, 0.0)

    slow = 30.0 / CV.MS_TO_KPH
    t = self._engage_at(eng, slow, t0=30000)
    self._pull(eng, t + 200, slow)
    self.assertAlmostEqual(eng.pedal_speed_kph, 90.0, places=0)

  def test_a_late_third_pull_still_drops_to_lateral(self):
    """Outside the window a lone pull keeps meaning "drop longitudinal".

    The resume deliberately shares double_pull_window_ms so that a late tap
    de-escalates rather than accelerating.
    """
    eng = self._make_engagement()
    fast = 100.0 / CV.MS_TO_KPH
    t = self._engage_at(eng, fast)

    self._pull(eng, t + WINDOW_MS + 100, fast)
    self.assertFalse(eng.enableLongControl)
    self.assertTrue(eng.cruiseEnabled, "lateral stays")
    self.assertEqual(eng.pedal_speed_kph, 0.0)

  def test_third_pull_without_a_remembered_speed_is_an_ordinary_double_pull(self):
    """Nothing to resume to: behave exactly as before rather than targeting 0."""
    eng = self._make_engagement()
    v = 55.0 / CV.MS_TO_KPH
    t = self._engage_at(eng, v)
    eng.last_set_speed_kph = 0.0

    self._pull(eng, t + 200, v)
    self.assertTrue(eng.enableLongControl)
    self.assertAlmostEqual(eng.pedal_speed_kph, 55.0, places=0)

  def test_resume_returns_to_an_adjusted_speed_not_the_engage_capture(self):
    """It is the speed held at hand-back that comes back, +/- adjustments included."""
    eng = self._make_engagement()
    v = 60.0 / CV.MS_TO_KPH
    self._engage_at(eng, v)

    # RES_ACCEL: +1 unit above the larger of target and current speed.
    eng.process_buttons(
      cruise_buttons=16, prev_cruise_buttons=IDLE, curr_time_ms=4000,
      v_ego=v, speed_units="KPH", use_pedal=True, pedal_long_allowed=True,
      long_control_allowed=True, real_brake_pressed=False)
    adjusted = eng.pedal_speed_kph
    self.assertGreater(adjusted, 60.0)

    eng.process_buttons(
      cruise_buttons=IDLE, prev_cruise_buttons=IDLE, curr_time_ms=5000,
      v_ego=v, speed_units="KPH", use_pedal=True, pedal_long_allowed=True,
      long_control_allowed=True, real_brake_pressed=True)
    self.assertEqual(eng.last_set_speed_kph, adjusted)

    slow = 20.0 / CV.MS_TO_KPH
    t = self._engage_at(eng, slow, t0=40000)
    self._pull(eng, t + 200, slow)
    self.assertAlmostEqual(eng.pedal_speed_kph, adjusted, places=3)

  def test_burst_is_reported_active_only_while_the_target_is_provisional(self):
    """What carstate holds the cluster value on.

    The window must cover every pull of a burst and close afterwards,
    otherwise the MAX box either flickers through the FSM's provisional steps
    or freezes on a stale number.
    """
    eng = self._make_engagement()
    v = 50.0 / CV.MS_TO_KPH
    self.assertFalse(eng.stalk_burst_active(0), "no pull yet")

    self._pull(eng, 1000, v)
    self.assertTrue(eng.stalk_burst_active(1000))
    self.assertTrue(eng.stalk_burst_active(1000 + WINDOW_MS - 1))
    self.assertFalse(eng.stalk_burst_active(1000 + WINDOW_MS))

    # Each further pull re-arms the hold for the rest of the burst.
    self._pull(eng, 1200, v)
    self.assertTrue(eng.stalk_burst_active(1200 + WINDOW_MS - 1))
    self._pull(eng, 1400, v)
    self.assertTrue(eng.stalk_burst_active(1400 + WINDOW_MS - 1))
    self.assertFalse(eng.stalk_burst_active(1400 + WINDOW_MS))

  def test_burst_covers_every_step_the_target_moves_through(self):
    """The whole triple-pull sequence must sit inside one hold.

    This is the flicker the hold exists to suppress: engaged at 100, the three
    pulls drive the target to 0, then to the current speed, then back to 100.
    """
    eng = self._make_engagement()
    fast = 100.0 / CV.MS_TO_KPH
    self._engage_at(eng, fast)

    eng.process_buttons(
      cruise_buttons=IDLE, prev_cruise_buttons=IDLE, curr_time_ms=5000,
      v_ego=fast, speed_units="KPH", use_pedal=True, pedal_long_allowed=True,
      long_control_allowed=True, real_brake_pressed=True)

    slow = 45.0 / CV.MS_TO_KPH
    seen = []
    for i, t in enumerate((9000, 9200, 9400)):
      self._pull(eng, t, slow)
      seen.append((eng.pedal_speed_kph, eng.stalk_burst_active(t + 20)))
      if i:
        self.assertTrue(eng.stalk_burst_active(t + 20))

    targets = [speed for speed, _ in seen]
    self.assertEqual(targets[0], 0.0, "first pull drops the target")
    self.assertAlmostEqual(targets[1], 45.0, places=0, msg="second captures current")
    self.assertAlmostEqual(targets[2], 100.0, places=0, msg="third resumes")
    self.assertTrue(all(active for _, active in seen),
                    "every step must fall inside one hold")

  def test_stock_cc_mode_is_untouched(self):
    """Without the pedal the DI owns the set speed and does its own resume."""
    eng = self._make_engagement()
    v = 80.0 / CV.MS_TO_KPH
    self._pull(eng, 1000, v, use_pedal=False)
    self._pull(eng, 1200, v, use_pedal=False)
    self.assertTrue(eng.cruiseEnabled)
    self.assertEqual(eng.pedal_speed_kph, 0.0)

    eng.last_set_speed_kph = 120.0
    self._pull(eng, 1400, v, use_pedal=False)
    self.assertEqual(eng.pedal_speed_kph, 0.0, "NAP must not invent a target here")


if __name__ == "__main__":
  unittest.main()
