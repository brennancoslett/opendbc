#!/usr/bin/env python3
"""Map-speed stalk pull: adopt the posted limit only when asked, only when sure.

The MCU's map limit is good on the interstate and flickery in town, so it is
never applied on its own. A fourth pull inside one stalk burst asks for it; the
debounce in MapSpeedLimit decides whether there is an answer to give.
"""
import unittest

from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.preap.engagement import PreAPEngagement
from opendbc.car.tesla.preap.map_speed import MapSpeedLimit, SENTINEL, STABLE_MS, STALE_MS

WINDOW_MS = 400
MAIN = 2
IDLE = 0


class TestMapSpeedLimit(unittest.TestCase):
  """The debounce that decides whether a limit is worth offering."""

  def _hold(self, tracker, value, t0=1000, ms=STABLE_MS, step=100):
    """Feed one value at 10 Hz for ms, returning the final timestamp."""
    t = t0
    ts = 0
    while t <= t0 + ms:
      ts += 1
      tracker.update(value, ts, t)
      t += step
    return t - step

  def test_a_held_value_becomes_usable(self):
    m = MapSpeedLimit()
    t = self._hold(m, 45.0)
    self.assertEqual(m.limit(t), 45.0)

  def test_a_value_is_not_usable_before_it_has_held(self):
    m = MapSpeedLimit()
    t = self._hold(m, 45.0, ms=STABLE_MS - 500)
    self.assertEqual(m.limit(t), 0.0)

  def test_the_25_placeholder_is_never_usable(self):
    """25 is what the MCU sends when it has no limit, not a 25 mph zone."""
    m = MapSpeedLimit()
    t = self._hold(m, SENTINEL, ms=STABLE_MS * 3)
    self.assertEqual(m.limit(t), 0.0)

  def test_a_change_withdraws_the_previous_limit(self):
    """Once the road has changed, the old limit is not a safer answer than none."""
    m = MapSpeedLimit()
    t = self._hold(m, 45.0)
    self.assertEqual(m.limit(t), 45.0)

    m.update(35.0, 999, t + 100)
    self.assertEqual(m.limit(t + 100), 0.0)

    t2 = self._hold(m, 35.0, t0=t + 100)
    self.assertEqual(m.limit(t2), 35.0)

  def test_flicker_never_settles(self):
    m = MapSpeedLimit()
    t = 1000
    for i in range(60):
      m.update(45.0 if i % 2 else 35.0, i + 1, t)
      t += 100
    self.assertEqual(m.limit(t), 0.0)

  def test_a_repeated_payload_goes_stale(self):
    """A quiet MCU leaves the parser repeating itself; that is not fresh data."""
    m = MapSpeedLimit()
    t = self._hold(m, 45.0)
    self.assertEqual(m.limit(t), 45.0)

    # Same ts_nanos: the parser is replaying its last frame.
    for _ in range(10):
      t += STALE_MS // 5
      m.update(45.0, 12345, t)
    self.assertEqual(m.limit(t), 0.0)

  def test_target_applies_the_offset_in_display_units(self):
    m = MapSpeedLimit()
    t = self._hold(m, 40.0)
    target = m.target_kph("MPH", 5, t)
    self.assertAlmostEqual(target, 45.0 * CV.MPH_TO_KPH, places=3)
    self.assertAlmostEqual(m.limit_ms("MPH", t), 40.0 * CV.MPH_TO_MS, places=3)

  def test_no_limit_means_no_target(self):
    m = MapSpeedLimit()
    self.assertEqual(m.target_kph("MPH", 5, 1000), 0.0)


class TestMapSpeedPull(unittest.TestCase):
  """The stalk gesture itself."""

  def _make_engagement(self):
    return PreAPEngagement(double_pull_enabled=True, double_pull_window_ms=WINDOW_MS)

  def _pull(self, eng, t, v_ego, map_speed_target_kph=0.0, use_pedal=True):
    """One full stalk pull. Returns the event raised on the press frame.

    The event lives exactly one frame, so the release clears it -- reading it
    off the engagement afterwards would always come back empty.
    """
    event = None
    for buttons, prev in ((MAIN, IDLE), (IDLE, MAIN)):
      eng.process_buttons(
        cruise_buttons=buttons, prev_cruise_buttons=prev,
        curr_time_ms=t if buttons == MAIN else t + 20,
        v_ego=v_ego, speed_units="MPH",
        use_pedal=use_pedal, pedal_long_allowed=use_pedal,
        long_control_allowed=True, real_brake_pressed=False,
        map_speed_target_kph=map_speed_target_kph)
      if buttons == MAIN:
        event = eng.map_speed_event
    return event

  def _engage_at(self, eng, v_ego, t0=1000):
    self._pull(eng, t0, v_ego)
    self._pull(eng, t0 + 200, v_ego)
    self.assertTrue(eng.enableLongControl)
    return t0 + 200

  def test_fourth_pull_takes_the_map_speed(self):
    eng = self._make_engagement()
    v = 35.0 * CV.MPH_TO_MS
    t = self._engage_at(eng, v)

    target = 50.0 * CV.MPH_TO_KPH               # 45 posted + 5
    self._pull(eng, t + 200, v)                 # third: resume
    event = self._pull(eng, t + 400, v, target)  # fourth: map speed

    self.assertAlmostEqual(eng.pedal_speed_kph, target, places=3)
    self.assertEqual(event, "mapSpeedApplied")
    self.assertTrue(eng.enableLongControl, "a map-speed pull must not drop longitudinal")

  def test_third_pull_still_resumes(self):
    """The new gesture sits above the resume without displacing it."""
    eng = self._make_engagement()
    fast = 65.0 * CV.MPH_TO_MS
    self._engage_at(eng, fast)
    remembered = eng.pedal_speed_kph

    eng.process_buttons(
      cruise_buttons=IDLE, prev_cruise_buttons=IDLE, curr_time_ms=5000,
      v_ego=30.0 * CV.MPH_TO_MS, speed_units="MPH", use_pedal=True,
      pedal_long_allowed=True, long_control_allowed=True, real_brake_pressed=True)
    self.assertFalse(eng.enableLongControl)

    slow = 30.0 * CV.MPH_TO_MS
    t = self._engage_at(eng, slow, t0=9000)
    event = self._pull(eng, t + 200, slow, map_speed_target_kph=50.0 * CV.MPH_TO_KPH)

    self.assertAlmostEqual(eng.pedal_speed_kph, remembered, places=0)
    self.assertIsNone(event, "third pull is a resume, not a map-speed pull")

  def test_fourth_pull_without_a_limit_changes_nothing(self):
    eng = self._make_engagement()
    v = 35.0 * CV.MPH_TO_MS
    t = self._engage_at(eng, v)
    before = eng.pedal_speed_kph

    self._pull(eng, t + 200, v)
    event = self._pull(eng, t + 400, v, map_speed_target_kph=0.0)

    self.assertEqual(eng.pedal_speed_kph, before)
    self.assertEqual(event, "mapSpeedUnavailable")
    self.assertTrue(eng.enableLongControl)

  def test_the_event_lasts_one_frame(self):
    """carState publishes it on the frame it is raised and never again."""
    eng = self._make_engagement()
    v = 35.0 * CV.MPH_TO_MS
    t = self._engage_at(eng, v)
    self._pull(eng, t + 200, v)
    event = self._pull(eng, t + 400, v, map_speed_target_kph=50.0 * CV.MPH_TO_KPH)
    self.assertEqual(event, "mapSpeedApplied")
    # The release frame inside that pull already cleared it.
    self.assertIsNone(eng.map_speed_event)

    eng.process_buttons(
      cruise_buttons=IDLE, prev_cruise_buttons=IDLE, curr_time_ms=t + 600,
      v_ego=v, speed_units="MPH", use_pedal=True, pedal_long_allowed=True,
      long_control_allowed=True, real_brake_pressed=False)
    self.assertIsNone(eng.map_speed_event)

  def test_a_slow_fourth_pull_is_a_new_burst_not_a_map_speed_request(self):
    """Outside the window a lone pull still de-escalates to lateral."""
    eng = self._make_engagement()
    v = 35.0 * CV.MPH_TO_MS
    t = self._engage_at(eng, v)
    self._pull(eng, t + 200, v)

    event = self._pull(eng, t + 200 + WINDOW_MS + 100, v,
                       map_speed_target_kph=50.0 * CV.MPH_TO_KPH)
    self.assertFalse(eng.enableLongControl)
    self.assertIsNone(event)

  def test_stock_cc_mode_leaves_the_di_alone(self):
    """Without the pedal the DI owns the set speed, so there is nothing to write."""
    eng = self._make_engagement()
    v = 35.0 * CV.MPH_TO_MS
    self._pull(eng, 1000, v, use_pedal=False)
    self._pull(eng, 1200, v, use_pedal=False)
    for i in range(2):
      event = self._pull(eng, 1400 + 200 * i, v, map_speed_target_kph=50.0 * CV.MPH_TO_KPH,
                         use_pedal=False)
      self.assertIsNone(event)
    self.assertEqual(eng.pedal_speed_kph, 0.0)


if __name__ == "__main__":
  unittest.main()
