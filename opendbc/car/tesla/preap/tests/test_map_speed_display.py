#!/usr/bin/env python3
"""The readout's view of the MCU map limit: hold across a gap, by distance.

Separate from the stalk-pull view, and deliberately looser. Nothing acts on
what the driver reads, so carrying a value the MCU has stopped reporting beats
a blank sign -- right up until the car has gone far enough that it is a claim
about a road it has never seen a limit for.
"""
import unittest

from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.preap.map_speed import (
  DISPLAY_HOLD_M,
  DISPLAY_STABLE_MS,
  MapSpeedLimit,
  SENTINEL,
  STALE_MS,
)

STEP_MS = 100
MPH_45 = 45.0


def _feed(tracker, value, ms, *, t0=1000, ts0=0, v_ego=0.0):
  """Feed one value at 10 Hz for ms. Returns (end_time_ms, end_ts)."""
  t, ts = t0, ts0
  while t <= t0 + ms:
    ts += 1
    tracker.update(value, ts, t, v_ego)
    t += STEP_MS
  return t - STEP_MS, ts


class TestDisplayDebounce(unittest.TestCase):
  def test_a_value_has_to_hold_before_it_reaches_the_sign(self):
    tracker = MapSpeedLimit()
    tracker.update(MPH_45, 1, 1000, 20.0)
    self.assertEqual(tracker.display, 0.0)

    _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=20.0)
    self.assertEqual(tracker.display, MPH_45)
    self.assertFalse(tracker.display_held)

  def test_the_sign_adopts_sooner_than_the_stalk_pull_does(self):
    tracker = MapSpeedLimit()
    _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=20.0)
    self.assertEqual(tracker.display, MPH_45)
    self.assertEqual(tracker.stable, 0.0)


class TestDisplayHold(unittest.TestCase):
  def _established(self, v_ego=20.0):
    tracker = MapSpeedLimit()
    end_t, end_ts = _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=v_ego)
    self.assertEqual(tracker.display, MPH_45)
    return tracker, end_t, end_ts

  def test_a_short_gap_keeps_the_number_and_marks_it_held(self):
    tracker, t, ts = self._established()
    # 2 s at 20 m/s is 40 m, far inside the budget.
    _feed(tracker, 0.0, 2000, t0=t + STEP_MS, ts0=ts, v_ego=20.0)

    self.assertEqual(tracker.display, MPH_45)
    self.assertTrue(tracker.display_held)

  def test_the_placeholder_is_a_gap_like_any_other(self):
    tracker, t, ts = self._established()
    _feed(tracker, SENTINEL, 2000, t0=t + STEP_MS, ts0=ts, v_ego=20.0)

    self.assertEqual(tracker.display, MPH_45)
    self.assertTrue(tracker.display_held)

  def test_the_sign_clears_once_the_budget_is_spent(self):
    tracker, t, ts = self._established()
    # Drive the whole budget out at 30 m/s, plus a margin.
    duration_ms = int((DISPLAY_HOLD_M / 30.0) * 1000) + 2000
    _feed(tracker, 0.0, duration_ms, t0=t + STEP_MS, ts0=ts, v_ego=30.0)

    self.assertEqual(tracker.display, 0.0)
    self.assertFalse(tracker.display_held)

  def test_standing_still_spends_nothing(self):
    tracker, t, ts = self._established()
    # Ten minutes at a light: far longer than any time-based hold would allow,
    # and the road has not changed.
    _feed(tracker, 0.0, 600_000, t0=t + STEP_MS, ts0=ts, v_ego=0.0)

    self.assertEqual(tracker.display, MPH_45)
    self.assertTrue(tracker.display_held)

  def test_a_fresh_reading_ends_the_hold_and_refills_the_budget(self):
    tracker, t, ts = self._established()
    t, ts = _feed(tracker, 0.0, 2000, t0=t + STEP_MS, ts0=ts, v_ego=30.0)
    self.assertTrue(tracker.display_held)

    t, ts = _feed(tracker, MPH_45, DISPLAY_STABLE_MS, t0=t + STEP_MS, ts0=ts, v_ego=30.0)
    self.assertFalse(tracker.display_held)

    # The budget is whole again, so the same gap is survivable a second time.
    _feed(tracker, 0.0, 2000, t0=t + STEP_MS, ts0=ts, v_ego=30.0)
    self.assertEqual(tracker.display, MPH_45)

  def test_a_new_limit_replaces_a_held_one(self):
    tracker, t, ts = self._established()
    t, ts = _feed(tracker, 0.0, 2000, t0=t + STEP_MS, ts0=ts, v_ego=20.0)
    _feed(tracker, 30.0, DISPLAY_STABLE_MS, t0=t + STEP_MS, ts0=ts, v_ego=20.0)

    self.assertEqual(tracker.display, 30.0)
    self.assertFalse(tracker.display_held)


class TestDisplayPublish(unittest.TestCase):
  def test_the_published_limit_is_in_metres_per_second(self):
    tracker = MapSpeedLimit()
    end_t, _ = _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=20.0)

    self.assertAlmostEqual(tracker.display_limit_ms("MPH", end_t), MPH_45 * CV.MPH_TO_MS)
    self.assertAlmostEqual(tracker.display_limit_ms("KPH", end_t), MPH_45 * CV.KPH_TO_MS)

  def test_a_silent_bus_blanks_the_sign(self):
    tracker = MapSpeedLimit()
    end_t, _ = _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=20.0)
    self.assertGreater(tracker.display_limit_ms("MPH", end_t), 0.0)

    # The parser is repeating its last payload: the value is not evidence.
    self.assertEqual(tracker.display_limit_ms("MPH", end_t + STALE_MS + 1), 0.0)
    self.assertEqual(tracker.display, 0.0)
    self.assertFalse(tracker.display_held)

  def test_the_raw_value_is_published_unfiltered(self):
    tracker = MapSpeedLimit()
    end_t, ts = _feed(tracker, MPH_45, DISPLAY_STABLE_MS, v_ego=20.0)

    # A gap holds the sign but the raw value follows the MCU straight down, so
    # the two together say whether a wrong number came from the bus or from us.
    _feed(tracker, SENTINEL, 2000, t0=end_t + STEP_MS, ts0=ts, v_ego=20.0)
    self.assertEqual(tracker.display, MPH_45)
    self.assertAlmostEqual(tracker.raw_ms("MPH"), SENTINEL * CV.MPH_TO_MS)

  def test_a_blank_sign_publishes_zero(self):
    tracker = MapSpeedLimit()
    end_t, _ = _feed(tracker, SENTINEL, 5000, v_ego=20.0)

    self.assertEqual(tracker.display_limit_ms("MPH", end_t), 0.0)


if __name__ == "__main__":
  unittest.main()
