#!/usr/bin/env python3
"""Where a driver's gas override ends.

The interceptor passes the driver's physical pedal straight to the drive unit
whenever openpilot is not transmitting, so the DI at which `gasPressed` clears
is the DI the car coasts at during a hand-back. On this car the drive unit does
not stop regenerating until about DI 12, so releasing at PEDAL_DI_PRESSED (2)
hands the whole lift-off over as braking.
"""
import pytest
from types import SimpleNamespace

from opendbc.car.tesla.preap.nap_conf import PEDAL_DI_PRESSED, PEDAL_DI_ZERO
from opendbc.car.tesla.preap import pedal_feedback as pf
from opendbc.car.tesla.preap.pedal_feedback import (
  GAS_PRESSED_HYSTERESIS_DI,
  PedalFeedback,
  _pedal_di_neutral,
)


def _lift_off_release_di(zero_torque_di, start_di=25.0, step=0.05):
  """Drive the pedal down from `start_di` and return the DI override ends at."""
  pedal = PedalFeedback()
  pedal.interceptor_value = start_di
  assert pedal.update_gas_pressed(zero_torque_di), "should latch while pressed"

  di = start_di
  while di > -5.0:
    di -= step
    pedal.interceptor_value = di
    if not pedal.update_gas_pressed(zero_torque_di):
      return di
  raise AssertionError("override never ended")


def test_override_ends_near_neutral_before_the_learner_has_run():
  """The bug this covers: the threshold used to wait on PedalZeroTorque.

  PedalZeroTorque only advances while openpilot already holds the pedal, so it
  cannot learn during an override and still reads its seed on a drive's first
  engagements. Passing PEDAL_DI_ZERO here is exactly that cold-start case; the
  override must still end near neutral rather than at PEDAL_DI_PRESSED.
  """
  neutral = _pedal_di_neutral()
  release_di = _lift_off_release_di(PEDAL_DI_ZERO)

  assert release_di == pytest.approx(neutral - GAS_PRESSED_HYSTERESIS_DI, abs=0.1)
  assert release_di > PEDAL_DI_PRESSED, (
    "released inside the regen band: the original defect, where a hand-back "
    + "coasted to a released pedal and pulled about -120 Nm"
  )


def test_a_converged_learner_above_neutral_still_wins():
  """The calibrated value is a floor, not a replacement for the learner."""
  learned = _pedal_di_neutral() + 4.0
  assert _lift_off_release_di(learned) == pytest.approx(
    learned - GAS_PRESSED_HYSTERESIS_DI, abs=0.1)


def test_threshold_is_hysteretic_so_a_resting_foot_cannot_chatter():
  neutral = _pedal_di_neutral()
  pedal = PedalFeedback()

  pedal.interceptor_value = neutral + 0.5
  assert pedal.update_gas_pressed(PEDAL_DI_ZERO)

  # Inside the hysteresis band the override holds rather than flapping.
  pedal.interceptor_value = neutral - GAS_PRESSED_HYSTERESIS_DI + 0.2
  assert pedal.update_gas_pressed(PEDAL_DI_ZERO)

  pedal.interceptor_value = neutral - GAS_PRESSED_HYSTERESIS_DI - 0.2
  assert not pedal.update_gas_pressed(PEDAL_DI_ZERO)

  # Re-arming takes the full threshold, not the hysteresis edge.
  pedal.interceptor_value = neutral - 0.2
  assert not pedal.update_gas_pressed(PEDAL_DI_ZERO)
  pedal.interceptor_value = neutral + 0.2
  assert pedal.update_gas_pressed(PEDAL_DI_ZERO)


def test_an_uncalibrated_config_cannot_widen_the_override_band(monkeypatch):
  """A config without calibration must fall back, never raise or read as 0."""
  monkeypatch.setattr(pf, "nap_conf", SimpleNamespace())
  assert _pedal_di_neutral() == pytest.approx(PEDAL_DI_PRESSED)
