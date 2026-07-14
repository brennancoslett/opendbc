#!/usr/bin/env python3
"""Pre-AP carstate update tests.

Regression coverage for the class of bug where update_preap writes a field on
`ret` (the CarState capnp struct) without a matching schema entry in car.capnp.
These writes look like ordinary Python assignment but silently require the
schema to agree; the first update call crashes card with AttributeError, which
leaves the panda in elm327 safe mode and surfaces as 'Unknown Vehicle Variant'
(canError) in the UI.

See vault/lessons/agent-failure-modes/capnp-schema-write-without-field.md
and the regression report d0cdc986c5d023f5|4a5ffc1c21 (2026-04-20).
"""
import unittest

from opendbc.car.car_helpers import interfaces


class TestPreAPCarStateUpdate(unittest.TestCase):

  def _make_interface(self):
    CarInterface = interfaces["TESLA_MODEL_S_PREAP"]
    CP = CarInterface.get_params("TESLA_MODEL_S_PREAP",
                                 {i: {} for i in range(8)},
                                 [],
                                 alpha_long=False, is_release=False, docs=False)
    return CarInterface(CP)

  def test_update_runs_without_crashing(self):
    """update() with empty CAN must not raise — exercises every ret.X write path."""
    CI = self._make_interface()
    # Ten iterations; mirrors upstream test_car_interfaces pattern and catches
    # issues that only appear after state has accumulated.
    for _ in range(10):
      CI.update([])

  def test_nap_specific_fields_on_carstate(self):
    """NAP-specific booleans written by update_preap must exist on the schema."""
    CI = self._make_interface()
    CS = CI.update([])
    for field in ("teslaCCEngaged", "teslaCCDisengaged", "teslaCCNotArmed",
                  "pedalMaxRegen", "pedalLongActive"):
      self.assertTrue(hasattr(CS, field), f"CarState schema missing {field}")


class TestVisionACCGasPressed(unittest.TestCase):
  """gasPressed must not be forged by the stock CC's own torque demand.

  While the Tesla DI drives the car under stock cruise, it authors DI_pedalPos
  itself — it carried a mean of ~20 (threshold is 2) across every ENABLED frame
  of the 2026-07-14 drive. Vision ACC is the only mode that claims op-long while
  a stock CC runs, so that signal read as a permanent driver gas override and
  dropped longActive the instant the CC engaged: vision ACC could never act
  (2 eligible frames out of 56489). Pedal mode is unaffected — it reads the
  interceptor — and plain no-pedal mode never claims op-long.
  """

  def _run(self, *, di_cruise_state, di_pedal_pos, vision_acc):
    from unittest.mock import PropertyMock, patch
    from opendbc.can import CANPacker
    from opendbc.car.car_helpers import interfaces
    from opendbc.car.tesla.values import CANBUS

    conf = "opendbc.car.tesla.preap.nap_conf.NAPConf"
    with patch(f"{conf}.vision_acc", new_callable=PropertyMock, return_value=vision_acc), \
         patch(f"{conf}.use_pedal", new_callable=PropertyMock, return_value=False):
      CarInterface = interfaces["TESLA_MODEL_S_PREAP"]
      CP = CarInterface.get_params("TESLA_MODEL_S_PREAP", {i: {} for i in range(8)}, [],
                                   alpha_long=False, is_release=False, docs=False)
      CI = CarInterface(CP)

      packer = CANPacker("tesla_preap")
      msgs = [
        packer.make_can_msg("DI_state", CANBUS.party,
                            {"DI_cruiseState": di_cruise_state, "DI_speedUnits": 1}),
        packer.make_can_msg("DI_torque1", CANBUS.party, {"DI_pedalPos": di_pedal_pos}),
      ]
      CS = None
      for _ in range(5):
        CS = CI.update([(0, msgs)])
      return CS

  def test_stock_cc_torque_demand_is_not_a_gas_press(self):
    # DI_cruiseState=2 (ENABLED) with the DI commanding ~20% pedal: not the driver
    CS = self._run(di_cruise_state=2, di_pedal_pos=20.0, vision_acc=True)
    self.assertFalse(CS.gasPressed,
                     "stock CC's own DI_pedalPos must not register as a driver gas press")

  def test_real_press_while_cruising_is_an_override(self):
    # DI reports a genuine driver press during cruise as OVERRIDE (4)
    CS = self._run(di_cruise_state=4, di_pedal_pos=40.0, vision_acc=True)
    self.assertTrue(CS.gasPressed, "DI OVERRIDE must surface as a driver gas press")

  def test_gas_press_still_detected_when_cruise_not_running(self):
    # STANDBY (1): the DI is not driving, so DI_pedalPos is the driver's foot again
    CS = self._run(di_cruise_state=1, di_pedal_pos=20.0, vision_acc=True)
    self.assertTrue(CS.gasPressed, "with cruise in STANDBY, DI_pedalPos is the driver")

  def test_untouched_when_vision_acc_off(self):
    # plain no-pedal mode keeps the raw DI_pedalPos behaviour
    CS = self._run(di_cruise_state=2, di_pedal_pos=20.0, vision_acc=False)
    self.assertTrue(CS.gasPressed)


if __name__ == "__main__":
  unittest.main()
