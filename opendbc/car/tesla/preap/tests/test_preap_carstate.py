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


class TestPreAPDIStateDecode(unittest.TestCase):
  """Pin the pre-AP DI_state field layout and the pedal semantics, both
  established by CAN measurement on the 2026-07-14 drives.

  Pre-AP firmware swaps two DI_state fields relative to the AP-era DBC this
  file was inherited from: bits 32-40 carry the displayed vehicle speed (they
  tracked vEgo 1:1 at all times, ENABLED included) and bits 48-55 the cruise
  set speed (held 41 while the car converged to and held 40.3 mph for 30 s;
  stepped 41->36 on a stalk-down and the car settled at 36). Reading the set
  speed from bits 32-40 made vision ACC see roughly half the true set speed,
  so it computed a large positive offset and demanded UP_2ND every 500 ms —
  under LIVE_TX that would ratchet the real set speed to the TX ceiling.
  tesla_preap.dbc now names the fields per the pre-AP layout.

  DI_pedalPos is the driver's accelerator, full stop — it maps linearly to
  DI_torqueDriver (33% -> +135 Nm, 0% -> -114 Nm regen) and lifting it decays
  speed toward the set speed even while ENABLED. It must stay wired to
  gasPressed in every mode: during stock-CC operation a pedal press is a
  genuine driver override (the DI never reports an OVERRIDE cruise state on
  this car — zero occurrences across both drives), and masking it would hide
  the driver's takeover from openpilot.
  """

  def _cs(self, *, di_cruise_state, di_pedal_pos, di_cruise_set, di_digital_speed=0,
          vision_acc=True):
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
                            {"DI_cruiseState": di_cruise_state, "DI_speedUnits": 0,  # 0 = MPH (1 = KPH)
                             "DI_cruiseSet": di_cruise_set,
                             "DI_digitalSpeed": di_digital_speed}),
        packer.make_can_msg("DI_torque1", CANBUS.party, {"DI_pedalPos": di_pedal_pos}),
      ]
      CS = None
      for _ in range(5):
        CS = CI.update([(0, msgs)])
      return CI.CS, CS

  def test_set_speed_reads_bits_48_not_vehicle_speed(self):
    # CC set at 40 mph while the (overriding) car does 55: the readback must be 40
    cs, _ = self._cs(di_cruise_state=2, di_pedal_pos=0.0,
                     di_cruise_set=40, di_digital_speed=55)
    self.assertAlmostEqual(cs.v_cruise_actual_kph, 40 * 1.609344, places=1)

  def test_plain_no_pedal_cruise_speed_reads_same_bits(self):
    # the pre-existing stock-CC display path must still see the true set speed
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=0.0,
                     di_cruise_set=40, di_digital_speed=55, vision_acc=False)
    self.assertAlmostEqual(CS.cruiseState.speed, 40 * 0.44704, places=2)

  def test_driver_press_during_stock_cc_is_a_gas_press(self):
    # accelerator override while ENABLED is real driver input — never mask it
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=20.0, di_cruise_set=40)
    self.assertTrue(CS.gasPressed, "driver accelerator press during stock CC must not be masked")

  def test_foot_off_during_stock_cc_is_not_a_gas_press(self):
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=0.0, di_cruise_set=40)
    self.assertFalse(CS.gasPressed)


if __name__ == "__main__":
  unittest.main()
