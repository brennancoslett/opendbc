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

  DI_pedalPos is CC-authored while the DI holds stock cruise, not the
  driver's accelerator: measurement on the 2026-07-14 drive 4 (a deliberate
  foot-off-the-pedal drive) found 100% of 31,580 ENABLED frames reading
  pedal-pressed by a raw threshold — physically impossible for a lifted
  foot, and it never once produced the DI's OVERRIDE cruise state (zero
  occurrences across every drive). So during ENABLED, gasPressed instead
  needs a real override signal: vEgo pulling meaningfully above the held
  set speed (the DI's own control never commands past its set). A fresh
  set-speed change (stalk step, or a future vision-ACC spoof) causes a few
  seconds of legitimate vEgo-above-set lag with a nonzero pedal reading from
  the same CC blend — drive 4 produced two ~3s false "override" runs
  starting on the exact frame the set changed — so detection is gated by a
  grace period after any DI_cruiseSet change. Validated against known
  ground truth: drive 2 (~3 real overrides driven) left exactly 3 isolated
  detections outside the grace window; drive 4 (foot off) left zero.
  Outside ENABLED (STANDBY/OFF) DI_pedalPos is the driver's accelerator
  again and the raw threshold applies directly.
  """

  def _cs(self, *, di_cruise_state, di_pedal_pos, di_cruise_set, di_digital_speed=0,
          esp_speed_kph=0.0, vision_acc=True, settle_set_change=True):
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
        packer.make_can_msg("ESP_B", CANBUS.party, {"ESP_vehicleSpeed": esp_speed_kph}),
      ]
      CS = None
      for _ in range(5):
        CS = CI.update([(0, msgs)])
      if settle_set_change:
        # Simulate the grace period having elapsed since the set speed last
        # changed, isolating the speed-above-set check from the set-change
        # transient it's designed to suppress.
        CI.CS.last_cruise_set_change_ms = 0
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

  def test_pedal_alone_during_stock_cc_is_not_a_gas_press(self):
    # CC-authored throttle blend at the held set speed must not be mistaken for a press
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=20.0, di_cruise_set=40,
                     esp_speed_kph=40 * 1.609344)
    self.assertFalse(CS.gasPressed, "CC's own pedal blend at the set speed must not be flagged")

  def test_foot_off_during_stock_cc_is_not_a_gas_press(self):
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=0.0, di_cruise_set=40)
    self.assertFalse(CS.gasPressed)

  def test_real_override_is_a_gas_press(self):
    # driver pushes past the held set speed with pedal down: genuine override
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=20.0, di_cruise_set=40,
                     esp_speed_kph=45 * 1.609344)
    self.assertTrue(CS.gasPressed, "vEgo above set speed with pedal pressed must be a gas press")

  def test_override_suppressed_right_after_a_set_change(self):
    # same speed-above-set + pedal signature, but within the grace window
    # right after DI_cruiseSet moved — this is the drive-4 false-positive case
    _, CS = self._cs(di_cruise_state=2, di_pedal_pos=20.0, di_cruise_set=40,
                     esp_speed_kph=45 * 1.609344, settle_set_change=False)
    self.assertFalse(CS.gasPressed, "vEgo-above-set lag right after a set change must not be flagged")


if __name__ == "__main__":
  unittest.main()
