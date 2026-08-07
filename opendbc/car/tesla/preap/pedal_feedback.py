from opendbc.car.carlog import carlog
from opendbc.car.tesla.preap.nap_conf import nap_conf, PEDAL_DI_PRESSED

PEDAL_TIMEOUT_MS = 500

# Hysteresis around the override threshold, in DI. Crossing it hands actuation
# between the driver and the controller, so a foot resting near the boundary
# must not chatter the authority handshake.
GAS_PRESSED_HYSTERESIS_DI = 1.5


def _pedal_di_neutral() -> float:
  """Calibrated zero-torque DI, or the nominal pressed threshold without one.

  Falls back to PEDAL_DI_PRESSED when the config object carries no
  calibration, so a partial config cannot widen the band the driver has to
  cross to take over.
  """
  return float(getattr(nap_conf, "pedal_di_neutral", PEDAL_DI_PRESSED))


class PedalFeedback:
  """Parses Comma Pedal GAS_SENSOR feedback and tracks pedal health."""

  def __init__(self):
    self._gas_pressed = False
    self.interceptor_value = 0.0
    self.interceptor_value2 = 0.0
    self.interceptor_state = 0
    self.idx = 0
    self.prev_idx = 0
    self.last_seen_ms = 0
    self.available = False
    self.timeout = True
    self.torque_level = 0.0

  def update(self, gas_sensor_msg, curr_time_ms):
    try:
      if not gas_sensor_msg:
        return False

      self.prev_idx = self.idx

      interceptor_gas = float(gas_sensor_msg.get("INTERCEPTOR_GAS", 0.0))
      interceptor_gas2 = float(gas_sensor_msg.get("INTERCEPTOR_GAS2", 0.0))
      self.interceptor_state = int(gas_sensor_msg.get("STATE", 0))
      self.idx = int(gas_sensor_msg.get("IDX", 0))

      self.interceptor_value = float(nap_conf.pedal_to_di(interceptor_gas))
      self.interceptor_value2 = float(nap_conf.pedal_to_di(interceptor_gas2))

      if self.idx != self.prev_idx:
        self.last_seen_ms = curr_time_ms

      self.timeout = (curr_time_ms - self.last_seen_ms) > PEDAL_TIMEOUT_MS
      self.available = (not self.timeout) and (self.interceptor_state == 0)
      return True

    except Exception:
      carlog.exception("Pedal feedback parse failed")
      self.available = False
      self.timeout = True
      return False

  def update_torque(self, di_torque1_msg):
    try:
      self.torque_level = di_torque1_msg.get("DI_torqueMotor", 0)
    except Exception:
      self.torque_level = 0.0

  def update_gas_pressed(self, zero_torque_di: float):
    """Latch driver override against the DI position that produces no torque.

    PEDAL_DI_PRESSED is 2, but the DI does not reach zero torque until around
    DI 13, so everything between the two is a request for regen. Treating that
    band as an override handed the driver's whole lift-off to the DI as
    braking: measured over a 22-minute drive, 25% of override frames carried
    negative motor torque and the median deceleration in the two seconds before
    hand-back was -1.18 m/s2. An override now means asking for more than
    neutral, so the controller resumes at the neutral point rather than after
    the car has already slowed.

    Below the threshold the controller keeps command, so a light rest on the
    pedal no longer produces regen -- it produces whatever the planner asked
    for, bounded by the same accel envelope as any other engaged frame.

    The floor is the *calibrated* neutral, not just the learner's estimate.
    PedalZeroTorque only advances while the controller already holds the
    pedal, so during an override it cannot learn, and early in a drive it
    still reads its seed -- which left this threshold at PEDAL_DI_PRESSED
    exactly on the first engagements, the ones the driver notices. Measured on
    a drive that ran this logic: the first three hand-backs released at DI
    0.49, -1.27 and -2.68 and pulled -121, -95 and -91 Nm, while later ones,
    after the learner had converged, released near DI 11 and pulled -3.8.
    """
    threshold = max(float(PEDAL_DI_PRESSED), _pedal_di_neutral(), float(zero_torque_di))
    if self._gas_pressed:
      self._gas_pressed = self.interceptor_value > threshold - GAS_PRESSED_HYSTERESIS_DI
    else:
      self._gas_pressed = self.interceptor_value > threshold
    return self._gas_pressed

  @property
  def gas_pressed(self):
    return self._gas_pressed
