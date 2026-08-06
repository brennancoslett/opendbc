#!/usr/bin/env python3
"""Fit the Virtual DAS feedforward table from a recorded drive.

`ff_table_default.py` is a synthetic fallback: it assumes accel 0 maps to
DI 0 and that ACCEL_MAX maps to the speed-dependent pedal ceiling. Neither
holds on a real car. The DI produces zero torque somewhere above a released
pedal (measured around DI 13 on a 2012-2014 Model S), and the pedal ceiling
is far hotter than the accel range it is supposed to span, so the fallback
under-drives small requests and over-drives large ones. The two integrators
downstream end up supplying the difference, which is what makes the pedal
hunt behind a lead.

This fits DI -> acceleration from a drive and writes the calibrated table the
runtime prefers:

  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python3 generate_ff_table.py \
      /data/media/0/realdata/<route> -o /data/vdas_ff_table.json

The fit is in DI units as the controller computes them, so it is valid for
whatever pedal calibration was in force during the drive. Recalibrating the
pedal changes the DI scale and invalidates the table -- regenerate after any
calibration change.
"""

import argparse
import glob
import json
import sys

import numpy as np

from opendbc.car.tesla.preap.nap_conf import (
  nap_conf, PEDAL_BP, PEDAL_MAX_VALUES, ACCEL_MAX, REGEN_MAX,
)
from opendbc.car.tesla.preap.ff_table_default import SPEED_BP, ACCEL_BP

# Comma Pedal GAS_COMMAND scaling, from teslacan.py
PEDAL_M1 = 0.050796813
PEDAL_D = -22.85856576
GAS_COMMAND_ID = 0x551

GRAVITY = 9.81
# Only fit frames where the controller held authority and the driver was out
# of the loop; anything else measures the driver's foot, not the map.
MIN_SPEED = 5.0  # m/s
MIN_SAMPLES_PER_BIN = 300
# A bin needs real spread in the commanded DI or the regression is fitting noise.
MIN_DI_SPREAD = 1.0


def _decode_gas_command(dat):
  """Returns (pedal_voltage, enable) from a GAS_COMMAND frame."""
  return ((dat[0] << 8 | dat[1]) * PEDAL_M1 + PEDAL_D, (dat[4] >> 7) & 1)


def collect(route_prefix):
  """Walk a route's rlogs and return the per-frame arrays the fit needs."""
  from openpilot.tools.lib.logreader import LogReader

  segments = sorted(glob.glob(route_prefix + "--*/rlog*"),
                    key=lambda p: int(p.split("--")[-1].split("/")[0]))
  if not segments:
    raise SystemExit(f"no rlogs under {route_prefix}--*/")

  rows = []
  state = {'accel': 0.0, 'long_active': 0, 'pitch': 0.0, 'di': 0.0, 'enabled': 0}
  for segment in segments:
    for msg in LogReader(segment):
      which = msg.which()
      if which == 'carControl':
        state['long_active'] = int(msg.carControl.longActive)
        orientation = list(msg.carControl.orientationNED)
        state['pitch'] = orientation[1] if len(orientation) > 1 else 0.0
      elif which == 'sendcan':
        for frame in msg.sendcan:
          if frame.address == GAS_COMMAND_ID:
            voltage, enable = _decode_gas_command(frame.dat)
            state['di'] = nap_conf.pedal_to_di(voltage)
            state['enabled'] = enable
      elif which == 'carState':
        cs = msg.carState
        if (state['enabled'] and state['long_active'] and not cs.gasPressed
            and not cs.brakePressed and not cs.standstill and cs.vEgo > MIN_SPEED):
          rows.append((cs.vEgo, cs.aEgo, state['pitch'], state['di']))
    print(f"  {segment}: {len(rows)} usable frames", file=sys.stderr)

  if not rows:
    raise SystemExit("no frames with the controller in command; nothing to fit")
  return np.array(rows).T  # v_ego, a_ego, pitch, pedal_di


def fit(v_ego, a_ego, pitch, pedal_di, speed_bp):
  """Regress grade-corrected acceleration on commanded DI, per speed bin.

  Returns (zero_accel_di, di_per_accel) sampled at speed_bp. Bins without
  enough evidence are filled by interpolation from the bins that had it, so a
  drive that never visited a speed still produces a monotonic table.
  """
  # Subtract the gravity component so a hilly drive does not bias the intercept.
  a_road = a_ego + GRAVITY * np.sin(pitch)

  edges = [0.0] + [(speed_bp[i] + speed_bp[i + 1]) / 2 for i in range(len(speed_bp) - 1)] + [1e3]
  centres, zero_di, di_per_accel = [], [], []
  for i, speed in enumerate(speed_bp):
    in_bin = (v_ego >= edges[i]) & (v_ego < edges[i + 1])
    if in_bin.sum() < MIN_SAMPLES_PER_BIN or np.std(pedal_di[in_bin]) < MIN_DI_SPREAD:
      print(f"  {speed:5.1f} m/s: skipped ({in_bin.sum()} samples)", file=sys.stderr)
      continue
    slope, intercept = np.polyfit(pedal_di[in_bin], a_road[in_bin], 1)
    if slope <= 0:
      print(f"  {speed:5.1f} m/s: skipped (non-physical slope {slope:+.4f})", file=sys.stderr)
      continue
    centres.append(speed)
    zero_di.append(-intercept / slope)
    di_per_accel.append(1.0 / slope)
    print(f"  {speed:5.1f} m/s: n={in_bin.sum():6d}  zero-accel DI {-intercept / slope:6.2f}" +
          f"  {1.0 / slope:5.1f} DI per m/s^2", file=sys.stderr)

  if len(centres) < 2:
    raise SystemExit("fewer than two speed bins had enough evidence; drive more of the range")
  return (np.interp(speed_bp, centres, zero_di),
          np.interp(speed_bp, centres, di_per_accel))


def zero_torque_blend(accel):
  """The runtime's zero-torque weighting, mirrored so it can be removed here."""
  if accel < 0:
    return float(np.clip((accel - REGEN_MAX) / (0.0 - REGEN_MAX), 0.0, 1.0))
  return float(1.0 - accel / ACCEL_MAX)


def build_table(zero_di, di_per_accel, speed_bp, accel_bp, zero_torque_nominal):
  """Turn the fit into table rows in the runtime's storage convention.

  FeedforwardModel adds `zero_torque_di * blend(accel)` on top of the stored
  value, so the learner's contribution has to come back out here. Rows are
  stored against `zero_torque_nominal`; if the learner settles somewhere else
  the table shifts with it, which is the intent -- the learner tracks the DI
  that actually zeroes torque, and the fit only supplies the slope and the
  residual road load.
  """
  floor, table = nap_conf.pedal_di_floor, []
  for speed, di_0, di_rate in zip(speed_bp, zero_di, di_per_accel, strict=True):
    ceiling = float(np.interp(speed, PEDAL_BP, PEDAL_MAX_VALUES))
    row = [float(np.clip(di_0 + accel * di_rate, floor, ceiling))
           - zero_torque_nominal * zero_torque_blend(accel) for accel in accel_bp]
    table.append([round(value, 2) for value in row])
  return table


def main():
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('route', help='route prefix, e.g. /data/media/0/realdata/0000005c--9a9eccc070')
  parser.add_argument('-o', '--output', default='/data/vdas_ff_table.json')
  parser.add_argument('--zero-torque', type=float, default=None,
                      help='DI the learner is expected to settle at (default: measure it from the drive)')
  parser.add_argument('--dry-run', action='store_true', help='print the table instead of writing it')
  args = parser.parse_args()

  print(f"collecting {args.route}", file=sys.stderr)
  v_ego, a_ego, pitch, pedal_di = collect(args.route)
  print(f"fitting {len(v_ego)} frames", file=sys.stderr)
  zero_di, di_per_accel = fit(v_ego, a_ego, pitch, pedal_di, SPEED_BP)

  # The learner anchors on zero motor torque, which sits below the DI that
  # holds a steady speed by however much road load costs. Use the low-speed
  # end of the fit, where road load is smallest, as the nominal anchor.
  zero_torque = args.zero_torque
  if zero_torque is None:
    moving = [di for speed, di in zip(SPEED_BP, zero_di, strict=True) if speed >= MIN_SPEED]
    zero_torque = float(min(moving)) if moving else 0.0
  print(f"zero-torque anchor: {zero_torque:.2f} DI", file=sys.stderr)

  table = build_table(zero_di, di_per_accel, SPEED_BP, ACCEL_BP, zero_torque)
  document = {'speed_bp': list(SPEED_BP), 'accel_bp': list(ACCEL_BP), 'table': table}

  header = "  speed |" + "".join(f"{accel:>8.1f}" for accel in ACCEL_BP)
  print(header, file=sys.stderr)
  for speed, row in zip(SPEED_BP, table, strict=True):
    print(f"  {speed:5.1f} |" + "".join(f"{value:>8.2f}" for value in row), file=sys.stderr)

  if args.dry_run:
    print(json.dumps(document, indent=2))
    return

  with open(args.output, 'w') as f:
    json.dump(document, f, indent=2)
  print(f"wrote {args.output}", file=sys.stderr)


if __name__ == '__main__':
  sys.exit(main() or 0)
