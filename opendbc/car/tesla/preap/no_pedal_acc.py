"""No-pedal ACC: stock-CC set-speed modulation.

Ports the Tinkla tesla-unity ACCController concept onto NAP's architecture.
openpilot's longitudinal planner runs normally (vision leads via the model
when no radar is fitted); with the standard zero-gain tuning, CC.actuators.accel
is the planner's target accel passed through as pure feedforward. This module
translates that request into spoofed cruise-stalk speed presses that walk the
DI's stock cruise set speed toward the planned speed, capped at the
driver-owned ceiling (engagement.pedal_speed_kph).

Decel authority is whatever the DI does when its set speed drops — motor
regen only, no friction brakes. Large decel requests CANCEL the stock CC
instead (coasting regen beats CC's shallow slew). The driver is the brakes.

Every decision is logged via carlog as well as transmitted, so a drive's
button history can be read back from the log either way. The mode is off by
default and gated by one toggle (NAPNoPedalACCEnabled, read at fingerprint
time — see preap/interface.py), so nothing here runs unless the driver turned
it on and rebooted. Independently of that, the tesla_preap.h safety mode
enforces the permitted button values and the TX rate.

A/B TUNING TELEMETRY
--------------------
Every log line is prefixed "NoPedalACC.tlm" (see _log_telemetry). One line at
5 Hz while modulating, 1 Hz otherwise, self-labeled with the projection value
so runs at different gains are directly comparable. To pull a drive's data:

    grep -h "NoPedalACC.tlm" /data/log/swaglog.* | sed 's/.*NoPedalACC.tlm //'

Fields (all key=value):
  f       carcontroller frame (100 Hz clock)
  proj    ACCEL_PROJECTION_S in effect this run (the A/B variable)
  region  1 = no-pedal ACC operating (op-long + longActive + no gas + di ENABLED)
  reason  why not modulating when region=0 (not_op_long / not_long_active /
          gas_pressed / di_standby / di_off), or the active sub-state
          (active / hard_brake_cancel / human_holdoff / auto_spacing)
  di      DI cruise state, long, gas, units  — engagement context
  vEgo    m/s   — offline dv/dt of this is the fallback achieved-accel signal
  aEgo    m/s^2 — ACHIEVED longitudinal accel (the closed-loop response)
  aReq    m/s^2 — CC.actuators.accel, what the planner WANTED (the target)
  ccSet   kph   — current DI set speed (v_cruise_actual_kph)
  desired kph   — target set speed = vEgo + aReq*proj (capped at ceiling)
  ceil    kph   — driver ceiling (pedal_speed_kph)
  off     kph   — desired - ccSet (the gap the button logic acts on)
  btn     the button decided this frame (none if in deadband)

The core A/B question — does raising proj make the DI accelerate more? — is
answered by aReq vs aEgo tracking (and off, the set-speed gap it produced).
"""
import time

from opendbc.car.carlog import carlog
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.tesla.values import CruiseButtons

# Stock CC on Pre-AP Model S only operates above ~17 mph (tesla-unity value)
MIN_CRUISE_SPEED_MS = 17.1 * CV.MPH_TO_MS

# Planner accel is projected this far ahead to form a target speed.
# Kept at 1.5 (tesla-unity value). A 2.0 bump was tried to make accel more
# perceptible, then reverted: drive 00000001--05edcaba20 (closed loop at 1.5)
# shows the DI does NOT respond to a larger set-speed gap. When
# the planner wanted accel>0.2 m/s^2, achieved-vs-gap was non-monotonic — the
# bulk (3322 frames at a 3-5 kph gap) delivered only ~0.05 m/s^2, while smaller
# and larger gaps scattered high (reverse causality: grade/lead already driving
# accel). Projection only enlarges the gap, and a bigger gap doesn't buy more
# accel here — the limiter is the DI's inherently soft response to set-speed
# steps, which no feedforward horizon fixes. Any change to this needs an on-road
# A/B, not a value derived from a log recorded at the old gain. The
# NoPedalACC.tlm telemetry (see module docstring) is the instrument for that A/B.
ACCEL_PROJECTION_S = 1.5

# Don't fight the driver: no automated press within this window of a human
# stalk action. tesla-unity used 3000, which is far longer than anything here
# needs and costs up to five press slots at exactly the moment the driver just
# moved the ceiling — so the set speed sits at its old value right after the
# driver asked for a new one. 500 is the smallest value that still clears every
# window it has to:
#   - the double-pull engage window (nap_conf caps it at 400ms), so we never
#     interleave a spoofed frame between the driver's two stalk pulls;
#   - SPOOF_ECHO_WINDOW_MS = 300 in engagement.py — if an RX echo of our own
#     press ever lands late enough to be read as a human press, the resulting
#     self-holdoff is one slot, not three seconds;
#   - it stays under AUTO_ACTION_SPACING_MS, so a driver press costs at most one
#     automated press slot rather than adding a second serialized delay.
HUMAN_ACTION_HOLDOFF_MS = 500
# Spacing between automated presses — the DI needs time to act on each step.
# Bumped 400 -> 600 for smoothness (drive 00000005--165bf7823d accel bookmark).
# Each 1-mph UP step makes the DI surge to ~+1.0 m/s^2 then decay to ~+0.35 over
# ~500 ms. At 400 ms we pressed ~2.5x/s and re-surged before the last one
# settled, holding ccSet ~1-1.4 kph above vEgo continuously even though the
# planner only wanted ~+0.3 m/s^2 — i.e. we over-pressed for the demand and the
# DI kept lurching. 600 ms (~1.7x/s) lets each surge settle and keeps ccSet
# closer to vEgo, so the accel is gentler and less pulsed. Note a floor on
# smoothness: even during a 1.1 s no-press gap the DI's own throttle ripples
# ~+-0.5 m/s^2, which stalk spoofing cannot remove. Iterate from the
# NoPedalACC.tlm cadence if 600 ms still feels jumpy.
AUTO_ACTION_SPACING_MS = 600

# Below this requested accel, stepped set-speed nudges can't track the
# request in time: cc_set_kph gets dragged down by each auto-press, which
# closes calc_button()'s offset_kph gap before it ever crosses the CANCEL
# threshold. Drive log 00000009--2dd6a2315a showed a -1.50 m/s^2 event
# (well under ACCEL_MIN=-3.48) still resolve to a half-step DN_1ST because
# of this self-correction. Bypass the offset math and holdoff/spacing gates
# entirely and drop CC now — the driver is the brakes.
ACCEL_CANCEL_THRESHOLD = -1.3

# Unmet-decel CANCEL. The stepped set-speed presses barely decelerate the car,
# so the right hand-off trigger is not a fixed decel level but "the planner
# wants to slow and the car ISN'T slowing" — i.e. the steps are failing. Fire
# when the planner demands decel (aReq < DECEL_DEMAND_THRESHOLD) AND the achieved
# accel falls short of the demand by DECEL_UNMET_GAP (aReq - aEgo < gap), held
# for DECEL_CANCEL_SUSTAIN_S — then drop CC to regen coast + driver.
#
# Why gap-based, not a fixed threshold (drives 00000006 and 00000007):
#   - A fixed -0.9 held ~44 mph straight at a lead for ~8 s while the planner
#     wanted -0.45 the whole time (00000007) — the steps did nothing and -0.45
#     never crossed -0.9. Too high.
#   - A low fixed threshold nuisance-cancels on genuine slowdowns the DI IS
#     handling. The gap term fixes both: it fires when decel is demanded but not
#     achieved (steps failing / closing on a lead without slowing), and stays
#     quiet when the car is actually decelerating (aEgo tracks aReq) or when the
#     "jumpy decel" is just DI ripple at aReq ~= 0.
# Hard braking still cancels instantly via ACCEL_CANCEL_THRESHOLD. And regardless
# of all this, regen coast tops out near -1.5 m/s^2: no-pedal ACC CANNOT stop for a
# stopped car — the driver is always the brake.
DECEL_DEMAND_THRESHOLD = -0.3   # m/s^2, planner is meaningfully asking to slow
DECEL_UNMET_GAP = -0.3          # m/s^2, achieved decel falls short of demand by this much
DECEL_CANCEL_SUSTAIN_S = 0.7    # s, held before handing off (hard braking bypasses via ACCEL_CANCEL_THRESHOLD)

# The unmet-decel arm above infers "our steps are failing", which presupposes we
# have been stepping. It must stay disarmed for this long after a driver stalk
# speed action: lowering the set speed drops the planner's target instantly, so
# a decel demand exists before the DI has been given any down-press, and that
# reads identically to a real failure. Deliberately much longer than
# HUMAN_ACTION_HOLDOFF_MS — deferring *presses* to the driver wants to be short,
# but deferring the *conclusion that the steps have failed* has to outlast the
# DI's response to a set-speed step. See the derivation at the use site.
DECEL_CANCEL_HUMAN_SETTLE_MS = 2500

# Ceiling-floor climb rate limit. The DI's response to a set-speed gap is
# bang-bang, not proportional — measured over 13559 ENABLED frames on
# 2026-07-26, median aEgo by gap: 0.00 below 1.8 kph, +0.14 at 1.8-2.2, +0.50 at
# 2.2-2.6, then flat at ~+0.64 all the way out past 8 kph. So the car can only
# be given 0 or about +0.6 m/s^2; there is no way to deliver the +0.12 the
# planner typically asks for during a climb except by duty-cycling.
#
# Unlimited, the floor presses every AUTO_ACTION_SPACING_MS, raising the set
# speed at 2.7 kph/s while aReq +0.12 only asks for 0.43 kph/s — six times too
# fast. The set speed runs away, the gap crosses the DI's ~2 kph threshold, the
# car surges at +0.6, overshoots, the planner asks for decel, we step down, and
# it repeats. That is the hunting Brennan bookmarked at 18:44:52 on drive
# 0000002e: ccSet swinging 56.3 <-> 64.4 with the gap parked right on the
# threshold, following far too close throughout.
#
# _floor_climb_interval_ms derives the spacing; this only bounds the degenerate
# tail. Across 3307 floor-climb-eligible frames aReq ran median +0.12, p90
# +0.20, with just 3.9% under +0.05 — and +0.05 is where the formula reaches
# 10 s. Below that the request is at the level of the planner's own equilibrium
# ripple rather than a real demand to speed up.
FLOOR_CLIMB_MAX_INTERVAL_MS = 10000

# A/B telemetry cadence at the 100 Hz carcontroller clock.
TLM_PERIOD_IN_FRAMES = 20    # 5 Hz while modulating (fine enough for accel dynamics)
TLM_PERIOD_OUT_FRAMES = 100  # 1 Hz otherwise, so non-engagement stays visible cheaply

# reason values that count as "no-pedal ACC is operating"
_ACTIVE_REASONS = ("active", "hard_brake_cancel", "sustained_decel_cancel", "human_holdoff", "auto_spacing")


def _current_time_millis():
  return int(round(time.time() * 1000))


def get_cc_step_kph(is_mph):
  """Stock CC set-speed steps in kph: (half press, full press).

  Imperial cars step 1/5 mph, metric cars 1/5 kph.
  """
  if is_mph:
    return 1 * CV.MPH_TO_KPH, 5 * CV.MPH_TO_KPH
  return 1.0, 5.0


class NoPedalACCController:
  """Per-frame stock-CC button decisions. TX ownership stays with StockCCSpoofer."""

  def __init__(self):
    self.last_auto_action_ms = 0
    self.last_floor_climb_ms = 0
    self.last_logged_button = None
    # telemetry scratch, refreshed every _decide()
    self._reason = "init"
    self._desired_kph = None
    self._offset_kph = None
    # timestamp (ms) the current sustained-decel demand began; 0 = not pending
    self._decel_demand_start_ms = 0
    carlog.info(
      "NoPedalACC config: proj=%.2f spacing=%dms holdoff=%dms settle=%dms cancel_thr=%+.2f min_cruise=%.1fmph",
      ACCEL_PROJECTION_S, AUTO_ACTION_SPACING_MS, HUMAN_ACTION_HOLDOFF_MS, DECEL_CANCEL_HUMAN_SETTLE_MS,
      ACCEL_CANCEL_THRESHOLD, MIN_CRUISE_SPEED_MS * CV.MS_TO_MPH)

  def update(self, CC, CS, frame):
    """Return the CruiseButtons value to spoof this frame, or None."""
    button = self._decide(CC, CS)
    self._log_decision(button, CC, CS)
    self._log_telemetry(button, CC, CS, frame)
    if button is not None:
      self.last_auto_action_ms = _current_time_millis()
    return button

  def _decide(self, CC, CS):
    self._reason = "active"
    self._desired_kph = None
    self._offset_kph = None

    # Engagement FSM must be in long mode (double-pull), openpilot long active
    if not (getattr(CS, "cruiseEnabled", False) and getattr(CS, "enableLongControl", False)):
      self._reason = "not_op_long"
      self._decel_demand_start_ms = 0
      return None
    if not CC.longActive:
      self._reason = "not_long_active"
      self._decel_demand_start_ms = 0
      return None
    # Driver on the accelerator: the DI holds CC through it; stay out
    if CS.out.gasPressed:
      self._reason = "gas_pressed"
      self._decel_demand_start_ms = 0
      return None
    # Only modulate a running stock CC — engaging it is StockCCSpoofer's job,
    # and after a CANCEL the driver must double-pull to rearm (no autoresume)
    di_state = getattr(CS, "di_cruise_state", "OFF")
    if di_state != "ENABLED":
      self._reason = "di_" + str(di_state).lower()
      self._decel_demand_start_ms = 0
      return None

    now = _current_time_millis()

    # Hard braking ahead: instant CANCEL (safety cutoff), bypass every gate and
    # skip calc_button()'s offset-based CANCEL path, which self-corrects too
    # fast to ever trigger.
    if CC.actuators.accel < ACCEL_CANCEL_THRESHOLD:
      self._reason = "hard_brake_cancel"
      self._decel_demand_start_ms = 0
      return CruiseButtons.CANCEL

    engagement = getattr(CS, "engagement", None)
    last_human_ms = getattr(engagement, "last_stalk_non_cancel_ms", -10000) if engagement else -10000

    # Unmet decel: the planner wants to slow but the car isn't (stepped decel is
    # failing) — hand off to regen coast + driver instead of holding speed into a
    # lead. Bypasses the holdoff/spacing gates below — a slowdown shouldn't wait.
    #
    # But it must not run right after the driver moves the set speed. Lowering
    # the ceiling drops the planner's target, so a decel demand appears
    # immediately while the DI has not been given a down-press yet — the exact
    # "wants to slow, isn't slowing" signature, produced by the driver rather
    # than by failing steps. Drive 0000002e: ceiling 72.4 -> 64.4 just before
    # f=13000 at 43.7 mph, CANCEL at f=13180 (1.8 s later) with `reason` reading
    # human_holdoff then auto_spacing the whole way — it concluded the steps had
    # failed without having taken one. CC dropped to regen coast and the car
    # bled from 43.7 mph toward 25.
    #
    # 2500 ms is measured, not guessed: across 14 downward ceiling changes in
    # that day's logs the DI began actually slowing (aEgo < -0.3) within 0.4-1.2 s
    # in 13 of them, with one 3.2 s outlier, and the false CANCEL landed at 1.8 s.
    # Hard braking is untouched — ACCEL_CANCEL_THRESHOLD above returns before
    # this, so a genuine emergency still hands off instantly.
    a_ego = float(getattr(CS.out, "aEgo", 0.0))
    a_req = float(CC.actuators.accel)
    settling = now - last_human_ms < DECEL_CANCEL_HUMAN_SETTLE_MS
    decel_unmet = a_req < DECEL_DEMAND_THRESHOLD and (a_req - a_ego) < DECEL_UNMET_GAP
    if decel_unmet and not settling:
      if self._decel_demand_start_ms == 0:
        self._decel_demand_start_ms = now
      elif now - self._decel_demand_start_ms >= DECEL_CANCEL_SUSTAIN_S * 1000:
        self._reason = "sustained_decel_cancel"
        return CruiseButtons.CANCEL
    else:
      self._decel_demand_start_ms = 0

    if now - last_human_ms < HUMAN_ACTION_HOLDOFF_MS:
      self._reason = "human_holdoff"
      return None
    if now - self.last_auto_action_ms < AUTO_ACTION_SPACING_MS:
      self._reason = "auto_spacing"
      return None

    cc_set_kph = getattr(CS, "v_cruise_actual_kph", 0.0)
    ceiling_kph = getattr(CS, "pedal_speed_kph", 0.0)
    a_req = float(CC.actuators.accel)
    desired_kph = CS.out.vEgo * CV.MS_TO_KPH + a_req * ACCEL_PROJECTION_S * CV.MS_TO_KPH
    # Reach the ceiling. The projected target undershoots near the max: as the
    # car approaches it, aReq -> 0 so desired -> vEgo, leaving the offset below
    # one press step and the set speed stalling under the driver's max
    # (drive 093aeba9a: ccSet 66.0 held while ceil 69.2, aReq +0.22, off +0.3).
    # When the planner isn't braking, floor the target at one step above the
    # current set so it keeps climbing to the ceiling (capped there); the big
    # accel ramp is unaffected because the projected target dominates then.
    #
    # The gate is strictly "the planner is not asking to slow at all". It must
    # not extend to mild decel demands, however small they look. A version
    # gated at DECEL_DEMAND_THRESHOLD (-0.3) drove drive 00000018 into a lead
    # twice: following at 50 mph the demand oscillated either side of -0.3, so
    # every frame it eased off, the floor re-raised the target to the ceiling
    # and cancelled out the down-presses — three DN_1ST then three UP_1ST inside
    # four seconds, set speed back to 82.1 kph, headway falling 1.22 s -> 0.61 s
    # with the DI still pulling because the set speed sat 3 mph above vEgo. Any
    # decel demand has to leave the target where the projection put it, which is
    # below the current set whenever the set speed is above the actual speed —
    # that is the only mechanism here that walks it back down.
    # The floor may only *amplify* the projection, never reverse it. Drive
    # 0000002e (18:14:51-56) is why: settled behind a lead at 60.0 kph with the
    # set at 61.2, the planner sat at aReq +0.01..+0.16 — "this speed is fine",
    # not "accelerate". The projection agreed, targeting 60.0, i.e. below the
    # current set. The old gate was `a_req >= 0.0` alone, so the floor threw
    # that away and substituted cc_set + one step. Each press raised the set,
    # which raised the next target: 7 UP presses in 3.6 s, set speed 61.2 ->
    # 72.4, and the DI delivered aEgo +0.8 against a request of +0.1 — a 7x
    # overshoot straight at the lead the car had just slowed for.
    #
    # Requiring the projection to already point at or above the current set
    # separates that from the stall the floor exists for (drive 093aeba9a:
    # ccSet 66.0, ceiling 69.2, aReq +0.22, projection 66.35 — genuinely above
    # the set, just short of calc_button's deadband). Anchoring the floor on
    # min(cc_set, vEgo) instead does NOT separate them: the set speed sits
    # ~1 kph above vEgo in both.
    floor_applied = False
    if a_req >= 0.0:
      half_kph, _ = get_cc_step_kph(CS.speed_units == "MPH")
      if desired_kph >= cc_set_kph:
        # Planner wants to gain on the current set. One step above it (+ a hair
        # to clear calc_button's offset >= half boundary), so it climbs at the
        # gentle half-step rate to the ceiling rather than jumping (targeting
        # the ceiling directly would give a full-step offset and 5 mph lurches
        # when far below it).
        floor_kph = cc_set_kph + half_kph + 0.05
        if desired_kph < floor_kph:
          # The floor, not the projection, is what would drive this press, so
          # rate-limit it. See FLOOR_CLIMB_* for why: unlimited, the floor
          # raises the set speed far faster than the planner asked, which is
          # what made the car hunt behind a lead on drive 0000002e 18:44:52.
          if now - self.last_floor_climb_ms >= self._floor_climb_interval_ms(a_req, half_kph):
            desired_kph = floor_kph
            floor_applied = True
        else:
          desired_kph = max(desired_kph, floor_kph)
      else:
        # Planner is content, but the set speed sits above the projection. That
        # is the NORMAL state while the car is still catching up to its set
        # speed, and it must not read as overspeed: drive 0000002e 18:39:06 had
        # vEgo 63.7, set 72.4, aReq +0.30 — a projection of 65.3, a 7.1 kph
        # "gap", and calc_button answered with a full 5 mph DECEL_2ND out of
        # nowhere. Hold the target at the set speed so no press is generated.
        #
        # Holding, not climbing, is the whole point. Climbing to cc_set + one
        # step is what ratcheted at 18:14:52 the day before: each press raised
        # the set, which raised the next target, 61.2 -> 72.4 kph in 3.6 s at a
        # lead. Hold keeps both failures out: no slam, no ratchet.
        desired_kph = cc_set_kph
    self._desired_kph = min(desired_kph, ceiling_kph)
    self._offset_kph = self._desired_kph - cc_set_kph
    button = self.calc_button(
      desired_kph=desired_kph,
      cc_set_kph=cc_set_kph,
      ceiling_kph=ceiling_kph,
      v_ego=CS.out.vEgo,
      is_mph=CS.speed_units == "MPH",
    )
    # Only a press the floor actually produced restarts its interval; a press
    # the projection would have made on its own is governed by normal spacing.
    if floor_applied and CruiseButtons.is_accel(button):
      self.last_floor_climb_ms = now
    return button

  @staticmethod
  def _floor_climb_interval_ms(a_req, half_kph):
    """Minimum spacing between floor-driven climb presses, in ms.

    Never raise the set speed faster than the planner's own requested
    acceleration would raise the actual speed: one step of set speed per
    step / a_req seconds. Derivation, from the measured DI response:

      interval = (step / closure_rate) * (A_on / a_req)
               = (half_kph / (A_on * 3.6)) * (A_on / a_req)
               = half_kph / (3.6 * a_req)

    A_on cancels, so this holds whatever the DI's gain turns out to be.
    """
    interval_ms = half_kph / (3.6 * max(a_req, 1e-3)) * 1000.0
    return min(max(interval_ms, AUTO_ACTION_SPACING_MS), FLOOR_CLIMB_MAX_INTERVAL_MS)

  @staticmethod
  def calc_button(desired_kph, cc_set_kph, ceiling_kph, v_ego, is_mph):
    """Pure decision table (tesla-unity _calc_button, modernized)."""
    half_kph, full_kph = get_cc_step_kph(is_mph)
    min_cruise_kph = MIN_CRUISE_SPEED_MS * CV.MS_TO_KPH

    desired_kph = min(desired_kph, ceiling_kph)

    # Below the stock CC floor there is nothing to modulate — drop CC, coast
    if desired_kph < min_cruise_kph:
      return CruiseButtons.CANCEL

    offset_kph = desired_kph - cc_set_kph

    button = None
    if offset_kph < -2 * full_kph and cc_set_kph > 0:
      # Way over: CC off gives the strongest available decel (coast/regen)
      button = CruiseButtons.CANCEL
    elif offset_kph < -0.6 * full_kph:
      button = CruiseButtons.DECEL_2ND
    elif offset_kph < -0.9 * half_kph:
      button = CruiseButtons.DECEL_SET
    elif v_ego > MIN_CRUISE_SPEED_MS:
      # Speed up only with headroom under the driver's ceiling
      if offset_kph >= full_kph and cc_set_kph + full_kph <= ceiling_kph + 0.1:
        button = CruiseButtons.RES_ACCEL_2ND
      elif offset_kph >= half_kph and cc_set_kph + half_kph <= ceiling_kph + 0.1:
        button = CruiseButtons.RES_ACCEL

    # SCCM crash guard (tesla-unity): repeated decel presses at the min cruise
    # speed crash the stalk module — CANCEL instead of stepping below it.
    if CruiseButtons.is_decel(button) and cc_set_kph - half_kph < min_cruise_kph:
      button = CruiseButtons.CANCEL

    return button

  def _log_decision(self, button, CC, CS):
    """Log decision changes, not every frame."""
    if button == self.last_logged_button:
      return
    self.last_logged_button = button
    if button is not None:
      carlog.warning(
        "NoPedalACC press %s | vEgo=%.1f accel=%+.2f ccSet=%.1fkph ceiling=%.1fkph di=%s longActive=%s",
        _BUTTON_NAMES.get(button, button), CS.out.vEgo, float(CC.actuators.accel),
        getattr(CS, "v_cruise_actual_kph", 0.0), getattr(CS, "pedal_speed_kph", 0.0),
        getattr(CS, "di_cruise_state", "OFF"), CC.longActive)

  def _log_telemetry(self, button, CC, CS, frame):
    """Continuous A/B tuning telemetry — see the module docstring for the schema.

    Throttled: 5 Hz while modulating, 1 Hz otherwise. The pair (aReq, aEgo) is
    the closed-loop tracking record; proj labels the run; reason exposes the
    engagement gating so 'why isn't it engaging' is answerable from the log.
    """
    region = self._reason in _ACTIVE_REASONS
    period = TLM_PERIOD_IN_FRAMES if region else TLM_PERIOD_OUT_FRAMES
    if frame % period != 0:
      return

    a_ego = float(getattr(CS.out, "aEgo", 0.0))
    desired = self._desired_kph if self._desired_kph is not None else -1.0
    offset = self._offset_kph if self._offset_kph is not None else 0.0

    carlog.info(
      "NoPedalACC.tlm f=%d proj=%.2f region=%d reason=%s di=%s long=%d gas=%d units=%s " +
      "vEgo=%.2f aEgo=%+.2f aReq=%+.2f ccSet=%.1f desired=%.1f ceil=%.1f off=%+.1f btn=%s",
      frame, ACCEL_PROJECTION_S, int(region), self._reason,
      getattr(CS, "di_cruise_state", "OFF"), int(CC.longActive), int(CS.out.gasPressed),
      getattr(CS, "speed_units", "?"),
      CS.out.vEgo, a_ego, float(CC.actuators.accel),
      getattr(CS, "v_cruise_actual_kph", 0.0), desired,
      getattr(CS, "pedal_speed_kph", 0.0), offset,
      _BUTTON_NAMES.get(button, "none" if button is None else str(button)))


_BUTTON_NAMES = {
  CruiseButtons.CANCEL: "CANCEL",
  CruiseButtons.RES_ACCEL: "UP_1ST",
  CruiseButtons.RES_ACCEL_2ND: "UP_2ND",
  CruiseButtons.DECEL_SET: "DN_1ST",
  CruiseButtons.DECEL_2ND: "DN_2ND",
}
