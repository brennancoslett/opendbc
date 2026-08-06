import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HistoricalMutation:
  name: str
  source_path: str
  original: bytes
  replacement: bytes
  test_node: str


MUTATIONS = (
  HistoricalMutation(
    name="four-reset-acquisition-window",
    source_path="opendbc/car/tesla/preap/carcontroller.py",
    original=b"  MAX_RESET_ATTEMPTS = 4\n",
    replacement=b"  MAX_RESET_ATTEMPTS = 3\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_pedal_authority.py::" +
      "test_acquisition_sends_four_resets_then_fails_on_next_update"
    ),
  ),
  HistoricalMutation(
    name="failed-state-late-acquire",
    source_path="opendbc/car/tesla/preap/carcontroller.py",
    original=(
      b"    if self.state == PedalAuthorityState.FAILED:\n" +
      b"      return PedalCommandAction.NONE\n"
    ),
    replacement=(
      b"    if self.state == PedalAuthorityState.FAILED:\n" +
      b"      self.state = PedalAuthorityState.INACTIVE\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_pedal_authority.py::" +
      "test_failed_request_cannot_late_acquire_and_rearms_only_after_falling_edge"
    ),
  ),
  HistoricalMutation(
    name="measured-acceleration-command-seed",
    source_path="opendbc/car/tesla/preap/carcontroller.py",
    original=b"          commanded_accel=0.0,\n",
    replacement=b"          commanded_accel=CS.out.aEgo,\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_engage_reset_starts_estimator_from_measured_acceleration"
    ),
  ),
  HistoricalMutation(
    name="observation-rewrites-command-state",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    self._track_measured_accel(a_ego)\n" +
      b"    self.inner_pid.reset()\n"
    ),
    replacement=(
      b"    self._track_measured_accel(a_ego)\n" +
      b"    self.jerk_limiter.reset(a_ego)\n" +
      b"    self.inner_pid.reset()\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVirtualDAS::" +
      "test_observe_does_not_mutate_commanded_jerk_state"
    ),
  ),
  HistoricalMutation(
    name="repeated-disabled-release",
    source_path="opendbc/car/tesla/preap/carcontroller.py",
    original=(
      b"      action = PedalCommandAction.RELEASE if self.state == PedalAuthorityState.ACTIVE " +
      b"else PedalCommandAction.NONE\n"
    ),
    replacement=b"      action = PedalCommandAction.RELEASE\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_pedal_authority.py::" +
      "test_healthy_request_acquires_then_releases_once_and_stays_silent"
    ),
  ),
  HistoricalMutation(
    name="pedal-failure-drops-lateral",
    source_path="opendbc/car/tesla/preap/engagement.py",
    original=(
      b"    self.pedal_unavailable = True\n" +
      b"    self._drop_longitudinal_keep_lateral()\n"
    ),
    replacement=(
      b"    self.pedal_unavailable = True\n" +
      b"    self.cruiseEnabled = False\n" +
      b"    self.enableLongControl = False\n" +
      b"    self.enableJustCC = False\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_pedal_authority.py::" +
      "test_controller_failure_drops_only_longitudinal_and_latches_unavailable"
    ),
  ),
  HistoricalMutation(
    name="generic-outer-integral-feedback",
    source_path="opendbc/car/tesla/preap/constants.py",
    original=b"PEDAL_LONG_KI_V = [0.0, 0.0, 0.0, 0.0]\n",
    replacement=b"PEDAL_LONG_KI_V = [0.05, 0.08, 0.10, 0.15]\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_longitudinal_tuning.py::" +
      "test_pedal_params_leave_generic_outer_feedback_disabled"
    ),
  ),
  HistoricalMutation(
    name="planner-feedforward-passthrough-disabled",
    source_path="opendbc/car/tesla/preap/interface.py",
    original=b"      ret.longitudinalTuning.kf = 1.0\n",
    replacement=b"      ret.longitudinalTuning.kf = 0.0\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_longitudinal_tuning.py::" +
      "test_pedal_params_leave_generic_outer_feedback_disabled"
    ),
  ),
  HistoricalMutation(
    name="sluggish-inner-acceleration-feedback",
    source_path="opendbc/car/tesla/preap/constants.py",
    original=b"VDAS_INNER_KI_V = [0.3, 0.2, 0.15]\n",
    replacement=b"VDAS_INNER_KI_V = [0.15, 0.10, 0.075]\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_inner_feedback_holds_cruise_against_sustained_road_load"
    ),
  ),
  HistoricalMutation(
    name="hard-inner-error-deadband-restored",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    error = self._gate_pid_error_noise(error, freeze_integrator)\n",
    replacement=(
      b"    if abs(error) < PID_ERROR_DEADBAND:\n" +
      b"      error = 0.0\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_persistent_sub_deadband_error_earns_residual_authority[0.0196]"
    ),
  ),
  HistoricalMutation(
    name="inner-error-noise-gate-call-bypassed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    error = self._gate_pid_error_noise(error, freeze_integrator)\n",
    replacement=b"    error = error\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_sub_deadband_sign_changing_noise_does_not_accumulate_residual_authority"
    ),
  ),
  HistoricalMutation(
    name="persistent-error-dwell-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    if self.persistent_error_elapsed_s < PID_PERSISTENT_ERROR_DWELL_S:\n",
    replacement=b"    if False:\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_sign_changing_sub_deadband_error_never_completes_dwell"
    ),
  ),
  HistoricalMutation(
    name="persistent-error-sign-reset-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    if error_sign != self.persistent_error_sign:\n" +
      b"      self.persistent_error_sign = error_sign\n" +
      b"      self.persistent_error_elapsed_s = self.dt\n"
    ),
    replacement=(
      b"    if error_sign != self.persistent_error_sign:\n" +
      b"      self.persistent_error_sign = error_sign\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_sign_changing_sub_deadband_error_never_completes_dwell"
    ),
  ),
  HistoricalMutation(
    name="persistent-error-freeze-reset-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    if freeze_integrator or error == 0.0:\n",
    replacement=b"    if error == 0.0:\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_persistent_error_dwell_restarts_after_freeze_observe_and_reset"
    ),
  ),
  HistoricalMutation(
    name="persistent-error-observe-reset-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    self.inner_pid.reset()\n" +
      b"    self._reset_negative_handoff()\n" +
      b"    self._reset_persistent_error()\n\n" +
      b"  def reset("
    ),
    replacement=(
      b"    self.inner_pid.reset()\n" +
      b"    self._reset_negative_handoff()\n\n" +
      b"  def reset("
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_persistent_error_dwell_restarts_after_freeze_observe_and_reset"
    ),
  ),
  HistoricalMutation(
    name="persistent-error-command-reset-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    self._reset_negative_handoff()\n" +
      b"    self._reset_persistent_error()\n\n" +
      b"  def _gate_pid_error_noise("
    ),
    replacement=(
      b"    self._reset_negative_handoff()\n\n" +
      b"  def _gate_pid_error_noise("
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestInnerPID::" +
      "test_persistent_error_dwell_restarts_after_freeze_observe_and_reset"
    ),
  ),
  HistoricalMutation(
    name="steady-grade-removed-from-effort",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"      a_limited + steady_grade_compensation + transient_pitch_compensation,\n"
    ),
    replacement=b"      a_limited + transient_pitch_compensation,\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_preserved_grade_handoff_holds_net_acceleration_for_1p5_seconds[0.5-15.0]"
    ),
  ),
  HistoricalMutation(
    name="steady-grade-subtracted-from-net-feedback",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    a_ego_filtered, j_ego = self._track_measured_accel(a_ego)\n"
    ),
    replacement=(
      b"    a_ego_filtered, j_ego = self._track_measured_accel(a_ego - steady_grade_compensation)\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_steady_grade_hold_does_not_double_compensate[0.5-15.0]"
    ),
  ),
  HistoricalMutation(
    name="engage-grade-bypasses-effort-envelope",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    effort_min, effort_max = accel_effort_limits or (REGEN_MAX, ACCEL_MAX)\n",
    replacement=b"    effort_min, effort_max = REGEN_MAX, ACCEL_MAX\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_engage_effort_limits_include_grade_compensation"
    ),
  ),
  HistoricalMutation(
    name="engage-grade-bypasses-pedal-slew-envelope",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    pedal_di = self._rate_limit(\n" +
      b"      pedal_di_bounded,\n" +
      b"      prev_pedal_di,\n" +
      b"      pedal_ramp_rate_up,\n" +
      b"      pedal_ramp_rate_down,\n" +
      b"    )\n"
    ),
    replacement=(
      b"    pedal_di = self._rate_limit(\n" +
      b"      pedal_di_bounded,\n" +
      b"      prev_pedal_di,\n" +
      b"      PEDAL_RAMP_RATE_UP,\n" +
      b"      pedal_ramp_rate_down,\n" +
      b"    )\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_engage_pedal_ramp_limit_applies_after_feedforward"
    ),
  ),
  HistoricalMutation(
    name="engage-slew-stops-before-pedal-catches-up",
    source_path="opendbc/car/tesla/preap/carcontroller.py",
    original=(
      b"          pedal_ramp_rate_up = (\n" +
      b"            ENGAGE_GRACE_PEDAL_RAMP_RATE_UP\n" +
      b"            if self.preap_long_handoff_slew_active\n" +
      b"            else PEDAL_RAMP_RATE_UP\n" +
      b"          )\n"
    ),
    replacement=b"          pedal_ramp_rate_up = PEDAL_RAMP_RATE_UP\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_pedal_authority.py::" +
      "test_non_timeout_gas_override_release_has_no_launch_for_1p5_seconds"
    ),
  ),
  HistoricalMutation(
    name="full-strength-transient-grade-overshoot",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"TRANSIENT_GRADE_GAIN = 0.4\n",
    replacement=b"TRANSIENT_GRADE_GAIN = 1.0\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_flat_to_grade_step_improves_on_no_pitch_baseline_at_1p5_seconds[0.4]"
    ),
  ),
  HistoricalMutation(
    name="steady-grade-pitch-outlier-unbounded",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    pitch = float(clip(orientation_ned[1], -maximum_pitch, maximum_pitch))\n",
    replacement=b"    pitch = orientation_ned[1]\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestGradeEstimator::" +
      "test_sustained_pitch_outlier_cannot_exceed_steady_grade_limit"
    ),
  ),
  HistoricalMutation(
    name="orientation-dropout-skips-short-hold",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    dropout_decay_elapsed_s = self.missing_orientation_elapsed_s - " +
      b"ORIENTATION_DROPOUT_HOLD_S\n"
    ),
    replacement=b"    dropout_decay_elapsed_s = self.missing_orientation_elapsed_s\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestGradeEstimator::" +
      "test_orientation_dropout_holds_then_decays_steady_grade[0.5-1.0]"
    ),
  ),
  HistoricalMutation(
    name="orientation-dropout-disables-bounded-decay",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    self.pitch_lp.x = self.pitch_before_dropout_rad * dropout_grade_scale\n",
    replacement=b"    self.pitch_lp.x = self.pitch_before_dropout_rad\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestGradeEstimator::" +
      "test_orientation_dropout_holds_then_decays_steady_grade[4.64-0.0]"
    ),
  ),
  HistoricalMutation(
    name="orientation-dropout-reset-retains-stale-state",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    self._clear_high_pass_state()\n" +
      b"    self.missing_orientation_elapsed_s = 0.0\n" +
      b"    self.pitch_before_dropout_rad = 0.0\n"
    ),
    replacement=b"",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestGradeEstimator::" +
      "test_reset_clears_grade"
    ),
  ),
  HistoricalMutation(
    name="legacy-positive-fallback-actuator-gain",
    source_path="opendbc/car/tesla/preap/ff_table_default.py",
    original=(
      b"    [-5.00, -3.33, -1.67,  0.00,  9.62, 19.24, 28.86, 38.48, 48.10],  # 20 m/s\n" +
      b"    [-5.00, -3.33, -1.67,  0.00, 10.66, 21.32, 31.98, 42.64, 53.30],  # 30 m/s\n"
    ),
    replacement=(
      b"    [-5.00, -3.33, -1.67,  0.00, 14.80, 29.60, 44.40, 59.20, 74.00],  # 20 m/s\n" +
      b"    [-5.00, -3.33, -1.67,  0.00, 16.40, 32.80, 49.20, 65.60, 82.00],  # 30 m/s\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_route_shaped_positive_transition_bounds_delivered_acceleration"
    ),
  ),
  HistoricalMutation(
    name="legacy-5mps-positive-fallback-transition",
    source_path="opendbc/car/tesla/preap/ff_table_default.py",
    original=b"    [-5.00, -3.33, -1.67,  0.00,  7.54, 15.08, 22.62, 30.16, 37.70],  # 5 m/s\n",
    replacement=b"    [-5.00, -3.33, -1.67,  0.00, 11.60, 23.20, 34.80, 46.40, 58.00],  # 5 m/s\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_speed_ramp_through_fallback_transition_is_smooth"
    ),
  ),
  HistoricalMutation(
    name="focused-job-skip-bypass",
    source_path=".github/workflows/tests.yml",
    original=(
      b"  tesla_preap_longitudinal_regression:\n" +
      b"    name: Tesla Pre-AP longitudinal regressions\n"
    ),
    replacement=(
      b"  tesla_preap_longitudinal_regression:\n" +
      b"    if: false\n" +
      b"    name: Tesla Pre-AP longitudinal regressions\n"
    ),
    test_node=(
      ".github/ci/test_tests_workflow.py::TestWorkflowContract::" +
      "test_focused_longitudinal_job_cannot_be_skipped_or_soft_failed"
    ),
  ),
  HistoricalMutation(
    name="focused-job-soft-failure-bypass",
    source_path=".github/workflows/tests.yml",
    original=(
      b"  tesla_preap_longitudinal_regression:\n" +
      b"    name: Tesla Pre-AP longitudinal regressions\n"
    ),
    replacement=(
      b"  tesla_preap_longitudinal_regression:\n" +
      b"    continue-on-error: true\n" +
      b"    name: Tesla Pre-AP longitudinal regressions\n"
    ),
    test_node=(
      ".github/ci/test_tests_workflow.py::TestWorkflowContract::" +
      "test_focused_longitudinal_job_cannot_be_skipped_or_soft_failed"
    ),
  ),
  HistoricalMutation(
    name="residual-trim-after-feedforward",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    pedal_di_unclipped = self._feedforward(accel_effort, v_ego)\n",
    replacement=(
      b"    pedal_di_unclipped = self._feedforward(base_accel_effort, v_ego) + accel_trim\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_residual_feedback_enters_feedforward_in_acceleration_domain"
    ),
  ),
  HistoricalMutation(
    name="retained-integral-without-authority-clamp",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    self.inner_pid.i = float(clip(\n" +
      b"      self.inner_pid.i,\n" +
      b"      self.inner_pid.neg_limit,\n" +
      b"      self.inner_pid.pos_limit,\n" +
      b"    ))\n"
    ),
    replacement=b"    self.inner_pid.i = self.inner_pid.i\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_retained_integral_is_clipped_to_remaining_acceleration_authority"
    ),
  ),
  HistoricalMutation(
    name="negative-handoff-integral-shaping-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"NEGATIVE_HANDOFF_INTEGRAL_SLEW = 0.25  # m/s\xc2\xb3\n",
    replacement=b"NEGATIVE_HANDOFF_INTEGRAL_SLEW = 100.0  # m/s\xc2\xb3\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_negative_command_handoff_keeps_grade_effort_separate"
    ),
  ),
  HistoricalMutation(
    name="negative-handoff-braking-effort-budget-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"      shared_effort_min = max(\n" +
      b"        effort_min,\n" +
      b"        previous_accel_effort - VDAS_DECEL_JERK_MAX * self.dt,\n" +
      b"      )\n"
    ),
    replacement=b"      shared_effort_min = effort_min\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_handoff_shares_command_jerk_budget_in_both_directions"
    ),
  ),
  HistoricalMutation(
    name="negative-handoff-recovery-effort-budget-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"      shared_effort_max = min(\n" +
      b"        effort_max,\n" +
      b"        previous_accel_effort + VDAS_ACCEL_JERK_MAX * self.dt,\n" +
      b"      )\n"
    ),
    replacement=b"      shared_effort_max = effort_max\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_handoff_shares_command_jerk_budget_in_both_directions"
    ),
  ),
  HistoricalMutation(
    name="negative-handoff-comfort-pedal-step-removed",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"NEGATIVE_HANDOFF_PEDAL_STEP = 0.50  # DI/update\n",
    replacement=b"NEGATIVE_HANDOFF_PEDAL_STEP = PEDAL_RAMP_RATE_DOWN  # DI/update\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_vdas_grade_control.py::" +
      "test_handoff_full_brake_and_recovery_stay_inside_comfort_pedal_step"
    ),
  ),
  HistoricalMutation(
    name="hidden-physical-profile-clipping",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=(
      b"    pedal_di_bounded = float(clip(pedal_di_unclipped, _pedal_di_floor(), max_pedal_value))\n"
    ),
    replacement=(
      b"    pedal_di_unclipped = float(clip(pedal_di_unclipped, _pedal_di_floor(), max_pedal_value))\n" +
      b"    pedal_di_bounded = pedal_di_unclipped\n"
    ),
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_physical_pedal_rail_freezes_and_unwinds_acceleration_integral"
    ),
  ),
  HistoricalMutation(
    name="final-slew-without-anti-windup",
    source_path="opendbc/car/tesla/preap/virtual_das.py",
    original=b"    if physical_bound_blocks_error or slew_bound_blocks_error:\n",
    replacement=b"    if physical_bound_blocks_error:\n",
    test_node=(
      "opendbc/car/tesla/preap/tests/test_virtual_das.py::TestVDASDomainBoundaries::" +
      "test_final_di_slew_backstop_freezes_and_unwinds_acceleration_integral"
    ),
  ),
)


class JUnitReportError(RuntimeError):
  pass


def run_pytest(
    repo_root: Path,
    test_nodes: tuple[str, ...],
    junit_path: Path,
) -> subprocess.CompletedProcess[str]:
  environment = os.environ.copy()
  environment["PYTHONDONTWRITEBYTECODE"] = "1"
  environment["PYTHONPATH"] = str(repo_root)
  return subprocess.run(
    [
      sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider",
      f"--junitxml={junit_path}", *test_nodes,
    ],
    cwd=repo_root,
    env=environment,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
  )


def copy_repo(destination: Path) -> None:
  ignored_names = shutil.ignore_patterns(
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".cache",
    ".hypothesis",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "*.pyc",
    "*.pyo",
  )
  shutil.copytree(REPO_ROOT, destination, ignore=ignored_names)


def apply_mutation(repo_root: Path, mutation: HistoricalMutation) -> None:
  source_path = repo_root / mutation.source_path
  source = source_path.read_bytes()
  match_count = source.count(mutation.original)
  if match_count != 1:
    raise RuntimeError(
      f"{mutation.name}: expected one source match in {mutation.source_path}, found {match_count}"
    )
  source_path.write_bytes(source.replace(mutation.original, mutation.replacement, 1))


def junit_testcases(junit_path: Path) -> list[ET.Element]:
  try:
    return list(ET.parse(junit_path).iter("testcase"))
  except (OSError, ET.ParseError) as exc:
    raise JUnitReportError(f"cannot read {junit_path.name}: {exc}") from exc


def has_only_assertion_failures(testcases: list[ET.Element]) -> bool:
  failures = [failure for testcase in testcases for failure in testcase.findall("failure")]
  errors = [error for testcase in testcases for error in testcase.findall("error")]
  return (
    bool(failures)
    and not errors
    and all(
      (failure.get("type") or "").endswith("AssertionError")
      or (failure.get("message") or "").startswith("AssertionError:")
      or "AssertionError" in (failure.text or "")
      for failure in failures
    )
  )


def main() -> int:
  baseline_nodes = tuple(dict.fromkeys(mutation.test_node for mutation in MUTATIONS))
  with tempfile.TemporaryDirectory(prefix="tesla-preap-longitudinal-mutations-") as temp_dir:
    temp_root = Path(temp_dir)
    baseline_xml = temp_root / "baseline.xml"
    baseline = run_pytest(REPO_ROOT, baseline_nodes, baseline_xml)
    if baseline.returncode != 0:
      print("BASELINE FAILED: longitudinal regression tests did not pass")
      print(baseline.stdout)
      return 1
    try:
      baseline_testcases = junit_testcases(baseline_xml)
    except JUnitReportError as exc:
      print(f"BASELINE INVALID: {exc}")
      return 1
    print(f"BASELINE PASS: {len(baseline_testcases)} tests across {len(baseline_nodes)} nodes")

    survivors = []
    for mutation in MUTATIONS:
      mutant_root = temp_root / mutation.name
      copy_repo(mutant_root)
      apply_mutation(mutant_root, mutation)

      junit_path = temp_root / f"{mutation.name}.xml"
      result = run_pytest(mutant_root, (mutation.test_node,), junit_path)
      try:
        mutation_testcases = junit_testcases(junit_path)
      except JUnitReportError as exc:
        print(f"INVALID: {mutation.name} {exc}")
        return 1
      if result.returncode == 1 and has_only_assertion_failures(mutation_testcases):
        print(f"KILLED: {mutation.name} [{mutation.test_node}]")
      elif result.returncode == 0:
        survivors.append(mutation.name)
        print(f"SURVIVED: {mutation.name} [{mutation.test_node}]")
      else:
        print(f"INVALID: {mutation.name} exited without assertion-only test failures "
              + f"(pytest status {result.returncode})")
        print(result.stdout)
        return 1

    if survivors:
      print(f"Historical mutations survived: {', '.join(survivors)}")
      return 1

  print(f"ALL KILLED: {len(MUTATIONS)} historical mutations")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
