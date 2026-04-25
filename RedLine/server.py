"""
ClinicalPilot — OpenEnv server (Environment).

Implements:
  reset()  → ClinicalObservation
  step()   → (ClinicalObservation, float, bool)
  state()  → EpisodeState

Reward design
─────────────
  +2   per completed protocol section (endpoint, criteria, stats, analysis)
  +1   every step that has ZERO consistency warnings   (dense signal)
  -2   per consistency warning                          (dense penalty)
  +5   if agent takes an action AFTER drift injection   (drift awareness)
  +15  FDA APPROVE
  +5   FDA REVISE  (partial credit)
  -10  FDA REJECT
  episode terminates at step 50 or after FDA review.
"""
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

from .models import (
    ClinicalAction,
    ClinicalObservation,
    ConsistencyWarning,
    EpisodeState,
    FDAVerdict,
    ProtocolState,
    ToolName,
)

GUIDANCE_V1 = (
    "FDA Guidance (v1): Primary endpoint must be Overall Survival (OS) or "
    "Progression-Free Survival (PFS) for oncology Phase 2 trials. "
    "Sample size must achieve power >= 0.80."
)

GUIDANCE_V2 = (
    "FDA Guidance UPDATE (v2 - SCHEMA DRIFT INJECTED): "
    "As of this update, Overall Response Rate (ORR) is now also acceptable as a "
    "primary endpoint for accelerated-approval oncology trials. "
    "Additionally, power requirement raised to >= 0.85. "
    "WARNING: If your protocol uses power < 0.85, you must revise the statistical plan."
)

VALID_PRIMARY_ENDPOINTS_V1 = {"os", "overall survival", "pfs", "progression-free survival"}
VALID_PRIMARY_ENDPOINTS_V2 = VALID_PRIMARY_ENDPOINTS_V1 | {
    "orr", "overall response rate", "objective response rate"
}

VALID_ANALYSIS_METHODS = {
    "kaplan-meier", "cox proportional hazards", "log-rank test",
    "recist criteria", "bayesian adaptive", "irrecist"
}


def _compute_sample_size(effect_size: float, alpha: float, power: float) -> int:
    if effect_size <= 0:
        return 9999
    z_alpha = {0.05: 1.96, 0.025: 2.24, 0.01: 2.576}.get(alpha, 1.96)
    z_beta  = {0.80: 0.842, 0.85: 1.036, 0.90: 1.282}.get(power, 0.842)
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    return max(10, math.ceil(n))


def _check_consistency(proto: ProtocolState, drift_injected: bool) -> list[ConsistencyWarning]:
    warnings: list[ConsistencyWarning] = []

    if proto.primary_endpoint:
        ep_lower = proto.primary_endpoint.lower().strip()
        valid_set = VALID_PRIMARY_ENDPOINTS_V2 if drift_injected else VALID_PRIMARY_ENDPOINTS_V1
        if ep_lower not in valid_set:
            warnings.append(ConsistencyWarning(
                field="primary_endpoint",
                message=(
                    f"'{proto.primary_endpoint}' is not an FDA-accepted primary endpoint. "
                    f"Accepted: {sorted(valid_set)}"
                )
            ))

    if proto.power is not None:
        min_power = 0.85 if drift_injected else 0.80
        if proto.power < min_power:
            warnings.append(ConsistencyWarning(
                field="power",
                message=(
                    f"Statistical power {proto.power} is below the required "
                    f"minimum of {min_power} (post-drift requirement)."
                )
            ))

    if proto.primary_endpoint and proto.analysis_methods:
        ep_lower = proto.primary_endpoint.lower()
        has_time_to_event = any(
            m in ("kaplan-meier", "cox proportional hazards", "log-rank test")
            for m in [m.lower() for m in proto.analysis_methods]
        )
        is_time_to_event_ep = any(
            kw in ep_lower for kw in ("survival", "pfs", "os")
        )
        if is_time_to_event_ep and not has_time_to_event:
            warnings.append(ConsistencyWarning(
                field="analysis_methods",
                message=(
                    "Time-to-event primary endpoint (OS/PFS) requires at least one "
                    "time-to-event analysis method (Kaplan-Meier, Cox, Log-Rank)."
                )
            ))

    if proto.sample_size and proto.effect_size and proto.power:
        expected = _compute_sample_size(proto.effect_size, proto.alpha or 0.05, proto.power)
        if abs(proto.sample_size - expected) > 5:
            warnings.append(ConsistencyWarning(
                field="sample_size",
                message=(
                    f"Stated sample size ({proto.sample_size}) doesn't match power "
                    f"calculation result ({expected}). Run power calc again or correct manually."
                )
            ))

    return warnings


def _score_completeness(proto: ProtocolState) -> float:
    score = 0.0
    if proto.primary_endpoint:
        score += 2.0
    if proto.inclusion_criteria or proto.exclusion_criteria:
        score += 2.0
    if proto.sample_size is not None:
        score += 2.0
    if proto.analysis_methods:
        score += 2.0
    return score


def _simulate_fda_review(proto: ProtocolState, drift_injected: bool) -> Tuple[FDAVerdict, str]:
    issues = []

    if not proto.primary_endpoint:
        issues.append("No primary endpoint defined.")
    elif proto.primary_endpoint.lower() not in (
        VALID_PRIMARY_ENDPOINTS_V2 if drift_injected else VALID_PRIMARY_ENDPOINTS_V1
    ):
        issues.append(
            f"Primary endpoint '{proto.primary_endpoint}' is not accepted under current guidance."
        )

    if not proto.inclusion_criteria:
        issues.append("Patient inclusion criteria are missing.")

    if proto.sample_size is None:
        issues.append("Sample size has not been determined.")

    if not proto.analysis_methods:
        issues.append("Statistical analysis plan is missing.")

    min_power = 0.85 if drift_injected else 0.80
    if proto.power is not None and proto.power < min_power:
        issues.append(
            f"Statistical power ({proto.power}) does not meet the required minimum ({min_power})."
        )
    elif proto.power is None:
        issues.append("Statistical power has not been specified.")

    if not issues:
        return FDAVerdict.APPROVE, (
            "FDA Complete Response: Protocol APPROVED. "
            "All required sections are complete and consistent."
        )
    elif len(issues) <= 2:
        return FDAVerdict.REVISE, (
            "FDA Complete Response Letter - REVISE REQUIRED.\n"
            "Issues identified:\n" + "\n".join(f"  - {i}" for i in issues)
        )
    else:
        return FDAVerdict.REJECT, (
            "FDA Complete Response Letter - REJECT.\n"
            "Multiple critical deficiencies:\n" + "\n".join(f"  - {i}" for i in issues)
        )


class ClinicalTrialEnv:
    """
    OpenEnv-compatible environment for clinical trial protocol design.

    Usage (sync):
        env = ClinicalTrialEnv()
        obs = env.reset()
        obs, reward, done = env.step(action)
    """

    def __init__(self, max_steps: int = 50, drift_step: Optional[int] = None):
        self._max_steps = max_steps
        self._drift_step = drift_step  # None -> random between 15-25
        self._state: EpisodeState = self._fresh_state()

    def _fresh_state(self) -> EpisodeState:
        drift = self._drift_step if self._drift_step is not None else random.randint(15, 25)
        return EpisodeState(max_steps=self._max_steps, drift_step=drift)

    def _build_obs(self, tool_result: str = "", reward: float = 0.0) -> ClinicalObservation:
        s = self._state
        proto = s.protocol
        warnings = _check_consistency(proto, proto.drift_injected)

        # Show drift alert from the moment drift fires until agent acknowledges it
        drift_alert = None
        if proto.drift_injected and not proto.drift_acknowledged:
            drift_alert = GUIDANCE_V2

        return ClinicalObservation(
            step=s.step,
            tool_result=tool_result,
            protocol_summary={
                "primary_endpoint":    proto.primary_endpoint,
                "secondary_endpoints": proto.secondary_endpoints,
                "inclusion_criteria":  proto.inclusion_criteria,
                "exclusion_criteria":  proto.exclusion_criteria,
                "sample_size":         proto.sample_size,
                "power":               proto.power,
                "analysis_methods":    proto.analysis_methods,
                "fda_verdict":         proto.fda_verdict,
                "guidance_version":    "v2" if proto.drift_injected else "v1",
            },
            consistency_warnings=warnings,
            fda_feedback=proto.fda_feedback_text,
            fda_verdict=proto.fda_verdict,
            schema_drift_alert=drift_alert,
            done=s.done,
            reward=reward,
        )

    def _tool_draft_endpoint(self, args: dict) -> str:
        endpoint      = str(args.get("endpoint", "")).strip()
        endpoint_type = str(args.get("endpoint_type", "primary")).lower()
        if not endpoint:
            return "ERROR: 'endpoint' argument is required."
        if endpoint_type == "primary":
            self._state.protocol.primary_endpoint = endpoint
            # Acknowledge drift only on a step AFTER the step drift fired,
            # so agent sees the alert on the injection step itself.
            if self._state.protocol.drift_injected and self._state.step > self._state.drift_step:
                self._state.protocol.drift_acknowledged = True
            return f"Primary endpoint set to: {endpoint}"
        else:
            self._state.protocol.secondary_endpoints.append(endpoint)
            return f"Secondary endpoint added: {endpoint}"

    def _tool_set_inclusion_criteria(self, args: dict) -> str:
        inclusion = args.get("criteria", [])
        exclusion = args.get("exclusion", [])
        if isinstance(inclusion, str):
            inclusion = [inclusion]
        if isinstance(exclusion, str):
            exclusion = [exclusion]
        self._state.protocol.inclusion_criteria = list(inclusion)
        self._state.protocol.exclusion_criteria = list(exclusion)
        return (
            f"Inclusion criteria set ({len(inclusion)} items). "
            f"Exclusion criteria set ({len(exclusion)} items)."
        )

    def _tool_run_power_calc(self, args: dict) -> str:
        try:
            effect_size = float(args.get("effect_size", 0.3))
            alpha       = float(args.get("alpha", 0.05))
            power       = float(args.get("power", 0.80))
        except (TypeError, ValueError) as e:
            return f"ERROR in power calculation inputs: {e}"

        n = _compute_sample_size(effect_size, alpha, power)
        self._state.protocol.effect_size = effect_size
        self._state.protocol.alpha       = alpha
        self._state.protocol.power       = power
        self._state.protocol.sample_size = n

        # Re-running power calc after drift fires = agent acknowledged drift
        if self._state.protocol.drift_injected and self._state.step > self._state.drift_step:
            self._state.protocol.drift_acknowledged = True

        return (
            f"Power calculation complete. "
            f"Effect size={effect_size}, alpha={alpha}, power={power} -> "
            f"N per arm={n} (total N={n*2})"
        )

    def _tool_draft_analysis_plan(self, args: dict) -> str:
        methods = args.get("methods", [])
        if isinstance(methods, str):
            methods = [methods]
        valid   = [m for m in methods if m.lower() in VALID_ANALYSIS_METHODS]
        invalid = [m for m in methods if m.lower() not in VALID_ANALYSIS_METHODS]
        self._state.protocol.analysis_methods = valid
        msg = f"Analysis plan drafted with methods: {valid}."
        if invalid:
            msg += f" WARNING: Unrecognized methods ignored: {invalid}."
        if self._state.protocol.drift_injected and self._state.step > self._state.drift_step:
            self._state.protocol.drift_acknowledged = True
        return msg

    def _tool_simulate_fda_review(self, _args: dict) -> str:
        # FIXED: content-based guard — blocks only if protocol is completely empty.
        # This allows the auto-demo to call FDA review after 6-7 meaningful steps.
        proto = self._state.protocol
        has_content = (
            proto.primary_endpoint is not None
            or bool(proto.inclusion_criteria)
            or proto.sample_size is not None
        )
        if not has_content:
            return (
                "FDA review requires at least a primary endpoint, inclusion criteria, "
                "and sample size to be defined. Complete your protocol first."
            )

        verdict, feedback = _simulate_fda_review(proto, proto.drift_injected)
        proto.fda_verdict       = verdict
        proto.fda_feedback_text = feedback
        proto.fda_review_called = True
        self._state.done = True   # episode ends after FDA review
        return feedback

    def reset(self) -> ClinicalObservation:
        self._state = self._fresh_state()
        return self._build_obs(
            tool_result=(
                "Episode started. You are designing a Phase 2 oncology trial. "
                "Current guidance: " + GUIDANCE_V1 + "\n"
                "Available tools: draft_endpoint, set_inclusion_criteria, "
                "run_power_calc, draft_analysis_plan, simulate_fda_review."
            )
        )

    def step(self, action: ClinicalAction) -> Tuple[ClinicalObservation, float, bool]:
        s = self._state

        if s.done:
            return self._build_obs("Episode already done. Call reset()."), 0.0, True

        # Inject schema drift
        if s.step >= s.drift_step and not s.protocol.drift_injected:
            s.protocol.drift_injected = True

        # Dispatch tool
        dispatch = {
            ToolName.DRAFT_ENDPOINT:         self._tool_draft_endpoint,
            ToolName.SET_INCLUSION_CRITERIA:  self._tool_set_inclusion_criteria,
            ToolName.RUN_POWER_CALC:          self._tool_run_power_calc,
            ToolName.DRAFT_ANALYSIS_PLAN:     self._tool_draft_analysis_plan,
            ToolName.SIMULATE_FDA_REVIEW:     self._tool_simulate_fda_review,
        }
        handler = dispatch.get(action.tool)
        if handler is None:
            tool_result = f"ERROR: Unknown tool '{action.tool}'."
        else:
            tool_result = handler(action.arguments)

        # Compute reward
        reward = 0.0

        # Dense: consistency reward/penalty every step
        warnings = _check_consistency(s.protocol, s.protocol.drift_injected)
        if not warnings:
            reward += 1.0
        else:
            reward -= 2.0 * len(warnings)

        # Completeness delta — penalize sections that have active warnings
        field_penalty_map = {
            "primary_endpoint": 2.0,
            "power":            2.0,
            "sample_size":      2.0,
            "analysis_methods": 2.0,
        }
        penalty = sum(field_penalty_map.get(w.field, 0.0) for w in warnings)
        completeness_now = max(0.0, _score_completeness(s.protocol) - penalty)
        prev_comp = sum(entry.get("comp_score", 0.0) for entry in s.action_log)
        reward += max(0.0, completeness_now - prev_comp)

        # Drift-awareness bonus — awarded exactly once
        if s.protocol.drift_injected and s.protocol.drift_acknowledged:
            if not any(e.get("drift_bonus") for e in s.action_log):
                reward += 5.0

        # Episodic FDA verdict reward
        if s.protocol.fda_verdict == FDAVerdict.APPROVE:
            reward += 15.0
        elif s.protocol.fda_verdict == FDAVerdict.REVISE:
            reward += 5.0
        elif s.protocol.fda_verdict == FDAVerdict.REJECT:
            reward -= 10.0

        # Log
        s.action_log.append({
            "step":       s.step,
            "tool":       action.tool,
            "arguments":  action.arguments,
            "reward":     reward,
            "comp_score": completeness_now,
            "drift_bonus": s.protocol.drift_injected and s.protocol.drift_acknowledged,
        })
        s.cumulative_reward += reward
        s.step += 1

        # Termination
        if s.step >= s.max_steps:
            s.done = True

        obs = self._build_obs(tool_result, reward)
        obs.done = s.done
        return obs, reward, s.done

    def state(self) -> EpisodeState:
        return self._state
