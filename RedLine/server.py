"""
RedLine — OpenEnv server (Environment) v0.5

Implements:
  reset()  → ClinicalObservation
  step()   → (ClinicalObservation, float, bool)
  state()  → EpisodeState

═══════════════════════════════════════════════════════
Rubric-based Reward System  (v0.5)
═══════════════════════════════════════════════════════

Reward = rubric_coherence() + rubric_efficiency() + rubric_drift() + rubric_outcome()

RUBRIC 1 — Protocol Coherence  (dense, every step)
  +1   zero consistency warnings
  −2   per warning (compound: 2 warnings = −4)
  +2   per protocol section first completed this episode (max +8 total)

RUBRIC 2 — Planning Efficiency  (dense)
  +0.5 per step remaining when episode ends with FDA APPROVE
  −0.5 per tool call that changed no protocol state (no-op)
  FDA review step is NEVER penalised as a no-op (terminal action)

RUBRIC 3 — Drift Adaptation  (sparse, event-driven)
  +5   agent calls run_power_calc(power>=0.85) after drift fires (once)
  −3   agent calls FDA with power < 0.85 after DRIFT_IGNORE_STEPS steps

RUBRIC 4 — Regulatory Outcome  (episodic)
  +15  FDA APPROVE   (requires all 4 sections)
  +5   FDA REVISE
  −10  FDA REJECT

═══════════════════════════════════════════════════════
All bug fixes
═══════════════════════════════════════════════════════
  FIX 1 — Drift acknowledgement:
    Removed step-number comparison from _tool_run_power_calc.
    Drift acknowledgement now fires purely on:
      drift_injected AND power >= 0.85 AND not already acknowledged
    drift_injected is set at the TOP of step() before dispatch,
    so it is already True by the time any handler runs on drift_step.
    No step comparison needed or correct.

  FIX 2 — Efficiency always -0.5:
    (a) Added fda_verdict to _PROTO_TRACKED_FIELDS so FDA call
        registers as a state change, not a no-op.
    (b) Added is_fda_step param to rubric_efficiency so the terminal
        FDA call is never penalised as a no-op regardless of (a).

  FIX 3 — FDA always REVISE:
    _normalise_tools() converts every element to ToolName before
    set-difference against REQUIRED_TOOLS_FOR_APPROVE.
    Also applied at the point of insertion in step() so
    self._tools_used always holds ToolName members.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
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

# ── Guidance strings ──────────────────────────────────────────────────────────

GUIDANCE_V1 = (
    "FDA Guidance (v1): Primary endpoint must be Overall Survival (OS) or "
    "Progression-Free Survival (PFS). Power >= 0.80 required."
)

GUIDANCE_V2 = (
    "FDA Guidance UPDATE (v2 — SCHEMA DRIFT INJECTED): "
    "ORR now accepted as primary endpoint for accelerated-approval trials. "
    "Power requirement raised to >= 0.85. Revise your statistical plan NOW."
)

# ── Valid vocabulary ──────────────────────────────────────────────────────────

VALID_PRIMARY_ENDPOINTS_V1 = {"os", "overall survival", "pfs", "progression-free survival"}
VALID_PRIMARY_ENDPOINTS_V2 = VALID_PRIMARY_ENDPOINTS_V1 | {
    "orr", "overall response rate", "objective response rate"
}

VALID_ANALYSIS_METHODS = {
    "kaplan-meier", "cox proportional hazards", "log-rank test",
    "recist criteria", "bayesian adaptive", "irrecist"
}

# ── Reward constants ──────────────────────────────────────────────────────────

SECTION_BONUS           =  2.0
WARNING_REWARD          =  1.0
WARNING_PENALTY         = -2.0

EFFICIENCY_STEP_BONUS   =  0.5   # per unused step on APPROVE
EFFICIENCY_NOOP_PENALTY = -0.5   # per tool call that changed nothing

DRIFT_ADAPT_BONUS       =  5.0
DRIFT_IGNORE_PENALTY    = -3.0
FDA_APPROVE_REWARD      = 15.0
FDA_REVISE_REWARD       =  5.0
FDA_REJECT_PENALTY      = -10.0

MIN_FDA_STEP       = 4
DRIFT_IGNORE_STEPS = 5
DEFAULT_DRIFT_STEP = 4

REQUIRED_TOOLS_FOR_APPROVE = {
    ToolName.DRAFT_ENDPOINT,
    ToolName.SET_INCLUSION_CRITERIA,
    ToolName.RUN_POWER_CALC,
    ToolName.DRAFT_ANALYSIS_PLAN,
}

# fda_verdict included so simulate_fda_review never reads as a no-op
_PROTO_TRACKED_FIELDS = (
    "primary_endpoint", "secondary_endpoints", "inclusion_criteria",
    "exclusion_criteria", "sample_size", "power", "analysis_methods",
    "fda_verdict",
)


# ── Rubric result ─────────────────────────────────────────────────────────────

@dataclass
class RubricScore:
    name: str
    score: float
    reason: str

    def __float__(self):
        return self.score


# ── Proto snapshot helpers ────────────────────────────────────────────────────

def _snapshot_proto(proto: ProtocolState) -> dict:
    return {f: copy.copy(getattr(proto, f, None)) for f in _PROTO_TRACKED_FIELDS}


def _proto_changed(before: dict, proto: ProtocolState) -> bool:
    for f in _PROTO_TRACKED_FIELDS:
        if before.get(f) != getattr(proto, f, None):
            return True
    return False


# ── Tool normalisation ────────────────────────────────────────────────────────

def _normalise_tools(tools_used: set) -> set:
    """Convert every element in tools_used to a ToolName enum member."""
    normalised = set()
    for t in tools_used:
        if isinstance(t, ToolName):
            normalised.add(t)
        else:
            try:
                normalised.add(ToolName(str(t)))
            except ValueError:
                pass
    return normalised


# ═══════════════════════════════════════════════════════════════════════════════
# RUBRICS  (pure functions — no side effects)
# ═══════════════════════════════════════════════════════════════════════════════

def rubric_coherence(
    proto: ProtocolState,
    drift_injected: bool,
    sections_awarded: set,
) -> Tuple[RubricScore, list, set]:
    warnings = _check_consistency(proto, drift_injected)

    dense_val    = WARNING_REWARD if not warnings else WARNING_PENALTY * len(warnings)
    dense_reason = "no warnings" if not warnings else f"{len(warnings)} warning(s)"

    new_sections = set()
    checks = {
        "endpoint": (proto.primary_endpoint is not None
                     and not any(w.field == "primary_endpoint" for w in warnings)),
        "criteria":  bool(proto.inclusion_criteria or proto.exclusion_criteria),
        "stats":    (proto.sample_size is not None
                     and not any(w.field in ("power", "sample_size") for w in warnings)),
        "analysis": (bool(proto.analysis_methods)
                     and not any(w.field == "analysis_methods" for w in warnings)),
    }
    for section, complete in checks.items():
        if complete and section not in sections_awarded:
            new_sections.add(section)

    bonus  = SECTION_BONUS * len(new_sections)
    total  = dense_val + bonus
    reason = f"dense={dense_val:+.1f}, new_sections={sorted(new_sections)}"

    return RubricScore("coherence", total, reason), warnings, sections_awarded | new_sections


def rubric_efficiency(
    proto_changed_this_step: bool,
    verdict: Optional[FDAVerdict],
    steps_taken: int,
    max_steps: int,
    is_fda_step: bool = False,
) -> RubricScore:
    """
    Step-budget + state-delta model.

    On APPROVE: +0.5 per unused step (rewards finishing early).
    On no-op (nothing changed): -0.5 (penalises spinning).
    FDA step always exempt from no-op penalty.
    """
    if verdict == FDAVerdict.APPROVE:
        steps_remaining = max(0, max_steps - steps_taken)
        bonus = steps_remaining * EFFICIENCY_STEP_BONUS
        return RubricScore(
            "efficiency",
            bonus,
            f"APPROVE in {steps_taken} steps → +{bonus:.1f} ({steps_remaining} unused × 0.5)"
        )

    if is_fda_step:
        return RubricScore("efficiency", 0.0, "FDA step — exempt from no-op penalty")

    if proto_changed_this_step:
        return RubricScore("efficiency", 0.0, "productive step")

    return RubricScore(
        "efficiency",
        EFFICIENCY_NOOP_PENALTY,
        "no-op: tool call changed no protocol state"
    )


def rubric_drift(
    proto: ProtocolState,
    drift_injected: bool,
    drift_step: int,
    current_step: int,
    bonus_awarded: bool,
) -> RubricScore:
    if not drift_injected:
        return RubricScore("drift", 0.0, "drift not yet active")

    # Bonus fires exactly once when drift_acknowledged becomes True
    # (set inside _tool_run_power_calc when power >= 0.85 after drift fires)
    if proto.drift_acknowledged and not bonus_awarded:
        return RubricScore(
            "drift",
            DRIFT_ADAPT_BONUS,
            "power revised to >=0.85 after drift — +5 bonus"
        )

    # Penalty: agent called FDA while ignoring the new power requirement
    steps_since = current_step - drift_step
    if (proto.fda_review_called
            and proto.power is not None and proto.power < 0.85
            and steps_since >= DRIFT_IGNORE_STEPS
            and not bonus_awarded):
        return RubricScore(
            "drift",
            DRIFT_IGNORE_PENALTY,
            f"FDA called with power={proto.power} < 0.85 after {steps_since} drift steps"
        )

    if bonus_awarded:
        return RubricScore("drift", 0.0, "drift bonus already awarded this episode")

    return RubricScore("drift", 0.0, "drift active — call run_power_calc(power>=0.85)")


def rubric_outcome(verdict: Optional[FDAVerdict]) -> RubricScore:
    if verdict is None:
        return RubricScore("outcome", 0.0, "no verdict yet")
    if verdict == FDAVerdict.APPROVE:
        return RubricScore("outcome", FDA_APPROVE_REWARD, "FDA APPROVE")
    if verdict == FDAVerdict.REVISE:
        return RubricScore("outcome", FDA_REVISE_REWARD, "FDA REVISE")
    return RubricScore("outcome", FDA_REJECT_PENALTY, "FDA REJECT")


# ═══════════════════════════════════════════════════════════════════════════════
# PURE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_sample_size(effect_size: float, alpha: float, power: float) -> int:
    if effect_size <= 0:
        return 9999
    z_alpha = {0.05: 1.96, 0.025: 2.24, 0.01: 2.576}.get(alpha, 1.96)
    z_beta  = {0.80: 0.842, 0.85: 1.036, 0.90: 1.282}.get(power, 0.842)
    return max(10, math.ceil(2 * ((z_alpha + z_beta) / effect_size) ** 2))


def _check_consistency(proto: ProtocolState, drift_injected: bool) -> list:
    warnings = []

    if proto.primary_endpoint:
        ep    = proto.primary_endpoint.lower().strip()
        valid = VALID_PRIMARY_ENDPOINTS_V2 if drift_injected else VALID_PRIMARY_ENDPOINTS_V1
        if ep not in valid:
            warnings.append(ConsistencyWarning(
                field="primary_endpoint",
                message=(
                    f"'{proto.primary_endpoint}' not FDA-accepted. "
                    f"Valid: {sorted(valid)}"
                ),
            ))

    if proto.power is not None:
        min_p = 0.85 if drift_injected else 0.80
        if proto.power < min_p:
            warnings.append(ConsistencyWarning(
                field="power",
                message=f"Power {proto.power} < minimum {min_p}",
            ))

    if proto.primary_endpoint and proto.analysis_methods:
        ep      = proto.primary_endpoint.lower()
        has_tte = any(
            m in ("kaplan-meier", "cox proportional hazards", "log-rank test")
            for m in [m.lower() for m in proto.analysis_methods]
        )
        if any(kw in ep for kw in ("survival", "pfs", "os")) and not has_tte:
            warnings.append(ConsistencyWarning(
                field="analysis_methods",
                message="OS/PFS endpoint requires time-to-event analysis (KM/Cox/Log-rank).",
            ))

    if proto.sample_size and proto.effect_size and proto.power:
        expected = _compute_sample_size(proto.effect_size, proto.alpha or 0.05, proto.power)
        if abs(proto.sample_size - expected) > 5:
            warnings.append(ConsistencyWarning(
                field="sample_size",
                message=f"N={proto.sample_size} doesn't match calc N={expected}.",
            ))

    return warnings


def _simulate_fda_review(
    proto: ProtocolState,
    drift_injected: bool,
    tools_used: set,
) -> Tuple[FDAVerdict, str]:
    # Normalise before set-difference so ToolName vs str mismatches
    # don't silently cause false "section not completed" issues
    tools_normalised = _normalise_tools(tools_used)

    issues = []

    missing_tools = sorted(
        REQUIRED_TOOLS_FOR_APPROVE - tools_normalised,
        key=lambda t: t.value
    )
    for tool in missing_tools:
        issues.append(f"Section '{tool.value}' not completed.")

    if not proto.primary_endpoint:
        issues.append("No primary endpoint defined.")
    elif proto.primary_endpoint.lower().strip() not in (
        VALID_PRIMARY_ENDPOINTS_V2 if drift_injected else VALID_PRIMARY_ENDPOINTS_V1
    ):
        issues.append(f"Endpoint '{proto.primary_endpoint}' not accepted.")

    if not proto.inclusion_criteria:
        issues.append("Inclusion criteria missing.")

    if proto.sample_size is None:
        issues.append("Sample size not determined.")

    if not proto.analysis_methods:
        issues.append("Analysis plan missing.")

    min_p = 0.85 if drift_injected else 0.80
    if proto.power is None:
        issues.append("Power not specified.")
    elif proto.power < min_p:
        issues.append(f"Power {proto.power} < required {min_p}.")

    if not issues:
        return FDAVerdict.APPROVE, "FDA Complete Response: APPROVED."
    if len(issues) <= 2:
        return FDAVerdict.REVISE, (
            "FDA CRL — REVISE REQUIRED.\n"
            + "\n".join(f"  • {i}" for i in issues)
        )
    return FDAVerdict.REJECT, (
        "FDA CRL — REJECT.\n"
        + "\n".join(f"  • {i}" for i in issues)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalTrialEnv:
    """OpenEnv-compatible environment for Phase 2 oncology protocol design."""

    def __init__(self, max_steps: int = 50, drift_step: Optional[int] = None):
        self._max_steps  = max_steps
        self._drift_step = drift_step if drift_step is not None else DEFAULT_DRIFT_STEP
        self._state: EpisodeState = self._fresh_state()
        self._reset_episode_state()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fresh_state(self) -> EpisodeState:
        return EpisodeState(max_steps=self._max_steps, drift_step=self._drift_step)

    def _reset_episode_state(self):
        self._sections_awarded:    set            = set()
        self._drift_bonus_awarded: bool           = False
        self._tools_used:          set            = set()
        self._proto_snapshot:      Optional[dict] = None

    def _build_obs(
        self,
        tool_result: str = "",
        reward: float = 0.0,
        rubrics: Optional[list] = None,
    ) -> ClinicalObservation:
        s, proto = self._state, self._state.protocol
        warnings    = _check_consistency(proto, proto.drift_injected)
        drift_alert = (
            GUIDANCE_V2
            if proto.drift_injected and not proto.drift_acknowledged
            else None
        )
        if rubrics:
            parts = [f"{r.name}={r.score:+.1f}({r.reason})" for r in rubrics if r.score != 0.0]
            if parts:
                tool_result += "\n[Rubrics] " + " | ".join(parts)
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

    # ── Tool handlers ─────────────────────────────────────────────────────────

    def _tool_draft_endpoint(self, args: dict) -> str:
        endpoint = str(args.get("endpoint", "")).strip()
        ep_type  = str(args.get("endpoint_type", "primary")).lower()
        if not endpoint:
            return "ERROR: 'endpoint' required."
        if ep_type == "primary":
            self._state.protocol.primary_endpoint = endpoint
            return f"Primary endpoint set: {endpoint}"
        self._state.protocol.secondary_endpoints.append(endpoint)
        return f"Secondary endpoint added: {endpoint}"

    def _tool_set_inclusion_criteria(self, args: dict) -> str:
        inc = args.get("criteria", [])
        exc = args.get("exclusion", [])
        if isinstance(inc, str): inc = [inc]
        if isinstance(exc, str): exc = [exc]
        self._state.protocol.inclusion_criteria = list(inc)
        self._state.protocol.exclusion_criteria = list(exc)
        return f"Criteria set: {len(inc)} inclusion, {len(exc)} exclusion."

    def _tool_run_power_calc(self, args: dict) -> str:
        try:
            es = float(args.get("effect_size", 0.3))
            a  = float(args.get("alpha",       0.05))
            pw = float(args.get("power",       0.80))
        except (TypeError, ValueError) as e:
            return f"ERROR: {e}"

        n = _compute_sample_size(es, a, pw)
        p = self._state.protocol
        p.effect_size, p.alpha, p.power, p.sample_size = es, a, pw, n

        # FIX 1: drift_acknowledged set purely on drift_injected + power >= 0.85.
        # No step comparison. drift_injected is already True when this handler
        # runs on the drift step (injection happens at top of step() before dispatch).
        if p.drift_injected and pw >= 0.85 and not p.drift_acknowledged:
            p.drift_acknowledged = True

        ack_msg = " [DRIFT ACKNOWLEDGED — +5 bonus incoming]" if p.drift_acknowledged else ""
        return (
            f"Power calc: es={es}, α={a}, power={pw} → "
            f"N/arm={n} (total={n * 2}){ack_msg}"
        )

    def _tool_draft_analysis_plan(self, args: dict) -> str:
        methods = args.get("methods", [])
        if isinstance(methods, str): methods = [methods]
        valid   = [m for m in methods if m.lower() in VALID_ANALYSIS_METHODS]
        invalid = [m for m in methods if m.lower() not in VALID_ANALYSIS_METHODS]
        self._state.protocol.analysis_methods = valid
        return (
            f"Analysis plan: {valid}."
            + (f" Ignored (unrecognised): {invalid}." if invalid else "")
        )

    def _tool_simulate_fda_review(self, _args: dict) -> str:
        s, proto = self._state, self._state.protocol

        enough_tools = len(self._tools_used) >= MIN_FDA_STEP
        enough_steps = s.step >= MIN_FDA_STEP

        if not enough_tools and not enough_steps:
            used_names = [t.value for t in _normalise_tools(self._tools_used)]
            return (
                f"FDA review not available yet. "
                f"Need >= {MIN_FDA_STEP} distinct tools used "
                f"(have {len(used_names)}: {used_names}) "
                f"OR >= {MIN_FDA_STEP} steps (at step {s.step})."
            )

        if not (proto.primary_endpoint or proto.inclusion_criteria or proto.sample_size):
            return "Complete at least one protocol section first."

        verdict, feedback = _simulate_fda_review(
            proto, proto.drift_injected, self._tools_used
        )
        proto.fda_verdict       = verdict
        proto.fda_feedback_text = feedback
        proto.fda_review_called = True
        s.done                  = True
        return feedback

    # ── Core loop ─────────────────────────────────────────────────────────────

    def reset(self) -> ClinicalObservation:
        self._state = self._fresh_state()
        self._reset_episode_state()
        return self._build_obs(
            "Episode started.\n"
            + GUIDANCE_V1 + "\n"
            + f"Drift fires at step {self._drift_step}. "
            + "Tools: draft_endpoint | set_inclusion_criteria | "
              "run_power_calc | draft_analysis_plan | simulate_fda_review"
        )

    def step(self, action: ClinicalAction) -> Tuple[ClinicalObservation, float, bool]:
        s = self._state

        # Guard: episode already finished
        if s.done:
            return self._build_obs("Episode done. Call reset() to start again."), 0.0, True

        # Inject drift at the configured step (before any tool runs)
        if s.step >= s.drift_step and not s.protocol.drift_injected:
            s.protocol.drift_injected = True

        # Normalise tool to ToolName so self._tools_used is always consistent
        tool_key = (
            action.tool
            if isinstance(action.tool, ToolName)
            else ToolName(str(action.tool))
        )
        self._tools_used.add(tool_key)

        # Snapshot BEFORE tool executes (efficiency delta check)
        snap_before = _snapshot_proto(s.protocol)

        # Dispatch
        dispatch = {
            ToolName.DRAFT_ENDPOINT:         self._tool_draft_endpoint,
            ToolName.SET_INCLUSION_CRITERIA:  self._tool_set_inclusion_criteria,
            ToolName.RUN_POWER_CALC:          self._tool_run_power_calc,
            ToolName.DRAFT_ANALYSIS_PLAN:     self._tool_draft_analysis_plan,
            ToolName.SIMULATE_FDA_REVIEW:     self._tool_simulate_fda_review,
        }
        handler     = dispatch.get(tool_key)
        tool_result = handler(action.arguments) if handler else "ERROR: unknown tool"

        changed = _proto_changed(snap_before, s.protocol)

        # ── Rubrics ───────────────────────────────────────────────────────────
        r1, warnings, self._sections_awarded = rubric_coherence(
            s.protocol, s.protocol.drift_injected, self._sections_awarded
        )
        r2 = rubric_efficiency(
            proto_changed_this_step=changed,
            verdict=s.protocol.fda_verdict,
            steps_taken=s.step + 1,
            max_steps=s.max_steps,
            is_fda_step=(tool_key == ToolName.SIMULATE_FDA_REVIEW),
        )
        r3 = rubric_drift(
            s.protocol, s.protocol.drift_injected,
            s.drift_step, s.step, self._drift_bonus_awarded
        )
        r4 = rubric_outcome(s.protocol.fda_verdict)

        if r3.score == DRIFT_ADAPT_BONUS:
            self._drift_bonus_awarded = True

        total = r1.score + r2.score + r3.score + r4.score

        s.action_log.append({
            "step":              s.step,
            "tool":              tool_key.value,
            "args":              action.arguments,
            "reward":            round(total, 3),
            "r_coherence":       round(r1.score, 3),
            "r_efficiency":      round(r2.score, 3),
            "r_drift":           round(r3.score, 3),
            "r_outcome":         round(r4.score, 3),
            "warnings":          len(warnings),
            "state_changed":     changed,
            "reason_coherence":  r1.reason,
            "reason_efficiency": r2.reason,
            "reason_drift":      r3.reason,
            "reason_outcome":    r4.reason,
        })

        s.cumulative_reward += total
        s.step              += 1

        if s.step >= s.max_steps:
            s.done = True

        obs      = self._build_obs(tool_result, total, [r1, r2, r3, r4])
        obs.done = s.done
        return obs, total, s.done

    # ── Public API ────────────────────────────────────────────────────────────

    def state(self) -> EpisodeState:
        return self._state

    def rubric_breakdown(self) -> dict:
        """Cumulative per-rubric totals for the current episode."""
        log = self._state.action_log
        return {
            "coherence":  round(sum(e.get("r_coherence",  0) for e in log), 3),
            "efficiency": round(sum(e.get("r_efficiency", 0) for e in log), 3),
            "drift":      round(sum(e.get("r_drift",      0) for e in log), 3),
            "outcome":    round(sum(e.get("r_outcome",    0) for e in log), 3),
            "total":      round(self._state.cumulative_reward,             3),
            "steps":      self._state.step,
            "tools_used": [t.value for t in _normalise_tools(self._tools_used)],
        }