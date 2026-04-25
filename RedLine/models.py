"""
ClinicalPilot — Pydantic models for OpenEnv.
Action / Observation / State for a Phase-2 oncology protocol designer.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums / Literals used as vocabulary for the agent
# ---------------------------------------------------------------------------

class ToolName(str, Enum):
    DRAFT_ENDPOINT         = "draft_endpoint"
    SET_INCLUSION_CRITERIA = "set_inclusion_criteria"
    RUN_POWER_CALC         = "run_power_calc"
    DRAFT_ANALYSIS_PLAN    = "draft_analysis_plan"
    SIMULATE_FDA_REVIEW    = "simulate_fda_review"


class FDAVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE  = "REVISE"
    REJECT  = "REJECT"


# ---------------------------------------------------------------------------
# Action — what the agent does at each step
# ---------------------------------------------------------------------------

class ClinicalAction(BaseModel):
    """Single agent action: choose a tool and supply arguments."""

    tool: ToolName = Field(..., description="Which tool to call.")
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific kwargs. See server.py for per-tool schemas.\n"
            "draft_endpoint          → {endpoint: str, endpoint_type: 'primary'|'secondary'}\n"
            "set_inclusion_criteria  → {criteria: List[str], exclusion: List[str]}\n"
            "run_power_calc          → {effect_size: float, alpha: float, power: float}\n"
            "draft_analysis_plan     → {methods: List[str]}\n"
            "simulate_fda_review     → {}   (episodic; only meaningful at step >= 30)\n"
        ),
    )


# ---------------------------------------------------------------------------
# Observation — what the agent sees after each step
# ---------------------------------------------------------------------------

class ConsistencyWarning(BaseModel):
    field: str
    message: str


class ClinicalObservation(BaseModel):
    """Returned by step() / reset()."""

    step: int = Field(0, description="Current step number (0-indexed).")
    tool_result: str = Field("", description="Human-readable result of the last tool call.")
    protocol_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current snapshot of the protocol state visible to the agent.",
    )
    consistency_warnings: List[ConsistencyWarning] = Field(
        default_factory=list,
        description="Rule-based warnings about contradictions in the current protocol.",
    )
    fda_feedback: Optional[str] = Field(
        None,
        description="Populated after simulate_fda_review is called.",
    )
    fda_verdict: Optional[FDAVerdict] = Field(None)
    schema_drift_alert: Optional[str] = Field(
        None,
        description="Non-None when a regulatory guidance update has been injected.",
    )
    done: bool = Field(False, description="True when episode ends.")
    reward: float = Field(0.0, description="Step-level reward (for logging).")


# ---------------------------------------------------------------------------
# State — full server-side state (not sent to agent verbatim)
# ---------------------------------------------------------------------------

class ProtocolState(BaseModel):
    """Full protocol document — fields are progressively filled."""

    # Section 1 — Endpoints
    primary_endpoint: Optional[str] = None
    secondary_endpoints: List[str] = Field(default_factory=list)

    # Section 2 — Population
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)

    # Section 3 — Statistics
    effect_size: Optional[float] = None
    alpha: Optional[float] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None          # computed by run_power_calc

    # Section 4 — Analysis Plan
    analysis_methods: List[str] = Field(default_factory=list)

    # Meta
    fda_review_called: bool = False
    fda_verdict: Optional[FDAVerdict] = None
    fda_feedback_text: Optional[str] = None

    # Schema drift — injected at step ~20
    drift_injected: bool = False
    drift_acknowledged: bool = False           # True once agent re-drafts after drift


class EpisodeState(BaseModel):
    """Full server state for one episode."""

    step: int = 0
    max_steps: int = 50
    protocol: ProtocolState = Field(default_factory=ProtocolState)
    action_log: List[Dict[str, Any]] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    drift_step: int = 20           # step at which guidance update fires
    done: bool = False
