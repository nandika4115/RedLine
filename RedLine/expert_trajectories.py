"""
expert_trajectories.py — 25 hand-authored episodes for SFT.

Format: each trajectory is a list of (prompt, completion) pairs.
The prompt is what the agent sees (observation as text).
The completion is the correct action as JSON.

These are used for SFT to create a pre-warmed baseline BEFORE RL.
"""

import json

# ---------------------------------------------------------------------------
# Helper — format observation into a text prompt for the LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a clinical trial protocol designer AI.
Your goal is to design a complete, FDA-compliant Phase 2 oncology clinical trial protocol.
At each step you must call exactly ONE tool with valid arguments.

Available tools:
- draft_endpoint(endpoint: str, endpoint_type: "primary"|"secondary")
- set_inclusion_criteria(criteria: List[str], exclusion: List[str])
- run_power_calc(effect_size: float, alpha: float, power: float)
- draft_analysis_plan(methods: List[str])
- simulate_fda_review()

IMPORTANT RULES (FDA Guidance v1):
- Primary endpoint must be "Overall Survival" or "Progression-Free Survival"
- Statistical power must be ≥ 0.80
- Analysis plan must include time-to-event methods for OS/PFS endpoints
- After step 20, FDA guidance may update — check for schema_drift_alert!

Respond with ONLY a JSON object: {"tool": "...", "arguments": {...}}"""


def make_prompt(obs_dict: dict) -> str:
    return (
        f"Current protocol state:\n{json.dumps(obs_dict['protocol_summary'], indent=2)}\n\n"
        f"Consistency warnings: {obs_dict.get('consistency_warnings', [])}\n"
        f"Schema drift alert: {obs_dict.get('schema_drift_alert', 'None')}\n"
        f"Step: {obs_dict.get('step', 0)}\n\n"
        f"What tool should you call next?"
    )


# ---------------------------------------------------------------------------
# Expert trajectory template — 10 steps of a perfect episode
# ---------------------------------------------------------------------------

def make_perfect_episode_no_drift() -> list[dict]:
    """A perfect 10-step episode with no schema drift."""
    steps = [
        # Step 0: Set primary endpoint
        {
            "prompt_obs": {
                "step": 0, "protocol_summary": {
                    "primary_endpoint": None, "inclusion_criteria": [],
                    "sample_size": None, "analysis_methods": [], "guidance_version": "v1"
                },
                "consistency_warnings": [], "schema_drift_alert": None
            },
            "action": {"tool": "draft_endpoint", "arguments": {"endpoint": "Overall Survival", "endpoint_type": "primary"}}
        },
        # Step 1: Add secondary endpoint
        {
            "prompt_obs": {
                "step": 1, "protocol_summary": {
                    "primary_endpoint": "Overall Survival", "inclusion_criteria": [],
                    "sample_size": None, "analysis_methods": [], "guidance_version": "v1"
                },
                "consistency_warnings": [], "schema_drift_alert": None
            },
            "action": {"tool": "draft_endpoint", "arguments": {"endpoint": "Quality of Life", "endpoint_type": "secondary"}}
        },
        # Step 2: Set inclusion/exclusion
        {
            "prompt_obs": {
                "step": 2, "protocol_summary": {
                    "primary_endpoint": "Overall Survival", "inclusion_criteria": [],
                    "sample_size": None, "analysis_methods": [], "guidance_version": "v1"
                },
                "consistency_warnings": [], "schema_drift_alert": None
            },
            "action": {
                "tool": "set_inclusion_criteria",
                "arguments": {
                    "criteria": ["ECOG PS 0-1", "Stage IIIB/IV NSCLC", "Age ≥ 18", "Measurable disease per RECIST 1.1"],
                    "exclusion": ["Prior platinum-based chemotherapy", "Active CNS metastases", "Autoimmune disease requiring systemic treatment"]
                }
            }
        },
        # Step 3: Run power calculation
        {
            "prompt_obs": {
                "step": 3, "protocol_summary": {
                    "primary_endpoint": "Overall Survival",
                    "inclusion_criteria": ["ECOG PS 0-1", "Stage IIIB/IV NSCLC"],
                    "sample_size": None, "analysis_methods": [], "guidance_version": "v1"
                },
                "consistency_warnings": [], "schema_drift_alert": None
            },
            "action": {
                "tool": "run_power_calc",
                "arguments": {"effect_size": 0.3, "alpha": 0.05, "power": 0.80}
            }
        },
        # Step 4: Draft analysis plan
        {
            "prompt_obs": {
                "step": 4, "protocol_summary": {
                    "primary_endpoint": "Overall Survival",
                    "inclusion_criteria": ["ECOG PS 0-1"],
                    "sample_size": 176, "analysis_methods": [], "guidance_version": "v1"
                },
                "consistency_warnings": [], "schema_drift_alert": None
            },
            "action": {
                "tool": "draft_analysis_plan",
                "arguments": {"methods": ["Kaplan-Meier", "Log-rank test", "Cox proportional hazards"]}
            }
        },
    ]
    return steps


def make_perfect_episode_with_drift() -> list[dict]:
    """A perfect episode that correctly responds to schema drift."""
    steps = make_perfect_episode_no_drift()

    # Step 5: Drift fires — agent sees it and revises power
    steps.append({
        "prompt_obs": {
            "step": 20,
            "protocol_summary": {
                "primary_endpoint": "Overall Survival",
                "inclusion_criteria": ["ECOG PS 0-1"],
                "sample_size": 176, "power": 0.80,
                "analysis_methods": ["Kaplan-Meier"],
                "guidance_version": "v2"
            },
            "consistency_warnings": [{"field": "power", "message": "Power 0.80 below new minimum 0.85"}],
            "schema_drift_alert": (
                "FDA Guidance UPDATE (v2 — SCHEMA DRIFT INJECTED): "
                "ORR is now acceptable as primary endpoint. "
                "Power requirement raised to ≥ 0.85."
            )
        },
        # Correct response: re-run power calc with higher power
        "action": {
            "tool": "run_power_calc",
            "arguments": {"effect_size": 0.3, "alpha": 0.05, "power": 0.85}
        }
    })
    # Step 6: Call FDA review
    steps.append({
        "prompt_obs": {
            "step": 32,
            "protocol_summary": {
                "primary_endpoint": "Overall Survival",
                "inclusion_criteria": ["ECOG PS 0-1"],
                "sample_size": 214, "power": 0.85,
                "analysis_methods": ["Kaplan-Meier", "Cox proportional hazards"],
                "guidance_version": "v2"
            },
            "consistency_warnings": [],
            "schema_drift_alert": None
        },
        "action": {"tool": "simulate_fda_review", "arguments": {}}
    })
    return steps


# ---------------------------------------------------------------------------
# Generate all SFT pairs
# ---------------------------------------------------------------------------

def generate_sft_dataset() -> list[dict]:
    """
    Returns a list of {"messages": [...]} dicts in HuggingFace chat format.
    """
    dataset = []

    # 15 no-drift episodes
    for _ in range(15):
        traj = make_perfect_episode_no_drift()
        for step_data in traj:
            prompt = make_prompt(step_data["prompt_obs"])
            completion = json.dumps(step_data["action"])
            dataset.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })

    # 10 drift-aware episodes
    for _ in range(10):
        traj = make_perfect_episode_with_drift()
        for step_data in traj:
            prompt = make_prompt(step_data["prompt_obs"])
            completion = json.dumps(step_data["action"])
            dataset.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                    {"role": "assistant", "content": completion}
                ]
            })

    return dataset


if __name__ == "__main__":
    import json as _json
    ds = generate_sft_dataset()
    print(f"Generated {len(ds)} SFT pairs")
    print("Sample:")
    print(_json.dumps(ds[0], indent=2))
