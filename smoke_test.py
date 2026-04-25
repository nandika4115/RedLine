"""
smoke_test.py — verify environment works end-to-end without GPU.
Run: python smoke_test.py
"""
import json
from RedLine.models import ClinicalAction, ToolName
from RedLine.server import ClinicalTrialEnv

def test_happy_path():
    """Full successful episode — no drift."""
    env = ClinicalTrialEnv(max_steps=50, drift_step=99)  # drift after episode ends
    obs = env.reset()
    assert obs.step == 0
    assert obs.protocol_summary["primary_endpoint"] is None

    steps = [
        ClinicalAction(tool=ToolName.DRAFT_ENDPOINT,
                       arguments={"endpoint": "Overall Survival", "endpoint_type": "primary"}),
        ClinicalAction(tool=ToolName.SET_INCLUSION_CRITERIA,
                       arguments={"criteria": ["ECOG PS 0-1", "Stage IIIB/IV NSCLC"],
                                  "exclusion": ["Prior platinum-based chemotherapy"]}),
        ClinicalAction(tool=ToolName.RUN_POWER_CALC,
                       arguments={"effect_size": 0.3, "alpha": 0.05, "power": 0.80}),
        ClinicalAction(tool=ToolName.DRAFT_ANALYSIS_PLAN,
                       arguments={"methods": ["Kaplan-Meier", "Log-rank test", "Cox proportional hazards"]}),
    ]
    cumulative = 0.0
    for action in steps:
        obs, reward, done = env.step(action)
        cumulative += reward
        assert not done, "Episode ended early"

    print(f"  After 4 steps: cumulative reward = {cumulative:.2f}")
    assert obs.protocol_summary["primary_endpoint"] == "Overall Survival"
    assert obs.protocol_summary["sample_size"] is not None

    # Fast-forward to step 30 for FDA review
    for _ in range(26):  # steps 4..29
        obs, reward, done = env.step(
            ClinicalAction(tool=ToolName.DRAFT_ENDPOINT,
                           arguments={"endpoint": "Quality of Life", "endpoint_type": "secondary"})
        )
        cumulative += reward
        if done:
            break

    # Call FDA review
    obs, reward, done = env.step(
        ClinicalAction(tool=ToolName.SIMULATE_FDA_REVIEW, arguments={})
    )
    cumulative += reward
    print(f"  FDA verdict: {obs.fda_verdict}")
    print(f"  Total reward: {cumulative:.2f}")
    assert done
    assert obs.fda_verdict is not None
    print("  ✅ Happy path PASSED")


def test_drift_detection():
    """Schema drift fires and agent responds correctly."""
    env = ClinicalTrialEnv(max_steps=50, drift_step=3)
    env.reset()

    # Steps 0-2: build partial protocol
    env.step(ClinicalAction(tool=ToolName.DRAFT_ENDPOINT,
                            arguments={"endpoint": "Overall Survival", "endpoint_type": "primary"}))
    env.step(ClinicalAction(tool=ToolName.RUN_POWER_CALC,
                            arguments={"effect_size": 0.3, "alpha": 0.05, "power": 0.80}))
    env.step(ClinicalAction(tool=ToolName.SET_INCLUSION_CRITERIA,
                            arguments={"criteria": ["ECOG PS 0-1"], "exclusion": []}))

    # Step 3: drift fires
    obs, reward, done = env.step(
        ClinicalAction(tool=ToolName.DRAFT_ANALYSIS_PLAN,
                       arguments={"methods": ["Kaplan-Meier"]})
    )
    assert obs.schema_drift_alert is not None, "Drift should have fired at step 3"
    print(f"  Drift alert: {obs.schema_drift_alert[:60]}...")

    # Find the power warning
    warn_fields = [w.field if hasattr(w, "field") else w["field"]
                   for w in obs.consistency_warnings]
    assert "power" in warn_fields, f"Expected power warning, got: {warn_fields}"
    print("  ✅ Drift detection PASSED")


def test_bad_endpoint_penalty():
    """Bad primary endpoint → consistency warning → negative reward."""
    env = ClinicalTrialEnv(max_steps=50, drift_step=99)
    env.reset()
    obs, reward, done = env.step(
        ClinicalAction(tool=ToolName.DRAFT_ENDPOINT,
                       arguments={"endpoint": "Tumor Size", "endpoint_type": "primary"})
    )
    assert reward < 0, f"Expected negative reward for bad endpoint, got {reward}"
    assert len(obs.consistency_warnings) > 0
    print(f"  Bad endpoint reward: {reward:.2f} ✅")


def test_fda_too_early():
    """FDA review called too early → refused, no termination."""
    env = ClinicalTrialEnv(max_steps=50, drift_step=99)
    env.reset()
    obs, reward, done = env.step(
        ClinicalAction(tool=ToolName.SIMULATE_FDA_REVIEW, arguments={})
    )
    assert not done, "Episode should not end when FDA called too early"
    print(f"  Early FDA result: '{obs.tool_result[:60]}...' ✅")


if __name__ == "__main__":
    print("Running RedLine smoke tests...")
    test_happy_path()
    test_drift_detection()
    test_bad_endpoint_penalty()
    test_fda_too_early()
    print("\n✅ All smoke tests passed!")
