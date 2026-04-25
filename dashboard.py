"""
dashboard.py — Gradio demo dashboard for ClinicalPilot.
Run: python dashboard.py
"""
from __future__ import annotations

import json

import gradio as gr

from RedLine.models import ClinicalAction, ToolName
from RedLine.server import ClinicalTrialEnv

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

env = ClinicalTrialEnv(max_steps=50)
reward_history: list[float] = []
step_log: list[str] = []


def reset_episode():
    global env, reward_history, step_log
    env = ClinicalTrialEnv(max_steps=50)
    reward_history = []
    step_log = []
    obs = env.reset()
    return _render(obs, 0.0, "Episode reset. Make your first tool call.", False)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _protocol_table(proto_dict: dict) -> str:
    icons = {
        "primary_endpoint":    "🎯",
        "secondary_endpoints": "📌",
        "inclusion_criteria":  "✅",
        "exclusion_criteria":  "🚫",
        "sample_size":         "📊",
        "power":               "⚡",
        "analysis_methods":    "📈",
        "fda_verdict":         "🏛️",
        "guidance_version":    "📋",
    }
    rows = ""
    for k, v in proto_dict.items():
        icon  = icons.get(k, "•")
        v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
        rows += f"<tr><td><b>{icon} {k}</b></td><td>{v_str}</td></tr>"
    return (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='width:100%;border-collapse:collapse'>"
        + rows + "</table>"
    )


def _warnings_html(warnings: list) -> str:
    if not warnings:
        return "<span style='color:green;font-weight:bold'>✅ No consistency warnings</span>"
    html = "<ul style='color:orange;margin:0'>"
    for w in warnings:
        field = w["field"]   if isinstance(w, dict) else w.field
        msg   = w["message"] if isinstance(w, dict) else w.message
        html += f"<li><b>{field}</b>: {msg}</li>"
    html += "</ul>"
    return html


def _reward_html(history: list[float]) -> str:
    if not history:
        return "No rewards yet."
    total = sum(history)
    filled = min(30, max(1, int(abs(total) / 2)))
    bar   = "█" * filled
    color = "green" if total >= 0 else "red"
    return (
        f"<span style='color:{color};font-size:1.1em'><b>Cumulative: {total:.1f}</b></span><br>"
        f"Last step: {history[-1]:+.1f}<br>"
        f"Steps taken: {len(history)}<br>"
        f"<pre style='margin:4px 0'>{bar}</pre>"
    )


def _drift_html(alert: str | None, ever_drifted: bool) -> str:
    if alert:
        # Drift just fired and agent hasn't acknowledged yet — show big warning
        return (
            "<div style='background:#fff3e0;padding:12px;"
            "border-left:4px solid orange;border-radius:4px'>"
            "<b style='color:orange;font-size:1.1em'>⚠️ SCHEMA DRIFT DETECTED</b><br>"
            f"<small>{alert}</small></div>"
        )
    if ever_drifted:
        # Drift happened and agent acknowledged — show confirmation
        return (
            "<div style='background:#e8f5e9;padding:8px;"
            "border-left:4px solid green;border-radius:4px'>"
            "<b style='color:green'>✅ Schema drift acknowledged — protocol updated</b>"
            "</div>"
        )
    return "<span style='color:gray'>No drift detected yet</span>"


def _render(obs, reward: float, message: str, done: bool):
    proto   = obs.protocol_summary
    warns   = [w.model_dump() if hasattr(w, "model_dump") else w
               for w in obs.consistency_warnings]
    drift   = obs.schema_drift_alert
    fda_txt = obs.fda_feedback or ""
    verdict = obs.fda_verdict

    # Detect whether drift ever fired (guidance_version flips to v2)
    ever_drifted = proto.get("guidance_version") == "v2"

    # FDA verdict box
    verdict_html = ""
    if verdict:
        verdict_str = str(verdict).split(".")[-1]
        color_map   = {"APPROVE": "green", "REVISE": "orange", "REJECT": "red"}
        color       = color_map.get(verdict_str, "gray")
        verdict_html = (
            f"<div style='padding:12px;background:#f5f5f5;"
            f"border-left:6px solid {color};border-radius:4px'>"
            f"<h2 style='color:{color};margin:0'>FDA Verdict: {verdict_str}</h2>"
            f"<pre style='white-space:pre-wrap;margin:8px 0 0'>{fda_txt}</pre>"
            "</div>"
        )

    # Step log — annotate special events
    log_entry = f"Step {obs.step}: {message} | reward={reward:+.1f}"
    step_log.append(log_entry)
    if drift:
        step_log.append("  ⚠️⚠️⚠️  SCHEMA DRIFT: power min raised to 0.85 — REVISE NOW")
    if verdict:
        step_log.append(f"  🏛️  FDA: {str(verdict).split('.')[-1]}")

    log_text = "\n".join(step_log[-20:])
    status   = f"Step {obs.step}/50 | {'✅ Done' if done else '🔄 Running'}"

    return (
        _protocol_table(proto),
        _warnings_html(warns),
        _reward_html(reward_history),
        _drift_html(drift, ever_drifted),
        verdict_html,
        log_text,
        status,
    )


# ---------------------------------------------------------------------------
# Action handlers — each maps a UI button to one env.step() call
# ---------------------------------------------------------------------------

def call_draft_endpoint(endpoint: str, ep_type: str):
    if not endpoint.strip():
        return reset_episode()
    obs, reward, done = env.step(ClinicalAction(
        tool=ToolName.DRAFT_ENDPOINT,
        arguments={"endpoint": endpoint.strip(), "endpoint_type": ep_type}
    ))
    reward_history.append(reward)
    return _render(obs, reward, f"draft_endpoint({endpoint}, {ep_type})", done)


def call_set_criteria(inclusion: str, exclusion: str):
    inc = [x.strip() for x in inclusion.split(",") if x.strip()]
    exc = [x.strip() for x in exclusion.split(",") if x.strip()]
    obs, reward, done = env.step(ClinicalAction(
        tool=ToolName.SET_INCLUSION_CRITERIA,
        arguments={"criteria": inc, "exclusion": exc}
    ))
    reward_history.append(reward)
    return _render(obs, reward, "set_inclusion_criteria(...)", done)


def call_power_calc(effect_size: float, alpha: float, power: float):
    obs, reward, done = env.step(ClinicalAction(
        tool=ToolName.RUN_POWER_CALC,
        arguments={"effect_size": effect_size, "alpha": alpha, "power": power}
    ))
    reward_history.append(reward)
    return _render(obs, reward, f"run_power_calc(es={effect_size}, a={alpha}, power={power})", done)


def call_analysis_plan(methods_str: str):
    methods = [m.strip() for m in methods_str.split(",") if m.strip()]
    obs, reward, done = env.step(ClinicalAction(
        tool=ToolName.DRAFT_ANALYSIS_PLAN,
        arguments={"methods": methods}
    ))
    reward_history.append(reward)
    return _render(obs, reward, f"draft_analysis_plan({methods})", done)


def call_fda_review():
    obs, reward, done = env.step(ClinicalAction(
        tool=ToolName.SIMULATE_FDA_REVIEW,
        arguments={}
    ))
    reward_history.append(reward)
    return _render(obs, reward, "simulate_fda_review()", done)


# ---------------------------------------------------------------------------
# Auto demo
# ---------------------------------------------------------------------------

def run_auto_demo():
    """
    Scripted 7-step demo showing the full arc:
      Steps 0-3: Build a valid protocol (endpoint, criteria, power=0.80, analysis)
      Step  4:   drift_step=4 fires — guidance updates, alert shown, power warning appears
      Step  5:   Agent detects drift, re-runs power=0.85 → +5 drift bonus
      Step  6:   simulate_fda_review() → APPROVE (+15)
    """
    global env, reward_history, step_log
    # drift_step=4: drift fires on step index 4 (the 5th action)
    env = ClinicalTrialEnv(max_steps=50, drift_step=4)
    reward_history = []
    step_log = []
    # Initialize the environment without calling `reset_episode()` which
    # would overwrite `env` (and its forced `drift_step`) with a default
    # instance. Call `env.reset()` directly to preserve `drift_step`.
    obs = env.reset()

    steps = [
        # 0: Set primary endpoint (pre-drift, v1 guidance)
        (ClinicalAction(
            tool=ToolName.DRAFT_ENDPOINT,
            arguments={"endpoint": "Overall Survival", "endpoint_type": "primary"}
        ), "Step 0: Set primary endpoint → Overall Survival"),

        # 1: Set inclusion/exclusion criteria
        (ClinicalAction(
            tool=ToolName.SET_INCLUSION_CRITERIA,
            arguments={
                "criteria": ["ECOG PS 0-1", "Stage IIIB/IV NSCLC", "Age >= 18"],
                "exclusion": ["Prior platinum therapy", "Active CNS metastases"]
            }
        ), "Step 1: Set inclusion/exclusion criteria"),

        # 2: Power calc under v1 guidance — power=0.80 is valid here
        (ClinicalAction(
            tool=ToolName.RUN_POWER_CALC,
            arguments={"effect_size": 0.3, "alpha": 0.05, "power": 0.80}
        ), "Step 2: Power calc (v1 guidance) — power=0.80, N=176"),

        # 3: Draft analysis plan
        (ClinicalAction(
            tool=ToolName.DRAFT_ANALYSIS_PLAN,
            arguments={"methods": ["Kaplan-Meier", "Log-rank test", "Cox proportional hazards"]}
        ), "Step 3: Draft analysis plan"),

        # 4: Any action on this step triggers drift injection (drift_step=4)
        #    The OBSERVATION returned will show schema_drift_alert + power warning
        (ClinicalAction(
            tool=ToolName.DRAFT_ENDPOINT,
            arguments={"endpoint": "Quality of Life", "endpoint_type": "secondary"}
        ), "Step 4: [DRIFT FIRES] Guidance v2 injected — power must now be >= 0.85"),

        # 5: Agent detects drift alert, revises power to 0.85 — drift_acknowledged=True, +5 bonus
        (ClinicalAction(
            tool=ToolName.RUN_POWER_CALC,
            arguments={"effect_size": 0.3, "alpha": 0.05, "power": 0.85}
        ), "Step 5: Agent responds to drift — power revised to 0.85, N=214 (+5 drift bonus)"),

        # 6: Call FDA review — should APPROVE (+15)
        (ClinicalAction(
            tool=ToolName.SIMULATE_FDA_REVIEW,
            arguments={}
        ), "Step 6: simulate_fda_review() — expect APPROVE"),
    ]

    last_render = _render(obs, 0.0, "Episode reset. Auto-demo starting.", False)
    for action, label in steps:
        obs, reward, done = env.step(action)
        reward_history.append(reward)
        last_render = _render(obs, reward, label, done)
        if done:
            break

    return last_render


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

OUTPUTS = None  # defined after component creation

with gr.Blocks(title="ClinicalPilot — OpenEnv Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏥 ClinicalPilot — Clinical Trial Protocol Design Environment
        **First RL training environment for Phase 2 oncology protocol design.**
        Design a complete clinical trial protocol step-by-step. Watch for FDA schema drift!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## Protocol State")
            protocol_html = gr.HTML()

        with gr.Column(scale=1):
            gr.Markdown("## Metrics")
            reward_html  = gr.HTML()
            drift_html   = gr.HTML()
            status_text  = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        warnings_html = gr.HTML()

    with gr.Row():
        fda_html = gr.HTML()

    gr.Markdown("## Step Log")
    step_log_box = gr.Textbox(label="Action Log", lines=10, interactive=False)

    OUTPUTS = [
        protocol_html, warnings_html, reward_html,
        drift_html, fda_html, step_log_box, status_text
    ]

    gr.Markdown("---")
    gr.Markdown("## Actions")

    with gr.Tab("1️⃣ Draft Endpoint"):
        with gr.Row():
            ep_input = gr.Textbox(label="Endpoint name", value="Overall Survival")
            ep_type  = gr.Dropdown(["primary", "secondary"], value="primary", label="Type")
        gr.Button("Call draft_endpoint").click(
            call_draft_endpoint, [ep_input, ep_type], OUTPUTS
        )

    with gr.Tab("2️⃣ Set Criteria"):
        inc_input = gr.Textbox(
            label="Inclusion (comma-separated)",
            value="ECOG PS 0-1, Stage IIIB/IV NSCLC"
        )
        exc_input = gr.Textbox(
            label="Exclusion (comma-separated)",
            value="Prior platinum-based chemotherapy"
        )
        gr.Button("Call set_inclusion_criteria").click(
            call_set_criteria, [inc_input, exc_input], OUTPUTS
        )

    with gr.Tab("3️⃣ Power Calculation"):
        with gr.Row():
            es_sl    = gr.Slider(0.1, 0.8, value=0.3, step=0.05, label="Effect size")
            alpha_sl = gr.Dropdown([0.01, 0.025, 0.05], value=0.05, label="Alpha")
            pow_sl   = gr.Slider(0.70, 0.95, value=0.80, step=0.05, label="Power")
        gr.Button("Call run_power_calc").click(
            call_power_calc, [es_sl, alpha_sl, pow_sl], OUTPUTS
        )

    with gr.Tab("4️⃣ Analysis Plan"):
        methods_input = gr.Textbox(
            label="Methods (comma-separated)",
            value="Kaplan-Meier, Log-rank test, Cox proportional hazards"
        )
        gr.Button("Call draft_analysis_plan").click(
            call_analysis_plan, [methods_input], OUTPUTS
        )

    with gr.Tab("5️⃣ FDA Review"):
        gr.Markdown(
            "**Triggers the FDA Complete Response Letter simulation. Ends the episode.**"
        )
        gr.Button("Call simulate_fda_review", variant="primary").click(
            call_fda_review, [], OUTPUTS
        )

    gr.Markdown("---")
    with gr.Row():
        gr.Button("🔄 Reset Episode").click(reset_episode, [], OUTPUTS)
        gr.Button("🤖 Run Auto Demo (shows drift detection)", variant="secondary").click(
            run_auto_demo, [], OUTPUTS
        )

    # Populate UI on first load
    demo.load(fn=reset_episode, inputs=[], outputs=OUTPUTS)


if __name__ == "__main__":
    print("ClinicalPilot dashboard starting on http://0.0.0.0:7860 ...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
