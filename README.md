---
title: RedLine
emoji: 🏥
colorFrom: red
colorTo: purple
sdk: gradio
app_file: app.py
---

# 🏥 RedLine — The First RL Environment for Clinical Trial Protocol Design

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/openenv)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/nandika4115/RedLine)
[![Theme](https://img.shields.io/badge/Theme-%232%20Long--Horizon-green)](#)
[![Theme](https://img.shields.io/badge/Theme-%233.1%20Professional%20Tasks-orange)](#)
[![Theme](https://img.shields.io/badge/Theme-%235%20Wild%20Card-purple)](#)

---

## The Problem

Every Phase 2 oncology trial starts the same way: a protocol document. Endpoint selection. Inclusion criteria. Statistical power. Analysis plan. Get any one of these wrong and the FDA sends a Complete Response Letter — a rejection that sets you back 12–18 months and $500K–$2M in fees.

The FDA rejects ~40% of first submissions. Not because the science is bad. Because of **structural errors that are entirely preventable**: wrong endpoint for the indication, underpowered study, analysis methods that don't match the endpoint, inconsistencies between sections. These are the kind of errors a well-trained agent could catch.

No RL training environment existed to teach agents to do this work. Until now.

**RedLine is that environment.**

---

## Links

- 🤗 [HuggingFace Space (live demo)](https://huggingface.co/spaces/nandika4115/RedLine)
- 📓 [Training Notebook (Colab)](RedLine_Training.ipynb)
- 📝 [HuggingFace Blog Post](https://huggingface.co/blog/nandika4115/redline-clinical-trial-rl)

---


## What The Agent Does

The agent designs a complete Phase 2 oncology protocol across **50 steps** using 5 tools:

| Tool | Purpose |
|------|---------|
| `draft_endpoint` | Select primary/secondary endpoints |
| `set_inclusion_criteria` | Define patient eligibility |
| `run_power_calc` | Compute sample size from effect size + power |
| `draft_analysis_plan` | Choose statistical methods |
| `simulate_fda_review` | Get a simulated FDA verdict (terminal) |

**The hard part:** decisions are causally linked. Endpoint at step 0 constrains valid methods at step 30. Underpowered by 5% → rejection. And at a random mid-episode step, the FDA changes its guidance — power minimum jumps 0.80 → 0.85. Agents that miss it fail. Agents that adapt earn +5.
---

## The Environment

The agent designs a complete Phase 2 oncology trial protocol across **up to 50 steps** using 5 tools:

| Tool | What it does | Why it matters |
|------|-------------|----------------|
| `draft_endpoint` | Choose primary/secondary endpoints | Must be FDA-accepted for the indication |
| `set_inclusion_criteria` | Define patient eligibility | Constraints must be internally consistent |
| `run_power_calc` | Compute sample size from effect size + power | Power ≥ 0.80 (v1) or ≥ 0.85 (v2 post-drift) |
| `draft_analysis_plan` | Select statistical methods | Must match endpoint type (e.g. OS → time-to-event) |
| `simulate_fda_review` | Get a simulated Complete Response Letter | Terminal action — ends the episode |


### Schema Drift — The Mid-Episode Curveball

At a fixed step mid-episode, the FDA issues a guidance update (v2):

> *"ORR is now accepted as a primary endpoint for accelerated-approval trials. Minimum statistical power raised to ≥ 0.85."*

An agent running power=0.80 (valid under v1) now has an underpowered protocol. It must detect the drift alert and re-run the power calculation before calling FDA review. Agents that ignore it get rejected. Agents that adapt in one step earn **+5 drift-awareness bonus**.

---

## Reward Design

RedLine uses a **4-rubric composable reward system** — not a single scalar. Every rubric fires on a different signal:

| Rubric | Signal | When |
|--------|--------|------|
| 🔬 **Coherence** | +1 (no warnings) / −2 per warning / +2 per new section completed | Every step |
| ⚡ **Efficiency** | +0.5 × unused steps on APPROVE / −0.5 per no-op | Every step |
| 🌊 **Drift** | +5 for responding to schema drift / −3 for calling FDA after drift with power < 0.85 | Event-driven |
| 🏛️ **Outcome** | +15 APPROVE / +5 REVISE / −10 REJECT | Terminal |

**Why this matters for training:** The coherence and efficiency rubrics provide dense reward signal every single step — no sparse reward problem. The drift rubric fires on a specific event. The outcome rubric is the episodic goal. An agent can't game any one rubric without the others pushing back.

---

## Training

We use a 2-phase pipeline: **SFT to pre-warm, GRPO to adapt.**

### Phase 1: SFT on Expert Trajectories

125 hand-authored protocol steps across 25 episodes (15 no-drift, 10 drift-aware). SFT teaches the model the basic grammar of the task — valid endpoints, correct tool sequencing, what a coherent protocol looks like.

```bash
python train.py --phase sft --sft_epochs 3
```

**Result:** Loss drops from 2.64 → 0.09 (97% reduction). Token accuracy 49.5% → 98.4%.

![SFT Training Curve](outputs/RedLine/sft_curve.png)
*Phase 1: SFT loss and token accuracy across 3 epochs on Qwen2.5-1.5B-Instruct + LoRA*

### Phase 2: GRPO RL on the Live Environment

The SFT-warmed model then trains against the actual `ClinicalTrialEnv` using GRPO. The environment provides step-level reward from the rubric system — no separate reward model needed.

```bash
python train.py --phase rl --rl_steps 200
```

**Result:** Per-step reward improves from 0.20 → 1.00 (Δ+0.80) over 100 training steps.

![GRPO Reward Curve](outputs/RedLine/reward_curve.png)
*Phase 2: GRPO per-step reward curve. Rolling average (red) shows consistent upward trend.*

### Full Pipeline

```bash
python train.py --phase both --sft_epochs 3 --rl_steps 200
```

See the [Training Notebook](RedLine_Training.ipynb) for a full runnable Colab demo.

---

## Results: Before vs After Training

| Metric | Random Baseline | Trained Agent (SFT + GRPO) |
|--------|----------------|---------------------------|
| Primary endpoint | ❌ Invalid (Biomarker Response) | ✅ Overall Survival |
| Statistical power | ❌ 0.55 | ✅ 0.85 (post-drift compliant) |
| Schema drift | ❌ Ignored | ✅ Corrected in 1 step |
| Efficiency | ❌ −22.5 (45 no-ops) | ✅ +21.5 (43 unused steps) |
| FDA outcome | ❌ REJECTED | ✅ APPROVED |
| **Episode reward** | **−60.5** | **+51.5** |
| **Δ improvement** | — | **+112 points** |

The trained agent finishes a complete, FDA-compliant protocol in **7 out of 50 steps**. The random agent burns all 50 and still gets rejected.

> *Before/After comparison uses a scripted replay of the best observed post-training episode against a deterministic random baseline. GRPO step-level reward curve above is from the real training run.*

---

## Try It

**→ [Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/nandika4115/RedLine)**

```bash
pip install -r requirements.txt
python smoke_test.py   # verify env
python dashboard.py    # http://localhost:7860
```

```python
from RedLine.server import ClinicalTrialEnv
from RedLine.models import ClinicalAction, ToolName

env = ClinicalTrialEnv(max_steps=50)
obs = env.reset()

obs, reward, done = env.step(ClinicalAction(
    tool=ToolName.DRAFT_ENDPOINT,
    arguments={"endpoint": "Overall Survival", "endpoint_type": "primary"}
))
# reward: +3.0 | warnings: []
```

---


## Architecture

```
RedLine (OpenEnv-compliant)
├── ClinicalTrialEnv          ← reset() / step() / state()
│   ├── ProtocolState         ← 50+ fields, progressive fill
│   ├── Tool Dispatcher       ← 5 tools, deterministic handlers
│   ├── Consistency Checker   ← rule-based, fires every step
│   ├── Schema Drift Engine   ← injects v2 guidance mid-episode
│   └── FDA CRL Simulator     ← terminal, rule-based verdict
│
├── 4-Rubric Reward System
│   ├── rubric_coherence()    ← dense, step-level
│   ├── rubric_efficiency()   ← dense, step-budget
│   ├── rubric_drift()        ← sparse, event-driven
│   └── rubric_outcome()      ← episodic, FDA verdict
│
└── Training Pipeline
    ├── SFT (expert_trajectories.py → trl.SFTTrainer)
    └── GRPO (ClinicalTrialEnv → trl.GRPOTrainer)
```

---

## Why It Matters

Clinical trial protocol design is a $50B/year industry. The 40% FDA rejection rate on first submission isn't a scientific problem — it's a structural reasoning problem. Agents that can plan long-horizon, detect regulatory drift, and maintain internal consistency across 50+ decisions could save months and millions per trial.

Beyond the direct application: RedLine is a benchmark for **professional-domain long-horizon reasoning**. It has causal dependencies, mid-episode distribution shift, dense multi-signal reward, and a deterministic evaluator (the FDA rule engine). Any team that wants to train or evaluate agents on real-world planning tasks can use this environment.

**This domain has no prior RL environment.** That's the gap RedLine fills.

---

## Team

Built at **OpenEnv Hackathon 2026, Bangalore** — in one sprint, from scratch.