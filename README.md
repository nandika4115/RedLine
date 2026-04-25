---
title: RedLine
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
---
# 🏥 RedLine — OpenEnv Environment

> **The first RL training environment for clinical trial protocol design.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://meta-pytorch.org/OpenEnv)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/YOUR_HF_USERNAME/RedLine)
[![Theme](https://img.shields.io/badge/Theme-%232%20Long--Horizon-green)](https://docs.google.com/document/d/...)

---

## Problem

Designing a Phase 2 clinical trial protocol takes **12–18 months** and costs **$500K–$2M** in consultant and regulatory fees. FDA first-submission failure rate is ~40%, mostly from structural errors: wrong endpoint selection, power miscalculation, inconsistent analysis plans.

No RL training environment existed to teach agents to do this. ClinicalPilot is that environment.

---

## What the Agent Does

The agent plans a Phase 2 oncology trial protocol across **50 steps** using 5 tools:

| Tool | Description |
|------|-------------|
| `draft_endpoint` | Select primary or secondary endpoints |
| `set_inclusion_criteria` | Define patient eligibility |
| `run_power_calc` | Compute required sample size |
| `draft_analysis_plan` | Select statistical methods |
| `simulate_fda_review` | Get simulated FDA Complete Response Letter |

### Causal Dependencies (Long-Horizon)
- Primary endpoint choice → constrains valid analysis methods
- Power level → determines sample size → affects FDA review
- Schema drift at step ~20 → forces revision of statistical plan

---

## Schema Drift (Patronus AI theme)

At a random step between step 15–25, the FDA issues a **guidance update**:
- ORR (Overall Response Rate) is now accepted as a primary endpoint
- Minimum statistical power raised from 0.80 → 0.85

Agents that detect and adapt score +5 drift-awareness bonus. Agents that ignore it fail FDA review.

---

## Reward Design

| Event | Reward |
|-------|--------|
| Complete protocol section (endpoint, criteria, stats, analysis) | +2 per section |
| Zero consistency warnings on a step | +1 |
| Each consistency warning | -2 |
| Detect & respond to schema drift | +5 |
| FDA APPROVE | +15 |
| FDA REVISE | +5 |
| FDA REJECT | -10 |

The **consistency checker** is deterministic and instant (rule-based) — provides dense, non-gameable reward signal every step.

---

## Results

| Metric | Random Baseline | Trained Agent |
|--------|----------------|---------------|
| Avg episode reward | ~-12 | ~+18 |
| FDA approval rate | ~5% | ~70% |
| Drift detection rate | ~10% | ~80% |

![Reward Curve](outputs/RedLine/reward_curve.png)
![Before vs After](outputs/RedLine/before_after.png)

---

## Training

Training uses **SFT → GRPO** (2-phase approach):

1. **SFT** on 125 expert-authored protocol steps → pre-warms the model
2. **GRPO** on the live environment → learns to handle novel states + drift

```bash
# Full training pipeline
python train.py --phase both --sft_epochs 3 --rl_steps 200
```

See the [Colab notebook](ClinicalPilot_Training.ipynb) for a runnable end-to-end demo.

---

## Quickstart

```bash
pip install git+https://huggingface.co/spaces/YOUR_HF_USERNAME/RedLine

from RedLine.server import ClinicalTrialEnv
from RedLine.models import ClinicalAction, ToolName

env = ClinicalTrialEnv(max_steps=50)
obs = env.reset()

action = ClinicalAction(
    tool=ToolName.DRAFT_ENDPOINT,
    arguments={"endpoint": "Overall Survival", "endpoint_type": "primary"}
)
obs, reward, done = env.step(action)
print(f"Reward: {reward}, Warnings: {obs.consistency_warnings}")
```

---

## Setup

```bash
pip install -r requirements.txt
python smoke_test.py          # verify everything works
python dashboard.py           # launch Gradio demo (port 7860)
```

---

## Architecture

```
ClinicalTrialEnv (OpenEnv)
├── Protocol State (JSON, 50+ fields)
├── Tool Dispatcher (5 tools)
├── Consistency Checker (deterministic, step-level reward)
├── Schema Drift Engine (fires at step 15–25)
└── FDA CRL Simulator (rule-based, episodic reward)

Agent
├── Trained with SFT on expert trajectories
└── Fine-tuned with GRPO on live environment
```

---

## Themes Covered

- **#2 Long-Horizon Planning**: Endpoint choice at step 0 constrains analysis plan at step 40
- **#3.1 Professional World Modeling**: Realistic FDA regulatory domain
- **#5 Wild Card**: Novel benchmark domain, no prior RL env exists
- **Patronus AI (Schema Drift)**: Mid-episode regulatory guidance update
- **Snorkel AI**: Expert agent feedback loop (see architecture)

---

## Links

- [HuggingFace Space](https://huggingface.co/spaces/YOUR_HF_USERNAME/RedLine)
- [Training Notebook (Colab)](ClinicalPilot_Training.ipynb)
- [HuggingFace Blog Post](https://huggingface.co/blog/YOUR_HF_USERNAME/clinical-pilot)

---

## Team

Built at OpenEnv Hackathon 2026, Bangalore.
