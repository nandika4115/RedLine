# RedLine: *When the rules change, so must the agent*

> A reinforcement learning environment where regulatory drift is the signal, not the noise.

---

## The problem it is solving

Clinical trial protocols take months, cost millions, and still fail regulatory review — not because the science is wrong, but because the *rules changed mid-process*. Endpoints shifted. Thresholds revised. A compliant decision in January becomes a violation by March.

Current AI systems can't handle this. They're optimized for one question: *is this output correct right now?* They have no mechanism to detect when the ground has moved — and adapt.

---

## The insight

We asked: what if adapting to changing rules wasn't a failure to avoid, but the objective to optimize?

> **Regulatory change is not noise. It is signal.**

Every regulated domain — clinical trials, finance, tax, compliance — operates under rules that evolve. Yet every AI system deployed in these domains assumes a static world. That assumption is wrong, and it fails silently.

---

## What RedLine does

RedLine is an RL environment where an agent plans under **evolving constraints**. Rules change mid-episode. Correct decisions become invalid. The agent isn't penalized for past choices made under old rules — it's evaluated on how fast it detects the shift and how cleanly it recovers.

The goal shifts from *correctness* to **dynamic consistency**.

| Phase | Objective |
|-------|-----------|
| Detect | Identify when prior assumptions are invalidated |
| Adapt | Restore compliance under the new rule set |
| Recover | Maintain consistency across the full decision history |

---

## Why it's different

RedLine is a system to treat regulatory schema drift as a first-class RL training signal — not an edge case, but the core objective.

Most agents are trained to be right. RedLine trains agents to **stay** right.

> Intelligence isn't about being correct once. It's about staying correct as the world changes.

---
## Future Work

RedLine is intended as a starting point for a broader class of adaptive systems. While demonstrated on clinical trials, the core idea naturally extends to domains like financial compliance and tax regulation. Future work includes integrating real-world regulatory data, moving toward learned evaluators, and scaling to longer, more complex decision processes. Together, this positions RedLine as a general framework for training agents in dynamic, rule-driven environments.
