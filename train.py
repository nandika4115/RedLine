"""
train.py — ClinicalPilot training script.

Phase 1: SFT on expert trajectories (pre-warms the model)
Phase 2: GRPO RL training against the ClinicalTrialEnv

Usage:
    python train.py --phase sft        # Run SFT only
    python train.py --phase rl         # Run RL only (requires SFT checkpoint)
    python train.py --phase both       # SFT then RL (recommended)

This script is designed to run in Google Colab with a T4/A100.
Model: Qwen/Qwen2.5-1.5B-Instruct (fits in 8GB VRAM with 4-bit quant)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

# ── Check for required packages ──────────────────────────────────────────────
try:
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
    from peft import LoraConfig, get_peft_model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        f"Missing dependency: {e}\n"
        "Install with: pip install trl transformers datasets peft accelerate bitsandbytes matplotlib"
    )

from RedLine.expert_trajectories import SYSTEM_PROMPT, generate_sft_dataset, make_prompt
from RedLine.models import ClinicalAction, ToolName
from RedLine.server import ClinicalTrialEnv

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("./outputs/clinical_pilot")
SFT_DIR    = OUTPUT_DIR / "sft_checkpoint"
RL_DIR     = OUTPUT_DIR / "rl_checkpoint"


# ── Utilities ────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id: str = MODEL_ID, use_4bit: bool = True):
    """Load model with optional 4-bit quantization for memory efficiency."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = None
    if use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def add_lora(model, r: int = 16, alpha: int = 32):
    """Attach LoRA adapters for efficient fine-tuning."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def save_reward_plot(rewards: list[float], path: str = "reward_curve.png"):
    """Save a reward curve plot to disk."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, linewidth=1.5, color="#2196F3", label="Episode reward")

    # Rolling average
    window = max(1, len(rewards) // 10)
    if len(rewards) >= window:
        rolling = [
            sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(rewards))
        ]
        ax.plot(rolling, linewidth=2.5, color="#F44336", label=f"Rolling avg (w={window})")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.set_title("ClinicalPilot — Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Reward curve saved to {path}")


# ── Detect TRL version for API compatibility ──────────────────────────────────

def _trl_version_tuple():
    try:
        import trl
        parts = trl.__version__.split(".")
        return tuple(int(x) for x in parts[:2])
    except Exception:
        return (0, 0)


# ── Phase 1: SFT ─────────────────────────────────────────────────────────────

def run_sft(num_epochs: int = 3):
    print("\n" + "=" * 60)
    print("PHASE 1: Supervised Fine-Tuning on Expert Trajectories")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    model = add_lora(model)

    # Build SFT dataset
    raw = generate_sft_dataset()
    print(f"  SFT pairs: {len(raw)}")
    dataset = Dataset.from_list(raw)

    # Format for SFT trainer
    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    dataset = dataset.map(lambda x: {"text": format_example(x)})

    # ── TRL >=0.13 renamed max_seq_length → max_length in SFTConfig ──────────
    trl_major, trl_minor = _trl_version_tuple()
    seq_len_kwarg = "max_length" if (trl_major, trl_minor) >= (0, 13) else "max_seq_length"

    config = SFTConfig(
        output_dir=str(SFT_DIR),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        **{seq_len_kwarg: 1024},
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    print("  Training SFT...")
    trainer.train()
    trainer.save_model(str(SFT_DIR))
    print(f"  SFT checkpoint saved to {SFT_DIR}")
    return str(SFT_DIR)


# ── Phase 2: GRPO RL ──────────────────────────────────────────────────────────

def build_env_reward_fn():
    """
    Returns a GRPO-compatible reward function.

    GRPO in TRL expects:
      reward_fn(completions: List[str], prompts: List[str], **kwargs) -> List[float]

    We simulate one env step per completion and return the step reward.
    """
    env = ClinicalTrialEnv(max_steps=50)
    env.reset()

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            # Parse the model's JSON action
            try:
                action_dict = json.loads(completion.strip())
                tool = action_dict.get("tool", "")
                args = action_dict.get("arguments", {})
                action = ClinicalAction(tool=ToolName(tool), arguments=args)
            except Exception:
                # Malformed JSON or invalid tool → big penalty
                rewards.append(-5.0)
                continue

            # Step the env and get reward
            try:
                _, step_reward, done = env.step(action)
                if done:
                    env.reset()   # auto-reset for continuous GRPO training
                rewards.append(step_reward)
            except Exception:
                rewards.append(-3.0)

        return rewards

    return reward_fn


def build_rl_dataset(tokenizer, n_samples: int = 200):
    """
    Build a dataset of prompts for GRPO training.
    Each sample is a partially-completed protocol state → agent must predict action.
    """
    env = ClinicalTrialEnv(max_steps=50)
    samples = []

    for _ in range(n_samples):
        obs = env.reset()
        # Random walk 0–15 steps to get diverse protocol states
        num_warmup = random.randint(0, 15)
        random_actions = [
            ClinicalAction(tool=ToolName.DRAFT_ENDPOINT,
                           arguments={"endpoint": random.choice(
                               ["Overall Survival", "Progression-Free Survival", "ORR"]),
                               "endpoint_type": "primary"}),
            ClinicalAction(tool=ToolName.SET_INCLUSION_CRITERIA,
                           arguments={"criteria": ["ECOG PS 0-1"], "exclusion": ["Prior chemo"]}),
            ClinicalAction(tool=ToolName.RUN_POWER_CALC,
                           arguments={"effect_size": 0.3, "alpha": 0.05, "power": 0.80}),
        ]
        for i in range(num_warmup):
            action = random.choice(random_actions)
            obs, _, done = env.step(action)
            if done:
                break

        # Build prompt
        obs_dict = obs.model_dump()
        user_msg = make_prompt(obs_dict)
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt_text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        samples.append({"prompt": prompt_text})

    return Dataset.from_list(samples)


def run_rl(sft_checkpoint: str | None = None, num_steps: int = 200):
    print("\n" + "=" * 60)
    print("PHASE 2: GRPO RL Training on ClinicalTrialEnv")
    print("=" * 60)

    model_path = sft_checkpoint or MODEL_ID
    print(f"  Loading from: {model_path}")

    model, tokenizer = load_model_and_tokenizer(model_path, use_4bit=True)
    if sft_checkpoint is None:
        model = add_lora(model)   # add LoRA if starting fresh

    reward_fn = build_env_reward_fn()
    rl_dataset = build_rl_dataset(tokenizer, n_samples=num_steps)

    config = GRPOConfig(
        output_dir=str(RL_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        max_new_tokens=128,
        temperature=0.8,
        report_to="none",
        save_strategy="steps",
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=rl_dataset,
    )

    print("  Training RL...")
    train_result = trainer.train()
    trainer.save_model(str(RL_DIR))

    # ── Save reward curve ────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_history = trainer.state.log_history
    rewards = [
        entry.get("train_reward", entry.get("reward", 0))
        for entry in log_history
        if "train_reward" in entry or "reward" in entry
    ]
    if rewards:
        save_reward_plot(rewards, str(OUTPUT_DIR / "reward_curve.png"))
    else:
        print("  Warning: no reward entries found in log_history to plot.")

    print(f"  RL checkpoint saved to {RL_DIR}")
    return str(RL_DIR)


# ── Evaluation (before/after comparison) ─────────────────────────────────────

def evaluate_agent(model_path: str, n_episodes: int = 10, label: str = "agent"):
    """Run n_episodes and return avg cumulative reward + FDA pass rate."""
    from transformers import AutoModelForCausalLM, pipeline

    pipe = pipeline(
        "text-generation",
        model=model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    env = ClinicalTrialEnv(max_steps=50)
    total_rewards = []
    fda_verdicts  = []

    for ep in range(n_episodes):
        obs = env.reset()
        cum_reward = 0.0

        for _step in range(50):
            obs_dict   = obs.model_dump()
            user_msg   = make_prompt(obs_dict)
            chat       = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
            prompt_text = pipe.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            output = pipe(
                prompt_text,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
            )
            completion = output[0]["generated_text"][len(prompt_text):]

            try:
                action_dict = json.loads(completion.strip())
                action = ClinicalAction(
                    tool=ToolName(action_dict["tool"]),
                    arguments=action_dict.get("arguments", {}),
                )
            except Exception:
                # Bad output → skip step
                continue

            obs, reward, done = env.step(action)
            cum_reward += reward
            if done:
                break

        total_rewards.append(cum_reward)
        state = env.state()
        fda_verdicts.append(state.protocol.fda_verdict)

    avg_reward = sum(total_rewards) / len(total_rewards)
    approve_rate = sum(1 for v in fda_verdicts if str(v) == "FDAVerdict.APPROVE") / n_episodes
    print(f"\n[{label}] Avg reward: {avg_reward:.2f} | FDA approve rate: {approve_rate:.0%}")
    return avg_reward, approve_rate


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClinicalPilot Training")
    parser.add_argument("--phase", choices=["sft", "rl", "both", "eval"], default="both")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--rl_steps",   type=int, default=200)
    parser.add_argument("--eval_model", type=str, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ("sft", "both"):
        sft_path = run_sft(num_epochs=args.sft_epochs)

    if args.phase in ("rl", "both"):
        sft_path = str(SFT_DIR) if SFT_DIR.exists() else None
        rl_path  = run_rl(sft_checkpoint=sft_path, num_steps=args.rl_steps)

    if args.phase == "eval":
        model_path = args.eval_model or str(RL_DIR)
        evaluate_agent(model_path, n_episodes=5, label="trained_agent")