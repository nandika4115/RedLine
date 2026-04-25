"""
train.py — ClinicalPilot training script.
Compatible with: TRL 1.2.0, Transformers 5.0.0, PyTorch 2.10+

Phase 1: SFT on expert trajectories (pre-warms the model)
Phase 2: GRPO RL training against the ClinicalTrialEnv

Usage:
    python train.py --phase sft        # Run SFT only
    python train.py --phase rl         # Run RL only
    python train.py --phase both       # SFT then RL (recommended)
    python train.py --phase eval       # Evaluate trained model
"""
from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

MODEL_ID   = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path("./outputs/clinical_pilot")
SFT_DIR    = OUTPUT_DIR / "sft_checkpoint"
RL_DIR     = OUTPUT_DIR / "rl_checkpoint"


def load_model_and_tokenizer(model_id: str = MODEL_ID, use_4bit: bool = True):
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
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, linewidth=1.5, color="#2196F3", label="Episode reward")
    window = max(1, len(rewards) // 10)
    if len(rewards) >= window:
        rolling = [
            sum(rewards[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(rewards))
        ]
        ax.plot(rolling, linewidth=2.5, color="#F44336", label=f"Rolling avg (w={window})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.set_title("ClinicalPilot — Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Reward curve saved to {path}")


def run_sft(num_epochs: int = 3):
    print("\n" + "=" * 60)
    print("PHASE 1: Supervised Fine-Tuning on Expert Trajectories")
    print("=" * 60)

    model, tokenizer = load_model_and_tokenizer()
    model = add_lora(model)

    raw = generate_sft_dataset()
    print(f"  SFT pairs: {len(raw)}")
    dataset = Dataset.from_list(raw)

    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    dataset = dataset.map(lambda x: {"text": format_example(x)})

    # TRL 1.2.0: max_seq_length -> max_length, warmup_ratio -> warmup_steps
    config = SFTConfig(
        output_dir=str(SFT_DIR),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        max_length=1024,
        dataset_text_field="text",
        report_to="none",
    )

    # TRL 1.2.0: 'tokenizer' -> 'processing_class'
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=config,
    )

    print("  Training SFT...")
    trainer.train()
    trainer.save_model(str(SFT_DIR))
    print(f"  SFT checkpoint saved to {SFT_DIR}")
    return str(SFT_DIR)


def build_env_reward_fn():
    env = ClinicalTrialEnv(max_steps=50)
    env.reset()

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            try:
                action_dict = json.loads(completion.strip())
                tool = action_dict.get("tool", "")
                args = action_dict.get("arguments", {})
                action = ClinicalAction(tool=ToolName(tool), arguments=args)
            except Exception:
                rewards.append(-5.0)
                continue
            try:
                _, step_reward, done = env.step(action)
                if done:
                    env.reset()
                rewards.append(step_reward)
            except Exception:
                rewards.append(-3.0)
        return rewards

    return reward_fn


def build_rl_dataset(tokenizer, n_samples: int = 200):
    env = ClinicalTrialEnv(max_steps=50)
    samples = []

    for _ in range(n_samples):
        obs = env.reset()
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
        for _ in range(num_warmup):
            action = random.choice(random_actions)
            obs, _, done = env.step(action)
            if done:
                break

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
        model = add_lora(model)

    reward_fn = build_env_reward_fn()
    rl_dataset = build_rl_dataset(tokenizer, n_samples=num_steps)

    # TRL 1.2.0: max_new_tokens and temperature removed from GRPOConfig
    config = GRPOConfig(
        output_dir=str(RL_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        report_to="none",
        save_strategy="steps",
        save_steps=50,
        bf16=torch.cuda.is_bf16_supported(),
    )

    # TRL 1.2.0: 'tokenizer' -> 'processing_class'
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=rl_dataset,
    )

    # Set generation params after init (TRL 1.x approach)
    gen_cfg = getattr(trainer, 'generation_config', None) or getattr(trainer.model, 'generation_config', None)
    if gen_cfg is not None:
        gen_cfg.max_new_tokens = 128
        gen_cfg.temperature = 0.8
        gen_cfg.do_sample = True

    print("  Training RL...")
    trainer.train()
    trainer.save_model(str(RL_DIR))

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


def evaluate_agent(model_path: str, n_episodes: int = 10, label: str = "agent"):
    from transformers import pipeline

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
        for _ in range(50):
            obs_dict  = obs.model_dump()
            user_msg  = make_prompt(obs_dict)
            chat      = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ]
            prompt_text = pipe.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            output = pipe(prompt_text, max_new_tokens=128, temperature=0.3, do_sample=True)
            completion = output[0]["generated_text"][len(prompt_text):]
            try:
                action_dict = json.loads(completion.strip())
                action = ClinicalAction(
                    tool=ToolName(action_dict["tool"]),
                    arguments=action_dict.get("arguments", {}),
                )
            except Exception:
                continue
            obs, reward, done = env.step(action)
            cum_reward += reward
            if done:
                break
        total_rewards.append(cum_reward)
        state = env.state()
        fda_verdicts.append(state.protocol.fda_verdict)

    avg_reward   = sum(total_rewards) / len(total_rewards)
    approve_rate = sum(1 for v in fda_verdicts if str(v) == "FDAVerdict.APPROVE") / n_episodes
    print(f"\n[{label}] Avg reward: {avg_reward:.2f} | FDA approve rate: {approve_rate:.0%}")
    return avg_reward, approve_rate


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