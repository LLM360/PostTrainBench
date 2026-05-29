#!/usr/bin/env python3
"""Locked SFT+LoRA training recipe for the data-engineering agent loop.

The agent's *only* variable is --data-path (and --output-dir / --max-steps /
--seed as small whitelisted knobs). All hyperparameters are fixed below.
Schema check is strict: each JSONL row must be
    {"messages": [{"role": "user", "content": str},
                  {"role": "assistant", "content": str}, ...]}
(optional leading system role is allowed).

The script trains a LoRA adapter, merges it into the base, and writes a full
HF checkpoint to --output-dir, plus train_manifest.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

# --- Locked hyperparameters --------------------------------------------------
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"
LR = 2e-4
LR_SCHEDULER = "cosine"
PER_DEVICE_BS = 2
GRAD_ACCUM = 8
MAX_SEQ_LEN = 2048
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 10

MAX_STEPS_DEFAULT = 2000
SEED_DEFAULT = 42
MAX_EPOCHS_CAP = 30  # hard cap on epochs regardless of --max-steps; raised from 5 because the empirical winner used ~20 epochs via continuation training


def parse_args() -> argparse.Namespace:
    # Whitelist: --data-path, --output-dir, --max-steps, --seed only.
    # --base-model is intentionally NOT a flag: the locked recipe always
    # trains $MODEL_TO_TRAIN. Argparse rejects any unknown arg.
    p = argparse.ArgumentParser(
        description="Locked SFT+LoRA trainer.", allow_abbrev=False
    )
    p.add_argument("--data-path", required=True, help="JSONL with 'messages' rows.")
    p.add_argument("--output-dir", default="final_model")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"row {i}: invalid JSON ({e})")
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                raise SystemExit(f"row {i}: 'messages' must be list of ≥2 entries")
            roles = [m.get("role") for m in msgs]
            if "user" not in roles or "assistant" not in roles:
                raise SystemExit(f"row {i}: needs both user and assistant turns")
            for j, m in enumerate(msgs):
                if not isinstance(m.get("content"), str) or not m["content"].strip():
                    raise SystemExit(f"row {i}.messages[{j}]: missing/empty content")
            rows.append({"messages": msgs})
    if not rows:
        raise SystemExit("empty dataset")
    return rows


def main() -> int:
    args = parse_args()
    base_model = os.environ.get("MODEL_TO_TRAIN")
    if not base_model:
        raise SystemExit("$MODEL_TO_TRAIN must be set (the locked recipe trains this model only)")
    set_seed(args.seed)

    data_path = Path(args.data_path).resolve()
    if not data_path.is_file():
        raise SystemExit(f"data file not found: {data_path}")

    rows = load_and_validate(data_path)
    ds = Dataset.from_list(rows)

    # Auto-cap max_steps to MAX_EPOCHS_CAP epochs so small datasets don't
    # over-train. effective_bs = PER_DEVICE_BS * GRAD_ACCUM.
    effective_bs = PER_DEVICE_BS * GRAD_ACCUM
    steps_per_epoch = max(1, (len(rows) + effective_bs - 1) // effective_bs)
    epoch_cap_steps = MAX_EPOCHS_CAP * steps_per_epoch
    effective_max_steps = min(args.max_steps, epoch_cap_steps)
    if effective_max_steps != args.max_steps:
        print(
            f"[train_sft] capping --max-steps {args.max_steps} → {effective_max_steps} "
            f"({MAX_EPOCHS_CAP} epochs on {len(rows)} rows, effective_bs={effective_bs})"
        )

    print(f"[train_sft] base={base_model} rows={len(rows)} steps={effective_max_steps}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tmp_out = Path(args.output_dir + "_lora_tmp")
    sft_cfg = SFTConfig(
        output_dir=str(tmp_out),
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_steps=effective_max_steps,
        max_length=MAX_SEQ_LEN,
        bf16=True,
        save_strategy="no",
        logging_steps=LOGGING_STEPS,
        seed=args.seed,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    actual_steps = int(trainer.state.global_step)
    final_loss = float(train_result.metrics.get("train_loss", float("nan")))

    print("[train_sft] freeing trainer/model memory before merge")
    peft_model = trainer.model
    del trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[train_sft] merging LoRA into base and saving final_model/")
    merged = peft_model.merge_and_unload()
    del peft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

    manifest = {
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_model": base_model,
        "data_path": str(data_path),
        "data_sha256": file_sha256(data_path),
        "row_count": len(rows),
        "actual_steps": actual_steps,
        "max_steps_arg": int(args.max_steps),
        "effective_max_steps": int(effective_max_steps),
        "final_train_loss": final_loss,
        "hyperparams": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "lr": LR,
            "lr_scheduler": LR_SCHEDULER,
            "per_device_bs": PER_DEVICE_BS,
            "grad_accum": GRAD_ACCUM,
            "max_seq_len": MAX_SEQ_LEN,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "seed": args.seed,
        },
    }
    with (out_dir / "train_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Clean up the LoRA-only tmp dir to save disk
    if tmp_out.exists():
        import shutil
        shutil.rmtree(tmp_out, ignore_errors=True)

    print(f"[train_sft] DONE → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
