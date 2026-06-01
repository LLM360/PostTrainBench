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
MAX_SEQ_LEN = 8192
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


def _check_prior_experiment_published(output_dir: str) -> None:
    """Refuse training if the most recent experiment isn't fully published.

    Logic: derive the experiment dir for the run we're about to start
    (parent of --output-dir if it ends in /final_model, otherwise the
    --output-dir itself's parent). Scan `experiments/` for the
    highest-numbered exp_NNN dir that is NOT the current one. If that
    prior exp has a `final_model/` directory (training completed) but
    no `eval_result.json` (full --limit -1 self-eval) AND no
    `.published` marker, refuse to train.
    """
    import re

    out = Path(output_dir).resolve()
    # Detect experiments root: typical paths are
    #   experiments/exp_007/final_model   -> exp dir is experiments/exp_007
    #   experiments/exp_007               -> exp dir is itself
    current_exp_dir = out.parent if out.name == "final_model" else out
    experiments_root = current_exp_dir.parent
    if not experiments_root.is_dir() or experiments_root.name != "experiments":
        return  # not in the standard layout; skip (e.g. ad-hoc runs)

    exp_pat = re.compile(r"^exp_(\d+)$")
    candidates = []
    for child in experiments_root.iterdir():
        if not child.is_dir():
            continue
        m = exp_pat.match(child.name)
        if not m:
            continue
        if child.resolve() == current_exp_dir.resolve():
            continue
        candidates.append((int(m.group(1)), child))
    if not candidates:
        return  # no prior experiments

    candidates.sort()
    prior_n, prior_dir = candidates[-1]

    has_final_model = (prior_dir / "final_model").is_dir()
    has_eval_result = (prior_dir / "eval_result.json").is_file()
    has_published_marker = (prior_dir / ".published").is_file()

    if has_final_model and not has_published_marker:
        if not has_eval_result:
            msg = (
                f"\n[train_sft V3 GATE] Refusing to train exp_{current_exp_dir.name.split('_')[-1]}: "
                f"prior experiment exp_{prior_n:03d} has final_model/ but no eval_result.json.\n"
                f"  You ran train_sft.py on exp_{prior_n:03d} but never ran the full-sample\n"
                f"  evaluator. Required next action:\n"
                f"    python3 evaluate.py --model-path experiments/exp_{prior_n:03d}/final_model \\\n"
                f"        --limit -1 --json-output-file experiments/exp_{prior_n:03d}/eval_result.json \\\n"
                f"        --max-tokens 128 --max-connections 8 --gpu-memory-utilization 0.8\n"
                f"  Then: python3 publish_experiment.py --exp-dir experiments/exp_{prior_n:03d}/\n"
                f"  Then you may start exp_{current_exp_dir.name.split('_')[-1]}.\n"
            )
            raise SystemExit(msg)
        else:
            msg = (
                f"\n[train_sft V3 GATE] Refusing to train exp_{current_exp_dir.name.split('_')[-1]}: "
                f"prior experiment exp_{prior_n:03d} has eval_result.json but was never published.\n"
                f"  Required next action:\n"
                f"    python3 publish_experiment.py --exp-dir experiments/exp_{prior_n:03d}/\n"
                f"  Then you may start exp_{current_exp_dir.name.split('_')[-1]}.\n"
            )
            raise SystemExit(msg)


def _format_preflight(base_model_path: str, tokenizer) -> None:
    """Catch chat_template/EOS misconfig before training, not after.

    V2 burned ~30 min of pilot time when the trained model's saved
    config listed only <|endoftext|> (151643) as EOS but the chat
    template ended turns with <|im_end|> (151645). The model
    overgenerated past the answer line and the agent fell into a
    debugging loop. This pre-flight surfaces the mismatch loudly
    before we spend an hour training.
    """
    print("[train_sft] === format pre-flight ===", flush=True)
    print(f"[train_sft] tokenizer.eos_token={tokenizer.eos_token!r} id={tokenizer.eos_token_id}", flush=True)
    print(f"[train_sft] tokenizer.pad_token={tokenizer.pad_token!r} id={tokenizer.pad_token_id}", flush=True)

    # Probe chat_template if present — render a one-turn convo and see what
    # special tokens it emits. Loud warn if it emits a token that's NOT in
    # the saved generation_config's eos_token_id list.
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}, {"role": "assistant", "content": "ok"}],
            tokenize=False,
        )
        # Find any special token strings in the rendered output that match
        # known EOS-like patterns.
        eos_like = []
        for tok in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"):
            if tok in rendered:
                eos_like.append(tok)
        if eos_like:
            print(f"[train_sft] chat_template uses these end-tokens: {eos_like}", flush=True)
            template_eos_ids = []
            for tok in eos_like:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != tokenizer.unk_token_id:
                    template_eos_ids.append((tok, tid))
            print(f"[train_sft] their token ids: {template_eos_ids}", flush=True)

            # Compare against base model's generation_config.json if present.
            base_path = Path(base_model_path)
            gen_cfg_path = base_path / "generation_config.json"
            if gen_cfg_path.is_file():
                gen_cfg = json.loads(gen_cfg_path.read_text())
                gen_eos = gen_cfg.get("eos_token_id")
                if gen_eos is None:
                    gen_eos = []
                elif isinstance(gen_eos, int):
                    gen_eos = [gen_eos]
                template_ids = [tid for _, tid in template_eos_ids]
                missing = [tid for tid in template_ids if tid not in gen_eos]
                if missing:
                    print(
                        f"[train_sft] WARNING: chat_template uses EOS-like ids {missing} "
                        f"not in base generation_config.eos_token_id={gen_eos}. "
                        f"The trained model may overgenerate past expected stop tokens. "
                        f"Consider widening eos_token_id in your training output or in your "
                        f"build_dataset.py to terminate assistant messages with <|endoftext|>.",
                        flush=True,
                    )
    except Exception as e:
        print(f"[train_sft] chat_template probe skipped ({e})", flush=True)
    print("[train_sft] === pre-flight done ===", flush=True)


def _require_matching_audit(data_path: Path) -> None:
    """Refuse to train if the sibling audit report is missing, failing, or stale.

    Looks for `dataset_audit_report.json` next to `data_path`. Requires:
      * report exists
      * report.pass is True
      * report.data_sha256 matches sha256(data_path)
    This prevents an agent from skipping the audit step or from editing
    data.jsonl after a passing audit was produced.
    """
    audit_path = data_path.with_name("dataset_audit_report.json")
    if not audit_path.is_file():
        raise SystemExit(
            f"refusing to train: no audit report at {audit_path}. "
            "Run dataset_audit.py first."
        )
    try:
        audit = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"refusing to train: cannot read {audit_path}: {e}")
    if not audit.get("pass", False):
        raise SystemExit(
            f"refusing to train: audit at {audit_path} did not pass. "
            "Fix the dataset and re-run dataset_audit.py."
        )
    audit_sha = audit.get("data_sha256", "")
    actual_sha = file_sha256(data_path)
    if not audit_sha:
        raise SystemExit(
            f"refusing to train: audit at {audit_path} is missing "
            "data_sha256. Re-run dataset_audit.py to produce a current report."
        )
    if audit_sha != actual_sha:
        raise SystemExit(
            f"refusing to train: audit report data_sha256={audit_sha!r} "
            f"does not match current {data_path.name} sha256={actual_sha!r}. "
            "The dataset has changed since it was audited; re-run "
            "dataset_audit.py."
        )


def main() -> int:
    args = parse_args()

    # V3 publish-discipline gate.
    # Refuse to train if there's an unpublished prior experiment: the agent
    # must complete the train → full-eval → publish cycle for exp_<N-1>
    # before starting exp_<N>. This kills the in-place fix-and-retry
    # anti-pattern that produced zero publishes in the V2 pilot.
    _check_prior_experiment_published(args.output_dir)

    base_model = os.environ.get("MODEL_TO_TRAIN")
    if not base_model:
        raise SystemExit("$MODEL_TO_TRAIN must be set (the locked recipe trains this model only)")
    set_seed(args.seed)

    data_path = Path(args.data_path).resolve()
    if not data_path.is_file():
        raise SystemExit(f"data file not found: {data_path}")

    _require_matching_audit(data_path)

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

    _format_preflight(base_model, tokenizer)

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
