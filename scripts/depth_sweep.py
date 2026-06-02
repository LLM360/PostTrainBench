#!/usr/bin/env python3
"""Offline training-DEPTH sweep — validate / retune the locked FIXED_EPOCHS.

Training depth (epochs) is LOCKED in the data-engineering loop: the agent cannot
tune it, so a run's outcome is attributable to the DATA, not a duration tune
(see src/eval/general/train_sft.py `FIXED_EPOCHS_CANONICAL` and the publish gate
in publish_experiment.py `CANONICAL_FIXED_EPOCHS`). This tool is the *human's*
way to pick a GOOD locked value: it trains the SAME dataset at several epoch
counts and full-evals each, so you can confirm the locked depth generalizes
across dataset sizes before committing to it.

It is intentionally OUTSIDE the agent's reach:
  * It overrides depth via $POSTTRAIN_FIXED_EPOCHS, which the agent's container
    env never sets.
  * It NEVER calls publish_experiment.py. Even if it did, the publish gate
    refuses any experiment trained at a non-canonical depth — so a sweep model
    can never reach the shared log.

After you pick a value, update BOTH:
  * train_sft.py     -> FIXED_EPOCHS_CANONICAL
  * publish_experiment.py -> CANONICAL_FIXED_EPOCHS
(they must stay in sync; the gate enforces the publish-side value).

Usage (offline, on a GPU node; NOT inside the agent loop):

    MODEL_TO_TRAIN=Qwen/Qwen3-1.7B-Base \\
    uv run python scripts/depth_sweep.py \\
        --data-path /path/to/dataset.jsonl \\
        --task gpqamain \\
        --epochs 8 14 20 26 \\
        --work-dir /tmp/depth_sweep \\
        --seed 42

Run it on a SMALL and a LARGE dataset to check the chosen depth holds for both.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SFT = REPO_ROOT / "src" / "eval" / "general" / "train_sft.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", required=True, help="JSONL dataset to train on (fixed across the sweep).")
    p.add_argument("--task", default="gpqamain", help="Benchmark task name (locates src/eval/tasks/<task>/evaluate.py).")
    p.add_argument("--epochs", type=int, nargs="+", required=True, help="Epoch counts to sweep, e.g. 8 14 20 26.")
    p.add_argument("--work-dir", default="/tmp/depth_sweep", help="Where per-epoch models + eval JSON go.")
    p.add_argument("--seed", type=int, default=42, help="Train seed held fixed across the sweep.")
    p.add_argument("--max-tokens", type=int, default=16000, help="Eval generation budget (match the harness: 16000).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--limit", type=int, default=-1, help="Eval sample limit (-1 = full set).")
    return p.parse_args()


def _accuracy_from_eval(eval_json: Path) -> float | None:
    """Best-effort accuracy pull from an evaluate.py JSON file (display only)."""
    try:
        data = json.loads(eval_json.read_text())
    except Exception:
        return None
    # Common shapes: {"accuracy": x} or inspect_ai results.scores[*].metrics.accuracy.value
    if isinstance(data.get("accuracy"), (int, float)):
        return float(data["accuracy"])
    try:
        for score in data.get("results", {}).get("scores", []):
            acc = score.get("metrics", {}).get("accuracy", {}).get("value")
            if isinstance(acc, (int, float)):
                return float(acc)
    except Exception:
        pass
    return None


def main() -> int:
    args = parse_args()
    if not os.environ.get("MODEL_TO_TRAIN"):
        raise SystemExit("Set $MODEL_TO_TRAIN (the base model train_sft.py trains).")
    evaluate_py = REPO_ROOT / "src" / "eval" / "tasks" / args.task / "evaluate.py"
    for path in (TRAIN_SFT, evaluate_py, Path(args.data_path)):
        if not path.exists():
            raise SystemExit(f"not found: {path}")

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, float | None]] = []

    for epochs in args.epochs:
        out_dir = work / f"epochs_{epochs}"
        eval_json = work / f"eval_epochs_{epochs}.json"
        print(f"\n=== depth sweep: {epochs} epochs ===", flush=True)

        env = dict(os.environ)
        env["POSTTRAIN_FIXED_EPOCHS"] = str(epochs)  # offline-only override; gate-refused if published
        train_cmd = [
            sys.executable, str(TRAIN_SFT),
            "--data-path", str(args.data_path),
            "--output-dir", str(out_dir),
            "--seed", str(args.seed),
        ]
        subprocess.run(train_cmd, env=env, check=True)

        eval_cmd = [
            sys.executable, str(evaluate_py),
            "--model-path", str(out_dir),
            "--limit", str(args.limit),
            "--json-output-file", str(eval_json),
            "--max-tokens", str(args.max_tokens),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        subprocess.run(eval_cmd, env=env, check=True)
        results.append((epochs, _accuracy_from_eval(eval_json)))

    print("\n=== DEPTH SWEEP SUMMARY (data fixed; vary epochs) ===")
    print(f"dataset: {args.data_path}")
    print(f"{'epochs':>8}  {'accuracy':>10}")
    for epochs, acc in results:
        print(f"{epochs:>8}  {('%.4f' % acc) if acc is not None else 'n/a':>10}")
    print(
        "\nPick the epoch count at the accuracy plateau (more epochs stop helping / "
        "start hurting), then set FIXED_EPOCHS_CANONICAL in train_sft.py AND "
        "CANONICAL_FIXED_EPOCHS in publish_experiment.py to it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
