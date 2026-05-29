#!/usr/bin/env python3
"""Pick the best of an 8-way data-engineering array run by official eval metric.

Scans every EVAL_DIR matching the (task, model, agent, hours, num_gpus) tuple
under POST_TRAIN_BENCH_RESULTS_DIR, reads its metrics.json (produced by the
existing eval phase in run_task.sh), and writes winner.json next to the
shared log CSV.

The winner is decided by the official benchmark metric, not by any peer
signal — that's the whole point of withholding eval scores from the shared
CSV during the run.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="evaluation task id, e.g. gpqamain")
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--results-dir", default=os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
    p.add_argument(
        "--metric-key",
        default=None,
        help="Specific key inside metrics.json to rank by. Default: first numeric value found.",
    )
    p.add_argument("--shared-log-dir", default=None)
    return p.parse_args()


def safe(s: str) -> str:
    return re.sub(r"[/:\[\]]", "_", s)


def load_metric(metrics_path: Path, key: str | None) -> tuple[str, float] | None:
    try:
        m = json.loads(metrics_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if key and key in m:
        try:
            return key, float(m[key])
        except (TypeError, ValueError):
            return None
    for k, v in m.items():
        try:
            return k, float(v)
        except (TypeError, ValueError):
            continue
    return None


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_dir)
    model_safe = safe(args.model)
    pattern = f"{args.task}_{model_safe}_*"

    entries: list[dict] = []
    for agent_dir in sorted(results_root.glob("*")):
        if not agent_dir.is_dir():
            continue
        for eval_dir in sorted(agent_dir.glob(pattern)):
            metrics_path = eval_dir / "metrics.json"
            picked = load_metric(metrics_path, args.metric_key)
            entry = {
                "eval_dir": str(eval_dir),
                "agent_dir": agent_dir.name,
                "cluster_id": eval_dir.name.split("_")[-1],
                "metric_key": picked[0] if picked else None,
                "metric_value": picked[1] if picked else None,
                "metrics_path_exists": metrics_path.exists(),
            }
            entries.append(entry)

    if not entries:
        print(f"no runs found under {results_root} matching {pattern}")
        return 1

    entries.sort(
        key=lambda e: (e["metric_value"] if e["metric_value"] is not None else float("-inf")),
        reverse=True,
    )
    winner = entries[0]

    if args.shared_log_dir:
        shared_dir = Path(args.shared_log_dir)
    else:
        shared_dir = results_root / "data_eng_shared" / f"{args.task}_{model_safe}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    winner_path = shared_dir / "winner.json"
    payload = {
        "task": args.task,
        "model": args.model,
        "winner": winner,
        "ranking": entries,
    }
    winner_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {winner_path}")
    if winner["metric_value"] is None:
        print("WARNING: winner has no parsed metric; check that runs completed.")
    else:
        print(
            f"winner: {winner['agent_dir']}/{Path(winner['eval_dir']).name} "
            f"with {winner['metric_key']}={winner['metric_value']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
