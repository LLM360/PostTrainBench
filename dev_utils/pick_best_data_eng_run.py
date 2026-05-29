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
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, help="evaluation task id, e.g. gpqamain")
    p.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-0.5B")
    p.add_argument("--results-dir", default=os.environ.get("POST_TRAIN_BENCH_RESULTS_DIR", "results"))
    p.add_argument(
        "--metric-key",
        required=True,
        help=(
            "Required. Key inside metrics.json to rank by, e.g. "
            "'accuracy' for gpqamain. There is no default — picking the "
            "'first numeric field' is too easy to misuse for a winner writer."
        ),
    )
    p.add_argument("--shared-log-dir", default=None)
    return p.parse_args()


def safe(s: str) -> str:
    return re.sub(r"[/:\[\]]", "_", s)


def load_metric(metrics_path: Path, key: str) -> tuple[str, float] | None:
    """Return (key, float_value) if metrics.json has `key` as a numeric value,
    else None. No 'first numeric field' fallback by design."""
    try:
        m = json.loads(metrics_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if key not in m:
        return None
    try:
        return key, float(m[key])
    except (TypeError, ValueError):
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
        print(f"no runs found under {results_root} matching {pattern}", file=sys.stderr)
        return 1

    # Surface which runs are missing the metric, then exclude them from
    # ranking. If *all* runs are missing it, fail loudly — we will not
    # write a winner.json against an empty ranking.
    missing = [e for e in entries if e["metric_value"] is None]
    valid = [e for e in entries if e["metric_value"] is not None]

    if missing:
        print(
            f"warning: {len(missing)}/{len(entries)} runs have no '{args.metric_key}' "
            "in metrics.json — excluding them from ranking:",
            file=sys.stderr,
        )
        for e in missing:
            reason = "metrics.json missing" if not e["metrics_path_exists"] else f"no '{args.metric_key}' field"
            print(f"  - {e['agent_dir']}/{Path(e['eval_dir']).name}: {reason}", file=sys.stderr)

    if not valid:
        print(
            f"error: no runs have a parsed '{args.metric_key}' metric; "
            "refusing to write winner.json. Check that eval phase completed "
            "and that --metric-key matches a numeric key produced by evaluate.py.",
            file=sys.stderr,
        )
        return 2

    valid.sort(key=lambda e: e["metric_value"], reverse=True)
    winner = valid[0]
    # Keep the full ranking (valid first, then missing) so the artifact still
    # records every run that was considered.
    full_ranking = valid + missing

    if args.shared_log_dir:
        shared_dir = Path(args.shared_log_dir)
    else:
        shared_dir = results_root / "data_eng_shared" / f"{args.task}_{model_safe}"
    shared_dir.mkdir(parents=True, exist_ok=True)
    winner_path = shared_dir / "winner.json"
    payload = {
        "task": args.task,
        "model": args.model,
        "metric_key": args.metric_key,
        "winner": winner,
        "ranking": full_ranking,
        "excluded_missing_metric": [e["eval_dir"] for e in missing],
    }
    winner_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {winner_path}")
    print(
        f"winner: {winner['agent_dir']}/{Path(winner['eval_dir']).name} "
        f"with {winner['metric_key']}={winner['metric_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
