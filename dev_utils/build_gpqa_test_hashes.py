#!/usr/bin/env python3
"""Build the GPQA decontamination index for dataset_audit.py.

Loads `Idavidrein/gpqa::gpqa_main::train` (the slice scored by the harness;
yes, GPQA's eval split is named "train"), normalizes every question, and
writes two artifacts to src/eval/tasks/gpqamain/task_context/:

  test_decontam.jsonl   — JSONL consumed by dataset_audit.py:
                          {"id": str, "sha256": str, "first_50": str,
                           "shingle_hashes": [str, ...]}
  test_hashes.txt       — human-readable summary:
                          <sha256>\\t<shingle_count>\\t<first_50>

Run once. Re-run when the dataset version bumps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset

SHINGLE_N = 8  # must match dataset_audit.py
PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = PUNCT_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def shingle_set(normalized: str, n: int = SHINGLE_N) -> set[str]:
    words = normalized.split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def shingle_hash(s: str) -> str:
    return hashlib.blake2s(s.encode("utf-8"), digest_size=6).hexdigest()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="src/eval/tasks/gpqamain/task_context",
        help="Directory to write test_decontam.jsonl and test_hashes.txt.",
    )
    p.add_argument("--dataset", default="Idavidrein/gpqa")
    p.add_argument("--config", default="gpqa_main")
    p.add_argument("--split", default="train")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ds = load_dataset(args.dataset, args.config, split=args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decontam_path = out_dir / "test_decontam.jsonl"
    summary_path = out_dir / "test_hashes.txt"

    n = 0
    with decontam_path.open("w") as fdec, summary_path.open("w") as fsum:
        fsum.write(f"# {args.dataset}::{args.config}::{args.split}\n")
        fsum.write("# format: <sha256>\\t<shingle_count>\\t<first_50_chars>\n")
        for rec in ds:
            q = str(rec["Question"])
            rec_id = str(rec.get("Record ID", f"row_{n}"))
            norm = normalize(q)
            sha = sha256_hex(norm)
            shingles = shingle_set(norm)
            sh_hashes = sorted({shingle_hash(s) for s in shingles})
            first_50 = q[:50].replace("\t", " ").replace("\n", " ")
            json.dump(
                {
                    "id": rec_id,
                    "sha256": sha,
                    "first_50": first_50,
                    "shingle_hashes": sh_hashes,
                },
                fdec,
            )
            fdec.write("\n")
            fsum.write(f"{sha}\t{len(sh_hashes)}\t{first_50}\n")
            n += 1
    print(f"wrote {n} test items → {decontam_path}")
    print(f"wrote summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
