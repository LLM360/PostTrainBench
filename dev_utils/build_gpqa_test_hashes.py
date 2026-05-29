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

Reproducibility:
  * --revision pins the HF dataset commit SHA so reruns are byte-identical.
    Default = the gpqa_main HEAD at PR #6 review time.
  * The normalization / shingle / hash functions below MUST match the
    canonical implementation in src/eval/general/dataset_audit.py. We do not
    `import` it here because the audit module lives in a parallel PR and the
    two scripts are intentionally runnable independently (this generator is
    typically run on a workstation; the audit runs inside agent containers).
    If you change one, change the other — the SHINGLE_N / regex / digest
    sizes are part of the audit contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

# Default pinned revision — HEAD of Idavidrein/gpqa main at PR review time.
# Override with --revision if the dataset bumps and you intend to rebuild.
DEFAULT_GPQA_REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"

# --- canonical decontam constants (keep in lockstep with dataset_audit.py) ---
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


# If the canonical audit module is importable, prefer its helpers to
# guarantee parity. We attempt the import lazily and fall back silently —
# this script is also used in environments where the audit module isn't
# present (e.g. a fresh checkout of the generator-only PR).
def _maybe_use_canonical_helpers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    try:
        from src.eval.general import dataset_audit as _audit  # type: ignore
    except Exception:
        return
    global normalize, shingle_set, shingle_hash, sha256_hex, SHINGLE_N
    for name in ("normalize", "shingle_set", "shingle_hash", "sha256_hex"):
        if hasattr(_audit, name):
            globals()[name] = getattr(_audit, name)
    if hasattr(_audit, "SHINGLE_N") and _audit.SHINGLE_N != SHINGLE_N:
        raise SystemExit(
            f"SHINGLE_N mismatch: local={SHINGLE_N} canonical={_audit.SHINGLE_N}; "
            "update this script to match dataset_audit.py before regenerating."
        )


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
    p.add_argument(
        "--revision",
        default=DEFAULT_GPQA_REVISION,
        help=(
            "HF dataset revision (commit SHA or tag). Pinned by default for "
            "byte-stable reruns. To find a fresh SHA: "
            "`curl -s https://huggingface.co/api/datasets/Idavidrein/gpqa | jq -r .sha`"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _maybe_use_canonical_helpers()
    ds = load_dataset(
        args.dataset, args.config, split=args.split, revision=args.revision
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    decontam_path = out_dir / "test_decontam.jsonl"
    summary_path = out_dir / "test_hashes.txt"

    n = 0
    with decontam_path.open("w") as fdec, summary_path.open("w") as fsum:
        fsum.write(f"# {args.dataset}::{args.config}::{args.split}@{args.revision}\n")
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
