#!/usr/bin/env python3
"""Hard-gate audit for data-engineering training datasets.

Refuses to allow training when:
  1) any row's user content overlaps the benchmark test set (sha256 exact
     match OR ≥ DECONTAM_SHINGLE_OVERLAP_MIN shared 13-grams with any test item)
  2) the dataset is monocultural (fails any of: distinct n-gram ratios,
     mean pairwise TF-IDF cosine distance, length coefficient-of-variation)

Writes a JSON report next to the dataset and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# --- Tunable thresholds (kept at top so we can adjust during pilot) ---------
# Decontam uses 8-grams (shorter shingles → more sensitive to paraphrases).
# Diversity discriminators: distinct_4g + mean_cos_dist are the strong ones;
# distinct_1g is size-sensitive (common words dominate large corpora) so its
# floor is set very low to avoid false fails on 1k+ row datasets.
SHINGLE_N = 8
DECONTAM_SHINGLE_OVERLAP_MIN = 3         # ≥ this many shared 8-grams = contamination
DIVERSITY_DISTINCT_1G_MIN = 0.03
DIVERSITY_DISTINCT_4G_MIN = 0.30
DIVERSITY_MEAN_COS_DIST_MIN = 0.35
DIVERSITY_LEN_CV_MIN = 0.15
DIVERSITY_SAMPLE_SIZE = 500
MIN_ROWS = 50
MAX_REPORTED_VIOLATIONS = 10

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


# --- Loading -----------------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
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
            rows.append(obj)
    return rows


def extract_user_content(row: dict, row_idx: int) -> str:
    if "messages" not in row:
        raise SystemExit(f"row {row_idx}: missing 'messages' key")
    msgs = row["messages"]
    if not isinstance(msgs, list) or len(msgs) < 2:
        raise SystemExit(f"row {row_idx}: 'messages' must be a list of ≥2 entries")
    user_msgs = [m for m in msgs if m.get("role") == "user"]
    if not user_msgs:
        raise SystemExit(f"row {row_idx}: no user message found")
    has_asst = any(m.get("role") == "assistant" for m in msgs)
    if not has_asst:
        raise SystemExit(f"row {row_idx}: no assistant message found")
    parts = [str(m.get("content", "")) for m in user_msgs]
    return "\n".join(parts)


# --- Decontamination ---------------------------------------------------------

def load_test_decontam(path: Path) -> tuple[set[str], dict[str, list[str]]]:
    """Returns (test_sha_set, shingle_hash -> [test_ids])."""
    sha_set: set[str] = set()
    shingle_index: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            sha_set.add(obj["sha256"])
            tid = obj["id"]
            for h in obj["shingle_hashes"]:
                shingle_index.setdefault(h, []).append(tid)
    return sha_set, shingle_index


def check_decontam(rows: list[dict], test_path: Path) -> dict:
    sha_set, shingle_index = load_test_decontam(test_path)
    violations: list[dict] = []
    for i, row in enumerate(rows):
        user = extract_user_content(row, i)
        norm = normalize(user)
        row_sha = sha256_hex(norm)
        if row_sha in sha_set:
            violations.append(
                {"row": i, "kind": "sha256_exact", "first_80": user[:80]}
            )
            if len(violations) >= MAX_REPORTED_VIOLATIONS:
                break
            continue
        shingles = shingle_set(norm)
        hits: dict[str, int] = {}
        for sh in shingles:
            h = shingle_hash(sh)
            for tid in shingle_index.get(h, ()):
                hits[tid] = hits.get(tid, 0) + 1
        if hits:
            tid, overlap = max(hits.items(), key=lambda x: x[1])
            if overlap >= DECONTAM_SHINGLE_OVERLAP_MIN:
                violations.append(
                    {
                        "row": i,
                        "kind": "shingle_overlap",
                        "test_id": tid,
                        "overlap": overlap,
                        "threshold": DECONTAM_SHINGLE_OVERLAP_MIN,
                        "first_80": user[:80],
                    }
                )
                if len(violations) >= MAX_REPORTED_VIOLATIONS:
                    break
    return {
        "pass": not violations,
        "violations": violations,
        "test_items_loaded": len(sha_set),
        "shingle_overlap_threshold": DECONTAM_SHINGLE_OVERLAP_MIN,
    }


# --- Diversity ---------------------------------------------------------------

def distinct_n_ratio(normalized_rows: list[str], n: int) -> float:
    ngrams: set[tuple] = set()
    total = 0
    for r in normalized_rows:
        words = r.split()
        if len(words) < n:
            continue
        for i in range(len(words) - n + 1):
            ngrams.add(tuple(words[i : i + n]))
            total += 1
    return len(ngrams) / total if total else 0.0


def mean_pairwise_cos_dist(rows: list[str], sample: int, seed: int) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances

    rng = random.Random(seed)
    pool = rows if len(rows) <= sample else rng.sample(rows, sample)
    if len(pool) < 2:
        return 0.0
    vec = TfidfVectorizer(max_features=10000).fit_transform(pool)
    dists = cosine_distances(vec)
    iu = np.triu_indices(dists.shape[0], k=1)
    return float(dists[iu].mean())


def length_cv(rows: list[str]) -> float:
    lens = np.array([len(r.split()) for r in rows], dtype=float)
    if lens.mean() == 0:
        return 0.0
    return float(lens.std() / lens.mean())


def check_diversity(user_texts: list[str], seed: int) -> dict:
    normalized = [normalize(t) for t in user_texts]
    d1 = distinct_n_ratio(normalized, 1)
    d4 = distinct_n_ratio(normalized, 4)
    cos_d = mean_pairwise_cos_dist(normalized, DIVERSITY_SAMPLE_SIZE, seed)
    lcv = length_cv(normalized)
    failures = []
    if d1 < DIVERSITY_DISTINCT_1G_MIN:
        failures.append(f"distinct_1g={d1:.3f} < {DIVERSITY_DISTINCT_1G_MIN}")
    if d4 < DIVERSITY_DISTINCT_4G_MIN:
        failures.append(f"distinct_4g={d4:.3f} < {DIVERSITY_DISTINCT_4G_MIN}")
    if cos_d < DIVERSITY_MEAN_COS_DIST_MIN:
        failures.append(f"mean_cos_dist={cos_d:.3f} < {DIVERSITY_MEAN_COS_DIST_MIN}")
    if lcv < DIVERSITY_LEN_CV_MIN:
        failures.append(f"len_cv={lcv:.3f} < {DIVERSITY_LEN_CV_MIN}")
    return {
        "pass": not failures,
        "distinct_1g": d1,
        "distinct_4g": d4,
        "mean_cos_dist": cos_d,
        "len_cv": lcv,
        "failures": failures,
        "sample_size": min(len(user_texts), DIVERSITY_SAMPLE_SIZE),
    }


# --- Main --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decontam + diversity hard-gate audit.")
    p.add_argument("--data-path", required=True, help="Training data JSONL (messages schema).")
    p.add_argument(
        "--test-decontam",
        default="task_context/test_decontam.jsonl",
        help="Path to test_decontam.jsonl shipped in task_context/.",
    )
    p.add_argument(
        "--report-path",
        default=None,
        help="Where to write the JSON report. Defaults to <data_path>.audit_report.json beside the data.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    test_path = Path(args.test_decontam).resolve()
    if not data_path.is_file():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 2
    if not test_path.is_file():
        print(f"ERROR: test_decontam file not found: {test_path}", file=sys.stderr)
        return 2

    rows = load_rows(data_path)
    if len(rows) < MIN_ROWS:
        report = {
            "pass": False,
            "data_path": str(data_path),
            "row_count": len(rows),
            "reason": f"too few rows ({len(rows)} < {MIN_ROWS})",
        }
        _write_report(report, args.report_path, data_path)
        print(f"AUDIT FAIL: too few rows ({len(rows)} < {MIN_ROWS})", file=sys.stderr)
        return 1

    user_texts = [extract_user_content(r, i) for i, r in enumerate(rows)]
    decontam = check_decontam(rows, test_path)
    diversity = check_diversity(user_texts, args.seed)

    passed = decontam["pass"] and diversity["pass"]
    report = {
        "pass": passed,
        "data_path": str(data_path),
        "row_count": len(rows),
        "decontam": decontam,
        "diversity": diversity,
        "thresholds": {
            "decontam_shingle_overlap_min": DECONTAM_SHINGLE_OVERLAP_MIN,
            "diversity_distinct_1g_min": DIVERSITY_DISTINCT_1G_MIN,
            "diversity_distinct_4g_min": DIVERSITY_DISTINCT_4G_MIN,
            "diversity_mean_cos_dist_min": DIVERSITY_MEAN_COS_DIST_MIN,
            "diversity_len_cv_min": DIVERSITY_LEN_CV_MIN,
            "min_rows": MIN_ROWS,
            "shingle_n": SHINGLE_N,
        },
    }
    _write_report(report, args.report_path, data_path)

    if passed:
        print(
            f"AUDIT PASS: {len(rows)} rows, distinct_1g={diversity['distinct_1g']:.3f}, "
            f"distinct_4g={diversity['distinct_4g']:.3f}, "
            f"mean_cos_dist={diversity['mean_cos_dist']:.3f}, "
            f"len_cv={diversity['len_cv']:.3f}"
        )
        return 0

    print("AUDIT FAIL:", file=sys.stderr)
    if not decontam["pass"]:
        print(f"  decontam: {len(decontam['violations'])} violation(s)", file=sys.stderr)
        for v in decontam["violations"][:5]:
            print(f"    row {v['row']} ({v['kind']}): {v.get('first_80', '')!r}", file=sys.stderr)
    if not diversity["pass"]:
        print("  diversity:", file=sys.stderr)
        for fail in diversity["failures"]:
            print(f"    {fail}", file=sys.stderr)
    return 1


def _write_report(report: dict, explicit_path: str | None, data_path: Path) -> None:
    if explicit_path:
        out = Path(explicit_path)
    else:
        out = data_path.with_name("dataset_audit_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
