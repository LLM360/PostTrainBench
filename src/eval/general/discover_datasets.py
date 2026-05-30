#!/usr/bin/env python3
"""V4 DYNAMIC Hugging Face dataset discovery (trending / popular / keyword).

PostTrainBench data-engineering loop V4. There is NO static curated menu of
datasets to pick from. Instead the agent must EXPLORE the live Hub and judge
relevance itself. This helper is step 1 of that loop:

    1. discover   -> this script: query the live Hub by keyword + sort signal,
                     hard-exclude any GPQA-tainted dataset (contamination
                     guard), and emit a COMPACT ranked digest (id, popularity
                     signals, tags, a 1-row sample + inferred field names) the
                     agent can read in ONE shot WITHOUT downloading anything.
    2. inspect    -> read the digest: which fields look like a multiple-choice
                     graduate-level question + answer letter?
    3. judge      -> teacher_judge.py: ask the teacher model whether a candidate
                     is actually relevant to the GPQA-style reasoning benchmark.
    4. synthesize -> teacher_synth.py: if NOTHING on the Hub is relevant, have
                     the teacher generate fresh step-by-step reasoning data.

Design goals: best-effort + time-boxed. The Hub being slow on one dataset must
never hang the whole run; every per-dataset probe is wrapped in try/except and
skipped on failure. Only a total inability to reach the Hub is fatal.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

# --- contamination guard -----------------------------------------------------
# Any candidate whose id / tags / card mentions this is HARD EXCLUDED so the
# agent can never train on GPQA-derived data (dataset_audit.py is the train-time
# gate; this is the discovery-time guard).
GPQA_MARKER = "gpqa"

# Map friendly --sort values to the HfApi sort key.
SORT_MAP = {
    "trending": "trending_score",
    "downloads": "downloads",
    "likes": "likes",
}

SAMPLE_CHARS = 400          # truncate the 1-row sample to ~this many chars
PROBE_TIMEOUT_S = 20.0      # soft per-dataset budget for sample loading


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _truncate(value: Any, limit: int = SAMPLE_CHARS) -> Any:
    """Truncate stringified values so the digest stays one-shot readable."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...<truncated>"
    if isinstance(value, (list, tuple)):
        s = json.dumps(value, ensure_ascii=False, default=str)
        return s if len(s) <= limit else s[:limit] + "...<truncated>"
    if isinstance(value, dict):
        return {k: _truncate(v, max(40, limit // max(1, len(value)))) for k, v in value.items()}
    return value


def _contaminated(ds_id: str, tags: list[str], card_text: str) -> bool:
    """True if the GPQA marker appears in id, any tag, or the card text."""
    blob = " ".join([ds_id or ""] + [str(t) for t in (tags or [])]).lower()
    if GPQA_MARKER in blob:
        return True
    if card_text and GPQA_MARKER in card_text.lower():
        return True
    return False


def _trim_tags(tags: list[str], max_tags: int = 12) -> list[str]:
    """Keep the most informative tags; drop noisy auto-tags, cap the count."""
    if not tags:
        return []
    cleaned: list[str] = []
    for t in tags:
        t = str(t)
        # region:* and arxiv:* tags are rarely useful for relevance judging.
        if t.startswith("region:") or t.startswith("arxiv:"):
            continue
        cleaned.append(t)
    return cleaned[:max_tags]


def _probe_sample(ds_id: str) -> dict[str, Any]:
    """Best-effort, time-boxed 1-row sample + inferred field names.

    Tries streaming first (cheapest — no full download), then falls back to a
    1-row split slice. Always returns a dict; on failure returns an 'error' key.
    """
    start = time.monotonic()
    # Strategy 1: streaming take(1) — does not download the whole dataset.
    try:
        from datasets import load_dataset  # imported lazily; container has it

        ds = load_dataset(ds_id, split="train", streaming=True)
        for row in ds:
            fields = sorted(row.keys()) if isinstance(row, dict) else []
            return {
                "fields": fields,
                "row": _truncate(row),
                "via": "streaming",
            }
        return {"fields": [], "row": None, "via": "streaming", "note": "empty stream"}
    except Exception as e_stream:  # noqa: BLE001 - best-effort, never fatal
        if time.monotonic() - start > PROBE_TIMEOUT_S:
            return {"error": f"timeout/stream-fail: {type(e_stream).__name__}: {e_stream}"}

    # Strategy 2: 1-row split slice (may resolve a different default split).
    for split in ("train[:1]", "test[:1]", "validation[:1]"):
        try:
            from datasets import load_dataset

            ds = load_dataset(ds_id, split=split)
            if len(ds) == 0:
                continue
            row = ds[0]
            fields = sorted(row.keys()) if isinstance(row, dict) else []
            return {"fields": fields, "row": _truncate(row), "via": f"slice:{split}"}
        except Exception:  # noqa: BLE001 - try next split / give up
            if time.monotonic() - start > PROBE_TIMEOUT_S:
                return {"error": "timeout while slicing splits"}
            continue
    return {"error": "could not load any sample split"}


def _num_rows_estimate(info: Any) -> Any:
    """Cheap row-count estimate from the listing metadata, if present."""
    # HfApi dataset objects sometimes expose card_data with dataset_info sizes;
    # keep this cheap and best-effort — return None if not trivially available.
    cd = getattr(info, "card_data", None)
    if cd is None:
        return None
    try:
        data = cd.to_dict() if hasattr(cd, "to_dict") else dict(cd)
    except Exception:  # noqa: BLE001
        return None
    di = data.get("dataset_info")
    if isinstance(di, dict):
        splits = di.get("splits")
        if isinstance(splits, list):
            total = 0
            seen = False
            for sp in splits:
                if isinstance(sp, dict) and isinstance(sp.get("num_examples"), int):
                    total += sp["num_examples"]
                    seen = True
            if seen:
                return total
    return None


def discover(api, keywords: list[str], sort_key: str, task_filter: str | None,
             limit: int, probe_samples: bool) -> list[dict[str, Any]]:
    """Query the Hub per keyword, merge, dedupe, exclude GPQA, rank, probe."""
    seen_ids: dict[str, Any] = {}
    excluded = 0

    # Over-fetch per keyword so that after merge + GPQA exclusion we still have
    # enough to fill --limit.
    per_kw = max(limit * 2, limit + 10)

    search_terms = keywords if keywords else [None]
    common_kwargs: dict[str, Any] = {"sort": sort_key, "direction": -1, "limit": per_kw}
    if task_filter:
        common_kwargs["task_categories"] = task_filter

    for term in search_terms:
        try:
            kwargs = dict(common_kwargs)
            if term:
                kwargs["search"] = term
            results = list(api.list_datasets(**kwargs))
        except TypeError:
            # Older/newer hub: task_categories arg may differ; retry with filter=.
            kwargs = {"sort": sort_key, "direction": -1, "limit": per_kw}
            if term:
                kwargs["search"] = term
            if task_filter:
                kwargs["filter"] = task_filter
            try:
                results = list(api.list_datasets(**kwargs))
            except Exception as e:  # noqa: BLE001
                log(f"[discover] keyword={term!r} search failed: {e}; continuing")
                continue
        except Exception as e:  # noqa: BLE001
            log(f"[discover] keyword={term!r} search failed: {e}; continuing")
            continue

        for ds in results:
            ds_id = getattr(ds, "id", None) or getattr(ds, "datasetId", None)
            if not ds_id or ds_id in seen_ids:
                continue
            tags = list(getattr(ds, "tags", []) or [])
            card_text = ""
            cd = getattr(ds, "card_data", None)
            if cd is not None:
                try:
                    card_text = json.dumps(cd.to_dict() if hasattr(cd, "to_dict") else dict(cd),
                                           default=str)
                except Exception:  # noqa: BLE001
                    card_text = str(cd)
            if _contaminated(ds_id, tags, card_text):
                log(f"[discover] EXCLUDED (gpqa contamination guard): {ds_id}")
                excluded += 1
                continue
            seen_ids[ds_id] = {
                "id": ds_id,
                "downloads": getattr(ds, "downloads", None),
                "likes": getattr(ds, "likes", None),
                "trending_score": getattr(ds, "trending_score", None),
                "tags": _trim_tags(tags),
                "last_modified": str(getattr(ds, "last_modified", "") or ""),
                "num_rows_estimate": _num_rows_estimate(ds),
            }

    candidates = list(seen_ids.values())

    # Rank by the chosen primary signal (desc). None sorts last.
    rank_field = {
        "trending_score": "trending_score",
        "downloads": "downloads",
        "likes": "likes",
    }[sort_key]

    def _key(c: dict[str, Any]) -> Any:
        v = c.get(rank_field)
        return v if isinstance(v, (int, float)) else -1

    candidates.sort(key=_key, reverse=True)
    candidates = candidates[:limit]

    log(f"[discover] {len(candidates)} candidates after merge/exclude "
        f"(excluded {excluded} gpqa-tainted); primary sort={rank_field}")

    if probe_samples:
        for c in candidates:
            sample = _probe_sample(c["id"])
            c["sample"] = sample
            if "error" in sample:
                log(f"[discover] sample probe failed for {c['id']}: {sample['error']}")
    return candidates


def main() -> int:
    p = argparse.ArgumentParser(
        description="V4 dynamic HF dataset discovery (trending/popular/keyword).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--keywords",
        default="graduate physics chemistry biology molecular genetics MCQ",
        help="space-separated keywords; one search per keyword, results merged",
    )
    p.add_argument("--sort", choices=list(SORT_MAP.keys()), default="trending",
                   help="primary popularity signal to rank by")
    p.add_argument("--task-filter", default=None,
                   help="optional task_categories filter, e.g. question-answering")
    p.add_argument("--limit", type=int, default=30, help="max candidates returned")
    p.add_argument("--out", default=None,
                   help="write digest JSON here (default: print to stdout)")
    p.add_argument("--no-sample", action="store_true",
                   help="skip the per-dataset 1-row sample probe (faster, less info)")
    args = p.parse_args()

    if args.limit <= 0:
        log("[fatal] --limit must be > 0")
        return 2

    keywords = [k for k in args.keywords.split() if k.strip()]
    sort_key = SORT_MAP[args.sort]

    try:
        from huggingface_hub import HfApi
    except Exception as e:  # noqa: BLE001
        log(f"[fatal] cannot import huggingface_hub: {e}")
        return 3

    api = HfApi()

    # Connectivity smoke test — if the Hub is unreachable, fail fast and loud.
    try:
        _ = list(api.list_datasets(limit=1))
    except Exception as e:  # noqa: BLE001
        log(f"[fatal] Hugging Face Hub unreachable: {type(e).__name__}: {e}")
        return 4

    candidates = discover(
        api=api,
        keywords=keywords,
        sort_key=sort_key,
        task_filter=args.task_filter,
        limit=args.limit,
        probe_samples=not args.no_sample,
    )

    if not candidates:
        log("[discover] no candidates found — consider broadening --keywords or "
            "synthesizing data with teacher_synth.py")

    digest = json.dumps(candidates, indent=2, ensure_ascii=False, default=str)
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(digest)
            log(f"[discover] wrote {len(candidates)} candidates -> {args.out}")
        except Exception as e:  # noqa: BLE001
            log(f"[fatal] could not write --out {args.out}: {e}")
            return 5
    else:
        print(digest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
