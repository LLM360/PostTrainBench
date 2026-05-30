#!/usr/bin/env python3
"""Batched LLM-as-judge using the teacher vLLM endpoint.

Filters candidate training rows for relevance + quality against a textual
criterion (default: graduate-level STEM MCQ suitable for GPQA-style training).
Each row is judged with ONE tiny teacher call returning strict JSON:
    {"keep": true/false, "reason": "<=15 words"}

The expensive part (the teacher round-trips) is parallelized with a thread
pool. The agent should call this ONCE on a candidate file before audit/train,
e.g. to drop off-topic or low-quality scraped rows.

CLI:
    python3 teacher_judge.py --in candidates.jsonl --out verdicts.jsonl \
        --batch 50 [--criterion "..."] [--max-workers 8] [--filter-out-rejected]

Inputs handled per JSONL row:
    - training-data obj with "messages": extract the user/question turn
    - raw obj with "question" (or "problem"/"prompt"/"text"): use that field

Outputs:
    - verdicts.jsonl: each original row augmented with _judge_keep / _judge_reason
    - <out>.kept.jsonl (only with --filter-out-rejected): kept rows, unaugmented-
      schema-preserving (the _judge_* keys are stripped so it pipes into audit/train)

Exit codes:
    0  success (verdicts written)
    2  usage / IO error (bad args, unreadable input, no rows, teacher unreachable)
Per-row failures never abort the run: the row is marked keep=false with
reason "judge_error" and processing continues.

Env:
    TEACHER_VLLM_URL    base_url for OpenAI-compatible endpoint (required)
    TEACHER_MODEL_NAME  model name to request (required)
    TEACHER_API_KEY     api key (default "EMPTY")
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import openai
except ImportError:  # pragma: no cover - exercised only when dep missing
    sys.stderr.write(
        "ERROR: the 'openai' python package is required but not importable.\n"
    )
    sys.exit(2)

# --- Tunables ---------------------------------------------------------------
QUESTION_TRUNC_CHARS = 800     # keep prompts tiny / cheap
JUDGE_MAX_TOKENS = 120         # short yes/no + reason only
JUDGE_TEMPERATURE = 0.0        # deterministic
REASON_MAX_WORDS = 15
# Fields searched (in order) when a row has no "messages" array.
RAW_TEXT_FIELDS = ("question", "problem", "prompt", "text", "input", "query")

DEFAULT_CRITERION = (
    "graduate-level STEM multiple-choice suitable for GPQA-style training"
)

SYSTEM_PROMPT = (
    "You are a strict data-quality judge. Given a CRITERION and a candidate "
    "training item, decide if the item is RELEVANT and HIGH-QUALITY per the "
    "criterion. Reply with ONLY one line of JSON and nothing else: "
    '{"keep": true|false, "reason": "<=15 words"}. '
    "Be conservative: if the item is off-topic, malformed, trivial, or "
    "low-quality, set keep=false."
)


# --- Env / client -----------------------------------------------------------

def build_client() -> tuple["openai.OpenAI", str]:
    """Build the teacher client from env. Exits(2) on missing config."""
    base_url = os.environ.get("TEACHER_VLLM_URL", "").strip()
    model = os.environ.get("TEACHER_MODEL_NAME", "").strip()
    api_key = os.environ.get("TEACHER_API_KEY", "").strip() or "EMPTY"
    if not base_url:
        sys.stderr.write("ERROR: TEACHER_VLLM_URL is not set in the environment.\n")
        sys.exit(2)
    if not model:
        sys.stderr.write("ERROR: TEACHER_MODEL_NAME is not set in the environment.\n")
        sys.exit(2)
    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    return client, model


def probe_endpoint(client: "openai.OpenAI", model: str) -> None:
    """Single cheap call to confirm the teacher is reachable. Exits(2) if down."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001 - surface any connection/auth error
        sys.stderr.write(
            f"ERROR: teacher endpoint unreachable ({type(exc).__name__}): {exc}\n"
            f"  base_url={os.environ.get('TEACHER_VLLM_URL')!r} model={model!r}\n"
        )
        sys.exit(2)


# --- Input handling ---------------------------------------------------------

def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        sys.stderr.write(f"ERROR: input file not found: {path}\n")
        sys.exit(2)
    rows: list[dict] = []
    try:
        with path.open() as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    sys.stderr.write(f"ERROR: line {i + 1} is not valid JSON: {exc}\n")
                    sys.exit(2)
                if not isinstance(obj, dict):
                    sys.stderr.write(f"ERROR: line {i + 1} is not a JSON object.\n")
                    sys.exit(2)
                rows.append(obj)
    except OSError as exc:
        sys.stderr.write(f"ERROR: could not read {path}: {exc}\n")
        sys.exit(2)
    if not rows:
        sys.stderr.write(f"ERROR: no rows found in {path}.\n")
        sys.exit(2)
    return rows


def extract_question_text(row: dict) -> str:
    """Pull the user/question text from either schema. Returns "" if absent."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        # Prefer the last user turn (the actual question).
        user_parts = [
            str(m.get("content", ""))
            for m in msgs
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_parts:
            return user_parts[-1].strip()
        # Fall back to concatenating any content if roles are unusual.
        any_parts = [
            str(m.get("content", "")) for m in msgs if isinstance(m, dict)
        ]
        if any_parts:
            return " ".join(p for p in any_parts if p).strip()
    # Raw schemas.
    for field in RAW_TEXT_FIELDS:
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
        # Some raw rows nest the question under a dict.
        if isinstance(val, dict):
            inner = val.get("text") or val.get("content")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
    return ""


# --- Judgment ---------------------------------------------------------------

def build_user_prompt(question: str, criterion: str) -> str:
    snippet = question[:QUESTION_TRUNC_CHARS]
    return (
        f"CRITERION: {criterion}\n\n"
        f"CANDIDATE ITEM (question text, truncated):\n{snippet}\n\n"
        'Reply with ONLY: {"keep": true|false, "reason": "<=15 words"}'
    )


def _clip_reason(reason: str) -> str:
    words = str(reason).split()
    if len(words) > REASON_MAX_WORDS:
        words = words[:REASON_MAX_WORDS]
    return " ".join(words)


def parse_verdict(text: str) -> tuple[bool, str]:
    """Robustly parse the teacher reply into (keep, reason).

    Tries strict JSON first, then a brace-substring JSON, then a keyword scan.
    """
    if text is None:
        return False, "judge_error"
    raw = text.strip()
    if not raw:
        return False, "judge_error"

    # 1) Strict JSON.
    obj = _try_json(raw)
    # 2) JSON substring (model wrapped it in prose / code fences).
    if obj is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = _try_json(raw[start : end + 1])

    if isinstance(obj, dict) and "keep" in obj:
        keep = _coerce_bool(obj.get("keep"))
        reason = _clip_reason(obj.get("reason") or ("kept" if keep else "rejected"))
        return keep, reason

    # 3) Keyword fallback (tolerate non-JSON / yes-no replies).
    low = raw.lower()
    # Look for explicit keep:/true|false or yes/no signals.
    if "keep" in low:
        if "true" in low and "false" not in low:
            return True, _clip_reason(raw)
        if "false" in low and "true" not in low:
            return False, _clip_reason(raw)
    # Plain yes/no.
    has_yes = _has_word(low, ("yes", "relevant", "keep", "high-quality", "good"))
    has_no = _has_word(low, ("no", "irrelevant", "reject", "discard", "drop", "low-quality"))
    if has_no and not has_yes:
        return False, _clip_reason(raw)
    if has_yes and not has_no:
        return True, _clip_reason(raw)
    # Ambiguous / unparseable -> conservative reject.
    return False, "unparseable_verdict"


def _try_json(s: str) -> Any:
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _coerce_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("true", "yes", "1", "y", "keep")
    return False


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def judge_one(
    client: "openai.OpenAI",
    model: str,
    question: str,
    criterion: str,
) -> tuple[bool, str]:
    if not question:
        return False, "no_question_text"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, criterion)},
            ],
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )
        content = resp.choices[0].message.content if resp.choices else ""
        return parse_verdict(content or "")
    except Exception:  # noqa: BLE001 - per-row failures must not abort the batch
        return False, "judge_error"


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batched LLM-as-judge (teacher vLLM) for relevance/quality filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input candidates JSONL.")
    p.add_argument("--out", dest="out_path", required=True, help="Output verdicts JSONL.")
    p.add_argument(
        "--batch",
        type=int,
        default=50,
        help="Rows submitted per logical batch (advisory; all rows are processed).",
    )
    p.add_argument(
        "--criterion",
        default=DEFAULT_CRITERION,
        help="Quality/relevance criterion the judge enforces.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread-pool size for parallel teacher calls.",
    )
    p.add_argument(
        "--filter-out-rejected",
        action="store_true",
        help="Also write <out>.kept.jsonl with only kept rows (schema-preserving).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch < 1:
        sys.stderr.write("ERROR: --batch must be >= 1.\n")
        return 2
    if args.max_workers < 1:
        sys.stderr.write("ERROR: --max-workers must be >= 1.\n")
        return 2

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    rows = load_rows(in_path)
    questions = [extract_question_text(r) for r in rows]

    client, model = build_client()
    probe_endpoint(client, model)

    verdicts: list[tuple[bool, str]] = [(False, "judge_error")] * len(rows)
    workers = min(args.max_workers, len(rows))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(judge_one, client, model, questions[i], args.criterion): i
            for i in range(len(rows))
        }
        done = 0
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                verdicts[idx] = fut.result()
            except Exception:  # noqa: BLE001 - defensive; judge_one already guards
                verdicts[idx] = (False, "judge_error")
            done += 1
            if done % max(1, args.batch) == 0:
                sys.stderr.write(f"  judged {done}/{len(rows)}...\n")
                sys.stderr.flush()

    # Write augmented verdicts.
    kept_count = 0
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row, (keep, reason) in zip(rows, verdicts):
                aug = dict(row)
                aug["_judge_keep"] = bool(keep)
                aug["_judge_reason"] = reason
                if keep:
                    kept_count += 1
                f.write(json.dumps(aug, ensure_ascii=False) + "\n")
    except OSError as exc:
        sys.stderr.write(f"ERROR: could not write {out_path}: {exc}\n")
        return 2

    # Optional kept-only file (strips _judge_* so it pipes into audit/train).
    if args.filter_out_rejected:
        kept_path = Path(str(out_path) + ".kept.jsonl")
        try:
            with kept_path.open("w") as f:
                for row, (keep, _reason) in zip(rows, verdicts):
                    if not keep:
                        continue
                    clean = {
                        k: v
                        for k, v in row.items()
                        if k not in ("_judge_keep", "_judge_reason")
                    }
                    f.write(json.dumps(clean, ensure_ascii=False) + "\n")
        except OSError as exc:
            sys.stderr.write(f"ERROR: could not write {kept_path}: {exc}\n")
            return 2
        sys.stderr.write(f"wrote kept-only file: {kept_path}\n")

    sys.stderr.write(f"kept {kept_count} / {len(rows)} rows (criterion: {args.criterion})\n")
    sys.stderr.write(f"verdicts written to: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
