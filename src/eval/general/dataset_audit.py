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

# --- Behavioral guard thresholds (optional, off-by-default for exit code) ----
# Catches the exp_004/exp_016 format-collapse class that decontam/diversity is
# blind to: the literal 'ANSWER: <LETTER>' marker is load-bearing learned
# answer-extraction behavior, and mass-rewriting assistant traces collapsed
# GPQA to ~0.13 even though those datasets PASSED the decontam/diversity gate.
BEHAVIOR_ANSWER_MARKER_RATE_MIN = 0.95   # absolute floor on rows ending ANSWER: <A-D>
BEHAVIOR_ANSWER_MARKER_DROP_MAX = 0.05   # max allowed drop vs parent marker rate
BEHAVIOR_LETTER_DIST_TOL = 0.15          # max abs per-letter fraction drift vs parent
BEHAVIOR_TRACE_CHANGED_FRAC_MAX = 0.5    # >this fraction of traces rewritten vs parent

PUNCT_RE = re.compile(r"[^\w\s]")
WS_RE = re.compile(r"\s+")
# Final 'ANSWER: <LETTER>' marker on its own line (case-insensitive, multiline).
ANSWER_MARKER_RE = re.compile(r"(?mi)^ANSWER:\s*([A-D])\s*$")


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


def file_sha256(path: Path) -> str:
    """Stream-hash a file. Matches train_sft.file_sha256 / publish_experiment.file_sha256."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


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
    """Return only the user-turn content. Used for diversity metrics.

    Diversity is computed on user turns only because including assistant-side
    rationales would artificially inflate distinct-ngram ratios and skew length
    statistics.
    """
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


def extract_all_content(row: dict, row_idx: int) -> str:
    """Return concatenated content of EVERY message (user + assistant + system).

    Decontamination runs on this combined string so a contaminated row cannot
    hide the benchmark text by placing it in a system or assistant turn.
    Schema validation (presence of 'messages', a user turn, an assistant turn)
    is delegated to extract_user_content() which we still call per row.
    """
    if "messages" not in row:
        raise SystemExit(f"row {row_idx}: missing 'messages' key")
    msgs = row["messages"]
    if not isinstance(msgs, list):
        raise SystemExit(f"row {row_idx}: 'messages' must be a list")
    parts = [str(m.get("content", "")) for m in msgs]
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


def _scan_text_for_contam(
    text: str,
    sha_set: set[str],
    shingle_index: dict[str, list[str]],
) -> dict | None:
    """Return a violation dict for `text` if it hits the test set, else None.

    The exact-SHA path normalizes the text and hashes; the shingle path
    splits into 8-grams and counts how many appear in the test shingle
    index. Both paths share the same normalize() so a short prompt that
    matches a test item exactly still flags via SHA even when shingles
    are too few to clear the overlap threshold.
    """
    norm = normalize(text)
    if not norm:
        return None
    row_sha = sha256_hex(norm)
    if row_sha in sha_set:
        return {"kind": "sha256_exact"}
    shingles = shingle_set(norm)
    hits: dict[str, int] = {}
    for sh in shingles:
        h = shingle_hash(sh)
        for tid in shingle_index.get(h, ()):
            hits[tid] = hits.get(tid, 0) + 1
    if hits:
        tid, overlap = max(hits.items(), key=lambda x: x[1])
        if overlap >= DECONTAM_SHINGLE_OVERLAP_MIN:
            return {
                "kind": "shingle_overlap",
                "test_id": tid,
                "overlap": overlap,
                "threshold": DECONTAM_SHINGLE_OVERLAP_MIN,
            }
    return None


def check_decontam(rows: list[dict], test_path: Path) -> dict:
    """Scan each message turn AND the combined concatenation against the test set.

    A contaminated row can hide the benchmark text in a system or assistant
    message, so we scan every message individually (defense against split
    hiding). We ALSO run the original combined-string scan as defense-in-
    depth so a contamination split across two adjacent turns whose shingle
    union covers the test item still trips the gate.

    Per-message scanning is required for the exact-SHA path: a row like
    user=<short test question>, assistant=<answer> never matches the test
    SHA when hashed as one combined string (the answer changes the hash),
    so without per-message hashing the exact-SHA check becomes useless for
    short prompts that don't have enough 8-grams to clear the shingle
    overlap threshold.
    """
    sha_set, shingle_index = load_test_decontam(test_path)
    violations: list[dict] = []

    def record(v: dict) -> bool:
        """Append violation; return True if MAX_REPORTED_VIOLATIONS reached."""
        violations.append(v)
        return len(violations) >= MAX_REPORTED_VIOLATIONS

    for i, row in enumerate(rows):
        flagged_this_row = False
        msgs = row.get("messages") if isinstance(row, dict) else None
        if isinstance(msgs, list):
            for msg_idx, m in enumerate(msgs):
                content = str(m.get("content", "") if isinstance(m, dict) else "")
                if not content:
                    continue
                role = m.get("role", "?") if isinstance(m, dict) else "?"
                hit = _scan_text_for_contam(content, sha_set, shingle_index)
                if hit is not None:
                    v = {
                        "row": i,
                        "scope": "per_message",
                        "role": role,
                        "message_index": msg_idx,
                        "first_80": content[:80],
                        **hit,
                    }
                    flagged_this_row = True
                    if record(v):
                        return {
                            "pass": False,
                            "violations": violations,
                            "test_items_loaded": len(sha_set),
                            "shingle_overlap_threshold": DECONTAM_SHINGLE_OVERLAP_MIN,
                            "scope": "per_message+combined",
                        }
        # Defense-in-depth: combined-all-messages scan. A contamination
        # that's split across two adjacent turns (so neither single
        # message clears the shingle threshold, but their union does) is
        # only caught here.
        if not flagged_this_row:
            combined = extract_all_content(row, i)
            hit = _scan_text_for_contam(combined, sha_set, shingle_index)
            if hit is not None:
                v = {
                    "row": i,
                    "scope": "combined_all_messages",
                    "first_80": combined[:80],
                    **hit,
                }
                if record(v):
                    break
    return {
        "pass": not violations,
        "violations": violations,
        "test_items_loaded": len(sha_set),
        "shingle_overlap_threshold": DECONTAM_SHINGLE_OVERLAP_MIN,
        "scope": "per_message+combined",
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


# --- Behavioral guard --------------------------------------------------------

def extract_assistant_content(row: dict) -> str:
    """Concatenate every assistant-turn content. Empty string if none/malformed.

    Used by the behavioral guard, which is best-effort and never raises (schema
    validation already happened via extract_user_content during the main pass).
    """
    if not isinstance(row, dict):
        return ""
    msgs = row.get("messages")
    if not isinstance(msgs, list):
        return ""
    parts = [
        str(m.get("content", ""))
        for m in msgs
        if isinstance(m, dict) and m.get("role") == "assistant"
    ]
    return "\n".join(parts)


def answer_marker_letter(assistant_text: str) -> str | None:
    """Return the LAST 'ANSWER: <LETTER>' marker letter (uppercased), else None.

    Matches a final answer line anywhere in the trace; takes the last match so a
    closing 'ANSWER: C' is what we score even if earlier lines mention letters.
    """
    matches = ANSWER_MARKER_RE.findall(assistant_text or "")
    return matches[-1].upper() if matches else None


def _prompt_key(row: dict) -> str:
    """Stable alignment key for a row: normalized concatenation of user turns."""
    try:
        return normalize(extract_user_content(row, -1))
    except SystemExit:
        # Best-effort: fall back to whatever user content we can scrape.
        if not isinstance(row, dict):
            return ""
        msgs = row.get("messages")
        if not isinstance(msgs, list):
            return ""
        parts = [
            str(m.get("content", ""))
            for m in msgs
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        return normalize("\n".join(parts))


_THINK_TAG_RE = re.compile(r"(?i)</?think>")


def _is_pure_think_wrapper(child_text: str, parent_text: str) -> bool:
    """True iff child_text is parent_text with ONLY a <think>...</think> wrapper
    added around the reasoning, preserving the same final ANSWER letter.

    The test UNWRAPS the child (removes the <think>/</think> TAGS but KEEPS the
    reasoning they enclose), drops the ANSWER line (re-emitted canonically after
    </think>), normalizes whitespace, and requires the result to equal the
    parent under the same treatment; AND the final ANSWER letter must match.
    Stripping <think>...</think> from the child must recover the parent's
    reasoning text — i.e. the only edit was the wrapper, not content damage.
    """
    c_letter = answer_marker_letter(child_text)
    p_letter = answer_marker_letter(parent_text)
    # The final answer letter must survive the migration. A None on either side
    # (no parseable letter) is NOT a pure wrapper — let the normal gates speak.
    if c_letter is None or p_letter is None or c_letter != p_letter:
        return False

    def reasoning_core(text: str) -> str:
        # Drop ANSWER: lines (re-emitted canonically after </think>) and the bare
        # <think>/</think> tags — KEEPING the reasoning text — then collapse
        # whitespace so the wrapper add/relocate is the only tolerated change.
        body = ANSWER_MARKER_RE.sub("", text or "")
        body = _THINK_TAG_RE.sub("", body)
        return WS_RE.sub(" ", body).strip().lower()

    return reasoning_core(child_text) == reasoning_core(parent_text)


def check_behavior(
    rows: list[dict],
    parent_rows: list[dict] | None,
    allow_think_migration: bool = False,
) -> dict:
    """Detect format-collapse: missing ANSWER markers, letter-dist drift, mass rewrites.

    Always best-effort and never raises. Returns a report section with a
    `high_risk` bool + human-readable `reasons`. The caller decides whether
    high_risk forces a non-zero exit (only under --strict-behavior).

    When `allow_think_migration` is set, the traces_changed_frac high_risk is
    NOT counted as a hard-fail trigger IF the rewrites are PURE <think> wrappers
    — i.e. for every changed-and-aligned row, stripping <think>...</think>
    recovers the parent's reasoning text AND the final ANSWER letter is
    unchanged. Genuine marker-rate / answer-letter damage still trips high_risk.
    """
    asst_texts = [extract_assistant_content(r) for r in rows]
    n = len(asst_texts)
    letters = [answer_marker_letter(t) for t in asst_texts]
    marked = sum(1 for x in letters if x is not None)
    marker_rate = (marked / n) if n else 0.0

    dist: dict[str, int] = {k: 0 for k in ("A", "B", "C", "D")}
    for x in letters:
        if x in dist:
            dist[x] += 1

    # --- THINK-HYGIENE HARD GATE (independent of --strict-behavior) ----------
    # Every THINKING-ONLY target MUST be exactly ONE closed <think>...</think>
    # block followed by exactly ONE 'ANSWER: <LETTER>' line, AFTER </think>, with
    # NO 'ANSWER:' anywhere inside <think>. We hard-fail (always, regardless of
    # the advisory --strict-behavior switch) any row that deviates from that
    # shape, because the eval scorer reads the FIRST '^ANSWER:' match while
    # answer_marker_letter() (this audit) reads the LAST — so a malformed row can
    # pass the marker check yet mis-score at eval. The required shape is REQUIRED:
    #   * MISSING <think>      — bare CoT / 'ANSWER: A'-only is NOT thinking-only.
    #   * MISSING/UNCLOSED </think> — an open <think> with no closer is malformed.
    #   * MULTIPLE <think>     — more than one opener (or closer) is malformed.
    #   * ANSWER not the single allowed line AFTER the single </think>:
    #       - any 'ANSWER:' before/inside <think>, or
    #       - more than one 'ANSWER:' marker in the row, or
    #       - zero 'ANSWER:' markers after </think>.
    # The existing in-think-ANSWER detection is preserved as one of these cases.
    missing_think_rows: list[int] = []          # zero '<think>' openers
    unclosed_think_rows: list[int] = []         # '<think>' present, no '</think>'
    multi_think_rows: list[int] = []            # >1 '<think>' or >1 '</think>'
    answer_inside_think_rows: list[int] = []    # 'ANSWER:' before/inside <think>
    multi_answer_marker_rows: list[int] = []    # >1 'ANSWER:' marker in the row
    missing_answer_after_think_rows: list[int] = []  # no 'ANSWER:' after </think>
    for idx, atext in enumerate(asst_texts):
        n_open = atext.count("<think>")
        n_close = atext.count("</think>")
        # Shape gate 1: there MUST be exactly one closed <think> block.
        if n_open == 0:
            missing_think_rows.append(idx)
            continue
        if n_close == 0:
            unclosed_think_rows.append(idx)
            continue
        if n_open > 1 or n_close > 1:
            multi_think_rows.append(idx)
            # still inspect ANSWER placement below using the first </think>.
        before_close = atext.split("</think>", 1)[0]
        after_close = atext.split("</think>", 1)[1]
        all_answers = ANSWER_MARKER_RE.findall(atext)
        # Shape gate 2: no 'ANSWER:' before/inside the (first) <think> block.
        if ANSWER_MARKER_RE.search(before_close):
            answer_inside_think_rows.append(idx)
        # Shape gate 3: exactly one 'ANSWER:' marker total.
        if len(all_answers) > 1:
            multi_answer_marker_rows.append(idx)
        # Shape gate 4: there IS an 'ANSWER:' line after the single </think>.
        elif not ANSWER_MARKER_RE.search(after_close):
            # (only flag the "missing after" case when there's exactly one or
            # zero markers; >1 is already covered by multi_answer_marker_rows.)
            missing_answer_after_think_rows.append(idx)
    think_hygiene_violation = bool(
        missing_think_rows
        or unclosed_think_rows
        or multi_think_rows
        or answer_inside_think_rows
        or multi_answer_marker_rows
        or missing_answer_after_think_rows
    )
    think_hygiene_reason = ""
    if think_hygiene_violation:
        think_hygiene_reason = (
            "thinking-only shape violation: "
            f"{len(missing_think_rows)} row(s) have NO <think> block, "
            f"{len(unclosed_think_rows)} row(s) have an unclosed/missing </think>, "
            f"{len(multi_think_rows)} row(s) have MULTIPLE <think> blocks, "
            f"{len(answer_inside_think_rows)} row(s) have an 'ANSWER:' marker "
            f"before/inside <think>, "
            f"{len(multi_answer_marker_rows)} row(s) have >1 'ANSWER:' marker, "
            f"{len(missing_answer_after_think_rows)} row(s) have NO 'ANSWER:' "
            "line after </think>. Every target must be exactly ONE closed "
            "<think>...</think> block then exactly ONE 'ANSWER: <LETTER>' line "
            "(after </think>, none inside). The eval scorer takes the FIRST "
            "^ANSWER: match while this audit's marker check takes the LAST, so a "
            "malformed row can pass the marker check yet mis-score at eval."
        )

    reasons: list[str] = []
    high_risk = False

    # (a) Absolute ANSWER-marker-rate floor.
    if marker_rate < BEHAVIOR_ANSWER_MARKER_RATE_MIN:
        high_risk = True
        reasons.append(
            f"answer_marker_rate={marker_rate:.3f} < {BEHAVIOR_ANSWER_MARKER_RATE_MIN} "
            f"(only {marked}/{n} rows end with 'ANSWER: <A-D>')"
        )

    parent = {
        "given": parent_rows is not None,
        "row_count": len(parent_rows) if parent_rows is not None else None,
    }

    if parent_rows is not None:
        p_asst = [extract_assistant_content(r) for r in parent_rows]
        pn = len(p_asst)
        p_letters = [answer_marker_letter(t) for t in p_asst]
        p_marked = sum(1 for x in p_letters if x is not None)
        p_marker_rate = (p_marked / pn) if pn else 0.0
        parent["answer_marker_rate"] = p_marker_rate

        # (a-cont) Material drop in marker rate vs parent.
        drop = p_marker_rate - marker_rate
        if drop > BEHAVIOR_ANSWER_MARKER_DROP_MAX:
            high_risk = True
            reasons.append(
                f"answer_marker_rate dropped {drop:.3f} vs parent "
                f"({p_marker_rate:.3f} -> {marker_rate:.3f}), "
                f"> {BEHAVIOR_ANSWER_MARKER_DROP_MAX}"
            )

        # (b) Answer-letter distribution drift (per-letter fraction) vs parent.
        p_dist: dict[str, int] = {k: 0 for k in ("A", "B", "C", "D")}
        for x in p_letters:
            if x in p_dist:
                p_dist[x] += 1
        parent["letter_dist"] = dict(p_dist)
        c_total = sum(dist.values())
        p_total = sum(p_dist.values())
        letter_drift: dict[str, float] = {}
        max_drift = 0.0
        max_drift_letter = None
        for k in ("A", "B", "C", "D"):
            cf = (dist[k] / c_total) if c_total else 0.0
            pf = (p_dist[k] / p_total) if p_total else 0.0
            d = abs(cf - pf)
            letter_drift[k] = d
            if d > max_drift:
                max_drift = d
                max_drift_letter = k
        if max_drift > BEHAVIOR_LETTER_DIST_TOL:
            high_risk = True
            reasons.append(
                f"answer_letter_dist drift={max_drift:.3f} (letter {max_drift_letter}) "
                f"> {BEHAVIOR_LETTER_DIST_TOL}"
            )

        # (c) Fraction of assistant traces that DIFFER from parent's aligned row.
        # Align by normalized user-prompt key; first occurrence wins on dup keys.
        p_index: dict[str, str] = {}
        for r in parent_rows:
            key = _prompt_key(r)
            if key and key not in p_index:
                p_index[key] = extract_assistant_content(r)
        aligned = 0
        changed = 0
        changed_pure_wrapper = 0
        changed_not_wrapper = 0
        for r, atext in zip(rows, asst_texts):
            key = _prompt_key(r)
            if key in p_index:
                aligned += 1
                if atext != p_index[key]:
                    changed += 1
                    if _is_pure_think_wrapper(atext, p_index[key]):
                        changed_pure_wrapper += 1
                    else:
                        changed_not_wrapper += 1
        changed_frac = (changed / aligned) if aligned else 0.0
        parent["aligned_rows"] = aligned
        parent["traces_changed"] = changed
        parent["traces_changed_frac"] = changed_frac
        parent["traces_changed_pure_wrapper"] = changed_pure_wrapper
        parent["traces_changed_not_wrapper"] = changed_not_wrapper
        if changed_frac > BEHAVIOR_TRACE_CHANGED_FRAC_MAX:
            # Under --allow-think-migration a mass rewrite is NOT a hard-fail IF
            # EVERY changed row is a pure <think> wrapper (parent reasoning +
            # same ANSWER letter recovered by stripping <think>). A single
            # genuinely-rewritten row (changed_not_wrapper > 0) still trips it,
            # so real marker/answer damage is never masked. The marker-rate and
            # letter-distribution sub-checks (a)/(b) above are independent and
            # still fire on genuine damage regardless of migration mode.
            is_pure_migration = (
                allow_think_migration and changed_not_wrapper == 0 and changed > 0
            )
            if is_pure_migration:
                parent["think_migration_allowed"] = True
                reasons.append(
                    f"traces_changed_frac={changed_frac:.3f} ({changed}/{aligned} "
                    f"aligned rows rewritten) > {BEHAVIOR_TRACE_CHANGED_FRAC_MAX} "
                    "but ALL changed rows are pure <think> wrappers (parent "
                    "reasoning + same ANSWER letter preserved); allowed under "
                    "--allow-think-migration (not counted as high_risk)"
                )
            else:
                high_risk = True
                detail = ""
                if allow_think_migration:
                    parent["think_migration_allowed"] = False
                    detail = (
                        f"; --allow-think-migration set but "
                        f"{changed_not_wrapper} changed row(s) are NOT pure "
                        "<think> wrappers (reasoning text or ANSWER letter "
                        "changed), so the migration exemption does NOT apply"
                    )
                reasons.append(
                    f"traces_changed_frac={changed_frac:.3f} ({changed}/{aligned} "
                    f"aligned rows rewritten) > {BEHAVIOR_TRACE_CHANGED_FRAC_MAX}"
                    f"{detail}"
                )

    return {
        "high_risk": high_risk,
        "reasons": reasons,
        "answer_marker_rate": marker_rate,
        "answer_marked_rows": marked,
        "row_count": n,
        "answer_letter_dist": dist,
        # THINK-HYGIENE HARD GATE (always enforced, independent of high_risk /
        # --strict-behavior). think_hygiene_violation True => audit MUST exit
        # non-zero. The *_rows lists are the offending row indices (capped for
        # the report); the counts reflect ALL offenders. Every target MUST be
        # exactly ONE closed <think>...</think> block then exactly ONE
        # 'ANSWER: <LETTER>' line after </think>.
        "missing_think_rows": missing_think_rows[:MAX_REPORTED_VIOLATIONS],
        "missing_think_count": len(missing_think_rows),
        "unclosed_think_rows": unclosed_think_rows[:MAX_REPORTED_VIOLATIONS],
        "unclosed_think_count": len(unclosed_think_rows),
        "multi_think_rows": multi_think_rows[:MAX_REPORTED_VIOLATIONS],
        "multi_think_count": len(multi_think_rows),
        "answer_inside_think_rows": answer_inside_think_rows[:MAX_REPORTED_VIOLATIONS],
        "answer_inside_think_count": len(answer_inside_think_rows),
        "multi_answer_marker_rows": multi_answer_marker_rows[:MAX_REPORTED_VIOLATIONS],
        "multi_answer_marker_count": len(multi_answer_marker_rows),
        "missing_answer_after_think_rows": missing_answer_after_think_rows[:MAX_REPORTED_VIOLATIONS],
        "missing_answer_after_think_count": len(missing_answer_after_think_rows),
        "think_hygiene_violation": think_hygiene_violation,
        "think_hygiene_reason": think_hygiene_reason,
        "parent": parent,
        "thresholds": {
            "answer_marker_rate_min": BEHAVIOR_ANSWER_MARKER_RATE_MIN,
            "answer_marker_drop_max": BEHAVIOR_ANSWER_MARKER_DROP_MAX,
            "letter_dist_tol": BEHAVIOR_LETTER_DIST_TOL,
            "trace_changed_frac_max": BEHAVIOR_TRACE_CHANGED_FRAC_MAX,
        },
    }


# --- Main --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decontam + diversity hard-gate audit.")
    p.add_argument("--data-path", required=True, help="Training data JSONL (messages schema).")
    p.add_argument(
        "--test-decontam",
        default="test_decontam.jsonl",
        help=(
            "Path to test_decontam.jsonl. Defaults to the file in the current "
            "working directory because run_task.sh flattens task_context/* into "
            "the agent's task root."
        ),
    )
    p.add_argument(
        "--report-path",
        default=None,
        help="Where to write the JSON report. Defaults to <data_path>.audit_report.json beside the data.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--parent-path",
        default=None,
        help=(
            "Optional parent dataset JSONL this one derives from. Enables the "
            "behavioral guard's drift checks (answer-marker-rate drop, "
            "answer-letter-distribution drift, fraction of assistant traces "
            "rewritten vs the aligned parent row)."
        ),
    )
    p.add_argument(
        "--strict-behavior",
        action="store_true",
        help=(
            "Make the behavioral guard a HARD gate: behavior.high_risk forces a "
            "non-zero exit. Off by default, in which case high_risk only WARNs "
            "to stderr and never changes the decontam/diversity pass/exit."
        ),
    )
    p.add_argument(
        "--allow-think-migration",
        action="store_true",
        help=(
            "One-time bare-CoT -> thinking-only migration mode. The "
            "traces_changed_frac mass-rewrite high_risk is NOT a hard fail when "
            "EVERY changed-and-aligned row is a PURE <think> wrapper of its "
            "parent (stripping <think>...</think> recovers the parent reasoning "
            "AND the same final ANSWER letter is preserved). Genuine "
            "marker-rate / answer-letter damage, or any non-wrapper rewrite, "
            "still hard-fails. Use ONLY for the wrapper-migration experiment."
        ),
    )
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

    # Bind the report to the exact dataset bytes — publish_experiment.py and
    # train_sft.py refuse to consume an audit report whose data_sha256 does
    # not match the current data.jsonl, preventing reuse of a stale passing
    # report after data.jsonl has changed.
    data_sha = file_sha256(data_path)

    rows = load_rows(data_path)
    if len(rows) < MIN_ROWS:
        report = {
            "pass": False,
            "data_path": str(data_path),
            "data_sha256": data_sha,
            "row_count": len(rows),
            "reason": f"too few rows ({len(rows)} < {MIN_ROWS})",
        }
        _write_report(report, args.report_path, data_path)
        print(f"AUDIT FAIL: too few rows ({len(rows)} < {MIN_ROWS})", file=sys.stderr)
        return 1

    user_texts = [extract_user_content(r, i) for i, r in enumerate(rows)]
    decontam = check_decontam(rows, test_path)
    diversity = check_diversity(user_texts, args.seed)

    # Optional behavioral guard. Best-effort, never raises. Loads the parent
    # dataset (if given) to compute drift; a missing/unreadable parent file is a
    # hard input error (exit 2), consistent with --data-path / --test-decontam.
    parent_rows: list[dict] | None = None
    if args.parent_path:
        parent_path = Path(args.parent_path).resolve()
        if not parent_path.is_file():
            print(f"ERROR: parent file not found: {parent_path}", file=sys.stderr)
            return 2
        parent_rows = load_rows(parent_path)
    behavior = check_behavior(
        rows, parent_rows, allow_think_migration=args.allow_think_migration
    )

    # IMPORTANT: behavior NEVER affects the decontam/diversity pass/exit unless
    # --strict-behavior is set. Existing gating is unchanged.
    passed = decontam["pass"] and diversity["pass"]
    report = {
        "pass": passed,
        "data_path": str(data_path),
        "data_sha256": data_sha,
        "row_count": len(rows),
        "decontam": decontam,
        "diversity": diversity,
        "behavior": behavior,
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

    # Always surface the behavioral guard verdict. High risk WARNs loudly to
    # stderr regardless of --strict-behavior; under --strict-behavior it also
    # forces a non-zero exit even if decontam/diversity passed.
    if behavior["high_risk"]:
        gate = "STRICT (hard gate)" if args.strict_behavior else "WARN ONLY (advisory)"
        print(
            f"BEHAVIOR HIGH-RISK [{gate}]: possible format-collapse "
            f"(exp_004/exp_016 class):",
            file=sys.stderr,
        )
        for reason in behavior["reasons"]:
            print(f"    {reason}", file=sys.stderr)
        if not args.strict_behavior:
            print(
                "    (advisory only; not failing the audit. Pass "
                "--strict-behavior to make this a hard gate.)",
                file=sys.stderr,
            )

    # THINK-HYGIENE HARD GATE: ALWAYS enforced, independent of --strict-behavior
    # AND of the decontam/diversity pass. Every target MUST be exactly ONE closed
    # <think>...</think> block then exactly ONE 'ANSWER: <LETTER>' line after
    # </think>. Bare CoT (no <think>), an unclosed/missing </think>, multiple
    # <think> blocks, an 'ANSWER:' before/inside <think>, >1 'ANSWER:' marker, or
    # no 'ANSWER:' after </think> all mis-score at eval (FIRST-match scorer) or
    # are simply not the thinking-only shape, so we refuse to allow training.
    # This intentionally short-circuits before the normal pass/exit logic below
    # so the advisory checks above can never mask it.
    if behavior["think_hygiene_violation"]:
        print(
            "AUDIT FAIL [THINK-HYGIENE HARD GATE]: thinking-only shape violation "
            "(always enforced, independent of --strict-behavior):",
            file=sys.stderr,
        )
        if behavior["missing_think_count"]:
            print(
                f"  {behavior['missing_think_count']} row(s) have NO <think> "
                f"block (bare CoT / answer-only is not thinking-only) (indices: "
                f"{behavior['missing_think_rows']})",
                file=sys.stderr,
            )
        if behavior["unclosed_think_count"]:
            print(
                f"  {behavior['unclosed_think_count']} row(s) have an unclosed/"
                f"missing </think> (indices: {behavior['unclosed_think_rows']})",
                file=sys.stderr,
            )
        if behavior["multi_think_count"]:
            print(
                f"  {behavior['multi_think_count']} row(s) have MULTIPLE <think> "
                f"blocks (indices: {behavior['multi_think_rows']})",
                file=sys.stderr,
            )
        if behavior["answer_inside_think_count"]:
            print(
                f"  {behavior['answer_inside_think_count']} row(s) have an "
                f"'ANSWER:' marker before/inside <think> (indices: "
                f"{behavior['answer_inside_think_rows']})",
                file=sys.stderr,
            )
        if behavior["multi_answer_marker_count"]:
            print(
                f"  {behavior['multi_answer_marker_count']} row(s) have >1 "
                f"'ANSWER:' marker (indices: "
                f"{behavior['multi_answer_marker_rows']})",
                file=sys.stderr,
            )
        if behavior["missing_answer_after_think_count"]:
            print(
                f"  {behavior['missing_answer_after_think_count']} row(s) have "
                f"NO 'ANSWER:' line after </think> (indices: "
                f"{behavior['missing_answer_after_think_rows']})",
                file=sys.stderr,
            )
        print(
            "  why: every THINKING-ONLY target must be exactly ONE closed "
            "<think>...</think> block then exactly ONE 'ANSWER: <LETTER>' line "
            "AFTER </think> and NONE inside <think>. The eval scorer takes the "
            "FIRST '^ANSWER:' match while this audit's marker check takes the "
            "LAST, so a malformed row can pass the marker check yet mis-score at "
            "eval. Scrub upstream (teacher_synth emits this shape by default).",
            file=sys.stderr,
        )
        return 1

    if passed:
        print(
            f"AUDIT PASS: {len(rows)} rows, distinct_1g={diversity['distinct_1g']:.3f}, "
            f"distinct_4g={diversity['distinct_4g']:.3f}, "
            f"mean_cos_dist={diversity['mean_cos_dist']:.3f}, "
            f"len_cv={diversity['len_cv']:.3f}"
        )
        # Strict behavior gate can flip an otherwise-passing audit to fail.
        if args.strict_behavior and behavior["high_risk"]:
            print(
                "AUDIT FAIL: --strict-behavior tripped by behavior.high_risk",
                file=sys.stderr,
            )
            return 1
        return 0

    print("AUDIT FAIL:", file=sys.stderr)
    if not decontam["pass"]:
        print(f"  decontam: {len(decontam['violations'])} violation(s)", file=sys.stderr)
        for v in decontam["violations"][:5]:
            scope = v.get("scope", "?")
            if scope == "per_message":
                where = (
                    f"row {v['row']} msg[{v.get('message_index')}] "
                    f"role={v.get('role', '?')!r}"
                )
            else:
                where = f"row {v['row']} ({scope})"
            print(
                f"    {where} ({v['kind']}): {v.get('first_80', '')!r}",
                file=sys.stderr,
            )
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
