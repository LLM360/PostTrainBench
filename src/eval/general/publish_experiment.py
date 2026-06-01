#!/usr/bin/env python3
"""Publish one experiment row to the shared CSV and to the local index.

Schema is closed: only the fields in SHARED_FIELDS are written, and any
field name starting with eval_ or score_ is rejected (defence in depth —
agents shouldn't be sharing eval scores during the loop).

The shared CSV path comes from $SHARED_LOG_CSV. flock is held on a sidecar
.lock file across the append.

Round 2 additions (hypothesis-driven experiment notes):
- notes.md is parsed for required sections: Parent, Hypothesis, Method,
  Conclusion (and optionally PivotReason).
- SHARED_FIELDS extended with parent_exp_id, hypothesis_short,
  conclusion_short, audit_pass.
- hypothesis_short / conclusion_short are numeric-score scrubbed before
  truncation to 200 chars.
- --audit-failed flag publishes a failure row without requiring
  data.jsonl / dataset_audit_report.json.
- Parent validation against local index.csv + shared CSV; cross-run
  pivots allowed when ## PivotReason is provided.
- Promoted data is persisted to <shared_dir>/promoted/<sha>.jsonl so
  future agents in future runs can fork the actual data.
- CSV schema is versioned via a `# schema_version=N` comment line. An
  on-disk file from an older schema is rotated aside under the flock on
  first write at the new version. The .lock sidecar path is stable
  across schema rolls so concurrent writers do not lose mutex.

Schema history:
- v1: original fields.
- v2: added parent_exp_id, hypothesis_short, conclusion_short, audit_pass,
       notes_excerpt.
- v3: added `promoted` column (cross-run handoff signal); strategy_short
       and notes_excerpt are now numeric-scrubbed identically to the other
       free-text fields.
- v4 (V2 discipline): self-eval is now mandatory and the publisher
       cross-checks the agent's stated scores against eval_result.json.
       New columns: eval_before, eval_after, outcome_improved,
       findings_short. ## Findings and ## Outcome become required sections
       (## Conclusion stays for backward compat as a legacy alias for
       ## Findings if Findings is missing). KNOWLEDGE.md presence + format
       checks. Backtrack-rule parent validation. --done flag with
       >=10-experiment floor and 10-experiment plateau. effective_promoted
       additionally requires outcome_improved == "yes".
- v5 (V4 multi-seed promotion margin): multiple eval seeds are accepted
       (--eval-results list or auto-glob of eval_result*.json in the exp dir).
       New columns: eval_after_mean, eval_after_std, n_eval_seeds. A candidate
       is outcome_improved=yes / effective_promoted ONLY when eval_after_mean
       beats the incumbent (eval_before) by more than max(PROMOTION_MARGIN_FLOOR,
       PROMOTION_STD_K * eval_after_std) — both env-overridable. Single
       eval_after still works (mean=value, std=0.0, n=1).
- v5.1 (multi-seed evidence hardening, PR #8): EVERY eval_result*.json that
       contributes to eval_after_mean is now validated with the SAME guards
       V4 applied only to the canonical eval_result.json — full-dataset sample
       count, generation max_tokens >= MIN_EVAL_MAX_TOKENS, and a parseable
       accuracy field. A smoke / low-token / malformed / old-shape seed file
       fails loud and can NEVER move the mean. Raw --eval-after floats are no
       longer file-backable evidence and are DROPPED from the promotion path:
       they are display-only and do not feed eval_after_mean / outcome_improved
       / promoted. The contributing file paths are logged (eval_sources
       provenance line) so a reviewer can audit what evidence the verdict
       rests on. No column / schema change (still schema_version=5).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

SCHEMA_VERSION = "5"
SCHEMA_HEADER_LINE = f"# schema_version={SCHEMA_VERSION}\n"

# V3 full-eval sample-count gate. Maps benchmark task name to the
# expected full-dataset sample count. Used to refuse partial-sample
# evals (e.g. --limit 5/50 smoke evals) at publish time.
TASK_FULL_SAMPLE_COUNT = {
    "gpqamain": 448,
    "gpqa_main": 448,
    "gpqa": 448,
}
DEFAULT_MIN_FULL_EVAL_SAMPLES = 200  # used when task not in table

# V4: SOURCE-NOVELTY GATE cadence. Every Nth experiment (exp_id index a
# multiple of this constant: exp_005, exp_010, ...) must introduce at
# least one data source this run has never used before. Tunable here.
SOURCE_NOVELTY_EVERY = 5

# V4: EVAL-TOKEN GUARD floor. The benchmark harness runs
# multiple_choice(cot=True) with --max-tokens 16000 and REWARDS reasoning.
# If we can prove the agent's self-eval was run with a generation budget
# smaller than this, the eval forbade reasoning and is invalid — refuse.
# Closes the V3 bug where the agent self-eval'd with --max-tokens 128.
MIN_EVAL_MAX_TOKENS = 4096

# V4 (multi-seed promotion margin). Single-seed scores were promotion noise
# in the V4 pilot: exp_007 (0.2879) beat exp_002 (0.2812) by ~1pt — within
# seed noise — yet 11 of 18 experiments forked from exp_007. A candidate now
# may only be marked outcome_improved=yes / effective_promoted if its
# multi-seed mean beats the incumbent by MORE than the noise band:
#
#   eval_after_mean > incumbent + max(PROMOTION_MARGIN_FLOOR,
#                                     PROMOTION_STD_K * eval_after_std)
#
# PROMOTION_MARGIN_FLOOR is the minimum absolute margin (guards against a
# zero-std single seed sneaking through on a 0.001 lead). PROMOTION_STD_K
# scales the across-seed std into the required margin. Both are overridable
# via the like-named environment variables.
def _env_float(name: str, default: float) -> float:
    """Read a float from os.environ[name], falling back to `default`.

    Non-parseable / empty values fall back to the default with a stderr
    warning so a typo'd override never silently disables the margin.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        print(
            f"[publish V4] WARNING: env {name}={raw!r} is not a float; "
            f"using default {default}.",
            file=sys.stderr,
        )
        return default


PROMOTION_MARGIN_FLOOR = _env_float("PROMOTION_MARGIN_FLOOR", 0.015)
PROMOTION_STD_K = _env_float("PROMOTION_STD_K", 1.0)

SHARED_FIELDS = [
    "agent_id",
    "cluster_id",
    "exp_id",
    "timestamp_utc",
    "strategy_short",
    "data_sources",
    "row_count",
    "diversity_distinct_1g",
    "diversity_distinct_4g",
    "diversity_mean_cos_dist",
    "diversity_len_cv",
    "decontam_pass",
    "dataset_sha256",
    "parent_exp_id",
    "hypothesis_short",
    "conclusion_short",
    "audit_pass",
    "promoted",
    # V2 (schema v4) additions. Inserted before notes_excerpt so existing
    # readers can keep dotted access by name; column order is part of the
    # schema bump.
    "eval_before",
    "eval_after",
    # V4 multi-seed promotion (schema v5). eval_after_mean/std summarize the
    # candidate's eval_result*.json across seeds; n_eval_seeds is how many
    # were averaged. Backward compat: a single eval_after still works ->
    # mean=value, std=0.0, n=1. Adding columns bumps SCHEMA_VERSION to 5 so
    # the existing schema guard rotates an older (v4) file aside on first
    # write rather than appending mis-aligned rows.
    "eval_after_mean",
    "eval_after_std",
    "n_eval_seeds",
    "outcome_improved",
    "findings_short",
    "notes_excerpt",
]

LOCAL_FIELDS = [
    "exp_id",
    "started_at_utc",
    "strategy_short",
    "row_count",
    "audit_pass",
    "promoted",
    "dataset_sha256",
    "parent_exp_id",
    "hypothesis_short",
    # V2: local index also tracks per-experiment outcome so the
    # backtrack rule and --done plateau can be evaluated cheaply.
    "eval_before",
    "eval_after",
    "outcome_improved",
]

FORBIDDEN_PREFIXES = ("eval_", "score_")
# Schema v4 (V2 discipline) intentionally publishes a small set of
# structured eval-related columns. These are exempt from the prefix-based
# guard while keeping the catch-all in place for any other eval_/score_
# field an author might be tempted to add later.
FORBIDDEN_EXEMPT = frozenset({
    "eval_before",
    "eval_after",
    # V4 multi-seed summary columns (n_eval_seeds has no forbidden prefix).
    "eval_after_mean",
    "eval_after_std",
})

REQUIRED_SECTIONS = ("Parent", "Hypothesis", "Method", "Findings", "Outcome")
# Conclusion is the v1/v2/v3 name; in v4 the equivalent semantic slot is
# ## Findings. If only Conclusion is present we accept it as a Findings
# alias for backward compatibility (older notes.md examples still in the
# wild) but new content should use ## Findings.
OPTIONAL_SECTIONS = ("PivotReason", "Conclusion")
ALL_SECTIONS = REQUIRED_SECTIONS + OPTIONAL_SECTIONS

_SECTION_RE = re.compile(
    r"^##\s+(Parent|Hypothesis|Method|Conclusion|Findings|Outcome|PivotReason)\s*$"
)

# Outcome is structured. Each of these keys appears on its own line as
# `key: value` (case-insensitive on the key, value taken verbatim).
_OUTCOME_KEY_RE = re.compile(
    r"^\s*(eval_before|eval_after|improved|confidence)\s*:\s*(.+?)\s*$",
    re.IGNORECASE,
)
_OUTCOME_IMPROVED_VALUES = {"yes", "no", "marginal"}
_OUTCOME_CONFIDENCE_VALUES = {"high", "med", "low"}

# KNOWLEDGE.md per-line format check.
_KNOWLEDGE_TAG_WHITELIST = {
    "#data",
    "#filter",
    "#format",
    "#training",
    "#audit",
    "#dead-end",
    "#meta",
}
_KNOWLEDGE_LINE_RE = re.compile(
    r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)\]\s+"
    r"\[(?P<exp>exp_\d+)\]\s+"
    r"\[(?P<tag>#[A-Za-z0-9_-]+)\]\s+"
    r"(?P<msg>.+)$"
)

# Override marker the agent must include in ## Method when forking from
# something other than the most-recent-improved parent.
_OVERRIDE_MARKER_RE = re.compile(r"\[override:[^\]]+\]", re.IGNORECASE)

# Numeric-score scrub patterns. Applied in order, before truncation.
_BARE_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_PERCENT_RE = re.compile(r"[%％]")
_SCORE_WORD_RE = re.compile(
    r"\b(accuracy|acc|score|stderr|metric|test_acc)\b",
    re.IGNORECASE,
)
_COMPARATIVE_PHRASES = [
    "much better",
    "much worse",
    "slight improvement",
    "better than",
    "worse than",
    "the best",
    "outperforms",
    "winning",
    "regression",
]
_COMPARATIVE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in _COMPARATIVE_PHRASES) + r")\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish an experiment row.")
    p.add_argument("--exp-dir", required=True, help="experiments/exp_<N>/")
    p.add_argument(
        "--data-sources",
        default="",
        help="Short comma-separated list of HF dataset ids and/or 'synthetic'.",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--promoted",
        action="store_true",
        help="Set if this experiment was promoted to final_model/.",
    )
    group.add_argument(
        "--audit-failed",
        action="store_true",
        help=(
            "Publish a failure row. Skips data.jsonl / "
            "dataset_audit_report.json existence checks; audit_pass=False."
        ),
    )
    p.add_argument(
        "--done",
        action="store_true",
        help=(
            "Declare this run finished. Requires >=10 published experiments "
            "AND the last 10 outcomes all != 'yes' (plateau). Refused otherwise."
        ),
    )
    # V4 multi-seed promotion. The seed set that feeds eval_after_mean /
    # eval_after_std / n_eval_seeds and the promotion-margin check is built
    # ONLY from validated, file-backed evals: --eval-results plus an auto-glob
    # of eval_result*.json in --exp-dir. With none supplied (and a single
    # eval_result.json present, the legacy path), n=1 / std=0.0 / mean=the
    # single validated score — fully backward compatible.
    p.add_argument(
        "--eval-after",
        action="append",
        type=float,
        default=None,
        metavar="ACC",
        help=(
            "DISPLAY-ONLY per-seed eval accuracy (float). NOTE: as of PR #8 "
            "raw --eval-after floats DO NOT feed eval_after_mean / promotion "
            "(unverified, not file-backed). To contribute a validated seed, "
            "pass its eval_result*.json via --eval-results instead."
        ),
    )
    p.add_argument(
        "--eval-results",
        default=None,
        metavar="PATHS",
        help=(
            "Comma- or whitespace-separated list of eval_result*.json files "
            "(one per seed). EACH is fully validated (full-dataset sample "
            "count, max-tokens, parseable accuracy) before its accuracy "
            "contributes to eval_after_mean. Merged with any auto-globbed "
            "eval_result*.json in --exp-dir. A non-comparable file fails loud."
        ),
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def parse_notes_sections(text: str) -> dict[str, str]:
    """Return {section_name: body_stripped} for any of the known sections.

    Headers match `^## (Parent|Hypothesis|Method|Conclusion|PivotReason)$`.
    A section's body runs until the next `## ` header or EOF.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for raw_line in text.splitlines():
        m = _SECTION_RE.match(raw_line.rstrip())
        if m:
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = m.group(1)
            buf = []
        elif current is not None:
            buf.append(raw_line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def scrub_numeric(text: str) -> str:
    """Strip numeric scores, percentages, score-words, and comparative phrases."""
    if not text:
        return text
    out = _PERCENT_RE.sub("", text)
    out = _COMPARATIVE_RE.sub("[redacted]", out)
    out = _SCORE_WORD_RE.sub("[score]", out)
    out = _BARE_NUMBER_RE.sub("[n]", out)
    return out


def short_field(text: str, limit: int = 200) -> str:
    """Collapse whitespace, scrub numeric content, then truncate."""
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text).strip()
    scrubbed = scrub_numeric(collapsed)
    # Re-collapse whitespace produced by removed % signs etc.
    scrubbed = re.sub(r"\s+", " ", scrubbed).strip()
    return scrubbed[:limit]


def extract_strategy(notes: str, sections: dict[str, str]) -> str:
    """First non-empty line of ## Hypothesis (preferred) or notes body.

    Falls back to the first content line of the notes body, excluding the
    `# exp_<N>` top-level header. Max 120 chars.
    """
    hyp = sections.get("Hypothesis", "").strip()
    if hyp:
        for line in hyp.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                return stripped[:120]
    for line in notes.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#\s*exp_\d+\s*$", stripped):
            continue
        cleaned = stripped.lstrip("#").strip()
        if cleaned:
            return cleaned[:120]
    return ""


def reject_forbidden(d: dict) -> None:
    bad = [
        k for k in d
        if any(k.startswith(p) for p in FORBIDDEN_PREFIXES) and k not in FORBIDDEN_EXEMPT
    ]
    if bad:
        raise SystemExit(
            f"refusing to write forbidden fields (eval_*/score_*): {bad}"
        )


def _peek_first_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.readline()
    except OSError:
        return ""


def _rotate_old_schema_file(csv_path: Path, old_version: str) -> None:
    """Rotate a CSV with a stale schema aside; KEEP the .lock file stable.

    The lock file path must not change: concurrent writers that already
    opened the old lock fd would otherwise lose mutual exclusion against
    new writers that open a freshly-created lock. We only rename the CSV
    itself.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rotated = csv_path.with_name(
        f"{csv_path.stem}.{old_version}.{ts}{csv_path.suffix}"
    )
    # os.replace is atomic on POSIX within the same filesystem.
    os.replace(str(csv_path), str(rotated))


_SCHEMA_VERSION_RE = re.compile(r"^#\s*schema_version=([0-9A-Za-z_.-]+)")


def _detect_schema_version(first_line: str) -> str | None:
    m = _SCHEMA_VERSION_RE.match(first_line)
    return m.group(1) if m else None


def append_with_flock(
    csv_path: Path,
    fieldnames: list[str],
    row: dict,
    versioned: bool = False,
) -> None:
    """Append one row under an exclusive flock on a sidecar .lock file.

    Uses os.open(O_CREAT|O_RDWR) — NOT a mode-"w" open — so concurrent writers
    sharing the same lock file path don't truncate each other's fd. mkdir runs
    *before* the flock so the parent dir is guaranteed to exist for every
    holder.

    When `versioned=True`, the file is preceded by a `# schema_version=<N>\n`
    line. If an existing file's first line is not the current schema header,
    it is rotated to `<stem>.<old_version>.<timestamp>.<ext>` *while holding
    the lock on the stable .lock path*, and a fresh file is written. The
    .lock file itself is never rotated — concurrent writers must continue
    to mutually exclude against the same lock path across schema rolls.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            needs_header = (
                not csv_path.exists() or csv_path.stat().st_size == 0
            )
            if versioned and not needs_header:
                first_line = _peek_first_line(csv_path)
                if not first_line.startswith(f"# schema_version={SCHEMA_VERSION}"):
                    old = _detect_schema_version(first_line) or "v0"
                    _rotate_old_schema_file(csv_path, old)
                    needs_header = True
            with csv_path.open("a", newline="") as cf:
                if needs_header and versioned:
                    cf.write(SCHEMA_HEADER_LINE)
                w = csv.DictWriter(cf, fieldnames=fieldnames)
                if needs_header:
                    w.writeheader()
                w.writerow(row)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _exp_number(exp_dir_name: str) -> int | None:
    m = re.match(r"exp_(\d+)", exp_dir_name)
    if not m:
        return None
    return int(m.group(1))


def _read_csv_rows(path: Path) -> list[dict]:
    """Read CSV rows, skipping an optional leading `# schema_version=...` comment."""
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            first = f.readline()
            if first.startswith("#"):
                # comment header — DictReader starts on next line
                reader = csv.DictReader(f)
            else:
                # rewind: re-open since DictReader needs the header line
                f.seek(0)
                reader = csv.DictReader(f)
            return list(reader)
    except OSError:
        return []


def _collect_local_parents(local_idx: Path) -> set[str]:
    """Bare exp_ids visible in the agent's own local index.csv."""
    valid: set[str] = set()
    for row in _read_csv_rows(local_idx):
        eid = row.get("exp_id")
        if eid:
            valid.add(eid)
    return valid


def _expected_backtrack_parent(local_idx: Path) -> str:
    """Return the most-recent local row whose outcome_improved == 'yes'.

    Returns 'none' if no such row exists. This is the parent the publisher
    will require by default; agents who want to fork from somewhere else
    must include an [override:<reason>] marker in ## Method.
    """
    rows = _read_csv_rows(local_idx)
    improved = [r for r in rows if (r.get("outcome_improved") or "").lower() == "yes"]
    if not improved:
        return "none"
    eid = improved[-1].get("exp_id") or "none"
    return eid


def _collect_shared_parents(shared_path: str | None) -> set[str]:
    """Slashed `<agent_id>/<exp_id>` ids from the shared CSV.

    Cross-agent references MUST be qualified with the publishing agent id
    because every agent independently produces exp_001, exp_002, ... so a
    bare exp_id is ambiguous across agents (PR #6's tree renderer also
    relies on the slashed form to disambiguate).
    """
    valid: set[str] = set()
    if not shared_path:
        return valid
    for row in _read_csv_rows(Path(shared_path)):
        eid = row.get("exp_id")
        aid = row.get("agent_id")
        if eid and aid:
            valid.add(f"{aid}/{eid}")
    return valid


_SLASHED_PARENT_RE = re.compile(r"^[A-Za-z0-9._\-]+/exp_\d+$")


def validate_parent(
    exp_dir: Path,
    sections: dict[str, str],
    local_idx: Path,
    shared_path: str | None,
    *,
    enforce_backtrack: bool = True,
) -> str:
    """Validate the ## Parent value and return it.

    Two accepted forms:
    - Bare `exp_NNN` — same-agent parent. Must exist in the agent's own
      local index.csv. The shared CSV is NOT consulted for the bare form
      because exp_NNN is ambiguous across agents.
    - Slashed `<agent_id>/exp_NNN` — cross-agent (or own past) parent. Must
      have a matching (agent_id, exp_id) row in $SHARED_LOG_CSV.
    - `none` — valid only for exp_001, or for exp_002+ with ## PivotReason.

    V2 backtrack rule (`enforce_backtrack=True`): for exp_002+ with a
    bare local parent, the expected parent is the most-recent local row
    with outcome_improved == 'yes'. Agents may fork from a different
    parent ONLY if ## Method contains an `[override:<reason>]` marker.
    Cross-agent slashed parents and PivotReason forks are exempt (the
    backtrack rule applies to local-only chains).
    """
    parent_raw = sections.get("Parent", "").strip()
    if not parent_raw:
        raise SystemExit(
            "notes.md is missing the ## Parent section. "
            "Use 'exp_<K>' (your own prior experiment), "
            "'<agent_id>/exp_<K>' (peer's experiment from $SHARED_LOG_CSV), "
            "or 'none' (requires ## PivotReason for exp_002+)."
        )
    # Take the first non-empty line of the section body as the value.
    parent_value = ""
    for line in parent_raw.splitlines():
        stripped = line.strip().lstrip("-").strip().lstrip("#").strip()
        # Drop trailing inline comments
        stripped = re.sub(r"\s+#.*$", "", stripped).strip()
        if stripped:
            parent_value = stripped
            break
    if not parent_value:
        raise SystemExit("notes.md ## Parent section is empty.")

    exp_n = _exp_number(exp_dir.name)
    if exp_n is None:
        raise SystemExit(
            f"could not parse exp number from directory name {exp_dir.name!r}"
        )

    def _check_value(value: str) -> bool:
        """Return True iff `value` is a recognized parent reference."""
        if "/" in value:
            if not _SLASHED_PARENT_RE.match(value):
                return False
            return value in _collect_shared_parents(shared_path)
        # Bare exp_NNN — local same-agent only.
        return value in _collect_local_parents(local_idx)

    def _fail(value: str) -> None:
        local_recent = sorted(_collect_local_parents(local_idx))[-10:]
        shared_recent = sorted(_collect_shared_parents(shared_path))[-10:]
        raise SystemExit(
            f"## Parent={value!r} not recognized. "
            "Use 'exp_NNN' for your own past experiments "
            "(must appear in local experiments/index.csv), or "
            "'<agent_id>/exp_NNN' for peers' experiments "
            "(must appear in $SHARED_LOG_CSV with matching agent_id). "
            f"Recent local: {local_recent or '(none)'}. "
            f"Recent shared: {shared_recent or '(none)'}."
        )

    if exp_n == 1:
        # exp_001: parent may be 'none' or a valid prior id (cross-run).
        if parent_value == "none":
            return parent_value
        if _check_value(parent_value):
            return parent_value
        # exp_001 with an unknown parent — same diagnostic as exp_002+.
        _fail(parent_value)

    # exp_002+
    pivot_reason = sections.get("PivotReason", "").strip()
    if parent_value == "none":
        if pivot_reason:
            return parent_value
        raise SystemExit(
            "exp_002+ may only have ## Parent: none when ## PivotReason "
            "is present and non-empty."
        )

    if not _check_value(parent_value):
        _fail(parent_value)

    # Backtrack rule (V2). Only enforced on local bare parents — slashed
    # cross-agent parents are exempt because we can't see peers' eval
    # results from here to know which one is "most-recent improved".
    if enforce_backtrack and "/" not in parent_value:
        expected = _expected_backtrack_parent(local_idx)
        if expected != "none" and parent_value != expected:
            method_body = sections.get("Method", "")
            if not _OVERRIDE_MARKER_RE.search(method_body):
                raise SystemExit(
                    f"## Parent={parent_value!r} violates the backtrack "
                    f"rule. Expected parent={expected!r} (most-recent local "
                    "experiment with outcome_improved='yes'). To fork from "
                    "a different parent (e.g., to escape a local optimum), "
                    "include an [override:<short reason>] marker in your "
                    "## Method section, then re-publish."
                )
            print(
                f"[publish] backtrack override accepted: parent={parent_value} "
                f"(expected={expected}); override marker found in ## Method.",
                file=sys.stderr,
            )
    return parent_value


def _validate_required_sections(
    sections: dict[str, str],
    *,
    audit_failed: bool,
) -> None:
    """Enforce required sections.

    Schema v4 promotes ## Findings and ## Outcome to required. For backward
    compatibility, ## Conclusion is accepted in lieu of ## Findings on
    audit-failed rows (where there's no eval to ground a real "findings"
    statement) and as a fallback for legacy notes.md templates.
    """
    if audit_failed:
        # Failure rows don't run eval, so Outcome can be skipped. Findings
        # may be empty too — Conclusion (legacy) is sufficient narrative.
        # Require at minimum: Parent, Hypothesis, Method, and one of
        # {Findings, Conclusion}.
        base = ["Parent", "Hypothesis", "Method"]
        missing = [s for s in base if not sections.get(s, "").strip()]
        if not (sections.get("Findings", "").strip() or sections.get("Conclusion", "").strip()):
            missing.append("Findings (or legacy Conclusion)")
        if missing:
            raise SystemExit(
                f"notes.md is missing required section(s): {missing}. "
                "On --audit-failed: ## Parent, ## Hypothesis, ## Method, "
                "and one of ## Findings or ## Conclusion are required."
            )
        return

    missing = [s for s in REQUIRED_SECTIONS if not sections.get(s, "").strip()]
    # Findings missing? Fall back to Conclusion (legacy alias).
    if "Findings" in missing and sections.get("Conclusion", "").strip():
        missing = [m for m in missing if m != "Findings"]
    if missing:
        raise SystemExit(
            f"notes.md is missing required section(s): {missing}. "
            "Required headers (exact): "
            "## Parent, ## Hypothesis, ## Method, ## Findings, ## Outcome "
            "(## PivotReason optional; ## Conclusion accepted as legacy "
            "alias for ## Findings)."
        )


def parse_outcome(body: str) -> dict[str, str]:
    """Parse the structured ## Outcome body into a dict.

    Expected keys: eval_before, eval_after, improved, confidence.
    Unknown lines are ignored. Returns lowercased values for the enum
    fields; eval_* values are kept as raw strings (caller validates the
    float-ness against the eval_result.json file).
    """
    parsed: dict[str, str] = {}
    for raw_line in body.splitlines():
        m = _OUTCOME_KEY_RE.match(raw_line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        if key in {"improved", "confidence"}:
            val = val.lower()
        parsed[key] = val
    return parsed


def _validate_outcome_section(
    sections: dict[str, str],
    *,
    audit_failed: bool,
    exp_n: int,
) -> dict[str, str]:
    """Parse + validate ## Outcome. Returns the parsed dict (possibly empty).

    On --audit-failed, Outcome is optional — return an empty dict.
    On normal publish for exp_001+, Outcome must be present and well-formed:
        eval_after must be a parseable float;
        improved must be in {yes, no, marginal};
        confidence (if present) must be in {high, med, low};
        eval_before must be 'none' (only allowed when exp_n == 1) OR a float.
    """
    if audit_failed:
        return {}

    body = sections.get("Outcome", "").strip()
    if not body:
        raise SystemExit(
            "notes.md is missing the ## Outcome section. V2 requires a "
            "structured Outcome block:\n"
            "  eval_before: <float or 'none' for exp_001>\n"
            "  eval_after: <float>\n"
            "  improved: yes | no | marginal\n"
            "  confidence: high | med | low"
        )
    parsed = parse_outcome(body)

    missing_keys = [k for k in ("eval_before", "eval_after", "improved") if k not in parsed]
    if missing_keys:
        raise SystemExit(
            f"## Outcome is missing required key(s): {missing_keys}. "
            "Expected exactly:\n"
            "  eval_before: <float or 'none' for exp_001>\n"
            "  eval_after: <float>\n"
            "  improved: yes | no | marginal\n"
            "  confidence: high | med | low"
        )

    improved = parsed.get("improved", "")
    if improved not in _OUTCOME_IMPROVED_VALUES:
        raise SystemExit(
            f"## Outcome improved={improved!r} is not valid. "
            f"Must be one of {sorted(_OUTCOME_IMPROVED_VALUES)}."
        )

    confidence = parsed.get("confidence")
    if confidence is not None and confidence not in _OUTCOME_CONFIDENCE_VALUES:
        raise SystemExit(
            f"## Outcome confidence={confidence!r} is not valid. "
            f"Must be one of {sorted(_OUTCOME_CONFIDENCE_VALUES)}."
        )

    # eval_after must be a float.
    try:
        float(parsed["eval_after"])
    except (TypeError, ValueError):
        raise SystemExit(
            f"## Outcome eval_after={parsed['eval_after']!r} is not a "
            "parseable float. Use a numeric accuracy value."
        )

    # eval_before: 'none' allowed only for exp_001.
    eb = parsed["eval_before"].lower()
    if eb == "none":
        if exp_n != 1:
            raise SystemExit(
                f"## Outcome eval_before='none' is only valid for exp_001 "
                f"(this is exp_{exp_n:03d}). Set eval_before to the parent's "
                "accuracy from its eval_result.json."
            )
    else:
        try:
            float(parsed["eval_before"])
        except (TypeError, ValueError):
            raise SystemExit(
                f"## Outcome eval_before={parsed['eval_before']!r} is not a "
                "parseable float (and not 'none')."
            )

    return parsed


def _check_eval_score_against_file(
    eval_path: Path,
    stated_value: str,
    *,
    field_name: str,
    tolerance: float = 0.01,
) -> None:
    """Cross-check a stated eval score in notes.md vs the JSON file on disk.

    The file is optional — if it doesn't exist we trust the agent (e.g.,
    cross-agent parent whose eval_result.json isn't visible). When it does
    exist, an absolute discrepancy > `tolerance` is a hard refuse.

    We look for 'accuracy' at the top level first, falling back to common
    inspect_ai metric layouts. The agent's stated value is a float string.
    """
    if not eval_path.is_file():
        return
    try:
        with eval_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[publish] warning: could not read {eval_path} to verify "
            f"{field_name}: {exc}",
            file=sys.stderr,
        )
        return

    file_score = _extract_accuracy(data)
    if file_score is None:
        print(
            f"[publish] warning: could not find an accuracy field in "
            f"{eval_path}; skipping {field_name} cross-check.",
            file=sys.stderr,
        )
        return

    try:
        stated_float = float(stated_value)
    except (TypeError, ValueError):
        # Already validated upstream — keep this defensive.
        return

    if abs(stated_float - file_score) > tolerance:
        raise SystemExit(
            f"## Outcome {field_name}={stated_float} disagrees with "
            f"{eval_path} accuracy={file_score} (delta={abs(stated_float - file_score):.4f} "
            f"> tolerance={tolerance}). Either fix notes.md or re-run "
            "evaluate.py to refresh eval_result.json."
        )


def _check_eval_used_full_dataset(eval_result_path: Path, task_name: str | None) -> None:
    """Refuse publish if eval_result.json was produced from a smoke eval.

    V2 pilot: agent ran --limit 5 / --limit 50 evals and wrote the JSON,
    then the framework happily accepted that as "the experiment's score".
    Smoke evals aren't valid plateau-detection signal. V3: only --limit -1
    counts.
    """
    data = json.loads(eval_result_path.read_text())
    # Probe several inspect_ai JSON shapes.
    samples = None
    eval_block = data.get("eval", {})
    if isinstance(eval_block, dict):
        dataset = eval_block.get("dataset")
        if isinstance(dataset, dict):
            samples = dataset.get("samples")
    if samples is None:
        results = data.get("results")
        if isinstance(results, list) and results:
            scores = results[0].get("scores")
            if isinstance(scores, list) and scores:
                scorer = scores[0].get("scorer", {})
                if isinstance(scorer, dict):
                    samples = scorer.get("samples")
    if samples is None:
        # Last-resort guess: count entries in `samples` list if present.
        samples_field = data.get("samples")
        if isinstance(samples_field, list):
            samples = len(samples_field)

    if samples is None:
        # Couldn't determine sample count — be conservative and warn loudly
        # rather than refuse, since unfamiliar inspect_ai versions may
        # restructure the JSON.
        print(
            f"[publish V3] WARNING: couldn't determine sample count from "
            f"{eval_result_path}; allowing but flagging in row.",
            file=sys.stderr,
        )
        return

    expected = TASK_FULL_SAMPLE_COUNT.get(task_name) if task_name else None
    threshold = expected if expected is not None else DEFAULT_MIN_FULL_EVAL_SAMPLES

    if samples < threshold:
        raise SystemExit(
            f"\n[publish V3 GATE] Refusing to publish: eval_result.json reports "
            f"only {samples} samples (need >= {threshold} for task='{task_name}').\n"
            f"  This looks like a smoke eval (--limit {samples}), not a full eval.\n"
            f"  Re-run with --limit -1 to use the full {expected or threshold} samples:\n"
            f"    python3 evaluate.py --model-path <model> --limit -1 \\\n"
            f"        --json-output-file <path-to-eval_result.json> ...\n"
        )


# V4: SOURCE-NOVELTY GATE -------------------------------------------------
def _parse_sources(raw: str) -> set[str]:
    """Split a data_sources string into a set of normalized source tokens.

    Comma-split, trimmed, lowercased. Empty tokens are dropped. Shared by
    the current experiment's --data-sources arg and the prior shared rows.
    """
    if not raw:
        return set()
    return {tok.strip().lower() for tok in raw.split(",") if tok.strip()}


def _collect_seen_sources(shared_path: str | None) -> set[str]:
    """Union of every prior row's data_sources across the shared CSV.

    This is the run-wide "what has already been tried" set. We read all
    prior rows (any agent, any exp) and accumulate their source tokens.
    """
    seen: set[str] = set()
    if not shared_path:
        return seen
    for row in _read_csv_rows(Path(shared_path)):
        seen |= _parse_sources(row.get("data_sources") or "")
    return seen


def _check_source_novelty(
    exp_n: int,
    current_sources: set[str],
    shared_path: str | None,
    *,
    audit_failed: bool,
) -> None:
    """V4 SOURCE-NOVELTY GATE: force a fresh data source every Nth experiment.

    RULE: when exp_n is a non-zero multiple of SOURCE_NOVELTY_EVERY
    (exp_005, exp_010, exp_015, ...), at least one of this experiment's
    data_sources must NOT already appear in the run-wide "seen sources"
    set (the union of all prior shared-CSV rows' data_sources). This forces
    the agent to periodically broaden the data distribution rather than
    endlessly re-permuting the same sources.

    Exemptions:
    - --audit-failed rows (no real dataset to diversify).
    - exp_001 (no priors — always exempt; also not a multiple of 5).
    - Any exp_n that is not a multiple of SOURCE_NOVELTY_EVERY.
    """
    if audit_failed:
        return
    if exp_n <= 1:
        return
    if SOURCE_NOVELTY_EVERY <= 0 or (exp_n % SOURCE_NOVELTY_EVERY) != 0:
        return

    seen = _collect_seen_sources(shared_path)
    novel = current_sources - seen
    if not novel:
        seen_list = sorted(seen) or ["(none)"]
        cur_list = sorted(current_sources) or ["(none)"]
        raise SystemExit(
            f"\n[publish V4 SOURCE-NOVELTY GATE] Refusing to publish exp_{exp_n:03d}: "
            f"every {SOURCE_NOVELTY_EVERY}th experiment must introduce at least one "
            "data source this run has NEVER used.\n"
            f"  This experiment's sources: {cur_list}\n"
            f"  Already-seen sources this run: {seen_list}\n"
            "  None of this experiment's sources are new. To satisfy the gate:\n"
            "    - run 'python3 discover_datasets.py' to find a new HF source "
            "(it auto-excludes anything mentioning 'gpqa'), OR\n"
            "    - run 'python3 teacher_synth.py --mode generate' to synthesize "
            "a fresh source, then\n"
            "  rebuild data.jsonl mixing in that new source and re-publish with "
            "the new source listed in --data-sources.\n"
        )
    print(
        f"[publish V4] source-novelty gate satisfied for exp_{exp_n:03d}: "
        f"new source(s) {sorted(novel)}.",
        file=sys.stderr,
    )


# V4: EVAL-TOKEN GUARD ----------------------------------------------------
def _extract_eval_max_tokens(data: dict) -> int | None:
    """Best-effort extraction of the generation max_tokens from an
    inspect_ai eval_result.json.

    Probes several common locations across inspect_ai versions:
      - top-level 'max_tokens'
      - eval.model_args.max_tokens
      - eval.config.max_tokens / eval.config.max_connections (max_tokens only)
      - plan/solver config: plan.steps[*].params.max_tokens (and
        plan.config.max_tokens), or a top-level 'config'.
    Returns the int value if found, else None (caller treats None as
    "unknown" — warn, do not refuse).
    """
    def _coerce(v) -> int | None:
        if isinstance(v, bool):  # bool is a subclass of int — reject
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str):
            try:
                return int(float(v.strip()))
            except (TypeError, ValueError):
                return None
        return None

    # Top-level.
    top = _coerce(data.get("max_tokens"))
    if top is not None:
        return top

    eval_block = data.get("eval")
    if isinstance(eval_block, dict):
        for sub_key in ("model_args", "config"):
            sub = eval_block.get(sub_key)
            if isinstance(sub, dict):
                got = _coerce(sub.get("max_tokens"))
                if got is not None:
                    return got

    # Top-level config block (some versions hoist generation config here).
    cfg = data.get("config")
    if isinstance(cfg, dict):
        got = _coerce(cfg.get("max_tokens"))
        if got is not None:
            return got

    # Plan / solver config: plan.config.max_tokens or
    # plan.steps[*].params.max_tokens.
    plan = data.get("plan")
    if isinstance(plan, dict):
        pcfg = plan.get("config")
        if isinstance(pcfg, dict):
            got = _coerce(pcfg.get("max_tokens"))
            if got is not None:
                return got
        steps = plan.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if not isinstance(step, dict):
                    continue
                params = step.get("params")
                if isinstance(params, dict):
                    got = _coerce(params.get("max_tokens"))
                    if got is not None:
                        return got
    return None


def _check_eval_max_tokens(eval_result_path: Path) -> None:
    """V4 EVAL-TOKEN GUARD: refuse evals run with too small a generation budget.

    Reads eval_result.json and best-effort extracts the generation
    max_tokens. If a value is found AND it is < MIN_EVAL_MAX_TOKENS, the
    eval forbade the chain-of-thought reasoning that the cot=True harness
    rewards — refuse. If max_tokens cannot be determined, emit a stderr
    WARNING but DO NOT refuse (conservative, like the sample-count check):
    we don't want a new false-refusal failure mode for inspect_ai JSON
    shapes we don't recognize.
    """
    if not eval_result_path.is_file():
        return
    try:
        data = json.loads(eval_result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[publish V4] WARNING: could not read {eval_result_path} to verify "
            f"eval max_tokens: {exc}; allowing without the token guard.",
            file=sys.stderr,
        )
        return

    max_tokens = _extract_eval_max_tokens(data)
    if max_tokens is None:
        print(
            f"[publish V4] WARNING: couldn't determine generation max_tokens from "
            f"{eval_result_path}; allowing but cannot confirm reasoning had room. "
            f"Ensure evaluate.py ran with --max-tokens 16000.",
            file=sys.stderr,
        )
        return

    if max_tokens < MIN_EVAL_MAX_TOKENS:
        raise SystemExit(
            f"\n[publish V4 EVAL-TOKEN GUARD] Refusing to publish: eval was run "
            f"with max_tokens={max_tokens} which is too small for cot=True "
            f"reasoning (need >= {MIN_EVAL_MAX_TOKENS}); re-run evaluate.py with "
            "--max-tokens 16000.\n"
        )
    print(
        f"[publish V4] eval-token guard satisfied: eval max_tokens={max_tokens} "
        f"(>= {MIN_EVAL_MAX_TOKENS}).",
        file=sys.stderr,
    )


def _extract_accuracy(data: dict) -> float | None:
    """Best-effort accuracy extraction from an evaluate.py JSON dump.

    inspect_ai writes results under varying shapes depending on the task
    (top-level "accuracy", or under "results"/"metrics"/"scores"). We try
    several common locations.
    """
    # Direct top-level.
    for key in ("accuracy", "acc", "score"):
        v = data.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # results -> scores -> {name: {accuracy: ...}}
    results = data.get("results") or {}
    if isinstance(results, dict):
        scores = results.get("scores") or []
        if isinstance(scores, list):
            for entry in scores:
                if isinstance(entry, dict):
                    metrics = entry.get("metrics") or {}
                    for mkey in ("accuracy", "acc"):
                        if mkey in metrics:
                            try:
                                return float(metrics[mkey].get("value", metrics[mkey]))
                            except (TypeError, ValueError, AttributeError):
                                pass
        # results -> accuracy directly.
        for key in ("accuracy", "acc"):
            v = results.get(key)
            if isinstance(v, (int, float)):
                return float(v)
    # metrics -> {accuracy: {value: ...}}
    metrics = data.get("metrics") or {}
    if isinstance(metrics, dict):
        for mkey in ("accuracy", "acc"):
            entry = metrics.get(mkey)
            if isinstance(entry, dict) and "value" in entry:
                try:
                    return float(entry["value"])
                except (TypeError, ValueError):
                    pass
            if isinstance(entry, (int, float)):
                return float(entry)
    return None


# V4: MULTI-SEED PROMOTION ------------------------------------------------
def _split_path_list(raw: str | None) -> list[str]:
    """Split a --eval-results value on commas and/or whitespace.

    Returns the list of non-empty trimmed tokens. Empty/None -> [].
    """
    if not raw:
        return []
    return [tok for tok in re.split(r"[,\s]+", raw.strip()) if tok]


def validate_eval_file(eval_path: Path, task_name: str | None) -> float:
    """Fully validate ONE eval_result*.json and return its grounded accuracy.

    Applies the SAME guards V4 applies to the canonical eval_result.json,
    so EVERY file that contributes to eval_after_mean is held to the same
    full-comparable-evidence bar:

    1. Readable + parseable JSON object (else: not full evidence → refuse).
    2. Full-dataset sample count (_check_eval_used_full_dataset) — a smoke /
       low-sample eval is refused.
    3. Generation max_tokens >= MIN_EVAL_MAX_TOKENS (_check_eval_max_tokens)
       — a low-token eval that forbade cot=True reasoning is refused.
    4. A recognizable accuracy field (_extract_accuracy). An old/malformed
       shape with no extractable accuracy is refused (it cannot ground a
       promotion number).

    Raises SystemExit (fail loud) on any of these so a non-comparable file
    can NEVER silently affect the multi-seed mean. The sample-count and
    token guards keep their own internal "couldn't determine → warn, don't
    refuse" conservatism for unfamiliar inspect_ai shapes; what we add here
    is that a file with NO extractable accuracy is always rejected.
    """
    if not eval_path.is_file():
        raise SystemExit(
            f"[publish V4 MULTI-SEED] eval result {eval_path} does not exist; "
            "cannot use it as promotion evidence."
        )
    try:
        data = json.loads(eval_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"[publish V4 MULTI-SEED] eval result {eval_path} is not readable / "
            f"parseable JSON ({exc}); it is not full comparable evidence and "
            "must not contribute to eval_after_mean. Re-run evaluate.py to "
            "regenerate it (or drop it from --eval-results)."
        )
    if not isinstance(data, dict):
        raise SystemExit(
            f"[publish V4 MULTI-SEED] eval result {eval_path} is not a JSON "
            "object (old/malformed shape); refusing to use it in the mean."
        )

    # Same full-dataset + token guards the canonical file gets. These raise
    # SystemExit on a smoke / low-token eval.
    _check_eval_used_full_dataset(eval_path, task_name)
    _check_eval_max_tokens(eval_path)

    acc = _extract_accuracy(data)
    if acc is None:
        raise SystemExit(
            f"[publish V4 MULTI-SEED] eval result {eval_path} has no "
            "recognizable accuracy field (old/malformed eval shape); refusing "
            "to use it in eval_after_mean. Re-run evaluate.py with the current "
            "harness so it writes a parseable accuracy."
        )
    return acc


def collect_eval_seed_scores(
    exp_dir: Path,
    cli_eval_results: str | None,
    task_name: str | None,
) -> tuple[list[float], list[str]]:
    """Collect FULLY-VALIDATED per-seed accuracies + their provenance.

    Only file-backed evals contribute to the mean, and every file is run
    through validate_eval_file (full-dataset count, max-tokens, accuracy
    extraction) so a smoke / low-token / malformed / old-shape file can
    NEVER affect eval_after_mean (it fails loud instead).

    Raw --eval-after floats are NOT consulted here: they are not file-backed
    and cannot be held to the score/full-sample/max-token checks, so they no
    longer feed the promotion math (see main() — they remain available only
    as a non-promotion display value via the ## Outcome eval_after column).

    Sources, unioned (de-dup of identical accuracies is intentionally NOT
    applied — two seeds can legitimately tie and both should count toward n;
    the SAME on-disk file is de-duped so the auto-glob doesn't double-count
    a path also named on --eval-results):
    1. --eval-results paths (comma/space list of eval_result*.json).
    2. Auto-glob of <exp_dir>/eval_result*.json (the canonical single-file
       eval_result.json plus any eval_result_seedK.json siblings).

    Returns (scores, sources) where sources is the list of contributing file
    paths (provenance) parallel to scores. The caller handles the legacy
    single-score fallback (## Outcome eval_after) only when scores is empty.
    """
    scores: list[float] = []
    sources: list[str] = []

    # 1. Explicit per-seed files. Track resolved paths so the auto-glob below
    #    doesn't count the same file twice.
    explicit: set[Path] = set()
    for tok in _split_path_list(cli_eval_results):
        p = Path(tok)
        if not p.is_absolute():
            # Resolve relative to CWD first, then to the exp dir as a
            # convenience (agents often pass bare 'eval_result_seed1.json').
            cand = p if p.is_file() else (exp_dir / tok)
        else:
            cand = p
        cand = cand.resolve()
        explicit.add(cand)
        # validate_eval_file fails loud on any non-comparable file.
        acc = validate_eval_file(cand, task_name)
        scores.append(acc)
        sources.append(str(cand))

    # 2. Auto-glob eval_result*.json in the exp dir.
    for p in sorted(exp_dir.glob("eval_result*.json")):
        if p.resolve() in explicit:
            continue
        acc = validate_eval_file(p, task_name)
        scores.append(acc)
        sources.append(str(p))

    return scores, sources


def compute_seed_stats(scores: list[float]) -> tuple[float | None, float, int]:
    """Return (mean, population_std, n) for a list of per-seed accuracies.

    - n == 0  -> (None, 0.0, 0): no seed signal at all.
    - n == 1  -> (score, 0.0, 1): single seed, zero spread (legacy behavior).
    - n >= 2  -> population std (ddof=0) so a 2-seed sample still yields a
      finite spread the margin can consume.
    """
    n = len(scores)
    if n == 0:
        return None, 0.0, 0
    mean = sum(scores) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((s - mean) ** 2 for s in scores) / n
    return mean, var ** 0.5, n


def promotion_margin(std: float) -> float:
    """The accuracy margin a candidate must clear to count as improved.

    margin = max(PROMOTION_MARGIN_FLOOR, PROMOTION_STD_K * std)

    The floor guards against a zero-std single seed promoting on a 0.001
    lead; the std term widens the bar when the across-seed spread is large.
    """
    return max(PROMOTION_MARGIN_FLOOR, PROMOTION_STD_K * float(std))


def passes_promotion_margin(
    eval_after_mean: float | None,
    incumbent_eval: float | None,
    eval_after_std: float,
) -> bool:
    """True iff the candidate beats the incumbent by more than the noise band.

    Returns False (refuse to call it an improvement) whenever we lack the
    numbers to make the call — a missing mean or a missing/non-float
    incumbent (eval_before='none', i.e. exp_001 with no parent) cannot
    establish a real improvement over an incumbent, so the margin is not met.

    The comparison is strict (>) so a candidate landing exactly on the bar
    does not promote.
    """
    if eval_after_mean is None or incumbent_eval is None:
        return False
    return eval_after_mean > incumbent_eval + promotion_margin(eval_after_std)


def _to_float_or_none(value: str | None) -> float | None:
    """Parse a float, returning None for None/empty/'none'/unparseable."""
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() == "none":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _validate_knowledge_md(
    knowledge_path: Path,
    exp_dir_name: str,
    exp_n: int,
) -> None:
    """Enforce KNOWLEDGE.md presence, per-line format, exp-tagging, and 1-3 line budget.

    Rules:
    - exp_001: file may be missing (Step 0 says "may show an empty file —
      that's fine"). If present, lines for exp_001 still get format-checked.
    - exp_002+: file MUST exist.
    - Of the last 5 non-empty lines, at LEAST 1 and AT MOST 3 must reference
      the current exp_NNN.
    - Each new line must match _KNOWLEDGE_LINE_RE with a tag from the
      whitelist.
    """
    if not knowledge_path.exists():
        if exp_n == 1:
            print(
                "[publish] note: experiments/KNOWLEDGE.md absent on exp_001 "
                "— allowed once, but you MUST create + append from exp_002.",
                file=sys.stderr,
            )
            return
        raise SystemExit(
            "experiments/KNOWLEDGE.md is missing. From exp_002 onward you "
            "MUST read this file at the start of every experiment and "
            "append 1-3 tagged lines at the end. Create it with at least "
            "one line referencing this experiment, format:\n"
            "  [YYYY-MM-DDTHH:MMZ] [exp_NNN] [#tag] one-sentence learning"
        )

    try:
        raw_lines = knowledge_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SystemExit(f"could not read {knowledge_path}: {exc}")

    nonempty = [line.rstrip() for line in raw_lines if line.strip()]
    # Tail-5 window. Few enough that overcounting is rare; lets us cheaply
    # detect "agent dumped 10 lines at once".
    tail = nonempty[-5:]
    exp_lines_in_tail = []
    for line in tail:
        m = _KNOWLEDGE_LINE_RE.match(line)
        if not m:
            # Bad-format line that references this exp must be flagged.
            if f"[{exp_dir_name}]" in line:
                raise SystemExit(
                    f"experiments/KNOWLEDGE.md has a malformed line for "
                    f"{exp_dir_name}: {line!r}. Format must be:\n"
                    "  [YYYY-MM-DDTHH:MMZ] [exp_NNN] [#tag] one-sentence learning"
                )
            continue
        if m.group("exp") != exp_dir_name:
            continue
        if m.group("tag") not in _KNOWLEDGE_TAG_WHITELIST:
            raise SystemExit(
                f"experiments/KNOWLEDGE.md line for {exp_dir_name} uses "
                f"non-whitelisted tag {m.group('tag')!r}. "
                f"Allowed: {sorted(_KNOWLEDGE_TAG_WHITELIST)}."
            )
        exp_lines_in_tail.append(line)

    if not exp_lines_in_tail:
        raise SystemExit(
            f"experiments/KNOWLEDGE.md has no line referencing {exp_dir_name} "
            "in its last 5 lines. Append 1-3 tagged lines for this "
            "experiment before publishing. Format:\n"
            "  [YYYY-MM-DDTHH:MMZ] [exp_NNN] [#tag] one-sentence learning"
        )
    if len(exp_lines_in_tail) > 3:
        raise SystemExit(
            f"too many KNOWLEDGE.md entries for this experiment "
            f"({len(exp_lines_in_tail)} found in tail) — keep to 1-3."
        )


def _validate_done_flag(local_idx: Path, exp_n: int) -> None:
    """Refuse --done if the floor or plateau condition isn't met.

    Floor: at least 10 published experiments (this one counts as the
    10th only after it lands in the index — we check existing count + 1).
    Plateau: among the LAST 10 outcomes (existing + this one), NONE may
    be 'yes'. The publisher hasn't written this row yet, so callers pass
    in the to-be-written outcome via the local_idx merge in main().
    """
    rows = _read_csv_rows(local_idx)
    # +1 for the row we're about to write.
    total_after = len(rows) + 1
    if total_after < 10:
        raise SystemExit(
            f"--done refused: only {total_after} experiments would be "
            "published (need >= 10). Keep iterating."
        )


def _check_done_plateau(
    local_idx: Path,
    *,
    current_outcome: str,
) -> None:
    """After computing the current outcome, verify the 10-experiment plateau.

    The current experiment counts as the most recent in the window. If
    ANY of the last 10 (including this one) has outcome_improved == 'yes',
    --done is refused with a diagnostic listing which exp_id still showed
    improvement.
    """
    rows = _read_csv_rows(local_idx)
    # Build the candidate "last 10" including the row we're about to write.
    pseudo = list(rows)
    pseudo.append({"exp_id": "<this>", "outcome_improved": current_outcome})
    tail = pseudo[-10:]
    if len(tail) < 10:
        raise SystemExit(
            f"--done refused: only {len(tail)} rows in the plateau window "
            "(need 10). Keep iterating."
        )
    yes_rows = [r for r in tail if (r.get("outcome_improved") or "").lower() == "yes"]
    if yes_rows:
        ids = [r.get("exp_id", "?") for r in yes_rows]
        raise SystemExit(
            f"--done refused: experiments {ids} in the last 10 still "
            "show outcome_improved='yes'; you have not plateaued. "
            "Keep iterating."
        )


def _persist_promoted_data(
    shared_path_str: str,
    data_path: Path,
    data_sha: str,
) -> Path | None:
    """Copy data.jsonl into <shared_dir>/promoted/<sha>.jsonl (idempotent)."""
    try:
        promoted_dir = Path(os.path.dirname(shared_path_str)) / "promoted"
        promoted_dir.mkdir(parents=True, exist_ok=True)
        target = promoted_dir / f"{data_sha}.jsonl"
        if not target.exists():
            shutil.copy2(str(data_path), str(target))
        return target
    except OSError as exc:
        print(
            f"[publish] warning: could not persist promoted data: {exc}",
            file=sys.stderr,
        )
        return None


def main() -> int:
    args = parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    notes_path = exp_dir / "notes.md"
    data_path = exp_dir / "data.jsonl"
    audit_path = exp_dir / "dataset_audit_report.json"
    eval_result_path = exp_dir / "eval_result.json"

    if not notes_path.is_file():
        raise SystemExit(f"missing required file in {exp_dir}: notes.md")

    if not args.audit_failed:
        for required in (data_path, audit_path):
            if not required.is_file():
                raise SystemExit(
                    f"missing required file in {exp_dir}: {required.name}"
                )

    notes = notes_path.read_text()
    sections = parse_notes_sections(notes)
    _validate_required_sections(sections, audit_failed=args.audit_failed)

    shared_path = os.environ.get("SHARED_LOG_CSV")
    local_idx = exp_dir.parent / "index.csv"

    # Parse exp number once for downstream use (KNOWLEDGE.md validation,
    # outcome 'none' rule, done floor).
    exp_n = _exp_number(exp_dir.name)
    if exp_n is None:
        raise SystemExit(
            f"could not parse exp number from {exp_dir.name!r}; expected exp_NNN"
        )

    parent_value = validate_parent(exp_dir, sections, local_idx, shared_path)

    # V4: SOURCE-NOVELTY GATE. Parse this experiment's data_sources and,
    # on every SOURCE_NOVELTY_EVERY-th experiment, require at least one
    # source not yet seen across the run's shared CSV. Audit-failed rows
    # and exp_001 are exempt (handled inside _check_source_novelty).
    current_sources = _parse_sources(args.data_sources)
    _check_source_novelty(
        exp_n, current_sources, shared_path, audit_failed=args.audit_failed
    )

    # V2: parse and validate the structured ## Outcome section.
    outcome = _validate_outcome_section(
        sections, audit_failed=args.audit_failed, exp_n=exp_n
    )
    outcome_improved = outcome.get("improved", "") if outcome else ""
    outcome_eval_before = outcome.get("eval_before", "") if outcome else ""
    outcome_eval_after = outcome.get("eval_after", "") if outcome else ""

    # Resolve the benchmark task name once (used both by the multi-seed
    # per-file validation below and by the canonical eval_result.json gate
    # later) so every eval file is held to the SAME full-sample count.
    task_name = os.environ.get("EVALUATION_TASK") or None
    if task_name:
        task_name = task_name.lower().strip()

    # V4: MULTI-SEED. Gather every FILE-BACKED per-seed accuracy we can see
    # (--eval-results list + auto-globbed eval_result*.json, INCLUDING the
    # canonical eval_result.json), running EACH file through the SAME
    # full-dataset / max-token / accuracy-extraction guards V4 applies to the
    # canonical file. Any smoke, low-token, malformed, or old-shape file fails
    # loud and can NEVER contribute to eval_after_mean.
    #
    # Raw --eval-after floats are deliberately NOT part of the promotion math:
    # they are not file-backed and cannot be validated, so an unverified CLI
    # number must not be able to satisfy the promotion margin. --eval-after
    # is retained only as a non-promotion display value (logged below; the
    # agent's own ## Outcome eval_after string is still what populates the
    # eval_after column).
    #
    # Backward compat: when no validated file-backed seed is found, fall back
    # to the single ## Outcome eval_after the agent stated (n=1, std=0.0).
    # The legacy single-seed publish path is unchanged: the lone canonical
    # eval_result.json is auto-globbed, validated identically, and yields
    # mean=its score / std=0.0 / n=1.
    seed_scores, seed_sources = collect_eval_seed_scores(
        exp_dir, args.eval_results, task_name
    )
    if not seed_scores:
        legacy = _to_float_or_none(outcome_eval_after)
        if legacy is not None:
            seed_scores = [legacy]
            seed_sources = ["<## Outcome eval_after (no file-backed seed)>"]
    # PROVENANCE: log exactly which files (or the legacy fallback) feed the
    # mean so a reviewer can audit what evidence the promotion verdict rests
    # on without a schema change.
    print(
        f"[publish V4] eval_sources (n={len(seed_scores)}) contributing to "
        f"eval_after_mean: {seed_sources or '(none)'}.",
        file=sys.stderr,
    )
    if args.eval_after:
        print(
            "[publish V4] NOTE: --eval-after floats "
            f"{args.eval_after} are display-only and DO NOT feed "
            "eval_after_mean / outcome_improved / promoted (unverified, not "
            "file-backed). Provide eval_result*.json via --eval-results to "
            "contribute additional validated seeds.",
            file=sys.stderr,
        )
    eval_after_mean, eval_after_std, n_eval_seeds = compute_seed_stats(seed_scores)
    if n_eval_seeds > 1:
        print(
            f"[publish V4] multi-seed: n={n_eval_seeds} "
            f"mean={eval_after_mean:.4f} std={eval_after_std:.4f} "
            f"scores={[round(s, 4) for s in seed_scores]}.",
            file=sys.stderr,
        )

    # V4 PROMOTION MARGIN: a candidate may only claim improvement
    # (outcome_improved=yes / and later effective_promoted) when its
    # across-seed mean beats the incumbent (eval_before, the parent's score)
    # by MORE than the noise band max(PROMOTION_MARGIN_FLOOR,
    # PROMOTION_STD_K * eval_after_std). This closes the V4 pilot failure
    # where exp_007 beat exp_002 by ~1pt on a single seed and the whole run
    # forked from it. The agent's stated ## Outcome improved:yes is now
    # necessary but not sufficient — it is downgraded to 'no' whenever the
    # margin is not cleared. This downgrade happens BEFORE the --done plateau
    # check and the backtrack-parent index write so both consume the
    # margin-gated verdict (a within-noise gain is NOT a real improvement).
    # All other gates (audit_pass/decontam_pass, source-novelty, eval-token,
    # full-dataset, KNOWLEDGE.md, backtrack parent) remain unchanged.
    incumbent_eval = _to_float_or_none(outcome_eval_before)
    margin_ok = passes_promotion_margin(
        eval_after_mean, incumbent_eval, eval_after_std
    )
    if outcome_improved == "yes" and not margin_ok:
        required = (
            (incumbent_eval + promotion_margin(eval_after_std))
            if incumbent_eval is not None and eval_after_mean is not None
            else None
        )
        print(
            "[publish V4 PROMOTION MARGIN] downgrading outcome_improved "
            f"'yes' -> 'no': eval_after_mean="
            f"{eval_after_mean if eval_after_mean is not None else 'n/a'} "
            f"incumbent(eval_before)="
            f"{incumbent_eval if incumbent_eval is not None else 'n/a'} "
            f"std={eval_after_std:.4f} n={n_eval_seeds} "
            f"margin={promotion_margin(eval_after_std):.4f} "
            f"(needed mean > {required if required is not None else 'incumbent+margin'}). "
            "Within-noise gains do not count as improvement.",
            file=sys.stderr,
        )
        outcome_improved = "no"

    # V4: EVAL-TOKEN GUARD. Closes the V3 bug at the publisher level: if
    # eval_result.json proves the self-eval ran with a generation budget
    # smaller than MIN_EVAL_MAX_TOKENS, reasoning was forbidden and the
    # eval is invalid. Runs whenever the file exists (independent of the
    # ## Outcome block), since it's a property of the eval run itself.
    if eval_result_path.is_file():
        _check_eval_max_tokens(eval_result_path)

    # V2: cross-check stated eval scores against eval_result.json on disk.
    if outcome:
        # V3: also refuse smoke evals — if eval_result.json reports fewer
        # samples than the task's full size, this isn't a valid plateau
        # signal. Task name (resolved once above) comes from $EVALUATION_TASK.
        # NOTE: the canonical eval_result.json is also auto-globbed and run
        # through validate_eval_file above (same guards), so this is the
        # backward-compat duplicate; the checks are idempotent.
        if eval_result_path.is_file():
            _check_eval_used_full_dataset(eval_result_path, task_name)
        _check_eval_score_against_file(
            eval_result_path,
            outcome_eval_after,
            field_name="eval_after",
        )
        # Parent-side check: only meaningful for own-experiment parents
        # whose eval_result.json lives locally. Skip if cross-agent.
        if "/" not in parent_value and parent_value not in ("", "none"):
            parent_eval_path = exp_dir.parent / parent_value / "eval_result.json"
            _check_eval_score_against_file(
                parent_eval_path,
                outcome_eval_before,
                field_name="eval_before",
            )

    # V2: KNOWLEDGE.md presence + per-line format check. Audit-failed rows
    # still must append a learning (often the most useful kind — what NOT
    # to do).
    knowledge_path = exp_dir.parent / "KNOWLEDGE.md"
    _validate_knowledge_md(knowledge_path, exp_dir.name, exp_n)

    # V2: --done floor (>=10 experiments). Plateau check fires after
    # outcome is computed; see below.
    if args.done:
        _validate_done_flag(local_idx, exp_n)
        _check_done_plateau(local_idx, current_outcome=outcome_improved)

    # Data + audit (skipped on --audit-failed)
    if args.audit_failed:
        data_sha = ""
        rows = 0
        diversity: dict = {}
        decontam: dict = {}
        audit_pass = False
        decontam_pass = False
    else:
        data_sha = file_sha256(data_path)
        rows = count_lines(data_path)
        audit = json.loads(audit_path.read_text())
        diversity = audit.get("diversity", {})
        decontam = audit.get("decontam", {})
        audit_pass = bool(audit.get("pass", False))
        decontam_pass = bool(decontam.get("pass", False))

        # Bind the audit report to the exact dataset bytes. A stale
        # passing report from an earlier version of data.jsonl must not
        # let a modified (potentially contaminated) dataset through.
        audit_sha = audit.get("data_sha256", "")
        if not audit_sha:
            raise SystemExit(
                f"audit report at {audit_path} is missing 'data_sha256'. "
                "Re-run dataset_audit.py to produce a current report."
            )
        if audit_sha != data_sha:
            raise SystemExit(
                f"audit report data_sha256={audit_sha!r} does not match "
                f"current {data_path.name} sha256={data_sha!r}. The dataset "
                "has changed since it was audited; re-run dataset_audit.py."
            )

    # Compute one effective promotion flag and use it everywhere: the shared
    # CSV row, the local index row, and the /shared_log/promoted/<sha>.jsonl
    # persistence guard. The schema treats promoted=True as the cross-run
    # handoff signal, so we MUST NOT advertise promoted=True on a row whose
    # audit or decontam check failed — peers would then fork a dataset that
    # never actually got persisted to /shared_log/promoted/.
    #
    # V2 additionally requires outcome_improved == 'yes': "promoted" means
    # VERIFIED improvement, not just "passed gates". outcome_improved has
    # already been margin-gated above (V4), so outcome_yes here reflects the
    # noise-band-aware verdict; effective_promoted then ANDs in the unchanged
    # audit_pass / decontam_pass gates.
    outcome_yes = (outcome_improved == "yes")
    effective_promoted = bool(
        args.promoted and audit_pass and decontam_pass and outcome_yes
    )
    if args.promoted and not effective_promoted:
        print(
            "[publish] WARNING: --promoted requested but "
            f"audit_pass={audit_pass} / decontam_pass={decontam_pass} / "
            f"outcome_improved={outcome_improved!r}; recording promoted=False",
            file=sys.stderr,
        )

    # All free-text fields written to the shared CSV are scrubbed the
    # same way — strategy_short and notes_excerpt would otherwise let eval
    # scores or comparative score language leak through. extract_strategy
    # returns the raw first line of ## Hypothesis (or the notes body), so
    # we route it through short_field() (which scrub_numeric()s + truncates)
    # rather than emitting it raw.
    #
    # V2: ## Findings is the new "what surprised me" section. For backward
    # compat with v3 readers, conclusion_short is sourced from Findings
    # (or legacy Conclusion if Findings missing). findings_short is the
    # same content under the new explicit name.
    strategy_short = short_field(extract_strategy(notes, sections), 200)
    notes_excerpt = short_field(notes, 200)
    hypothesis_short = short_field(sections.get("Hypothesis", ""), 200)
    findings_raw = sections.get("Findings", "") or sections.get("Conclusion", "")
    findings_short = short_field(findings_raw, 200)
    # conclusion_short kept for v3 readers; same content as findings_short.
    conclusion_short = findings_short

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    shared_row = {
        "agent_id": os.environ.get("AGENT_ID", "unknown"),
        "cluster_id": os.environ.get("CLUSTER_ID", "0"),
        "exp_id": exp_dir.name,
        "timestamp_utc": now,
        "strategy_short": strategy_short,
        "data_sources": args.data_sources[:200],
        "row_count": rows,
        "diversity_distinct_1g": _fmt(diversity.get("distinct_1g")),
        "diversity_distinct_4g": _fmt(diversity.get("distinct_4g")),
        "diversity_mean_cos_dist": _fmt(diversity.get("mean_cos_dist")),
        "diversity_len_cv": _fmt(diversity.get("len_cv")),
        "decontam_pass": decontam_pass,
        "dataset_sha256": data_sha,
        "parent_exp_id": parent_value,
        "hypothesis_short": hypothesis_short,
        "conclusion_short": conclusion_short,
        "audit_pass": audit_pass,
        # Promoted rows are the cross-run handoff signal — peers in future
        # runs look for promoted=True to find datasets worth forking.
        "promoted": effective_promoted,
        # V2 schema v4 additions. eval_before/eval_after are emitted as raw
        # strings (the agent's own numbers) — they intentionally bypass the
        # numeric scrubber that hides scores from peer free-text fields,
        # because V2's design decision is that the agent's own grounded
        # judgment is now visible. Peer rows still keep findings_short
        # scrubbed (no comparative phrases), and these structured columns
        # are the canonical place to look for "did it improve" signal.
        "eval_before": outcome_eval_before,
        "eval_after": outcome_eval_after,
        # V4 multi-seed summary. _fmt renders None -> "" and floats to 4dp,
        # matching the diversity columns; n_eval_seeds is a plain int. A
        # single-seed publish records mean=eval_after, std=0.0000, n=1.
        "eval_after_mean": _fmt(eval_after_mean),
        "eval_after_std": _fmt(eval_after_std),
        "n_eval_seeds": n_eval_seeds,
        "outcome_improved": outcome_improved,
        "findings_short": findings_short,
        "notes_excerpt": notes_excerpt,
    }
    reject_forbidden(shared_row)

    if shared_path:
        append_with_flock(
            Path(shared_path), SHARED_FIELDS, shared_row, versioned=True
        )
        print(f"[publish] shared row → {shared_path}")
    else:
        print(
            "[publish] SHARED_LOG_CSV unset; skipping shared write",
            file=sys.stderr,
        )

    local_row = {
        "exp_id": exp_dir.name,
        "started_at_utc": now,
        "strategy_short": strategy_short,
        "row_count": rows,
        "audit_pass": audit_pass,
        "promoted": effective_promoted,
        "dataset_sha256": data_sha,
        "parent_exp_id": parent_value,
        "hypothesis_short": hypothesis_short,
        # V2: outcome fields propagated into local index so backtrack
        # parent selection and --done plateau checks can read them
        # directly without re-parsing notes.md.
        "eval_before": outcome_eval_before,
        "eval_after": outcome_eval_after,
        "outcome_improved": outcome_improved,
    }
    reject_forbidden(local_row)
    append_with_flock(local_idx, LOCAL_FIELDS, local_row, versioned=False)
    print(f"[publish] local row → {local_idx}")

    if args.done:
        print(
            f"[publish] DONE accepted: {len(_read_csv_rows(local_idx))} "
            "experiments published, 10-experiment plateau confirmed.",
            file=sys.stderr,
        )

    # Persist promoted data for cross-run reuse.
    if effective_promoted and shared_path and data_sha:
        target = _persist_promoted_data(shared_path, data_path, data_sha)
        if target is not None:
            print(f"[publish] promoted data → {target}")

    # V3: write the .published marker. train_sft.py's V3 gate looks for
    # this file on the prior experiment to allow the next training run.
    try:
        (exp_dir / ".published").touch()
    except OSError as exc:
        print(
            f"[publish] warning: could not write .published marker to "
            f"{exp_dir / '.published'}: {exc}. The next train_sft.py call "
            "may refuse to start.",
            file=sys.stderr,
        )

    return 0


def _fmt(v: float | None) -> str:
    if v is None:
        return ""
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    sys.exit(main())
