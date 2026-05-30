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

SCHEMA_VERSION = "4"
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

    # V2: parse and validate the structured ## Outcome section.
    outcome = _validate_outcome_section(
        sections, audit_failed=args.audit_failed, exp_n=exp_n
    )
    outcome_improved = outcome.get("improved", "") if outcome else ""
    outcome_eval_before = outcome.get("eval_before", "") if outcome else ""
    outcome_eval_after = outcome.get("eval_after", "") if outcome else ""

    # V2: cross-check stated eval scores against eval_result.json on disk.
    if outcome:
        # V3: also refuse smoke evals — if eval_result.json reports fewer
        # samples than the task's full size, this isn't a valid plateau
        # signal. Task name comes from $EVALUATION_TASK if set.
        if eval_result_path.is_file():
            task_name = os.environ.get("EVALUATION_TASK") or None
            if task_name:
                task_name = task_name.lower().strip()
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
    # VERIFIED improvement, not just "passed gates".
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
