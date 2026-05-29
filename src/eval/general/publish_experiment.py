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

SCHEMA_VERSION = "3"
SCHEMA_HEADER_LINE = f"# schema_version={SCHEMA_VERSION}\n"

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
]

FORBIDDEN_PREFIXES = ("eval_", "score_")

REQUIRED_SECTIONS = ("Parent", "Hypothesis", "Method", "Conclusion")
OPTIONAL_SECTIONS = ("PivotReason",)
ALL_SECTIONS = REQUIRED_SECTIONS + OPTIONAL_SECTIONS

_SECTION_RE = re.compile(r"^##\s+(Parent|Hypothesis|Method|Conclusion|PivotReason)\s*$")

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
    bad = [k for k in d if any(k.startswith(p) for p in FORBIDDEN_PREFIXES)]
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
) -> str:
    """Validate the ## Parent value and return it.

    Two accepted forms:
    - Bare `exp_NNN` — same-agent parent. Must exist in the agent's own
      local index.csv. The shared CSV is NOT consulted for the bare form
      because exp_NNN is ambiguous across agents.
    - Slashed `<agent_id>/exp_NNN` — cross-agent (or own past) parent. Must
      have a matching (agent_id, exp_id) row in $SHARED_LOG_CSV.
    - `none` — valid only for exp_001, or for exp_002+ with ## PivotReason.
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

    if _check_value(parent_value):
        return parent_value
    _fail(parent_value)
    return parent_value  # unreachable; _fail raises


def _validate_required_sections(
    sections: dict[str, str],
    *,
    audit_failed: bool,
) -> None:
    """Enforce required sections. Audit-failed still needs Parent + Hypothesis."""
    if audit_failed:
        required = ("Parent", "Hypothesis", "Method", "Conclusion")
    else:
        required = REQUIRED_SECTIONS
    missing = [s for s in required if not sections.get(s, "").strip()]
    if missing:
        raise SystemExit(
            f"notes.md is missing required section(s): {missing}. "
            "Required headers (exact): "
            "## Parent, ## Hypothesis, ## Method, ## Conclusion "
            "(## PivotReason optional; required for cross-run pivots)."
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

    parent_value = validate_parent(exp_dir, sections, local_idx, shared_path)

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

    # All four free-text fields written to the shared CSV are scrubbed the
    # same way — strategy_short and notes_excerpt would otherwise let eval
    # scores or comparative score language leak through. extract_strategy
    # returns the raw first line of ## Hypothesis (or the notes body), so
    # we route it through short_field() (which scrub_numeric()s + truncates)
    # rather than emitting it raw.
    strategy_short = short_field(extract_strategy(notes, sections), 200)
    notes_excerpt = short_field(notes, 200)
    hypothesis_short = short_field(sections.get("Hypothesis", ""), 200)
    conclusion_short = short_field(sections.get("Conclusion", ""), 200)

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
        "promoted": bool(args.promoted),
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
        "promoted": bool(args.promoted),
        "dataset_sha256": data_sha,
        "parent_exp_id": parent_value,
        "hypothesis_short": hypothesis_short,
    }
    reject_forbidden(local_row)
    append_with_flock(local_idx, LOCAL_FIELDS, local_row, versioned=False)
    print(f"[publish] local row → {local_idx}")

    # Persist promoted data for cross-run reuse.
    if args.promoted and audit_pass and shared_path and data_sha:
        target = _persist_promoted_data(shared_path, data_path, data_sha)
        if target is not None:
            print(f"[publish] promoted data → {target}")

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
