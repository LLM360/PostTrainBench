#!/usr/bin/env python3
"""Visualize experiment lineage from shared_log.csv as an ASCII tree.

Groups rows by agent_id and draws per-agent trees linked via parent_exp_id.
Tolerant of legacy rows missing the v2 columns (parent_exp_id, hypothesis_short,
conclusion_short, audit_pass) — those are rendered as a flat [legacy] list.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from collections import defaultdict

DEFAULT_CSV = "results/data_eng_shared/gpqamain_Qwen_Qwen3-1.7B-Base/shared_log.csv"


def parse_csv(path):
    """Read CSV skipping `#` comment lines. Returns list[dict]."""
    if not os.path.exists(path):
        print(f"error: csv not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return [r for r in reader if r.get("agent_id")]


def truthy(v):
    return str(v).strip().lower() in ("true", "t", "1", "yes", "y", "pass")


def wrap_block(label, text, width, indent):
    """Wrap `text` with continuation lines indented under `indent`."""
    text = (text or "").strip().replace("\n", " ").replace("\r", " ")
    if not text:
        return []
    first = f"{indent}{label}: "
    body_width = max(20, width - len(first))
    wrapped = textwrap.wrap(text, width=body_width) or [""]
    out = [first + wrapped[0]]
    pad = " " * len(first)
    for line in wrapped[1:]:
        out.append(pad + line)
    return out


def render_node(row, depth, last_stack, width, xagent_note=""):
    """Render a single node with proper tree connectors."""
    # Build the tree prefix from last_stack (list of bools for each ancestor depth)
    prefix = ""
    for is_last_ancestor in last_stack[:-1]:
        prefix += "    " if is_last_ancestor else "│   "
    if last_stack:
        prefix += "└── " if last_stack[-1] else "├── "

    audit = "T" if truthy(row.get("audit_pass", "")) else (
        "F" if row.get("audit_pass") not in (None, "") else "?")
    promoted = "T" if truthy(row.get("decontam_pass", "")) else "F"
    legacy_tag = " [legacy]" if row.get("_legacy") else ""
    orphan_tag = row.get("_orphan_tag", "")
    xa = f" {xagent_note}" if xagent_note else ""
    strategy = (row.get("strategy_short") or "").strip() or "(no strategy)"

    header = (f"{prefix}{row['exp_id']} [audit_pass={audit}, promoted={promoted}]"
              f"{legacy_tag}{orphan_tag}{xa} strategy={strategy}")
    # Wrap header if too wide
    if len(header) > width:
        hwrap = textwrap.wrap(header, width=width,
                              subsequent_indent=" " * (len(prefix) + 4))
        lines = hwrap
    else:
        lines = [header]

    # Continuation indent for hypothesis/conclusion
    cont_prefix = ""
    for is_last_ancestor in last_stack[:-1]:
        cont_prefix += "    " if is_last_ancestor else "│   "
    if last_stack:
        cont_prefix += "    " if last_stack[-1] else "│   "
    cont_prefix += "    "

    lines.extend(wrap_block("hypothesis", row.get("hypothesis_short", ""), width, cont_prefix))
    lines.extend(wrap_block("conclusion", row.get("conclusion_short", ""), width, cont_prefix))
    return lines


def build_and_render(rows, width):
    """Build per-agent trees.

    Every agent gets its own section (even if it has no roots — e.g. all its
    rows hang off another agent's experiment). Cross-agent parents are
    rendered IN THE CHILD'S OWN SECTION as a reference marker — never by
    attaching the child under the foreign parent's section. That way no row
    is silently moved out of its agent's section.

    Visited tracking is per-rendered-tree (not global), so the same node can
    appear in multiple agents' sections when referenced cross-agent. A
    separate parent-chain cycle detector breaks loops within a single tree.
    """
    # Index all (agent_id, exp_id) tuples
    nodes = {(r["agent_id"], r["exp_id"]): r for r in rows}
    # children_of[(agent, exp)] -> list of (child_row, parent_was_xref:bool)
    # Children are kept under their *resolved* parent key but we only render
    # them inside the CHILD'S own agent section. Foreign-parent edges are
    # rendered as references, not as tree children.
    children_of = defaultdict(list)
    roots_by_agent = defaultdict(list)
    parent_counts = defaultdict(int)
    agents_with_rows = set()
    xagent_forks = 0
    orphan_count = 0

    for r in rows:
        agents_with_rows.add(r["agent_id"])
        pid = (r.get("parent_exp_id") or "").strip()
        if not pid or pid.lower() == "none":
            roots_by_agent[r["agent_id"]].append(r)
            continue
        # parent is scoped per agent unless explicit "agent/exp" form is used
        if "/" in pid:
            pa, pe = pid.split("/", 1)
        else:
            pa, pe = r["agent_id"], pid

        if (pa, pe) in nodes:
            parent_counts[(pa, pe)] += 1
            if pa != r["agent_id"]:
                # Cross-agent parent: keep this row as a ROOT in its OWN
                # agent's section, tagged with a [xref] marker. Do not
                # attach it under the foreign agent's tree.
                r["_xref_parent"] = f"{pa}/{pe}"
                roots_by_agent[r["agent_id"]].append(r)
                xagent_forks += 1
            else:
                # Same-agent parent: standard tree edge.
                children_of[(pa, pe)].append(r)
        else:
            r["_orphan_tag"] = f" [orphan parent: {pid}]"
            roots_by_agent[r["agent_id"]].append(r)
            orphan_count += 1

    out_lines = []
    cycle_count = 0

    def dfs(row, last_stack, visited_in_tree):
        """Render a same-agent subtree. visited_in_tree is per-tree, so the
        same exp can appear in multiple sections; cycles within one tree are
        broken with a clear marker."""
        nonlocal cycle_count
        key = (row["agent_id"], row["exp_id"])
        if key in visited_in_tree:
            cycle_count += 1
            out_lines.append(f"  !! cycle detected at {key}; breaking")
            return
        visited_in_tree = visited_in_tree | {key}
        xref_note = ""
        if row.get("_xref_parent"):
            xref_note = f"[xref parent: {row['_xref_parent']}]"
        out_lines.extend(render_node(row, len(last_stack), last_stack, width, xref_note))
        kids = children_of.get(key, [])
        for i, child in enumerate(kids):
            is_last = (i == len(kids) - 1)
            dfs(child, last_stack + [is_last], visited_in_tree)

    # Build a section for EVERY agent that has rows, even if all its rows
    # are cross-agent children (no roots in the strict sense). Such agents
    # will have entries in roots_by_agent because xref children become
    # roots in their own section.
    for agent in sorted(agents_with_rows):
        agent_rows = [r for r in rows if r["agent_id"] == agent]
        out_lines.append("")
        out_lines.append(f"=== {agent} ({len(agent_rows)} experiments) ===")
        roots = roots_by_agent.get(agent, [])
        if not roots:
            # Shouldn't normally happen now (all rows surface as roots or
            # children within their own agent), but guard against it.
            out_lines.append("  (no roots — every row attached cross-agent and was lost)")
            continue
        for root in roots:
            dfs(root, [], set())

    # Footer
    total = len(rows)
    promoted = sum(1 for r in rows if truthy(r.get("decontam_pass", "")))
    audit_pass = sum(1 for r in rows if truthy(r.get("audit_pass", "")))
    most_forked = max(parent_counts.items(), key=lambda kv: kv[1], default=None)
    out_lines.append("")
    out_lines.append("---")
    out_lines.append(f"Total experiments: {total}    Promoted: {promoted}    Audit-pass: {audit_pass}")
    out_lines.append(f"Cross-agent forks: {xagent_forks}")
    if most_forked:
        (pa, pe), n = most_forked
        out_lines.append(f"Most-forked parent: {pa}/{pe} ({n} children)")
    else:
        out_lines.append("Most-forked parent: (none)")
    out_lines.append(f"Orphan/missing-parent rows: {orphan_count}")
    if cycle_count:
        out_lines.append(f"Cycles broken during render: {cycle_count}")
    return out_lines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path to shared_log.csv")
    ap.add_argument("--agent", default=None, help="substring filter on agent_id")
    ap.add_argument("--width", type=int, default=80, help="wrap width")
    args = ap.parse_args()

    rows = parse_csv(args.csv)
    if not rows:
        print("No experiments published yet")
        return 0

    # Detect legacy rows (missing v2 columns)
    for r in rows:
        if "parent_exp_id" not in r or r.get("parent_exp_id") is None:
            r["_legacy"] = True
            r["parent_exp_id"] = "none"
        else:
            r["_legacy"] = False

    if args.agent:
        rows = [r for r in rows if args.agent in r["agent_id"]]
        if not rows:
            print(f"No experiments match agent filter: {args.agent}")
            return 0

    for line in build_and_render(rows, args.width):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
