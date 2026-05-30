#!/usr/bin/env python3
"""Teacher-vLLM reasoning-distillation + synthetic-MCQ helper (V4 headline tool).

WHY — REASONING DISTILLATION IS THE POINT. GPQA is scored by inspect_ai's
multiple_choice(cot=True) solver with a DEFAULT --max-tokens 16000: the model is
REWARDED for a long step-by-step chain of thought before a single 'ANSWER:
<LETTER>' line. So the lever is NOT memorising answers; it is teaching the small
base model to *think* like a strong reasoner. The teacher (MiniMax-M2.7) is a
strong reasoning model; we use it to produce REASONING TRACES and SFT the
student on them. The training target (the assistant message) carries the
reasoning, so the student learns the reasoning behaviour, not just the letter —
teacher CoT -> student weights. (A V3 bug trained on answer-only data because
self-eval used --max-tokens 128, which forbids reasoning; V4 keeps reasoning in
both the target and the eval.)

Two modes:
  RATIONALE (default) — distill teacher reasoning over EXISTING MCQ rows:
    python3 teacher_synth.py --mode rationale --in questions.jsonl --out distilled.jsonl
    Each input row (question + options) is re-solved by the teacher with a full
    CoT ending in 'ANSWER: <LETTER>'. USER content = MCQ in the exact eval
    template; ASSISTANT content = the teacher's reasoning trace.
  GENERATE — invent NOVEL graduate-level 4-option MCQs (when HF discovery is dry):
    python3 teacher_synth.py --mode generate --n 200 --topic "..." --out generated.jsonl
    Brand-new MCQs (NOT copied from any benchmark) with full reasoning +
    'ANSWER: <LETTER>', rotating subfields to avoid monoculture (the diversity
    audit in dataset_audit.py rejects monoculture).

Output schema (consumed by train_sft.py, gated by dataset_audit.py):
  {"messages":[{"role":"user","content":<MCQ in eval template>},
               {"role":"assistant","content":<reasoning ... ANSWER: X>}]}

Operational: ThreadPoolExecutor parallelism; rows written incrementally to
<out>.partial then atomically renamed to <out>; --resume skips rows already in
<out>.partial; endpoint unreachable -> nonzero exit; per-row failure -> log +
skip + continue (a slow/flaky batch must never make the agent rage-quit). Reads
TEACHER_VLLM_URL / TEACHER_MODEL_NAME / TEACHER_API_KEY from env. max_tokens
defaults to 2048 (reasoning needs room; matches train_sft MAX_SEQ_LEN ceiling);
temperature 0.2 rationale / 0.7 generate. Sample + read output before training
(does the rationale reach the claimed letter?) and run dataset_audit.py first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openai  # type: ignore
except Exception as exc:  # pragma: no cover - import guard
    sys.stderr.write(f"[teacher_synth] FATAL: could not import openai: {exc}\n")
    sys.exit(2)

LETTERS = ["A", "B", "C", "D"]
ANSWER_RE = re.compile(r"^\s*ANSWER:\s*([A-D])\s*$", re.IGNORECASE | re.MULTILINE)

# Subfields rotated in GENERATE mode so the teacher does not collapse onto its
# favourites (e.g. all thermodynamics). The diversity audit rejects monoculture.
SUBFIELDS = [
    "quantum mechanics", "electromagnetism", "statistical mechanics",
    "condensed matter physics", "particle / high-energy physics",
    "general relativity / astrophysics", "optics and photonics",
    "nuclear physics", "fluid dynamics", "thermodynamics",
    "organic chemistry reaction mechanisms", "inorganic / coordination chemistry",
    "physical chemistry / chemical kinetics", "quantum chemistry / spectroscopy",
    "analytical chemistry", "electrochemistry", "polymer chemistry",
    "molecular biology / gene regulation", "genetics and genomics",
    "biochemistry / enzyme kinetics", "cell biology / signaling",
    "immunology", "microbiology", "evolutionary biology / phylogenetics",
    "neuroscience", "structural biology",
]


# --- prompt construction -----------------------------------------------------
def build_eval_template(question: str, options: list[str]) -> str:
    """Render an MCQ in the exact format inspect_ai multiple_choice(cot=True) uses
    (instruction asking for 'ANSWER: $LETTER' + think step by step, then the
    question and lettered choices) so the student trains on the eval surface form."""
    letters = ", ".join(LETTERS[: len(options)])
    lines = [
        "Answer the following multiple choice question. The entire response "
        "should be in the format of 'ANSWER: $LETTER' (without quotes) where "
        f"LETTER is one of {letters}. Think step by step before answering.",
        "",
        question.strip(),
        "",
    ]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}) {str(opt).strip()}")
    return "\n".join(lines)


RATIONALE_SYSTEM = (
    "You are a meticulous graduate-level science tutor (physics, chemistry, "
    "biology). Solve the multiple-choice question with rigorous, explicit "
    "step-by-step reasoning: state relevant principles, do the quantitative "
    "work (unit analysis, derivations, order-of-magnitude checks) where "
    "useful, and eliminate distractors. Do NOT restate the question. End your "
    "response with a final line that is EXACTLY 'ANSWER: <LETTER>' where "
    "<LETTER> is one of A, B, C, D and nothing follows it."
)

GENERATE_SYSTEM = (
    "You are an expert exam author creating ORIGINAL graduate-level "
    "multiple-choice science questions. Your questions must be genuinely "
    "novel — DO NOT reproduce, paraphrase, or closely imitate any existing "
    "benchmark item (GPQA, MMLU, etc.). Each question has exactly four "
    "options (A-D) with one correct answer and three plausible, "
    "discriminating distractors. You always show full step-by-step worked "
    "reasoning before the answer."
)


def build_generate_user(subfield: str) -> str:
    return (
        f"Create ONE original, hard, graduate-level multiple-choice question in "
        f"the subfield: {subfield}.\n\n"
        "Requirements:\n"
        "- Exactly four answer options labelled A) B) C) D), one correct.\n"
        "- Graduate difficulty (quantitative or mechanistic, not trivia).\n"
        "- Must be your own invention, not copied from any known benchmark.\n\n"
        "Respond with VALID JSON only (no markdown fences), with this shape:\n"
        '{"question": "<the question stem, no option letters>",\n'
        ' "options": ["<A text>", "<B text>", "<C text>", "<D text>"],\n'
        ' "reasoning": "<full step-by-step reasoning that ends by naming the '
        'correct option>",\n'
        ' "answer": "<one of A B C D>"}'
    )


# --- input parsing -----------------------------------------------------------
def extract_question_options(row: dict) -> tuple[str, list[str]] | None:
    """Pull (question, [4 options]) from a heterogeneous input row.

    Accepts:
      * {"messages":[{"role":"user","content": <already eval-templated MCQ>}, ...]}
        -> parse the question + 'A) ... B) ...' lines back out.
      * {"question"/"Question": str, "options"/"choices": [..]} forms.
      * {"question": str, "Correct Answer"/"Incorrect Answer N": ...} forms.
    Returns None if it cannot recover a question + >=2 options.
    """
    # Case 1: messages row whose user content is an eval-templated MCQ.
    if isinstance(row.get("messages"), list):
        user = next(
            (m for m in row["messages"] if isinstance(m, dict) and m.get("role") == "user"),
            None,
        )
        if user and isinstance(user.get("content"), str):
            parsed = _parse_templated_mcq(user["content"])
            if parsed:
                return parsed

    # Case 2: explicit question + options/choices list.
    q = row.get("question") or row.get("Question") or row.get("prompt")
    opts = row.get("options") or row.get("choices")
    if isinstance(q, str) and isinstance(opts, list) and len(opts) >= 2:
        return q, [str(o) for o in opts[:4]]

    # Case 3: GPQA-style separate correct/incorrect fields.
    if isinstance(q, str) and row.get("Correct Answer") is not None:
        opts3 = [
            str(row.get("Correct Answer", "")),
            str(row.get("Incorrect Answer 1", "")),
            str(row.get("Incorrect Answer 2", "")),
            str(row.get("Incorrect Answer 3", "")),
        ]
        opts3 = [o for o in opts3 if o]
        if len(opts3) >= 2:
            return q, opts3

    return None


def _parse_templated_mcq(content: str) -> tuple[str, list[str]] | None:
    """Recover (question, options) from an eval-templated MCQ string."""
    opt_line = re.compile(r"^\s*([A-D])\)\s*(.+?)\s*$")
    options: list[str] = []
    q_lines: list[str] = []
    saw_option = False
    for line in content.splitlines():
        m = opt_line.match(line)
        if m:
            saw_option = True
            options.append(m.group(2))
            continue
        if saw_option:
            continue  # ignore trailing chatter after options begin
        # skip the boilerplate instruction line
        if line.strip().lower().startswith("answer the following multiple choice"):
            continue
        q_lines.append(line)
    question = "\n".join(q_lines).strip()
    if question and len(options) >= 2:
        return question, options[:4]
    return None


# --- teacher calls -----------------------------------------------------------
def make_client() -> "openai.OpenAI":
    base_url = os.environ.get("TEACHER_VLLM_URL")
    if not base_url:
        sys.stderr.write("[teacher_synth] FATAL: TEACHER_VLLM_URL is not set.\n")
        sys.exit(2)
    api_key = os.environ.get("TEACHER_API_KEY") or "EMPTY"
    return openai.OpenAI(base_url=base_url, api_key=api_key)


def teacher_model() -> str:
    model = os.environ.get("TEACHER_MODEL_NAME")
    if not model:
        sys.stderr.write("[teacher_synth] FATAL: TEACHER_MODEL_NAME is not set.\n")
        sys.exit(2)
    return model


def chat(client, model, system, user, *, max_tokens, temperature) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def healthcheck(client, model) -> None:
    """One tiny call so a dead endpoint fails fast with a nonzero exit."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=4,
            temperature=0.0,
        )
    except Exception as exc:
        sys.stderr.write(
            f"[teacher_synth] FATAL: teacher endpoint unreachable "
            f"({type(exc).__name__}: {exc}). Check TEACHER_VLLM_URL.\n"
        )
        sys.exit(3)


def ensure_answer_line(text: str) -> str | None:
    """Validate the trace ends with a parseable 'ANSWER: <LETTER>' line.

    Returns the normalised text (with a canonical final 'ANSWER: X' line) or
    None if no valid answer letter is present.
    """
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    letter = matches[-1].upper()
    if letter not in LETTERS:
        return None
    # Normalise: strip any trailing content after the last answer and append a
    # single canonical answer line so the student always sees the exact shape.
    body = ANSWER_RE.sub("", text).rstrip()
    return f"{body}\n\nANSWER: {letter}"


# --- per-row workers ---------------------------------------------------------
def do_rationale(client, model, row, *, max_tokens, temperature) -> dict | None:
    qo = extract_question_options(row)
    if qo is None:
        raise ValueError("could not extract question/options from row")
    question, options = qo
    user_mcq = build_eval_template(question, options)
    raw = chat(
        client, model, RATIONALE_SYSTEM, user_mcq,
        max_tokens=max_tokens, temperature=temperature,
    )
    trace = ensure_answer_line(raw)
    if trace is None:
        raise ValueError("teacher response had no parseable 'ANSWER: <LETTER>' line")
    return {
        "messages": [
            {"role": "user", "content": user_mcq},
            {"role": "assistant", "content": trace},
        ]
    }


def do_generate(client, model, subfield, *, max_tokens, temperature) -> dict | None:
    raw = chat(
        client, model, GENERATE_SYSTEM, build_generate_user(subfield),
        max_tokens=max_tokens, temperature=temperature,
    )
    obj = _loose_json(raw)
    if not isinstance(obj, dict):
        raise ValueError("teacher did not return a JSON object")
    question = obj.get("question")
    options = obj.get("options")
    reasoning = obj.get("reasoning") or ""
    answer = str(obj.get("answer", "")).strip().upper()[:1]
    if not (isinstance(question, str) and isinstance(options, list) and len(options) == 4):
        raise ValueError("generated item missing question or 4 options")
    if answer not in LETTERS:
        raise ValueError(f"generated item has invalid answer letter: {obj.get('answer')!r}")
    user_mcq = build_eval_template(question, [str(o) for o in options])
    assistant = ensure_answer_line(f"{reasoning.strip()}\n\nANSWER: {answer}")
    if assistant is None:
        raise ValueError("could not assemble a valid answer line for generated item")
    return {
        "messages": [
            {"role": "user", "content": user_mcq},
            {"role": "assistant", "content": assistant},
        ]
    }


def _loose_json(text: str) -> object:
    """Parse JSON that may be wrapped in markdown fences or prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # Fallback: grab the outermost {...} block.
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(t[start : end + 1])
    raise ValueError("no JSON object found in teacher response")


# --- IO / resume / atomic write ----------------------------------------------
def load_resume_keys(partial_path: str) -> set[str]:
    """Return the set of already-completed row keys from <out>.partial."""
    done: set[str] = set()
    if not os.path.exists(partial_path):
        return done
    with open(partial_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            user = next(
                (m["content"] for m in obj.get("messages", [])
                 if isinstance(m, dict) and m.get("role") == "user"),
                None,
            )
            if isinstance(user, str):
                done.add(user)
    return done


def read_input_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                sys.stderr.write(f"[teacher_synth] skipping malformed input line {n}: {exc}\n")
    return rows


# --- main run ----------------------------------------------------------------
def run(args) -> int:
    client = make_client()
    model = teacher_model()
    healthcheck(client, model)

    partial_path = args.out + ".partial"
    resume_keys = load_resume_keys(partial_path) if args.resume else set()
    if not args.resume and os.path.exists(partial_path):
        os.remove(partial_path)

    # Build the work list of (key, callable) jobs.
    jobs: list = []
    if args.mode == "rationale":
        if not args.in_path:
            sys.stderr.write("[teacher_synth] FATAL: --in is required in rationale mode.\n")
            return 2
        rows = read_input_rows(args.in_path)
        if not rows:
            sys.stderr.write("[teacher_synth] FATAL: no usable input rows.\n")
            return 2
        for row in rows:
            qo = extract_question_options(row)
            if qo is None:
                sys.stderr.write("[teacher_synth] skip: row missing question/options.\n")
                continue
            key = build_eval_template(qo[0], qo[1])
            if key in resume_keys:
                continue
            jobs.append((key, row))
        worker = lambda r: do_rationale(  # noqa: E731
            client, model, r, max_tokens=args.max_tokens, temperature=args.temperature
        )
    else:  # generate
        for i in range(args.n):
            subfield = SUBFIELDS[i % len(SUBFIELDS)]
            jobs.append((f"gen::{i}::{subfield}", subfield))
        worker = lambda s: do_generate(  # noqa: E731
            client, model, s, max_tokens=args.max_tokens, temperature=args.temperature
        )

    total = len(jobs)
    if total == 0:
        sys.stderr.write("[teacher_synth] nothing to do (all rows resumed or empty).\n")
        # Still produce <out> by promoting any existing partial.
        if os.path.exists(partial_path):
            os.replace(partial_path, args.out)
        return 0

    sys.stderr.write(
        f"[teacher_synth] mode={args.mode} jobs={total} "
        f"workers={args.max_workers} resumed={len(resume_keys)}\n"
    )

    write_lock = threading.Lock()
    done = ok = failed = 0
    out_fh = open(partial_path, "a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futs = {pool.submit(worker, payload): key for key, payload in jobs}
            for fut in as_completed(futs):
                done += 1
                try:
                    result = fut.result()
                    if result is not None:
                        with write_lock:
                            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                            out_fh.flush()
                        ok += 1
                    else:
                        failed += 1
                except Exception as exc:
                    failed += 1
                    sys.stderr.write(
                        f"[teacher_synth] row failed ({type(exc).__name__}: {exc})\n"
                    )
                if done % max(1, args.progress_every) == 0 or done == total:
                    sys.stderr.write(
                        f"[teacher_synth] progress {done}/{total} "
                        f"ok={ok} failed={failed}\n"
                    )
    finally:
        out_fh.close()

    if ok == 0:
        sys.stderr.write(
            "[teacher_synth] FATAL: produced zero rows (all jobs failed). "
            "Leaving .partial for inspection; not promoting to <out>.\n"
        )
        return 4

    os.replace(partial_path, args.out)  # atomic rename
    sys.stderr.write(
        f"[teacher_synth] DONE: wrote {ok} rows to {args.out} (failed={failed}).\n"
    )
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Teacher-vLLM reasoning-distillation / synthetic-MCQ helper.",
        allow_abbrev=False,
    )
    p.add_argument("--mode", choices=["rationale", "generate"], default="rationale")
    p.add_argument("--in", dest="in_path", default=None,
                   help="Input JSONL of MCQ rows (rationale mode).")
    p.add_argument("--out", required=True, help="Output JSONL of training rows.")
    p.add_argument("--n", type=int, default=200,
                   help="Number of MCQs to invent (generate mode).")
    p.add_argument("--topic", default="graduate physics/chemistry/biology",
                   help="High-level topic guidance (generate mode).")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Reasoning needs room; matches train_sft MAX_SEQ_LEN ceiling.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Default 0.2 (rationale) / 0.7 (generate).")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress to stderr every N completed rows.")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in <out>.partial.")
    args = p.parse_args()
    if args.temperature is None:
        args.temperature = 0.2 if args.mode == "rationale" else 0.7
    if args.max_workers < 1:
        p.error("--max-workers must be >= 1")
    if args.mode == "generate" and args.n < 1:
        p.error("--n must be >= 1 in generate mode")
    return args


def main() -> None:
    args = parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
