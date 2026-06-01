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
    audit in dataset_audit.py rejects monoculture). The teacher replies in the
    SAME line-anchored delimiter format the rationale path round-trips
    (QUESTION: / A) B) C) D) / REASONING: / ANSWER: <LETTER>), NOT strict JSON —
    the V4 pilot's JSON path kept 0 rows because MiniMax-M2.7 emits a long
    <think>...</think> preamble + prose that never parsed. We strip <think>,
    reuse the rationale parser/ANSWER_RE, then SELF-CONSISTENCY VERIFY: the
    teacher re-solves each item BLIND (gold hidden) and the item is kept only if
    the blind letter matches the generated gold. Optional --avoid <jsonl> drops
    near-duplicates against an existing pool; --difficulty-filter is a documented
    no-op stub (needs a base-model endpoint).

THINKING-ONLY TARGET SHAPE (BOTH modes, emitted by default): the assistant
target is always
  '<think>\n{reasoning}\n</think>\n\nANSWER: X'
— exactly ONE 'ANSWER: <LETTER>' line, AFTER </think>, and NO 'ANSWER:' anywhere
inside <think>. The eval scorer takes the FIRST ^ANSWER: match while the audit
guard takes the LAST, so a stray in-think ANSWER would pass audit but mis-score
at eval; ensure_answer_line scrubs every ANSWER line out of the <think> body and
re-emits a single canonical one after </think>. By default (--require-think) a
teacher trace with no closed </think> is REJECTED (one retry, then the row is
skipped — we do NOT synthesize a <think> wrapper around a trace that lacks one);
with --no-require-think the bare reasoning is WRAPPED in <think>...</think>
instead of rejected. enforce_seq_budget tail-trims
the <think> body (never the ANSWER line) so prompt + target fits MAX_SEQ_LEN=8192
(TRL right-truncates the tail, which would otherwise drop the ANSWER line).

Output schema (consumed by train_sft.py, gated by dataset_audit.py):
  {"messages":[{"role":"user","content":<MCQ in eval template>},
               {"role":"assistant","content":"<think>\\n...\\n</think>\\n\\nANSWER: X"}]}

Operational: ThreadPoolExecutor parallelism; rows written incrementally to
<out>.partial then atomically renamed to <out>; --resume skips rows already in
<out>.partial; endpoint unreachable -> nonzero exit; per-row failure -> log +
skip + continue (a slow/flaky batch must never make the agent rage-quit). Reads
TEACHER_VLLM_URL / TEACHER_MODEL_NAME / TEACHER_API_KEY from env. max_tokens
defaults to 6144 (the reasoning think-budget, sized so prompt + <think> + ANSWER
fits train_sft MAX_SEQ_LEN=8192; the per-call cap is reduced dynamically to
leave room for the prompt + ANSWER line); temperature 0.2 rationale / 0.7
generate. Sample + read output before training
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
# train_sft renders prompt+target with the model's chat template and feeds the
# whole sequence (including <think>) into the loss; TRL right-truncates the TAIL,
# so an over-budget target loses its trailing ANSWER line. Keep prompt+<think>+
# ANSWER under this ceiling (see enforce_seq_budget).
MAX_SEQ_LEN = 8192
ANSWER_RE = re.compile(r"^\s*ANSWER:\s*([A-D])\s*$", re.IGNORECASE | re.MULTILINE)
# Superset of the eval scorer's first-match ANSWER pattern, plus indented
# variants and any trailing junk after the colon. Used to SCRUB stray ANSWER
# lines out of the <think> body so the eval scorer (which takes the FIRST
# ^ANSWER:) and the audit guard (which takes the LAST) can never disagree.
ANSWER_LINE_LOOSE_RE = re.compile(r"^[ \t]*ANSWER[ \t]*:.*$", re.IGNORECASE | re.MULTILINE)
# Strip the teacher's chain-of-thought preamble before parsing. MiniMax-M2.7
# wraps its scratch work in <think>...</think>; in the V4 pilot that preamble
# exhausted max_tokens before any JSON appeared, so every generate parse failed.
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# Anchors for the line-anchored generate format (QUESTION: / REASONING:).
QUESTION_HDR_RE = re.compile(r"^\s*QUESTION:\s*", re.IGNORECASE)
REASONING_HDR_RE = re.compile(r"^\s*REASONING:\s*", re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (and any dangling, unclosed opener)."""
    t = THINK_RE.sub("", text)
    # Tolerant fallback: an UNCLOSED <think> (truncated by max_tokens) — drop
    # everything up to and including the last opener so the delimiters survive.
    low = t.lower()
    idx = low.rfind("<think>")
    if idx != -1:
        t = t[idx + len("<think>"):]
    return t.strip()

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
    "useful, and eliminate distractors. Do NOT restate the question. Wrap ALL "
    "of your reasoning in a single <think>...</think> block. After the closing "
    "</think>, write a final line that is EXACTLY 'ANSWER: <LETTER>' where "
    "<LETTER> is one of A, B, C, D and nothing follows it. NEVER write "
    "'ANSWER:' anywhere inside the <think> block — the answer line appears only "
    "once, after </think>."
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


def build_generate_user(subfield: str, topic: str | None = None) -> str:
    # Use the SAME line-anchored delimiter shape the rationale path already
    # round-trips (plain question text + 'A) ...' option lines + a final
    # 'ANSWER: <LETTER>' line, parsed by _parse_templated_mcq + ANSWER_RE).
    # Strict JSON failed in the V4 pilot because the teacher emits a long
    # <think>...</think> preamble + prose, so no JSON ever parsed. Delimiters
    # survive that prose: we strip <think> and key off the anchors.
    #
    # `topic` is the high-level domain guidance from --topic; `subfield` is the
    # rotated SUBFIELDS entry that keeps the batch diverse. We constrain the
    # question to the subfield (for diversity) WITHIN the requested topic so
    # --topic actually steers generation instead of being ignored.
    if topic:
        scope = (
            f"the subfield: {subfield} (within the broader topic area: {topic})"
        )
    else:
        scope = f"the subfield: {subfield}"
    return (
        f"Create ONE original, hard, graduate-level multiple-choice question in "
        f"{scope}.\n\n"
        "Requirements:\n"
        "- Exactly four answer options labelled A) B) C) D), one correct.\n"
        "- Graduate difficulty (quantitative or mechanistic, not trivia).\n"
        "- Must be your own invention, not copied from any known benchmark.\n\n"
        "Respond using EXACTLY this line-anchored format (no markdown, no JSON):\n"
        "QUESTION: <the question stem on one or more lines, no option letters>\n"
        "A) <option A text>\n"
        "B) <option B text>\n"
        "C) <option C text>\n"
        "D) <option D text>\n"
        "REASONING: <full step-by-step reasoning that ends by naming the "
        "correct option>\n"
        "ANSWER: <one of A B C D>\n"
        "The final line MUST be exactly 'ANSWER: <LETTER>' and nothing after it."
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


def ensure_answer_line(text: str, require_think: bool = True) -> str | None:
    """Normalise a trace into the canonical THINKING-ONLY target shape and the
    single choke point used by BOTH rationale and generate modes.

    Canonical shape (exactly):
        '<think>\\n{reasoning}\\n</think>\\n\\nANSWER: {letter}'
    Guarantees exactly ONE 'ANSWER: <LETTER>' line, placed AFTER </think>, with
    NO 'ANSWER:' anywhere inside <think>. (The eval scorer takes the FIRST
    ^ANSWER: match while the audit guard takes the LAST — so a stray in-think
    ANSWER passes audit but mis-scores at eval; we scrub it upstream here.)

    When `require_think` (default, matches --require-think): a trace with NO
    '</think>' (missing closer / unclosed <think>) is REJECTED — return None.
    Per LOCKED decision the caller retries once then SKIPS on None; we do NOT
    synthesize a <think> wrapper for a teacher trace that lacks one.

    When `require_think` is False (--no-require-think): a trace lacking a closed
    </think> is NOT rejected; instead the bare reasoning (everything before the
    final ANSWER line, with any stray ANSWER lines scrubbed) is WRAPPED in a
    '<think>...</think>' block so the output is still the canonical shape.

    Returns the canonical text, or None if there is no valid A-D answer letter.
    """
    low = text.lower()
    close_idx = low.rfind("</think>")
    if close_idx == -1:
        if require_think:
            # Missing closer / unclosed <think>: reject (caller retries -> skip).
            return None
        # --no-require-think: synthesize a <think> wrapper around the bare
        # reasoning. Treat the whole text as the think body: everything up to a
        # final ANSWER line is the reasoning; the last A-D ANSWER match is the
        # letter. Drop any dangling unclosed '<think>' opener from the body.
        all_matches = ANSWER_RE.findall(text)
        if not all_matches:
            return None
        letter = all_matches[-1].upper()
        if letter not in LETTERS:
            return None
        think_inner = ANSWER_LINE_LOOSE_RE.sub("", text)
        # Strip any dangling/unclosed <think> opener tags from the body.
        think_inner = re.sub(r"(?i)</?think>", "", think_inner)
        return f"<think>\n{think_inner.strip()}\n</think>\n\nANSWER: {letter}"
    open_idx = low.find("<think>")
    inner_start = open_idx + len("<think>") if open_idx != -1 else 0
    think_inner = text[inner_start:close_idx]
    post = text[close_idx + len("</think>"):]

    # Letter = LAST ANSWER_RE match in the post-think region; else last in the
    # full text. None if no A-D letter at all.
    post_matches = ANSWER_RE.findall(post)
    if post_matches:
        letter = post_matches[-1].upper()
    else:
        all_matches = ANSWER_RE.findall(text)
        if not all_matches:
            return None
        letter = all_matches[-1].upper()
    if letter not in LETTERS:
        return None

    # SCRUB every ANSWER:-style line out of the think body so the only ANSWER
    # line in the target is the canonical one after </think>.
    think_inner = ANSWER_LINE_LOOSE_RE.sub("", think_inner)
    return f"<think>\n{think_inner.strip()}\n</think>\n\nANSWER: {letter}"


# Lazily-initialised tokenizer cache for enforce_seq_budget (None = not yet
# attempted; False = attempted and unavailable/offline -> use chars/4 fallback).
_TOKENIZER = None


def _get_tokenizer():
    """Lazily load the student tokenizer (Qwen/Qwen3-1.7B-Base). Returns the
    tokenizer, or None if transformers is missing / the model is unavailable
    offline (caller then falls back to a chars/4 token estimate)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer  # type: ignore

            _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
        except Exception:
            _TOKENIZER = False
    return _TOKENIZER or None


def _count_tokens(text: str) -> int:
    """Token count via the student tokenizer, or a chars/4 estimate offline."""
    tok = _get_tokenizer()
    if tok is None:
        return (len(text) + 3) // 4
    try:
        return len(tok.encode(text))
    except Exception:
        return (len(text) + 3) // 4


def _prompt_token_len(user_prompt: str) -> int:
    """Tokens consumed by the rendered prompt. Uses the chat template when the
    tokenizer is available so the budget reflects what train_sft actually sees;
    falls back to a chars/4 estimate of the raw prompt offline."""
    tok = _get_tokenizer()
    if tok is None:
        return (len(user_prompt) + 3) // 4
    try:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(ids)
    except Exception:
        try:
            return len(tok.encode(user_prompt))
        except Exception:
            return (len(user_prompt) + 3) // 4


def enforce_seq_budget(
    user_prompt: str,
    assistant: str,
    max_seq_len: int = MAX_SEQ_LEN,
    margin: int = 64,
) -> str:
    """Keep prompt + assistant target under the train_sft sequence ceiling by
    TAIL-TRIMMING the <think> body only — the trailing 'ANSWER: <LETTER>' line
    is NEVER the casualty (TRL right-truncates the tail, so a dropped ANSWER
    would silently corrupt the label). The reasoning is trimmed from its end and
    the block is reclosed as '\\n</think>\\n\\nANSWER: {letter}'.

    Returns the (possibly trimmed) assistant target. If assistant is not in the
    canonical <think>...</think> + ANSWER shape (e.g. ensure_answer_line already
    returned None upstream) it is returned unchanged.
    """
    low = assistant.lower()
    open_idx = low.find("<think>")
    close_idx = low.rfind("</think>")
    if open_idx == -1 or close_idx == -1:
        return assistant
    post = assistant[close_idx + len("</think>"):]
    ans_matches = ANSWER_RE.findall(post)
    if not ans_matches:
        return assistant
    letter = ans_matches[-1].upper()
    inner_start = open_idx + len("<think>")
    think_inner = assistant[inner_start:close_idx].strip()

    prompt_tokens = _prompt_token_len(user_prompt)
    budget = max_seq_len - prompt_tokens - margin
    if budget < 1:
        budget = 1

    def assemble(inner: str) -> str:
        return f"<think>\n{inner}\n</think>\n\nANSWER: {letter}"

    if _count_tokens(assemble(think_inner)) <= budget:
        return assemble(think_inner)

    # Binary-search the longest think-body prefix that still fits the budget,
    # working in tokens via the tokenizer (or chars/4 fallback). Always reclose
    # so the ANSWER line survives.
    tok = _get_tokenizer()
    if tok is not None:
        try:
            ids = tok.encode(think_inner)
        except Exception:
            ids = None
    else:
        ids = None

    if ids is not None:
        lo, hi = 0, len(ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            try:
                cand_inner = tok.decode(ids[:mid]).strip()
            except Exception:
                cand_inner = think_inner[: mid * 4].strip()
            if _count_tokens(assemble(cand_inner)) <= budget:
                lo = mid
            else:
                hi = mid - 1
        try:
            trimmed = tok.decode(ids[:lo]).strip()
        except Exception:
            trimmed = think_inner[: lo * 4].strip()
        return assemble(trimmed)

    # chars/4 fallback: shrink the char prefix until it fits.
    lo, hi = 0, len(think_inner)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _count_tokens(assemble(think_inner[:mid].strip())) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return assemble(think_inner[:lo].strip())


# --- per-row workers ---------------------------------------------------------
def do_rationale(client, model, row, *, max_tokens, temperature,
                 max_seq_len=MAX_SEQ_LEN, require_think=True) -> dict | None:
    qo = extract_question_options(row)
    if qo is None:
        raise ValueError("could not extract question/options from row")
    question, options = qo
    user_mcq = build_eval_template(question, options)

    # Size the think-budget so prompt + <think> + ANSWER fits MAX_SEQ_LEN.
    prompt_tokens = _prompt_token_len(user_mcq)
    chat_max_tokens = min(max_tokens, max(256, max_seq_len - prompt_tokens - 48))

    # The teacher emits its OWN <think>...</think> (per RATIONALE_SYSTEM); pass
    # the raw response straight into the think-aware choke point. With
    # require_think (default) a bad trace (missing </think>) -> ensure_answer_line
    # None -> ONE retry, then skip. With --no-require-think the bare reasoning is
    # wrapped in <think> instead of rejected (so a missing closer is tolerated).
    trace = None
    for _attempt in range(2):
        raw = chat(
            client, model, RATIONALE_SYSTEM, user_mcq,
            max_tokens=chat_max_tokens, temperature=temperature,
        )
        trace = ensure_answer_line(raw, require_think=require_think)
        if trace is not None:
            break
    if trace is None:
        raise ValueError(
            "teacher response had no closed <think>/parseable 'ANSWER: <LETTER>' "
            "line after one retry"
        )
    trace = enforce_seq_budget(user_mcq, trace, max_seq_len=max_seq_len)
    return {
        "messages": [
            {"role": "user", "content": user_mcq},
            {"role": "assistant", "content": trace},
        ]
    }


def blind_resolve_letter(client, model, question, options, *, max_tokens) -> str | None:
    """Re-solve a generated MCQ BLIND (gold never shown) for self-consistency.

    Renders the same eval template + RATIONALE_SYSTEM the student trains on,
    asks the teacher to solve it cold, and returns the parsed ANSWER letter (or
    None if the teacher produced no parseable answer). Low temperature so the
    verdict is the teacher's most-confident answer, not a sample.
    """
    user_mcq = build_eval_template(question, options)
    raw = chat(
        client, model, RATIONALE_SYSTEM, user_mcq,
        max_tokens=max_tokens, temperature=0.0,
    )
    matches = ANSWER_RE.findall(strip_think(raw))
    if not matches:
        return None
    letter = matches[-1].upper()
    return letter if letter in LETTERS else None


def do_generate(client, model, subfield, *, max_tokens, temperature,
                avoid_keys=None, self_consistency=True,
                max_seq_len=MAX_SEQ_LEN, topic=None, require_think=True) -> dict | None:
    raw = chat(
        client, model, GENERATE_SYSTEM, build_generate_user(subfield, topic=topic),
        max_tokens=max_tokens, temperature=temperature,
    )
    question, options, reasoning, answer = parse_generated_item(raw)
    options = [str(o) for o in options]

    # (c) Dedup against an existing pool (normalized question text).
    if avoid_keys is not None and normalize_question(question) in avoid_keys:
        raise ValueError("generated item duplicates an item in --avoid pool")

    # (b) Self-consistency: teacher must re-derive the SAME letter blind, else
    # the item's gold is unreliable and we drop it. Blind re-solve uses the FULL
    # budget (thinking allowed; we only read the parsed letter).
    if self_consistency:
        verdict = blind_resolve_letter(
            client, model, question, options, max_tokens=max_tokens,
        )
        if verdict is None:
            raise ValueError("self-consistency: blind re-solve produced no answer")
        if verdict != answer:
            raise ValueError(
                f"self-consistency: blind re-solve {verdict} != generated gold {answer}"
            )

    user_mcq = build_eval_template(question, options)
    # parse_generated_item returns plain-prose REASONING (the generate format has
    # no <think> tags); WRAP it so the target is THINKING-ONLY shaped, then run
    # it through the same think-aware choke point both modes share. The wrapper
    # always produces a closed </think>, so require_think is satisfied regardless
    # of the flag; we thread it through for a consistent contract.
    assistant = ensure_answer_line(
        f"<think>\n{reasoning.strip()}\n</think>\n\nANSWER: {answer}",
        require_think=require_think,
    )
    if assistant is None:
        raise ValueError("could not assemble a valid answer line for generated item")
    assistant = enforce_seq_budget(user_mcq, assistant, max_seq_len=max_seq_len)
    return {
        "messages": [
            {"role": "user", "content": user_mcq},
            {"role": "assistant", "content": assistant},
        ]
    }


def parse_generated_item(raw: str) -> tuple[str, list[str], str, str]:
    """Parse a generate-mode teacher response in the line-anchored delimiter
    format (QUESTION: / A) B) C) D) / REASONING: / ANSWER: <LETTER>).

    Returns (question, options, reasoning, answer_letter). Reuses the existing
    primitives: _parse_templated_mcq recovers (question, options) from the
    'QUESTION: ... A) ... D) ...' block, ANSWER_RE recovers the gold letter.
    Strips <think>...</think> first; raises ValueError if it cannot recover a
    well-formed 4-option item with a valid answer letter.
    """
    text = strip_think(raw)
    if not text:
        raise ValueError("empty teacher response after stripping <think>")

    # Recover the answer letter (last ANSWER: line wins, same as ensure_answer_line).
    ans_matches = ANSWER_RE.findall(text)
    if not ans_matches:
        raise ValueError("no parseable 'ANSWER: <LETTER>' line in generated item")
    answer = ans_matches[-1].upper()
    if answer not in LETTERS:
        raise ValueError(f"generated item has invalid answer letter: {answer!r}")

    # Split QUESTION/options block from the REASONING block on the anchors.
    q_block_lines: list[str] = []
    reasoning_lines: list[str] = []
    in_reasoning = False
    for line in text.splitlines():
        if not in_reasoning and REASONING_HDR_RE.match(line):
            in_reasoning = True
            reasoning_lines.append(REASONING_HDR_RE.sub("", line, count=1))
            continue
        if in_reasoning:
            # Drop the trailing ANSWER: line out of the reasoning body; it is
            # re-appended canonically by ensure_answer_line downstream.
            if ANSWER_RE.match(line):
                continue
            reasoning_lines.append(line)
            continue
        # In the QUESTION/options region: drop the 'QUESTION:' header word so
        # _parse_templated_mcq sees plain question text + 'A) ...' option lines.
        q_block_lines.append(QUESTION_HDR_RE.sub("", line, count=1))

    qo = _parse_templated_mcq("\n".join(q_block_lines))
    if qo is None or len(qo[1]) < 4:
        raise ValueError("generated item missing question or 4 options")
    question, options = qo[0], qo[1][:4]
    reasoning = "\n".join(reasoning_lines).strip()
    return question, options, reasoning, answer


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


def normalize_question(question: str) -> str:
    """Normalize a question stem for dedup: lowercase, collapse whitespace,
    drop non-alphanumerics. Matches near-duplicate phrasings the V4 discovery
    monoculture produced (e.g. repeated 'graduate ... multiple choice exam')."""
    return re.sub(r"[^a-z0-9]+", " ", str(question).lower()).strip()


def load_avoid_keys(path: str) -> set[str]:
    """Build a set of normalized question stems from an existing pool JSONL so
    generate mode can skip near-duplicates. Handles both the messages schema
    (parse the user MCQ back to a question) and explicit question/options rows;
    malformed lines are skipped."""
    keys: set[str] = set()
    if not path:
        return keys
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            qo = extract_question_options(row)
            if qo is not None:
                keys.add(normalize_question(qo[0]))
    return keys


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
            client, model, r, max_tokens=args.max_tokens,
            temperature=args.temperature, max_seq_len=args.max_seq_len,
            require_think=args.require_think,
        )
    else:  # generate
        avoid_keys = load_avoid_keys(args.avoid) if args.avoid else None
        if args.avoid:
            sys.stderr.write(
                f"[teacher_synth] loaded {len(avoid_keys or [])} avoid keys "
                f"from {args.avoid}\n"
            )
        if args.difficulty_filter:
            # (d) Documented no-op stub: a true difficulty filter needs a
            # base-model endpoint to score student-solvability; not wired here.
            sys.stderr.write(
                "[teacher_synth] --difficulty-filter is a documented no-op stub "
                "(needs a base-model endpoint); self-consistency is the active gate.\n"
            )
        for i in range(args.n):
            subfield = SUBFIELDS[i % len(SUBFIELDS)]
            jobs.append((f"gen::{i}::{subfield}", subfield))
        self_consistency = not args.no_self_consistency
        worker = lambda s: do_generate(  # noqa: E731
            client, model, s, max_tokens=args.max_tokens, temperature=args.temperature,
            avoid_keys=avoid_keys, self_consistency=self_consistency,
            max_seq_len=args.max_seq_len, topic=args.topic,
            require_think=args.require_think,
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
    p.add_argument("--max-tokens", type=int, default=6144,
                   help="Reasoning think-budget: the ceiling on generated tokens "
                        "for the teacher's <think> reasoning, sized so "
                        "prompt + <think> + 'ANSWER: X' fits train_sft "
                        "MAX_SEQ_LEN=8192. The effective per-call cap is reduced "
                        "dynamically to leave room for the prompt + ANSWER line.")
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN,
                   help="train_sft sequence ceiling; targets are tail-trimmed "
                        "(think body only, never the ANSWER line) to fit.")
    p.add_argument("--require-think", dest="require_think",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Require a closed <think>...</think> in the target shape "
                        "(default True): a trace missing </think> is rejected "
                        "(one retry, then skip). With --no-require-think the bare "
                        "reasoning is instead WRAPPED in <think>...</think> so the "
                        "output stays thinking-only-shaped (leave ON for "
                        "thinking-only training).")
    p.add_argument("--temperature", type=float, default=None,
                   help="Default 0.2 (rationale) / 0.7 (generate).")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress to stderr every N completed rows.")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already present in <out>.partial.")
    p.add_argument("--avoid", default=None,
                   help="Optional JSONL pool; generate-mode questions whose "
                        "normalized stem matches one in the pool are dropped.")
    p.add_argument("--difficulty-filter", action="store_true",
                   help="Documented no-op stub (generate mode): a real filter "
                        "needs a base-model endpoint to score student-solvability.")
    p.add_argument("--no-self-consistency", action="store_true",
                   help="Disable generate-mode blind re-solve verification "
                        "(self-consistency is ON by default).")
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
