# GPQA (Main) — Benchmark Overview

GPQA is a **graduate-level multiple-choice science Q&A** benchmark across
physics, chemistry, and biology. Questions are designed to be hard even with
internet access ("Google-proof"). Reference paper:
[Rein et al., 2023](https://arxiv.org/abs/2311.12022).

## What the harness actually scores

The harness loads `Idavidrein/gpqa::gpqa_main::train` (yes, the only split is
named "train" — that **is** the eval set). Each item has fields:

```
{
  "Question": "...",
  "Correct Answer": "...",
  "Incorrect Answer 1": "...",
  "Incorrect Answer 2": "...",
  "Incorrect Answer 3": "...",
  "Record ID": "rec_..."
}
```

Choices are shuffled at scoring time. The solver chain is
`multiple_choice(cot=True)` then `choice()` — i.e., the model is given the
question + four labeled options and must produce a chain of thought and a
single-letter choice. Scoring is exact-letter accuracy.

## What good training data looks like for this benchmark

You are constructing supervised messages-jsonl rows. Good rows for GPQA:

- **Graduate-level science Q&A** in physics, chemistry, biology (organic,
  inorganic, biochem, molecular bio, genetics, ecology, condensed matter,
  high-energy, astrophysics, etc.).
- **Long-form, reasoned answers.** The model is scored on multiple-choice but
  uses chain-of-thought to get there — train it to write the kind of CoT
  that lands on the right answer.
- **Multiple-choice format is fine but not required.** Mix free-form Q→A
  rationales with MCQ-style four-option items where you've written out the
  rationale.
- **Quantitative reasoning** matters: unit analysis, order-of-magnitude
  estimates, multi-step derivations.
- **Vocabulary diversity**: the diversity audit will reject monoculture (e.g.,
  all questions about thermodynamics, or all questions written in the same
  voice). Mix subfields.

## What good training data does NOT look like

- **GPQA questions themselves.** `dataset_audit.py` will refuse any row whose
  normalized user content matches a test question's sha256 or shares ≥ 8
  13-gram shingles with one. Do not try to lightly paraphrase GPQA items.
- **Pure trivia / general knowledge.** GPQA is graduate-level — high-school
  science MCQs do little.
- **Synthetic monoculture.** If you ask the teacher to "generate a graduate
  physics question" 5000 times, you'll get 5000 thermodynamics questions in
  the same voice. The diversity audit will fail you.
- **Code, poetry, conversation, RLHF preference pairs, etc.** Wrong shape.

## Suggested HF datasets to mine (non-exhaustive)

These are public, non-overlapping with GPQA's test items in spirit. You must
still pass the audit — overlap by accident is checked.

- `cais/mmlu` (the STEM subjects only — physics, chemistry, biology branches)
- `openbookqa` / `sciq` (lower-difficulty but useful for breadth)
- `allenai/sciq`
- `EleutherAI/hendrycks_math` (for quantitative-reasoning style)
- `tiger-lab/MathInstruct` (multistep CoT)
- `camel-ai/physics`, `camel-ai/chemistry`, `camel-ai/biology` (synthetic
  graduate-level Q&A — already MCQ-shaped in places)
- arxiv-derived corpora for distillation prompts

Always pass it through `dataset_audit.py` before training.

## Suggested synthesis patterns for the teacher vLLM

- **CoT distillation**: take a non-GPQA seed question, prompt the teacher
  for a detailed worked solution, use `(question, teacher_solution)` as the
  training row.
- **MCQ rewriting**: take a free-form Q&A, ask the teacher to convert it
  into a four-option MCQ with one correct answer + three plausible
  distractors and a worked rationale.
- **Subfield rotation**: rotate the asked subfield (electromagnetism →
  quantum → stat mech → enzyme kinetics → phylogenetics → ...) explicitly
  in your synthesis prompts, otherwise the teacher will collapse onto its
  favourites and the diversity audit will fail you.

## Sanity checks you can run

- `python task/evaluate.py --model-path <model> --limit 5` — run the base
  model on 5 items to see failure modes. **Do not** use this loop to chase
  the eval score; it is a sanity tool only.
- `bash timer.sh` — remaining time.
