#!/bin/bash
# Codex CLI variant with xhigh reasoning + "fast" service tier.
# Uses ChatGPT auth from ~/.codex/auth.json (no API key path).
#
# Auth: this agent reuses the ChatGPT-auth credentials documented for
# `codex_non_api`. run_task.sh copies agents/${AGENT}/auth.json into
# /home/ben/.codex/auth.json, and falls back to
# agents/codex_non_api/auth.json for any codex_* agent that doesn't
# ship its own auth file, so the documented `codex login` setup is
# sufficient — no manual duplication.
#
# V3: respawn-loop. If codex exits while the SLURM timer still has time,
# we re-launch it with a continuation note appended to the prompt. The
# framework — not the agent's whim — decides when the run ends.

unset ANTHROPIC_API_KEY
unset GEMINI_API_KEY

# Clear API keys so the CLI uses the ChatGPT Pro auth from auth.json
export CODEX_API_KEY=""
export OPENAI_API_KEY=""

# Force ChatGPT auth method (not API key)
if ! grep -q "forced_login_method" ~/.codex/config.toml 2>/dev/null; then
    printf '\nforced_login_method = "chatgpt"\n' >> ~/.codex/config.toml
fi

# Match the rest of the repo's codex agents: write reasoning effort
# (and the service-tier override that defines this variant) into the
# config file via prepend, rather than passing them on the CLI. The
# previous `-c 'reasoning.effort="xhigh"'` form was not the key codex
# 0.134.0 reads (it expects `model_reasoning_effort`), so that override
# was silently ignored and the variant ran at default reasoning effort.
file=/home/ben/.codex/config.toml
tmp="$(mktemp)"
printf 'model_reasoning_effort = "xhigh"\nservice_tier = "fast"\n\n' > "$tmp"
[ -f "$file" ] && cat "$file" >> "$tmp"
mv "$tmp" "$file"

CONTINUATION_NOTE='
[SOLVE.SH RESPAWN] Your previous codex session exited but the SLURM timer
still has time remaining. Resume your work. Check git/file state of
experiments/ — find the highest exp_NNN. If it has final_model/ but no
eval_result.json, run the full eval next. If eval_result.json exists but
no .published, run publish_experiment.py next. If everything is published,
start exp_<N+1>. NEVER STOP.
'

# Locate timer.sh. run_task.sh installs it at /home/ben/task/timer.sh,
# but fall back to CWD so this is robust if the layout changes.
TIMER_SH=/home/ben/task/timer.sh
[ -f "$TIMER_SH" ] || TIMER_SH=./timer.sh

ATTEMPT=0
while true; do
    # Timer check. create_timer.sh emits "Timer expired!" on expiry,
    # otherwise two lines: "Remaining time (hours:minutes):" and "H:MM".
    # We also treat a "0:00" remaining reading as expired.
    if [ -x "$TIMER_SH" ] || [ -f "$TIMER_SH" ]; then
        TIMER_OUT="$(bash "$TIMER_SH" 2>&1 || true)"
    else
        TIMER_OUT=""
    fi
    if echo "$TIMER_OUT" | grep -qiE "Timer expired|TIME_UP|^0:00$| 0:00$"; then
        echo "[solve.sh] timer says time is up ($TIMER_OUT). Exiting respawn loop."
        break
    fi

    ATTEMPT=$((ATTEMPT + 1))
    echo "[solve.sh] launching codex attempt #$ATTEMPT at $(date -u +%FT%TZ)"

    # On attempt #1, normally use the bare base prompt. BUT if the durable
    # experiments/ tree already contains prior exp_NNN dirs, this is a RESUMED
    # allocation (e.g. SLURM preempt+requeue started a fresh job on the same
    # durable EVAL_DIR) — treat attempt #1 as a continuation so codex resumes
    # from the highest exp_NNN instead of restarting at exp_001. Empty
    # experiments/ (true exp_001) and default-prompt runs (no experiments/ dir,
    # so the glob never matches) keep the bare prompt.
    EXP_DIR=/home/ben/task/experiments
    if [ "$ATTEMPT" -eq 1 ] && ! ls -d "$EXP_DIR"/exp_* >/dev/null 2>&1; then
        EFFECTIVE_PROMPT="$PROMPT"
    else
        EFFECTIVE_PROMPT="$PROMPT

$CONTINUATION_NOTE"
    fi

    codex --search exec --json \
        -c model_reasoning_summary=detailed \
        --skip-git-repo-check --yolo \
        --model "$AGENT_CONFIG" "$EFFECTIVE_PROMPT" \
        || echo "[solve.sh] codex attempt #$ATTEMPT exited with $? at $(date -u +%FT%TZ)"

    # Small back-off so we don't busy-respawn if codex is rapidly crashing.
    sleep 5
done

echo "[solve.sh] done. Total codex attempts: $ATTEMPT"
