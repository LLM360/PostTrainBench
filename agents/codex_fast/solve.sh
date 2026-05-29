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

codex --search exec --json \
    -c model_reasoning_summary=detailed \
    --skip-git-repo-check --yolo \
    --model "$AGENT_CONFIG" "$PROMPT"
