#!/bin/bash
# Wrapper around evaluate.py for data-engineering agent runs.
#
# Why this exists: the bind-mounted Python env at /opt/env contains the
# `vllm` CLI binary at /opt/env/local/bin/vllm, and run_task.sh injects
# that directory into PATH via `apptainer exec --env PATH=...`. However,
# the codex CLI runs every shell command through `bash -lc "..."` (login
# shell), which sources /etc/profile + ~/.bashrc and *overwrites* PATH
# with the container's defaults — stripping out /opt/env/local/bin. As a
# result the agent sees `vllm: command not found` and inspect_ai cannot
# spawn its local vLLM server.
#
# This wrapper re-asserts the bind-mounted env on PATH and forwards all
# arguments to evaluate.py. Agents should call `bash eval.sh ...` instead
# of `python3 evaluate.py ...` for self-evals.
export PATH="/opt/env/local/bin:/opt/env/bin:${PATH}"
exec python3 /home/ben/task/evaluate.py "$@"
