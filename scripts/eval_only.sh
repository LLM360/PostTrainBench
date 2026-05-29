#!/bin/bash
# Standalone eval for a previously-trained final_model. Use when the harness's
# in-job eval phase fails (e.g., env-var mismatch like missing
# VLLM_DEFAULT_SERVER_ARGS) but the model on disk is fine.
#
# Usage:
#   sbatch --qos=k2p --account=k2p scripts/eval_only.sh <EVAL_DIR>
#   e.g. sbatch ... scripts/eval_only.sh \
#       results/codex_fast_gpt-5.5_6h_8gpu/gpqamain_Qwen_Qwen3-1.7B-Base_0

#SBATCH --job-name=data_eng_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --partition=main
#SBATCH --output=logs/eval_only_%j.out
#SBATCH --error=logs/eval_only_%j.err

set -eo pipefail

# Resolve REPO_ROOT from the script's location FIRST, so the source line below
# (and everything after) works regardless of the submitter's CWD. The script
# lives at <repo>/scripts/eval_only.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-slurm}"
source "${REPO_ROOT}/src/commit_utils/set_env_vars.sh"

# Normalize POST_TRAIN_BENCH_CONTAINERS_DIR to an absolute path so this
# script works when submitted from outside the repo. The default in
# set_env_vars.sh is the relative "containers".
case "${POST_TRAIN_BENCH_CONTAINERS_DIR}" in
    /*) ;;  # already absolute, leave alone
    *)  POST_TRAIN_BENCH_CONTAINERS_DIR="${REPO_ROOT}/${POST_TRAIN_BENCH_CONTAINERS_DIR}" ;;
esac
export POST_TRAIN_BENCH_CONTAINERS_DIR

set -u

EVAL_DIR="${1:-}"
if [ -z "$EVAL_DIR" ] || [ ! -d "$EVAL_DIR/final_model" ]; then
    echo "usage: sbatch $0 <eval_dir-with-final_model-inside>" >&2
    exit 2
fi

EVAL_DIR=$(realpath "$EVAL_DIR")
TASK_GUESS=$(basename "$EVAL_DIR" | sed 's/_[^_]*$//' | awk -F'_' '{print $1}')
EVALUATION_TASK="${EVALUATION_TASK:-${TASK_GUESS:-gpqamain}}"

# eval_only.sh exists specifically to run the evaluator, which needs the
# python deps shipped in POSTTRAIN_ENV_DIR. Fail loudly if it's missing
# rather than silently binding an empty path over /opt/env (which would
# leave PATH/PYTHONPATH pointing at a non-existent env inside the container).
if [ -z "${POSTTRAIN_ENV_DIR:-}" ] || [ ! -d "${POSTTRAIN_ENV_DIR}" ]; then
    echo "[eval_only] ERROR: POSTTRAIN_ENV_DIR is not set to an existing directory." >&2
    echo "[eval_only]   The evaluator needs python deps bound at /opt/env." >&2
    echo "[eval_only]   Export POSTTRAIN_ENV_DIR=<path-to-env> in your shell," >&2
    echo "[eval_only]   or set it in src/commit_utils/set_env_vars.sh." >&2
    exit 3
fi

echo "[eval_only] REPO_ROOT=$REPO_ROOT"
echo "[eval_only] EVAL_DIR=$EVAL_DIR"
echo "[eval_only] EVALUATION_TASK=$EVALUATION_TASK"
echo "[eval_only] model size: $(du -sh "$EVAL_DIR/final_model/model.safetensors" 2>/dev/null | cut -f1)"

TMP_HF_CACHE="/tmp/hf_cache_$$"

# If EVAL_DIR is already inside REPO_ROOT, the repo bind covers it. Otherwise
# we need an explicit bind so --model-path / --json-output-file are visible
# inside the container.
extra_binds=()
case "$EVAL_DIR" in
    "$REPO_ROOT"/*) ;;  # already covered by repo bind
    *) extra_binds+=( --bind "${EVAL_DIR}:${EVAL_DIR}" );;
esac

apptainer exec \
    --nv \
    --env PATH="/opt/env/local/bin:/opt/env/bin:$PATH" \
    --env PYTHONPATH="/opt/env/local/lib/python${POSTTRAIN_PYTHON_VERSION}/dist-packages" \
    --env "HF_HOME=${TMP_HF_CACHE}" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env VLLM_API_KEY="inspectai" \
    --env VLLM_DEFAULT_SERVER_ARGS='{"enforce_eager": true}' \
    --env PYTHONNOUSERSITE="1" \
    --writable-tmpfs \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --bind "${HF_HOME}:${TMP_HF_CACHE}" \
    --bind "${POSTTRAIN_ENV_DIR}:/opt/env" \
    "${extra_binds[@]}" \
    --pwd "${REPO_ROOT}/src/eval/tasks/${EVALUATION_TASK}" \
    "${POST_TRAIN_BENCH_CONTAINERS_DIR}/vllm_debug.sif" \
    python evaluate.py \
        --model-path "$EVAL_DIR/final_model" \
        --templates-dir ../../../../src/eval/templates \
        --limit -1 \
        --max-tokens 8000 \
        --gpu-memory-utilization 0.85 \
        --max-connections 4 \
        --json-output-file "$EVAL_DIR/metrics.json"

echo "[eval_only] DONE"
echo "[eval_only] metrics:"
cat "$EVAL_DIR/metrics.json" 2>/dev/null || echo "(no metrics.json produced)"
