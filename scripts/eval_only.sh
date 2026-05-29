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
export POST_TRAIN_BENCH_JOB_SCHEDULER="${POST_TRAIN_BENCH_JOB_SCHEDULER:-slurm}"
source src/commit_utils/set_env_vars.sh
set -u

EVAL_DIR="${1:-}"
if [ -z "$EVAL_DIR" ] || [ ! -d "$EVAL_DIR/final_model" ]; then
    echo "usage: sbatch $0 <eval_dir-with-final_model-inside>" >&2
    exit 2
fi

EVAL_DIR=$(realpath "$EVAL_DIR")
TASK_GUESS=$(basename "$EVAL_DIR" | sed 's/_[^_]*$//' | awk -F'_' '{print $1}')
EVALUATION_TASK="${EVALUATION_TASK:-${TASK_GUESS:-gpqamain}}"

echo "[eval_only] EVAL_DIR=$EVAL_DIR"
echo "[eval_only] EVALUATION_TASK=$EVALUATION_TASK"
echo "[eval_only] model size: $(du -sh "$EVAL_DIR/final_model/model.safetensors" 2>/dev/null | cut -f1)"

REPO_ROOT="$(pwd)"
TMP_HF_CACHE="/tmp/hf_cache_$$"

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
