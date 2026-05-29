#!/bin/bash
# Submit a SLURM (array) job running the data-engineering agent loop.
#
# Preconditions:
#   * The submitter must `mkdir -p logs` in $PWD before sbatch — Slurm
#     needs the stdout/stderr directory to exist at submission time. A
#     fresh clone will not have it; this script can't create it for you
#     because it runs after sbatch has already opened the output files.
#   * Cluster-specific options (partition / qos / reservation / account)
#     are NOT baked in. Pass them via sbatch flags, e.g.:
#         sbatch -p main --qos=k2m --reservation=moe --account=k2m \
#                --array=0-7 scripts/submit_data_eng_array.sh ...
#     This keeps the script site-agnostic.
#   * Only single-GPU allocations are supported by default. The SBATCH
#     directive below requests `--gres=gpu:1`. To use multiple GPUs,
#     override the gres on the sbatch command line AND pass --num-gpus
#     with a matching value:
#         sbatch --gres=gpu:8 ... scripts/submit_data_eng_array.sh \
#             --num-gpus 8 ...
#
# Usage:
#   mkdir -p logs
#   sbatch -p main --qos=k2m --account=k2m --array=0-0 \
#       scripts/submit_data_eng_array.sh \
#       --task gpqamain \
#       --agents codex \
#       --model Qwen/Qwen3-1.7B-Base \
#       --hours 6 \
#       --agent-config gpt-5.5 \
#       --teacher-url http://fs-mbz-gpu-349:8000/v1 \
#       --teacher-model MiniMax-M2.7 \
#       --teacher-key EMPTY
#
# Array index ($SLURM_ARRAY_TASK_ID) becomes the cluster_id passed to
# run_task.sh. The chosen agent for index i is agents[i % len(agents)] —
# so --agents claude,codex with --array=0-7 gives 4 claude + 4 codex.

#SBATCH --job-name=data_eng_agent
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/data_eng_%A_%a.out
#SBATCH --error=logs/data_eng_%A_%a.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

TASK=""
AGENTS="claude"
MODEL=""
HOURS="6"
AGENT_CONFIG=""
NUM_GPUS="${NUM_GPUS:-1}"
TEACHER_URL="${TEACHER_VLLM_URL:-}"
TEACHER_MODEL="${TEACHER_MODEL_NAME:-}"
TEACHER_KEY="${TEACHER_API_KEY:-EMPTY}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task) TASK="$2"; shift 2;;
        --agents) AGENTS="$2"; shift 2;;
        --model) MODEL="$2"; shift 2;;
        --hours) HOURS="$2"; shift 2;;
        --agent-config) AGENT_CONFIG="$2"; shift 2;;
        --num-gpus) NUM_GPUS="$2"; shift 2;;
        --teacher-url) TEACHER_URL="$2"; shift 2;;
        --teacher-model) TEACHER_MODEL="$2"; shift 2;;
        --teacher-key) TEACHER_KEY="$2"; shift 2;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

if [[ -z "$TASK" || -z "$MODEL" || -z "$AGENT_CONFIG" ]]; then
    echo "usage: sbatch [-p PART --qos=Q --account=A --reservation=R] $0 \\" >&2
    echo "         --task <task> --model <hf_id> --hours <h> --agent-config <model> \\" >&2
    echo "         [--agents claude,codex] [--num-gpus N (requires matching sbatch --gres=gpu:N)] \\" >&2
    echo "         [--teacher-url URL --teacher-model NAME --teacher-key KEY]" >&2
    exit 2
fi

# Guard: the SBATCH directive above only allocates 1 GPU. If the caller asks
# for more GPUs, they must override the gres at sbatch time
# (`sbatch --gres=gpu:N ...`). We can't detect that override from inside the
# job — SLURM_GPUS_ON_NODE is the closest signal — so if it's available we
# cross-check; otherwise we fall back to a strict NUM_GPUS==1 check.
if [[ "$NUM_GPUS" != "1" ]]; then
    detected_gpus="${SLURM_GPUS_ON_NODE:-}"
    if [[ -z "$detected_gpus" ]]; then
        echo "error: --num-gpus=$NUM_GPUS requested but this script's #SBATCH" >&2
        echo "       directive allocates only 1 GPU. Either:" >&2
        echo "         (a) re-run with --num-gpus 1, or" >&2
        echo "         (b) override the allocation at sbatch time:" >&2
        echo "             sbatch --gres=gpu:$NUM_GPUS $0 --num-gpus $NUM_GPUS ..." >&2
        exit 2
    fi
    if [[ "$detected_gpus" != "$NUM_GPUS" ]]; then
        echo "error: --num-gpus=$NUM_GPUS but SLURM allocated $detected_gpus GPUs" >&2
        echo "       (SLURM_GPUS_ON_NODE=$detected_gpus). Override the gres to" >&2
        echo "       match: sbatch --gres=gpu:$NUM_GPUS ..." >&2
        exit 2
    fi
fi

IFS=',' read -ra AGENT_LIST <<< "$AGENTS"
N_AGENTS=${#AGENT_LIST[@]}
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CHOSEN_AGENT="${AGENT_LIST[$((IDX % N_AGENTS))]}"

export POST_TRAIN_BENCH_PROMPT="data_eng_prompt"
export POST_TRAIN_BENCH_CONTAINER_NAME="${POST_TRAIN_BENCH_CONTAINER_NAME:-ubuntu-24.04-python}"
export TEACHER_VLLM_URL="$TEACHER_URL"
export TEACHER_MODEL_NAME="$TEACHER_MODEL"
export TEACHER_API_KEY="$TEACHER_KEY"
export POST_TRAIN_BENCH_JOB_SCHEDULER="slurm"

echo "[$(date -u +%FT%TZ)] cluster_id=${IDX} agent=${CHOSEN_AGENT} task=${TASK} model=${MODEL} hours=${HOURS} agent_config=${AGENT_CONFIG}"
echo "  teacher: ${TEACHER_VLLM_URL} model=${TEACHER_MODEL_NAME}"

bash src/run_task.sh "$TASK" "$CHOSEN_AGENT" "$MODEL" "$IDX" "$HOURS" "$AGENT_CONFIG" "$NUM_GPUS"
