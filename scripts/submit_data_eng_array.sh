#!/bin/bash
# Submit a SLURM (array) job running the data-engineering agent loop.
#
# Usage:
#   sbatch --array=0-0 scripts/submit_data_eng_array.sh \
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
#SBATCH --partition=main
#SBATCH --qos=k2m
#SBATCH --reservation=moe
#SBATCH --account=k2m
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
    echo "usage: sbatch $0 --task <task> --model <hf_id> --hours <h> --agent-config <model> [--agents claude,codex] [--num-gpus 8] [--teacher-url URL --teacher-model NAME --teacher-key KEY]" >&2
    exit 2
fi

mkdir -p logs

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
