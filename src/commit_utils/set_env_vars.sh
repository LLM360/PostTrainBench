if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    source /etc/profile.d/modules.sh
fi

export HF_HOME_NEW="/home/ben/hf_cache"

# Helper function: sets variable to default if unset or "UNDEFINED"
set_default() {
    local var_name="${1:-}"
    local default_value="${2:-}"
    local current_value
    eval "current_value=\"\${$var_name:-}\""
    
    if [ -z "$current_value" ] || [ "$current_value" = "UNDEFINED" ]; then
        export "$var_name"="$default_value"
    fi
}

set_default HF_HOME "$HOME/.cache/huggingface"
set_default POST_TRAIN_BENCH_RESULTS_DIR "results"
set_default POST_TRAIN_BENCH_CONTAINERS_DIR "containers"
set_default POST_TRAIN_BENCH_CONTAINER_NAME "standard"
set_default POST_TRAIN_BENCH_PROMPT "prompt"
set_default POST_TRAIN_BENCH_JOB_SCHEDULER "htcondor"
set_default POST_TRAIN_BENCH_EXPERIMENT_NAME ""

# Teacher vLLM endpoint used by the data-engineering agent loop for synthetic
# data generation. Populate these when running with
# POST_TRAIN_BENCH_PROMPT=data_eng_prompt.
set_default TEACHER_VLLM_URL ""
set_default TEACHER_MODEL_NAME ""
set_default TEACHER_API_KEY ""

# Bind-mounted Python env (vllm + transformers + trl + peft + accelerate +
# claude/codex/etc. CLIs) for use with ubuntu-24.04-python.sif when no
# fakeroot-built standard.sif is available. Built once with pip --prefix.
# Empty default — only the data-eng prompt path consumes this, and run_task.sh
# refuses to bind-mount it unless the user explicitly points at a real path.
set_default POSTTRAIN_ENV_DIR ""
set_default POSTTRAIN_PYTHON_VERSION "3.12"

# Stable host path that holds previously-trained checkpoints used as warm-start
# bases. Bound into the apptainer container at /base_models so MODEL_TO_TRAIN
# can be a path inside that mount. Empty default for the same reason as
# POSTTRAIN_ENV_DIR — opt-in only.
set_default BASE_MODELS_DIR ""

export PYTHONNOUSERSITE=1

if [ "${POST_TRAIN_BENCH_JOB_SCHEDULER}" = "htcondor_mpi-is" ]; then
    SAVE_PATH="$PATH"
    module load cuda/12.1
    export PATH="$PATH:$SAVE_PATH"
    hash -r
fi