#!/usr/bin/env bash
# Run from the repository: bash scripts/test_slurm.sh 7639
set -euo pipefail

if [[ "${1:-}" != "--batch" ]]; then
    # Git identifies the root even if the package was extracted into an extra folder.
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
    SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
    export MODEL_JOB_ID="${1:?Usage: bash scripts/test_slurm.sh TRAINING_JOB_ID [PROMPTS_FILE]}"
    [[ "$MODEL_JOB_ID" =~ ^[0-9]+$ ]] || { echo 'Training job ID must contain digits only.' >&2; exit 1; }
    PARTITION="${PARTITION:-large-llm}"
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
    CPUS="${CPUS:-4}"
    RAM="${RAM:-32G}"
    GPU_RESOURCE="${GPU_RESOURCE:-gpu:1}"
    export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
    export MODELS_ROOT="${MODELS_ROOT:-$PROJECT_ROOT/models}"
    export TEST_OUTPUT_ROOT="${TEST_OUTPUT_ROOT:-$PROJECT_ROOT/test_results}"
    export LYRICS_BASE_MODEL="${LYRICS_BASE_MODEL:-$PROJECT_ROOT/pretrained/Qwen2.5-0.5B}"
    PROMPTS_FILE="${2:-$PROJECT_ROOT/test_5/prompts.json}"
    export PROMPTS_FILE="$(realpath -- "$PROMPTS_FILE")"
    for path in "$PYTHON_BIN" "$MODELS_ROOT" "$TEST_OUTPUT_ROOT" "$LYRICS_BASE_MODEL"; do
        [[ "$path" == /* ]] || { echo "Use an absolute path: $path" >&2; exit 1; }
    done
    [[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 1; }
    for file in "$MODELS_ROOT/run-$MODEL_JOB_ID/best_model.pt" \
                "$MODELS_ROOT/run-$MODEL_JOB_ID/run_config.json" \
                "$MODELS_ROOT/run-$MODEL_JOB_ID/tokenizer_config.json" \
                "$LYRICS_BASE_MODEL/config.json" "$PROMPTS_FILE" \
                "$PROJECT_ROOT/test_5/batch_test.py"; do
        [[ -s "$file" ]] || { echo "Missing or empty file: $file" >&2; exit 1; }
    done
    mkdir -p "$TEST_OUTPUT_ROOT/logs"
    extra=()
    [[ -z "${ACCOUNT:-}" ]] || extra+=(--account="$ACCOUNT")
    [[ -z "${QOS:-}" ]] || extra+=(--qos="$QOS")
    echo "Testing model from training job: $MODEL_JOB_ID"
    echo "Prompts: $PROMPTS_FILE"
    sbatch --job-name="lyrics-test-$MODEL_JOB_ID" --partition="$PARTITION" \
        --time="$TIME_LIMIT" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" \
        --gres="$GPU_RESOURCE" --mem="$RAM" --export=ALL --chdir="$PROJECT_ROOT" \
        --output="$TEST_OUTPUT_ROOT/logs/%x-%j.out" \
        --error="$TEST_OUTPUT_ROOT/logs/%x-%j.err" \
        "${extra[@]}" "$SCRIPT_PATH" --batch "$PROJECT_ROOT"
    exit 0
fi

: "${SLURM_JOB_ID:?Must run through Slurm}"
PROJECT_ROOT="${2:?Missing project root}"
cd "$PROJECT_ROOT"
RUN_DIR="$TEST_OUTPUT_ROOT/model-$MODEL_JOB_ID/test-$SLURM_JOB_ID"
mkdir -p "$(dirname -- "$RUN_DIR")"
mkdir "$RUN_DIR" # Never overwrite a previous execution of this job.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
trap 'rc=$?; echo "Ended: $(date -Is) exit_code=$rc"; exit "$rc"' EXIT

echo "Started: $(date -Is) test job: $SLURM_JOB_ID model job: $MODEL_JOB_ID"
echo "Outputs: $RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git diff -- test_5 scripts > "$RUN_DIR/working_changes.diff"
"$PYTHON_BIN" -m pip freeze > "$RUN_DIR/packages.txt"
srun "$PYTHON_BIN" -u "$PROJECT_ROOT/test_5/batch_test.py" \
    --model-job-id "$MODEL_JOB_ID" --models-root "$MODELS_ROOT" \
    --base-model "$LYRICS_BASE_MODEL" --prompts "$PROMPTS_FILE" \
    --output-dir "$RUN_DIR" --device cuda \
    --lyric-tokens "${LYRIC_TOKENS:-128}" --steps "${DENOISING_STEPS:-32}" \
    --temperature "${TEMPERATURE:-0.8}" --top-k "${TOP_K:-50}" --seed "${SEED:-42}"
