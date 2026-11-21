#!/usr/bin/env bash
# Run with: bash scripts/train_slurm.sh
# This submits itself through sbatch; the training section runs on a compute node.
set -euo pipefail

if [[ "${1:-}" != "--batch" ]]; then
    PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
    # Override these through environment variables or edit them in your editor.
    PARTITION="${PARTITION:-large-llm}"
    TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
    CPUS="${CPUS:-8}"
    RAM="${RAM:-64G}"                 # System RAM, not VRAM.
    GPU_RESOURCE="${GPU_RESOURCE:-gpu:1}" # Site may require gpu:TYPE:1.
    OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/models}"
    export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
    export LYRICS_BASE_MODEL="${LYRICS_BASE_MODEL:-$PROJECT_ROOT/pretrained/Qwen2.5-0.5B}"
    export LYRICS_DATASET="${LYRICS_DATASET:-$PROJECT_ROOT/Dataset_maker_1/train_data.jsonl}"

    [[ "$PARTITION" != large-llm ]] || { echo 'Set PARTITION to your actual GPU partition.' >&2; exit 1; }
    for path in "$OUTPUT_ROOT" "$PYTHON_BIN" "$LYRICS_BASE_MODEL" "$LYRICS_DATASET"; do
        [[ "$path" == /* ]] || { echo "Use absolute paths: $path" >&2; exit 1; }
    done
    [[ -x "$PYTHON_BIN" ]] || { echo "Python not found: $PYTHON_BIN" >&2; exit 1; }
    [[ -f "$LYRICS_BASE_MODEL/config.json" ]] || { echo "Download Qwen into $LYRICS_BASE_MODEL first." >&2; exit 1; }
    [[ -s "$LYRICS_DATASET" ]] || { echo "Dataset missing: $LYRICS_DATASET" >&2; exit 1; }
    if head -n 1 "$LYRICS_DATASET" | grep -q '^version https://git-lfs.github.com/spec/'; then
        echo 'Dataset is a Git LFS pointer. Fetch the actual JSONL before submitting.' >&2
        exit 1
    fi
    mkdir -p "$OUTPUT_ROOT/logs"
    OUTPUT_ROOT="$(cd -- "$OUTPUT_ROOT" && pwd -P)"
    export OUTPUT_ROOT
    echo "Repository root: $PROJECT_ROOT"
    echo "Output root: $OUTPUT_ROOT"
    echo 'Training mode: fresh training from local Qwen (no checkpoint resume).'
    extra=()
    [[ -z "${ACCOUNT:-}" ]] || extra+=(--account="$ACCOUNT")
    [[ -z "${QOS:-}" ]] || extra+=(--qos="$QOS")
    sbatch --job-name=lyrics-train --partition="$PARTITION" \
        --time="$TIME_LIMIT" --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" \
        --gres="$GPU_RESOURCE" --mem="$RAM" --export=ALL \
        --chdir="$PROJECT_ROOT" \
        --output="$OUTPUT_ROOT/logs/%x-%j.out" \
        --error="$OUTPUT_ROOT/logs/%x-%j.err" \
        "${extra[@]}" "$PROJECT_ROOT/scripts/train_slurm.sh" --batch "$PROJECT_ROOT"
    exit 0
fi

: "${SLURM_JOB_ID:?Batch section must run through Slurm}"
PROJECT_ROOT="${2:?Missing repository root}"
cd "$PROJECT_ROOT"
export LYRICS_OUTPUT_DIR="$OUTPUT_ROOT/run-$SLURM_JOB_ID"
# Refuse to overwrite an earlier execution of this allocation (e.g. a requeue).
mkdir "$LYRICS_OUTPUT_DIR"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$LYRICS_OUTPUT_DIR/matplotlib"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
trap 'rc=$?; echo "Ended: $(date -Is) exit_code=$rc"; exit "$rc"' EXIT

echo "Started: $(date -Is)"
echo "Job: $SLURM_JOB_ID node: $(hostname)"
echo "Repository root: $PROJECT_ROOT"
echo "Outputs: $LYRICS_OUTPUT_DIR"
echo "Python: $PYTHON_BIN"
git rev-parse HEAD > "$LYRICS_OUTPUT_DIR/git_commit.txt"
git diff -- training_4 scripts > "$LYRICS_OUTPUT_DIR/working_changes.diff"
"$PYTHON_BIN" -m pip freeze > "$LYRICS_OUTPUT_DIR/packages.txt"
nvidia-smi
# Fail instead of silently falling back to CPU.
srun "$PYTHON_BIN" -u -c 'import torch, matplotlib, transformers; assert torch.cuda.is_available(), "CUDA unavailable"; print("GPU:", torch.cuda.get_device_name(0)); print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30); print("Torch:", torch.__version__); print("Transformers:", transformers.__version__)'
srun "$PYTHON_BIN" -u training_4/training.py
echo "Training finished. Outputs: $LYRICS_OUTPUT_DIR"
