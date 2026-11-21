# Running the current hybrid trainer on Slurm

Prepared against repository commit 1dbb7c28219bb756c0529f932d64ae84b868d350.
This launches training_4/training.py, starting from base Qwen. It DOES NOT resume
an existing diffusion checkpoint. The current trainer has no resume loader.
No cluster job has been submitted or GPU training validated by this setup.

## Install these changes

Copy the files into their matching repository paths, commit and push them from
your editor, then pull on the cluster. The two Python replacements are required:
exporting output variables alone cannot change the original Python constants.
Keep shell files in LF format (the included .gitattributes enforces this).

## Paths

Run `git rev-parse --show-toplevel` anywhere inside the cluster checkout to see
its absolute root. The launcher resolves the same root from its own location,
then passes it to the Slurm batch job. It works from outside the checkout too.
Default output root: <repository></repository>/models. Each job gets run-JOB_ID under it.
Use OUTPUT_ROOT=/absolute/persistent/path/models to select another location.
The repository, Python environment, data, pretrained model and output storage
must be accessible on the compute node, not just the login node.

## Prepare data

The repository JSONL is tracked using Git LFS. From the repository root, run:

```bash
git lfs pull --include='Dataset_maker_1/train_data.jsonl'
```

If LFS is not available, transfer the actual dataset through your cluster's
approved file-transfer method. The launcher rejects a remaining LFS pointer.

## Download Qwen before training

Cluster internet policy is UNKNOWN. Check your site's documentation/admin.
A successful login-node download does not prove compute-node network access.
Perform this download only on an allowed internet-enabled node, or download on
your own computer and transfer the complete directory to shared storage.
From the repository root, with the intended Python environment activated:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-0.5B",
    local_dir="pretrained/Qwen2.5-0.5B",
)
PY
```

This tests a real full download, rather than only connectivity to the website.
The job uses that local path and HF_HUB_OFFLINE=1. A missing file causes a
failure instead of an attempted download. Do not commit model weights to Git.
For repeatable provisioning, pass a specific Hugging Face commit as revision.

## Submit

Discover your partition with `sinfo`. Activate the existing cluster Python
environment first (including any site-required modules). Do not blindly upgrade
Torch: use a CUDA-enabled installation compatible with the cluster.
The repository requirements file does not list matplotlib, so verify that it
is installed alongside torch, transformers, numpy and tqdm.

From the repository root, replace REAL_GPU_PARTITION:

```bash
PARTITION=REAL_GPU_PARTITION \
TIME_LIMIT=04:00:00 \
PYTHON_BIN="$(command -v python)" \
bash scripts/train_slurm.sh
```

Optional submission variables: ACCOUNT, QOS, CPUS (8), RAM (64G),
GPU_RESOURCE (gpu:1), OUTPUT_ROOT, LYRICS_BASE_MODEL, LYRICS_DATASET.
Use absolute paths for all path overrides. A site may require a typed GPU
request to select the 96 GB GPU; gpu:1 alone does not ensure that device.

Optional training variables: LYRICS_BATCH_SIZE, LYRICS_ACCUM_STEPS,
LYRICS_EPOCHS, LYRICS_NUM_WORKERS. Defaults remain 2, 8, 20, 0. The learning
rate, sequence length and objective remain unchanged. These settings are not
yet tuned to the larger GPU. More VRAM does not justify silently changing the
effective batch size or training recipe.

Partition/time/resource variables become sbatch command-line arguments before
submission. Slurm does not expand Bash variables inside #SBATCH comments.
The job uses SLURM_JOB_ID and SLURM_CPUS_PER_TASK while running.

## Monitor

```bash
squeue -u "$USER"
tail -f models/logs/lyrics-train-JOB_ID.out
tail -n 80 models/logs/lyrics-train-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

Replace JOB_ID with the submitted number. Ctrl+C stops tail, not training.

Each run directory contains tokenizer files, best_model.pt after an improving
completed epoch, metrics.csv (every batch), training_curves.png (every epoch),
run_config.json, git_commit.txt, working_changes.diff, and packages.txt.
The existing plot uses the noninteractive Agg backend: SSH needs no display.
The graph's per-step values are minibatch metrics, not optimizer-update metrics.
CSV records both batch and optimizer step explicitly.

## Remaining limitations

The first full model checkpoint is saved only after the first epoch completes.
Later checkpoints are saved only when epoch training loss improves. Timeout or
cancellation before an epoch ends can lose all training since the last saved
best epoch. CSV survives independently, but it cannot reconstruct model weights.
There is no periodic/latest checkpoint, resume loader, or graceful time-limit
checkpoint in this version. Do not treat a re-submission as a resumed run.
Confirm whether you want fresh training or actual checkpoint continuation before
committing substantial GPU time. This launcher refuses to reuse an existing job
output directory, including automatic requeues, to prevent accidental overwrites.
