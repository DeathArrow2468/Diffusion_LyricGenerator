# Batch testing a trained diffusion model

## Place the files

Merge the package contents into your existing repository, preserving paths:

- scripts/test_slurm.sh
- test_5/batch_test.py
- test_5/prompts.json
- test_5/test_batch_io.py
- test_5/BATCH_TESTING.md

These are additions; no training files or existing interactive tester are replaced.
Keep one maintained launcher at scripts/test_slurm.sh. Save .sh files with LF line
endings. The launcher resolves the repository through Git rather than assuming
its parent directory is the root, and submits its own actual path.
Add `test_results/` to your repository .gitignore if you don't want result files
committed. The package does not replace your existing .gitignore.

## Write prompts

Edit test_5/prompts.json in your usual editor. It is a JSON array of strings:

```json
[
  "Write a hopeful pop song about starting over.",
  "Write a song with these themes:\nRain, an empty station, and saying goodbye."
]
```

Use commas between strings, no trailing comma, and escape embedded quotes as \".
TXT input is also supported: one prompt per nonempty line. JSON supports multiline
prompts. Duplicate prompts are preserved as separate requests. This is serial
inference on one GPU, with one model load for the whole list. It is not GPU tensor
batching or distributed inference. The complete prompt list is held in memory.

## Submit from the cluster checkout

Activate the same Python environment/modules that worked for training, then:

```bash
PARTITION=large-llm \
TIME_LIMIT=01:00:00 \
PYTHON_BIN="$(command -v python)" \
bash scripts/test_slurm.sh 7639
```

7639 is the TRAINING job ID identifying models/run-7639/best_model.pt. Slurm
assigns a different job ID to this testing run. Test after training finishes so
best_model.pt is no longer being overwritten. The loaded checkpoint is the best
saved epoch according to training loss, not necessarily the final epoch.

The tester loads the saved tokenizer and run_config.json from the training run,
uses max_seq_len from checkpoint metadata when present, and loads the entire
model_state_dict strictly. It requires the original Qwen files in
pretrained/Qwen2.5-0.5B because the current model wrapper initializes from Qwen
before the trained state replaces its parameters. Hub access is disabled.

For a different prompt list:

```bash
PYTHON_BIN="$(command -v python)" \
bash scripts/test_slurm.sh 7639 test_5/my_prompts.txt
```

For another trained model, change 7639 to its training job ID.

## Output

For example, if the new TEST job ID is 7700:

- test_results/model-7639/test-7700/results.csv
- test_results/model-7639/test-7700/prompts.json (copy of prompts read by the job)
- test_results/model-7639/test-7700/test_config.json
- test_results/model-7639/test-7700/git_commit.txt
- test_results/model-7639/test-7700/working_changes.diff
- test_results/model-7639/test-7700/packages.txt
- test_results/logs/lyrics-test-7639-7700.out
- test_results/logs/lyrics-test-7639-7700.err

CSV has exactly three columns: prompt, output, model_job_id.
model_job_id is 7639 on every row in this example, not 7700. CSV quoting preserves
lyric line breaks, commas, quotes, and Unicode. There are no generated baseline
Qwen comparisons in this file.

Rows are flushed and synced after each prompt to results.partial.csv. When every
prompt succeeds, it is renamed to results.csv. On an error, the job fails and the
partial file preserves completed rows. There is no automatic retry/resume; a new
submission writes to a new testing job directory. A requeue using the same job
ID refuses to overwrite its earlier output.

Inputs are read when the job starts; don't edit prompts/code or overwrite the
checkpoint while this test is queued/running. The run metadata saves generation
settings and checkpoint epoch/loss for reference.

## Monitor

```bash
squeue -u "$USER"
tail -f test_results/logs/lyrics-test-7639-7700.out
```

Ctrl+C stops following the log, not the Slurm job. You can disconnect SSH.

## Optional settings

Pass these as environment assignments before `bash scripts/test_slurm.sh 7639`:

| Variable | Default | Meaning |
|---|---|---|
| PARTITION | large-llm | Cluster partition |
| TIME_LIMIT | 01:00:00 | Test runtime limit |
| CPUS | 4 | CPUs |
| RAM | 32G | Host RAM, not VRAM |
| GPU_RESOURCE | gpu:1 | GPU resource request |
| ACCOUNT / QOS | unset | Optional site settings |
| LYRIC_TOKENS | 128 | Generated lyric block length |
| DENOISING_STEPS | 32 | Denoising iterations, capped by lyric tokens |
| TEMPERATURE | 0.8 | Sampling temperature; 0 chooses highest-scoring tokens |
| TOP_K | 50 | Sample from top K; 0 disables top-K filtering |
| SEED | 42 | Prompt i uses SEED + i, starting at i=0 |
| MODELS_ROOT | repository/models | Absolute training output root |
| TEST_OUTPUT_ROOT | repository/test_results | Absolute testing output root |
| LYRICS_BASE_MODEL | repository/pretrained/Qwen2.5-0.5B | Absolute Qwen directory |
| PYTHON_BIN | python on PATH | Absolute Python executable |

The generation defaults and random-position unmasking follow the existing
interactive test_5/test_lyrics.py sampler. The attention mask is constructed from
sequence length to distinguish a real EOS from padding when they share an ID.
The saved training maximum sequence length is respected; all prompts are checked
for available token space before the model is loaded. Larger prompts may require
lower LYRIC_TOKENS. Increasing it does not increase trained context length.

## Validation performed

Bash and Python syntax checks; five CPU-only tests covering JSON/TXT input,
invalid input, CSV multiline/Unicode round-trip, partial results on failure, and
overwrite protection. Mocked scheduler checks verified partition, paths with
spaces, invocation outside the checkout, and missing-checkpoint rejection.
GPU checkpoint loading/generation has not been executed in the authoring environment.

Run the CPU checks with:

```bash
python -m unittest discover -s test_5 -p test_batch_io.py
```
