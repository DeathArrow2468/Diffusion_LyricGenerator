# Hybrid loss implementation

Implements the primary objective in `hybrid_loss_specification(1).md`:

```text
L = alpha * L_AR + (1 - alpha) * L_distance
q_i = rho / |M| + (1 - rho) * softmax_M(-(d_i - 1) * ln(2) / h)
d_i = absolute_position_i - prefix_end
```

Section 9's third objective and beta are intentionally absent. Cross-entropy
remains the per-token error; the new code changes its weighting and combines
it with a separate causal exercise. This is an experimental supervised objective.

## Files

- `loss_config.py`: validated loss and sampling hyperparameters. Edit `LOSS_CONFIG` defaults here.
- `hybrid_loss.py`: shifted AR CE, per-example distance weights, and reductions.
- `hybrid_batch.py`: clean inputs, prefix sampling, target blocks, corruption, labels.
- `training.py`: shared CPU/CUDA training with two passes through the same weights.
- `training_cpu.py`: explicitly selects CPU and calls the shared trainer.
- `../diffusion_model_3/model.py`: explicit `mode="ar"` or `mode="denoising"` attention.

## Settings

| Setting | Default | Meaning |
|---|---:|---|
| alpha | 0.7 | AR coefficient; denoising gets 0.3 |
| rho | 0.1 | Uniform share of denoising weights |
| half_life_start | 16 | Initial distance half-life, in tokens |
| half_life_end | 16 | Final half-life; equal endpoints mean fixed weighting |
| max_target_tokens | 256 | Maximum denoising block length |
| prefix_policy | uniform | Uniformly select 0 through N-1 clean lyric-prefix tokens |
| noise_min / noise_max | 0.1 / 1.0 | Uniform noise-level range; masking probability equals noise |
| ignore_index | -100 | Unscored label sentinel |

These defaults are illustrative, not validated optimal settings. Set
`prefix_policy="prompt_only"` to always place the boundary at SEP. Set the two
half-lives to different positive values for a linear curriculum. Progress is
completed optimizer steps divided by planned optimizer steps, not batch index
or diffusion timestep. `S` is calculated from epochs, dataset length, batch size,
and gradient accumulation rather than duplicated in the loss config.

Batch size, sequence length, optimizer learning rate, epochs, paths, and gradient
accumulation remain in `config.py`. The existing dataset path is machine-specific:
set `DATASET_PATH` before running. Optimizer weight decay (0.01) and gradient norm
clipping (1.0) retain the existing trainer settings.

## Recorded data choices

Each example is `BOS (if available) + prompt + SEP + lyrics`. The prompt is fixed
conditioning; AR targets are all retained lyric tokens, including the first one
predicted from SEP. Thus this is lyric-conditional AR CE, not CE on prompt text.
EOS is not appended or trained here, preserving the project's lyric-only scope.
Supporting learned stopping requires a separate matched training/sampling change.

The denoising boundary is the end of the prompt plus a sampled clean lyric prefix.
It is stored as a zero-based absolute index. Target distance is from that fixed
boundary, **not** from the nearest unmasked word or rank among selected masks.
The actual block is clipped to remaining lyrics. Tokens after the block are
removed from denoising context but remain in the separate AR example.

Independent Bernoulli masking uses `m(tau)=tau`. If no position is selected,
one random block position is masked. This documented fallback is not rejection
resampling and changes the mask distribution. Empty lyric examples are rejected.
Long prompts are right-truncated to reserve SEP and at least one lyric token;
lyrics are then right-truncated to the remaining space. Attention is built from
lengths, so a shared PAD/EOS ID cannot accidentally hide a real position.

Both branches average per example, then over examples. Padding, prompt labels,
and unmasked denoising tokens do not enter the denominators. Distance coefficients
are computed with FP32 softmax and sum to one per example. Empty target sets raise
an error instead of silently producing NaNs or changing batch weighting.

## Devices and memory

Collation uses CPU tensors for DataLoader compatibility. The trainer moves every
batch tensor to `device`; loss position tensors use the labels' device and model
attention tensors use the inputs' device. CPU uses FP32. CUDA uses BF16 when
supported, otherwise FP32. No CUDA hardware was available for validation.

The AR pass uses clean inputs, shifted targets, strictly causal attention, and no
timestep embedding. The denoising pass uses corrupted inputs, same-position
targets, bidirectional attention, and the existing timestep embedding. Both share
all backbone and vocabulary-head weights. The wrapper targets the current
Qwen-style `.model` / `.lm_head` architecture; it is not a generic GPT-2 adapter.

The trainer backpropagates each weighted branch separately before taking an
optimizer step. This sums the same two gradient contributions while releasing
each graph before the next forward. A short final accumulation group is weighted
by its actual number of examples. Alpha endpoints skip the unused forward pass;
its logged component is then zero (not a measured loss).

Checkpoints record loss config, completed/planned steps, scheduler state, and
component losses in addition to model/optimizer state. Automatic resume is not
implemented. When adding resume, restore those fields and the original schedule
horizon together. Changing h over training can change the training-loss scale;
lowest training loss is only a checkpoint heuristic, not a quality guarantee.

## Run

From the repository root:

```bash
python -m training_4.training_cpu   # CPU
python -m training_4.training       # CUDA if available, otherwise CPU
python -m unittest discover -s tests -v
```

Tests use synthetic tensors and a small randomly initialized Qwen; they do not
need model downloads. They check the document's numerical example, absolute
sparse distances, gradients, reductions, schedule, masking fallback, attention
isolation, a CPU optimizer update, and trainer/checkpoint integration. They do
not establish lyric quality or full-model memory usage.
