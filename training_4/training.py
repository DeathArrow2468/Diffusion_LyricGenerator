import math
import json
from dataclasses import asdict
from pathlib import Path
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Resolve project imports from the script location, not the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from training_4 import config
from loss_func_1.config_loss import LOSS_CONFIG
from loss_func_1.hybrid_loss import HybridLoss
from loss_func_1.hybrid_batch import HybridCollateFn
from diffusion_model_3.model import BidirectionalDiffusionTransformer


def setup_tokenizer():
    """Use this training config explicitly; no ambiguous `import config`."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens":
        config.STRUCTURAL_TOKENS + [config.SEP_TOKEN, config.MASK_TOKEN]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class LyricDiffusionDataset(Dataset):
    """Same prompt/lyrics JSONL format; independent of the old masking module."""
    def __init__(self, path):
        with open(path, encoding="utf-8") as handle:
            self.data = [json.loads(line) for line in handle if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def set_seed(seed):
    """Makes training more repeatable between runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer(model):
    """
    AdamW is the gold standard for Transformers.
    We do not apply weight decay to biases or normalisation parameters.
    """

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        is_bias = name.endswith("bias")
        is_normalization = (
            "norm" in name.lower()
            or "ln_" in name.lower()
        )

        if is_bias or is_normalization:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": 0.01,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=config.LEARNING_RATE,
    )


def save_training_curves(
    step_losses,
    step_accuracies,
    epoch_losses,
    epoch_accuracies,
    steps_per_epoch,
    save_dir,
):
    """Plots loss and masked-token accuracy over the full training run."""

    steps = list(range(1, len(step_losses) + 1))
    epoch_x = [steps_per_epoch * (i + 1) for i in range(len(epoch_losses))]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(steps, step_losses, alpha=0.3, label="Per-step loss")
    ax_loss.plot(epoch_x, epoch_losses, marker="o", label="Epoch average")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(steps, step_accuracies, alpha=0.3, label="Per-step accuracy")
    ax_acc.plot(epoch_x, epoch_accuracies, marker="o", label="Epoch average")
    ax_acc.set_title("Masked-Token Accuracy")
    ax_acc.set_xlabel("Training step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    fig.tight_layout()

    graph_path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)

    print(f"Training graph saved to {graph_path}")


def train(device=None):
    # ==========================================
    # 1. INITIALIZATION & SETUP
    # ==========================================
    set_seed(getattr(config, "SEED", 42))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Automatically detect if you have an Nvidia GPU (CUDA) available, otherwise use CPU
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    print(f"Executing training on: {device.type.upper()}")

    print("Loading Tokenizer and Data Engine...")

    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.DATASET_PATH)

    if not len(dataset):
        raise ValueError("Training dataset is empty")
    if config.GRAD_ACCUM_STEPS < 1 or config.EPOCHS < 1:
        raise ValueError("GRAD_ACCUM_STEPS and EPOCHS must be positive")

    collate_fn = HybridCollateFn(
        tokenizer=tokenizer,
        max_len=config.MAX_SEQ_LEN,
        config=LOSS_CONFIG,
        sep_token=config.SEP_TOKEN,
        mask_token=config.MASK_TOKEN,
    )

    objective = HybridLoss(LOSS_CONFIG).to(device)



    # The DataLoader is our engine. It grabs raw JSON rows, passes them to collate_fn,
    # and hands us perfectly formatted, masked tensors.
    dataloader = DataLoader(
        dataset,
        batch_size=config.MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=getattr(config, "NUM_WORKERS", 0),
        pin_memory=device.type == "cuda",
    )

    print("Loading Bidirectional Diffusion Model...")

    model = BidirectionalDiffusionTransformer(
        config.MODEL_NAME,
        len(tokenizer),
        dtype=torch.float32,
    ).to(device)

    # All trainable weights must be stored in FP32.
    # BF16 is used only inside autocast for forward-pass calculations.
    trainable_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    if trainable_dtypes != {torch.float32}:
        raise RuntimeError(
            "The model is not stored fully in FP32. "
            "Check model.py and load Qwen with dtype=torch.float32."
        )

    # ==========================================
    # 2. OPTIMIZER & SCHEDULER
    # ==========================================
    optimizer = create_optimizer(model)

    # Calculate exactly how many times we will update the weights across all epochs.
    # We use ceil because the final partial gradient-accumulation group also performs an update.
    updates_per_epoch = math.ceil(
        len(dataloader) / config.GRAD_ACCUM_STEPS
    )

    total_steps = updates_per_epoch * config.EPOCHS

    # The Warmup Scheduler:
    # Instead of blasting the model with the full learning rate on step 1
    # (which destroys pre-trained weights), this starts the LR at 0.0,
    # slowly ramps it up, then slowly decays it.
    warmup_steps = min(
        config.WARMUP_STEPS,
        max(1, total_steps // 10),
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Save the exact tokenizer used for this training run.
    # The trained model cannot be loaded safely with a different tokenizer.
    tokenizer.save_pretrained(config.CHECKPOINT_DIR)

    print(
        "Starting Training! "
        f"Effective Batch Size: "
        f"{config.MICRO_BATCH_SIZE * config.GRAD_ACCUM_STEPS}"
    )

    print(
        f"Dataset examples: {len(dataset)} | "
        f"Total optimizer updates: {total_steps}"
    )

    # We track the best loss to ensure we only save models that are actively improving.
    # This is temporary: once we add a validation split, use validation loss instead.
    best_loss = float("inf")

    completed_steps = 0
    step_losses = []
    step_accuracies = []
    epoch_losses = []
    epoch_accuracies = []

    # ==========================================
    # 3. THE CORE TRAINING LOOP
    # ==========================================
    for epoch in range(config.EPOCHS):
        model.train()  # Put the model in training mode (enables dropout, etc.)

        total_loss = 0.0
        examples_seen = 0
        total_ar = 0.0
        total_distance = 0.0
        epoch_correct = 0
        epoch_masked = 0
        optimizer.zero_grad(set_to_none=True)

        # Create a terminal progress bar
        progress_bar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}/{config.EPOCHS}",
        )

        for step, batch in progress_bar:
            batch = {key: value.to(device, non_blocking=device.type == "cuda")
                     for key, value in batch.items()}
            batch_size = batch["input_ids"].size(0)
            group_start = (step // config.GRAD_ACCUM_STEPS) * config.GRAD_ACCUM_STEPS
            group_examples = min(
                config.GRAD_ACCUM_STEPS * config.MICRO_BATCH_SIZE,
                len(dataset) - group_start * config.MICRO_BATCH_SIZE,
            )
            accumulation_scale = batch_size / group_examples
            ar_value = 0.0
            distance_value = 0.0
            correct = masked_count = 0
            h_used = objective.half_life(completed_steps, total_steps)

            # Pass 1: clean text and causal attention. Update gradients, not weights.
            if LOSS_CONFIG.alpha > 0:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    ar_logits = model(batch["ar_input_ids"], batch["ar_attention_mask"], mode="ar")
                    ar_loss = objective.ar_loss(ar_logits, batch["ar_labels"])
                if not torch.isfinite(ar_loss):
                    raise RuntimeError(f"Invalid AR loss at epoch {epoch+1}, batch {step+1}")
                (LOSS_CONFIG.alpha * ar_loss * accumulation_scale).backward()
                ar_value = ar_loss.detach().item()
                del ar_logits, ar_loss

            # Pass 2: corrupted block and bidirectional attention. Same model weights.
            if LOSS_CONFIG.alpha < 1:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=use_bf16):
                    logits = model(batch["input_ids"], batch["attention_mask"],
                                   batch["timesteps"], mode="denoising")
                    distance_loss = objective.distance_loss(
                        logits, batch["labels"], batch["prefix_end"],
                        completed_steps, total_steps,
                    )
                if not torch.isfinite(distance_loss):
                    raise RuntimeError(f"Invalid distance loss at epoch {epoch+1}, batch {step+1}")
                with torch.no_grad():
                    masked_positions = batch["labels"].ne(LOSS_CONFIG.ignore_index)
                    correct = ((logits.argmax(-1) == batch["labels"]) & masked_positions).sum().item()
                    masked_count = masked_positions.sum().item()
                ((1 - LOSS_CONFIG.alpha) * distance_loss * accumulation_scale).backward()
                distance_value = distance_loss.detach().item()
                del logits, distance_loss

            # For logging only: both weighted gradients have already accumulated.
            # Do NOT call backward() again on this combined scalar.
            unscaled_loss = LOSS_CONFIG.alpha * ar_value + (1 - LOSS_CONFIG.alpha) * distance_value
            total_loss += unscaled_loss * batch_size
            total_ar += ar_value * batch_size
            total_distance += distance_value * batch_size
            examples_seen += batch_size
            epoch_correct += correct
            epoch_masked += masked_count
            step_losses.append(unscaled_loss)
            step_accuracies.append(correct / masked_count if masked_count else float("nan"))

            # --- GRADIENT ACCUMULATION & CLIPPING ---
            # Only update the actual model weights once we have accumulated a full macro-batch
            should_update = (
                (step + 1) % config.GRAD_ACCUM_STEPS == 0
                or (step + 1) == len(dataloader)
            )

            if should_update:
                # Gradient Clipping: If the math resulted in a massive gradient spike,
                # we cap it at a maximum norm of 1.0. This prevents the model from exploding.
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()  # Apply the weight updates
                scheduler.step()  # Slightly adjust the learning rate
                completed_steps += 1
                optimizer.zero_grad(set_to_none=True)  # Wipe the accumulated gradients clean

            # Update progress bar to show both loss and the dynamically changing learning rate
            progress_bar.set_postfix(
                {
                    "loss": f"{unscaled_loss:.4f}",
                    "ar": f"{ar_value:.4f}",
                    "distance": f"{distance_value:.4f}",
                    "ar_weighted": f"{LOSS_CONFIG.alpha * ar_value:.4f}",
                    "den_weighted": f"{(1-LOSS_CONFIG.alpha) * distance_value:.4f}",
                    "h": h_used,
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # ==========================================
        # 4. BEST LOSS CHECKPOINTING
        # ==========================================
        avg_loss = total_loss / examples_seen

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(epoch_correct / epoch_masked if epoch_masked else float("nan"))

        print(
            f"Epoch {epoch + 1} completed. "
            f"Average Loss: {avg_loss:.4f}"
        )

        # We only save the file if this epoch performed strictly better
        # than our previous best.
        if avg_loss < best_loss:
            print(
                f"Loss improved from {best_loss:.4f} "
                f"to {avg_loss:.4f}. Saving new best model..."
            )

            best_loss = avg_loss

            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                "best_model.pt",
            )

            # Save a dictionary containing everything needed to use or resume the model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "completed_steps": completed_steps,
                    "total_steps": total_steps,
                    "loss_config": asdict(LOSS_CONFIG),
                    "loss_components": {"ar": total_ar / examples_seen,
                                        "distance": total_distance / examples_seen},
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "tokenizer_length": len(tokenizer),
                    "model_name": config.MODEL_NAME,
                    "max_seq_len": config.MAX_SEQ_LEN,
                },
                checkpoint_path,
            )

            print(f"Checkpoint saved to {checkpoint_path}\n")

        else:
            print(
                f"Loss did not improve from {best_loss:.4f}. "
                "Skipping checkpoint.\n"
            )

    save_training_curves(
        step_losses,
        step_accuracies,
        epoch_losses,
        epoch_accuracies,
        len(dataloader),
        config.CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    train()
