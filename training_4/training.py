# 04_training/train.py
import math
import os
import random
import sys

import matplotlib  
matplotlib.use("Agg")
import matplotlib.pyplot as plt  

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import config

# Add the parent directory to Python's path so we can import our previous modules
sys.path.append(os.path.abspath(".."))

from masking_tokenization_2.tokenizer import (
    setup_tokenizer,
    LyricDiffusionDataset,
    MaskingCollateFn,
)
from diffusion_model_3.model import (
    BidirectionalDiffusionTransformer,
)


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


def train():
    # ==========================================
    # 1. INITIALIZATION & SETUP
    # ==========================================
    set_seed(getattr(config, "SEED", 42))

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Automatically detect if you have an Nvidia GPU (CUDA) available, otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # if device.type != "cuda":
    #     raise RuntimeError(
    #         "This full Qwen FP32 training configuration needs a CUDA GPU."
    #     )

    print(f"Executing training on: {device.type.upper()}")

    print("Loading Tokenizer and Data Engine...")

    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.DATASET_PATH)

    collate_fn = MaskingCollateFn(
        tokenizer=tokenizer,
        max_len=config.MAX_SEQ_LEN,
        mask_epsilon=getattr(config, "MASK_EPSILON", 1e-3),
        full_mask_probability=getattr(
            config,
            "FULL_MASK_PROBABILITY",
            0.10,
        ),
    )

    # The DataLoader is our engine. It grabs raw JSON rows, passes them to collate_fn,
    # and hands us perfectly formatted, masked tensors.
    dataloader = DataLoader(
        dataset,
        batch_size=config.MICRO_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=getattr(config, "NUM_WORKERS", 0),
        pin_memory=True,
    )

    print("Loading Bidirectional Diffusion Model...")

    model = BidirectionalDiffusionTransformer(
        config.MODEL_NAME,
        len(tokenizer),
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
            # Move all our data from system RAM to GPU VRAM
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )

            timesteps = batch["timesteps"].to(
                device,
                non_blocking=True,
            )

            loss_weights = batch["loss_weights"].to(
                device,
                non_blocking=True,
            )

            target_lengths = batch["target_lengths"].to(
                device,
                non_blocking=True,
            )

            # --- AUTOMATIC MIXED PRECISION (AMP) ---
            # The model's actual stored weights stay in float32.
            # BF16 is used only for suitable GPU calculations, which saves VRAM and speeds training.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float32,
                enabled=True,
            ):
                # 1. Forward Pass: Push the noisy sequence through the bidirectional architecture
                logits = model(
                    input_ids,
                    attention_mask,
                    timesteps,
                )

                # 2. Loss Calculation
                # logits shape is [Batch, Seq_Len, Vocab_Size].
                # labels shape is [Batch, Seq_Len].
                #
                # CrossEntropy completely ignores prompt, padding, and clean lyric positions,
                # because those labels contain -100. It grades ONLY [MASK] positions.
                token_losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(labels)

                # Each masked token is weighted by 1 / mask_ratio.
                # This stops a highly masked song from unfairly dominating the loss.
                #
                # Dividing by target_lengths gives every lyric example equal importance,
                # even when songs contain different numbers of lyric tokens.
                example_losses = (
                    (token_losses * loss_weights).sum(dim=1)
                    / target_lengths
                )

                unscaled_loss = example_losses.mean()

            with torch.no_grad():  
                masked_positions = labels != -100  
                correct = (  
                    (logits.argmax(dim=-1) == labels) & masked_positions  
                ).sum().item()  
                masked_count = masked_positions.sum().item()  

            epoch_correct += correct  
            epoch_masked += masked_count  
            step_losses.append(unscaled_loss.item())  
            step_accuracies.append(  
                correct / masked_count if masked_count > 0 else float("nan")  
            )  

            if not torch.isfinite(unscaled_loss):
                raise RuntimeError(
                    f"Loss became invalid at epoch {epoch + 1}, "
                    f"step {step + 1}: {unscaled_loss.item()}"
                )

            # 3. Gradient Accumulation Math
            # If we accumulate 8 times, each micro-batch contributes 1/8th
            # of the final gradient. The final partial group uses its true size.
            current_group_start = (
                step // config.GRAD_ACCUM_STEPS
            ) * config.GRAD_ACCUM_STEPS

            current_group_size = min(
                config.GRAD_ACCUM_STEPS,
                len(dataloader) - current_group_start,
            )

            scaled_loss = unscaled_loss / current_group_size

            # 4. Backward Pass: Calculate the gradients (how much to adjust the weights)
            scaled_loss.backward()

            # Keep a running total of the true loss for our end-of-epoch average
            total_loss += unscaled_loss.item()

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
                optimizer.zero_grad(set_to_none=True)  # Wipe the accumulated gradients clean

            # Update progress bar to show both loss and the dynamically changing learning rate
            progress_bar.set_postfix(
                {
                    "loss": f"{unscaled_loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        # ==========================================
        # 4. BEST LOSS CHECKPOINTING
        # ==========================================
        avg_loss = total_loss / len(dataloader)

        epoch_losses.append(avg_loss)  
        epoch_accuracies.append(epoch_correct / max(epoch_masked, 1))  

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