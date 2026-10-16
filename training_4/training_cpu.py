# 04_training/train.py
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import config

# Add the parent directory to Python's path so we can import our previous modules
sys.path.append(os.path.abspath('..'))
from masking_tokenization_2.tokenizer import setup_tokenizer, LyricDiffusionDataset, MaskingCollateFn
from diffusion_model_3.model import BidirectionalDiffusionTransformer

def train():
    # ==========================================
    # 1. INITIALIZATION & SETUP
    # ==========================================
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Automatically detect if you have an Nvidia GPU (CUDA) available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing training on: {device.type.upper()}")

    print("Loading Tokenizer and Data Engine...")
    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.DATASET_PATH)
    collate_fn = MaskingCollateFn(tokenizer, config.MAX_SEQ_LEN)

    # The DataLoader is our engine. It grabs raw JSON rows, passes them to collate_fn, 
    # and hands us perfectly formatted, masked tensors.
    dataloader = DataLoader(dataset, batch_size=config.MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print("Loading Bidirectional Diffusion Model...")
    model = BidirectionalDiffusionTransformer(config.MODEL_NAME, len(tokenizer)).to(device)

    # ==========================================
    # 2. OPTIMIZER & SCHEDULER (THE FIX)
    # ==========================================
    # AdamW is the gold standard for Transformers. We add a tiny bit of weight_decay (0.01) 
    # to prevent the model weights from growing too large and overfitting.
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    # Calculate exactly how many times we will update the weights across all epochs.
    # We divide by GRAD_ACCUM_STEPS because we only step the optimizer once per macro-batch.
    total_steps = (len(dataloader) // config.GRAD_ACCUM_STEPS) * config.EPOCHS

    # The Warmup Scheduler: 
    # Instead of blasting the model with the full learning rate on step 1 (which destroys pre-trained weights),
    # this starts the LR at 0.0 and slowly ramps it up over the first 100 steps, then slowly decays it.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.WARMUP_STEPS, 
        num_training_steps=total_steps
    ) 

    print(f"Starting Training! Effective Batch Size: {config.MICRO_BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    
    # We track the best loss to ensure we only save models that are actively improving
    best_loss = float('inf')

    # ==========================================
    # 3. THE CORE TRAINING LOOP
    # ==========================================
    for epoch in range(config.EPOCHS):
        model.train() # Put the model in training mode (enables dropout, etc.)
        total_loss = 0

        # Create a beautiful terminal progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for step, batch in progress_bar:
            # Move all our data from system RAM to the active device (CPU or GPU)
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timesteps = batch['timesteps'].to(device)

            # --- AUTOMATIC MIXED PRECISION (AMP) ---
            # device.type dynamically inserts 'cuda' or 'cpu'.
            # We explicitly cast the forward pass to bfloat16. This matches our model weights
            # and prevents datatype crash errors (Float vs BFloat16).
            with torch.amp.autocast(device.type, dtype=torch.bfloat16):
                
                # 1. Forward Pass: Push the noisy sequence through the bidirectional architecture
                logits = model(input_ids, attention_mask, timesteps)

                # 2. Loss Calculation
                # logits shape is [Batch, Seq_Len, Vocab_Size]. labels shape is [Batch, Seq_Len].
                # We flatten both using .view(-1) so CrossEntropy can compare them word-by-word.
                # Because we filled 'labels' with -100 at all clean positions in Stage 2,
                # CrossEntropy completely ignores the prompt and padding, and ONLY grades the [MASK] tokens.
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )

                # 3. Gradient Accumulation Math
                # If we accumulate 8 times, each micro-batch only contributes 1/8th of the final gradient.
                loss = loss / config.GRAD_ACCUM_STEPS

            # 4. Backward Pass: Calculate the gradients (how much to adjust the weights)
            loss.backward()
            
            # Keep a running total of the true loss for our end-of-epoch average
            total_loss += loss.item() * config.GRAD_ACCUM_STEPS

            # --- GRADIENT ACCUMULATION & CLIPPING ---
            # Only update the actual model weights once we have accumulated a full macro-batch
            if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                
                # Gradient Clipping: If the math resulted in a massive gradient spike, 
                # we cap it at a maximum norm of 1.0. This prevents the model from exploding.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()     # Apply the weight updates
                scheduler.step()     # Slightly adjust the learning rate
                optimizer.zero_grad() # Wipe the accumulated gradients clean for the next batch

            # Update progress bar to show both loss and the dynamically changing learning rate
            progress_bar.set_postfix({
                'loss': f"{loss.item() * config.GRAD_ACCUM_STEPS:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # ==========================================
        # 4. BEST LOSS CHECKPOINTING
        # ==========================================
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # We only save the file if this epoch performed strictly better than our previous best.
        # This guarantees that 'best_model.pt' is always the most capable version of your model.
        if avg_loss < best_loss:
            print(f"Loss improved from {best_loss:.4f} to {avg_loss:.4f}. Saving new best model...")
            best_loss = avg_loss
            
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            
            # Save a dictionary containing everything needed to use or resume the model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            print(f"Checkpoint saved to {checkpoint_path}\n")
        else:
            print(f"Loss did not improve from {best_loss:.4f}. Skipping checkpoint.\n")

if __name__ == "__main__":
    train()