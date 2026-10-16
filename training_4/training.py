# 04_training/train.py
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import config

# Add the parent directory to Python's path so we can import our previous modules
# Adjust the folder names below to exactly match your local folder structure.
sys.path.append(os.path.abspath('..'))
from masking_tokenization_2.tokenizer import setup_tokenizer, LyricDiffusionDataset, MaskingCollateFn
from diffusion_model_3.model import BidirectionalDiffusionTransformer

def train():
    # 1. Initialization
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Tokenizer and Data Engine...")
    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.DATASET_PATH)
    collate_fn = MaskingCollateFn(tokenizer, config.MAX_SEQ_LEN)
    
    # We use our micro-batch size here to prevent Out Of Memory (OOM) crashes
    dataloader = DataLoader(dataset, batch_size=config.MICRO_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print("Loading Bidirectional Diffusion Model...")
    model = BidirectionalDiffusionTransformer(config.MODEL_NAME, len(tokenizer)).to(device)
    
    # 2. Optimizer Setup
    # AdamW is the standard optimizer for Transformer models. 
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Starting Training! Effective Batch Size: {config.MICRO_BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    
    # 3. The Core Training Loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        # tqdm creates a nice progress bar in the terminal
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for step, batch in progress_bar:
            # Move all tensors to the GPU
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            timesteps = batch['timesteps'].to(device)
            
            # --- AUTOMATIC MIXED PRECISION (AMP) ---
            # We explicitly cast the math to bfloat16. This matches our model weights,
            # prevents datatype crashes (like the Float/BFloat16 error), and uses 
            # less VRAM while maintaining training stability.
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                
                # Forward Pass: Push the masked lyrics and prompt through the model
                logits = model(input_ids, attention_mask, timesteps)
                
                # --- LOSS CALCULATION ---
                # logits shape is [Batch, Seq_Len, Vocab_Size]. 
                # labels shape is [Batch, Seq_Len].
                # We flatten both so CrossEntropy can compare them word-by-word.
                # Because we filled 'labels' with -100 at all non-masked positions in Stage 2,
                # CrossEntropy automatically ignores the prompt and padding, and ONLY penalizes 
                # the model for mistakes made on the [MASK] tokens.
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    labels.view(-1)
                )
                
                # Scale the loss down by our accumulation steps.
                # If we accumulate 8 times, each micro-batch contributes 1/8th of the gradient.
                loss = loss / config.GRAD_ACCUM_STEPS
            
            # Backward Pass: Calculate the gradients (the adjustments needed for the weights)
            loss.backward()
            
            total_loss += loss.item() * config.GRAD_ACCUM_STEPS
            
            # --- GRADIENT ACCUMULATION ---
            # Only update the actual model weights once we have accumulated enough micro-batches
            if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            # Update the terminal progress bar with the current loss
            progress_bar.set_postfix({'loss': f"{loss.item() * config.GRAD_ACCUM_STEPS:.4f}"})
            
        # 4. Checkpointing
        # At the end of each epoch, save the weights so you can load them for inference
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"diffusion_epoch_{epoch+1}.pt")
        
        # We save the model's state_dict (the learned weights) and the optimizer state 
        # so training can be resumed later if needed.
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        print(f"Checkpoint saved to {checkpoint_path}\n")

if __name__ == "__main__":
    # If the system throws an import error, Python might be confused about folder names starting with numbers.
    # If that happens, rename your folders to remove the "01_", "02_" prefixes.
    train()