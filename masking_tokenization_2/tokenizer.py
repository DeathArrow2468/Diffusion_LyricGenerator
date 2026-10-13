# 02_dataset_and_masking/dataset.py
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import config

def setup_tokenizer():
    # 1. Download the pre-trained BPE dictionary for our specific model (e.g., Qwen)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # 2. Combine our custom song structure tokens with the SEP and MASK tokens
    new_tokens = config.STRUCTURAL_TOKENS + [config.SEP_TOKEN, config.MASK_TOKEN]
    
    # 3. Inject them into the tokenizer so they are treated as atomic IDs
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    
    # 4. Handle padding token if the base model doesn't have one defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer

class LyricDiffusionDataset(Dataset):
    """A standard PyTorch Dataset. It loads the JSONL file from disk into memory."""
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

class MaskingCollateFn:
    """
    The 'collate_fn' intercepts the raw data right before it goes to the GPU.
    This applies the smart truncation and dynamic diffusion masking.
    """
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Cache special token IDs in memory to avoid slow string lookups during training
        self.sep_id = tokenizer.convert_tokens_to_ids(config.SEP_TOKEN)
        self.mask_id = tokenizer.convert_tokens_to_ids(config.MASK_TOKEN)
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        attention_masks_list = []
        timesteps_list = []

        for item in batch:
            # Step 1: Convert text into integers (IDs) without adding default special tokens yet
            prompt_tokens = self.tokenizer.encode(item["prompt"], add_special_tokens=False)
            lyric_tokens = self.tokenizer.encode(item["lyrics"], add_special_tokens=False)
            
            # Step 2: Safely handle missing BOS/EOS tokens (The Fix for Qwen models)
            # If the tokenizer lacks a BOS token (like Qwen), this creates an empty list []
            # instead of crashing with a NoneType error when building the tensor.
            bos_list = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
            eos_list = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            
            # Step 3: Smart Truncation
            # Calculate exactly how many tokens the prompt, BOS, EOS, and SEP will consume.
            reserved_space = len(prompt_tokens) + len(bos_list) + 1 + len(eos_list) # 1 is for [SEP]
            available_lyric_space = self.max_len - reserved_space

            # Chop off the end of the lyrics if they exceed the remaining space.
            # This mathematically guarantees the [SEP] and prompt are never truncated.
            if len(lyric_tokens) > available_lyric_space:
                lyric_tokens = lyric_tokens[:available_lyric_space]

            # Step 4: Assemble the Sequence safely using list concatenation
            seq = bos_list + prompt_tokens + [self.sep_id] + lyric_tokens + eos_list

            # Step 5: Pad sequence with pad_token up to max_len
            if len(seq) < self.max_len:
                seq = seq + [self.pad_id] * (self.max_len - len(seq))
                
            seq_tensor = torch.tensor(seq, dtype=torch.long)
            
            # Step 6: Find the exact integer index of the [SEP] boundary
            sep_idx = (seq_tensor == self.sep_id).nonzero(as_tuple=True)[0]
            sep_pos = sep_idx[0].item()
            
            # Step 7: Roll the diffusion timestep (t)
            # t controls how much of the sequence gets masked (0.1 = 10%, 1.0 = 100%)
            t = random.uniform(0.1, 1.0) 
            mask_ratio = t
            timesteps_list.append(t)
            
            # Step 8: Initialize the labels tensor
            # PyTorch's CrossEntropyLoss completely ignores any target labeled as -100.
            labels = torch.full_like(seq_tensor, -100)
            
            # Step 9: Find valid indices allowed to be masked
            # Must be strictly AFTER the [SEP] position, and cannot be PAD or EOS tokens.
            lyric_indices = torch.where(
                (torch.arange(self.max_len) > sep_pos) & 
                (seq_tensor != self.pad_id) & 
                (seq_tensor != self.tokenizer.eos_token_id)
            )[0]
            
            num_to_mask = max(1, int(len(lyric_indices) * mask_ratio))
            
            if len(lyric_indices) > 0:
                # Shuffle valid indices and pick the exact number to mask
                mask_indices = lyric_indices[torch.randperm(len(lyric_indices))[:num_to_mask]]
                
                # Copy the original, correct token IDs into the -100 label tensor ONLY at masked spots
                labels[mask_indices] = seq_tensor[mask_indices]
                
                # Replace the original token in the input sequence with the [MASK] ID
                seq_tensor[mask_indices] = self.mask_id

            # Step 10: Create the Attention Mask (1 for real data, 0 for padding)
            attention_mask = (seq_tensor != self.pad_id).long()

            input_ids_list.append(seq_tensor)
            labels_list.append(labels)
            attention_masks_list.append(attention_mask)

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_masks_list),
            "timesteps": torch.tensor(timesteps_list, dtype=torch.float32)
        }


# --- Quick Local Test ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("1. Loading Tokenizer and Dataset...")
    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.TRAIN_DATA)
    collate_fn = MaskingCollateFn(tokenizer, config.MAX_SEQ_LEN)
    
    print("2. Firing up the DataLoader engine...")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print("3. Pulling exactly ONE batch through the masking engine...\n")
    batch = next(iter(dataloader))
    
    print(f"Input IDs shape: {batch['input_ids'].shape} --> (Batch Size, Sequence Length)")
    print(f"Labels shape:    {batch['labels'].shape}")
    print(f"Timesteps (t):   {batch['timesteps']}\n")
    
    print("--- Let's see what the GPU actually reads for Song 1 ---")
    decoded = tokenizer.decode(batch['input_ids'][0])
    print(decoded)