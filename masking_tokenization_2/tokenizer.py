# 02_dataset_and_masking/dataset.py
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import config


def setup_tokenizer():
    # 1. Download the pre-trained BPE dictionary for our specific model (e.g., Qwen)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # 2. Combine our custom song structure tokens with the SEP and MASK tokens
    new_tokens = (
        config.STRUCTURAL_TOKENS
        + [config.SEP_TOKEN, config.MASK_TOKEN]
    )

    # 3. Inject them into the tokenizer so they are treated as atomic IDs
    tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens}
    )

    # 4. Handle padding token if the base model doesn't have one defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class LyricDiffusionDataset(Dataset):
    """A standard PyTorch Dataset. It loads the JSONL file from disk into memory."""

    def __init__(self, jsonl_path):
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                item = json.loads(line)

                prompt = str(item.get("prompt", "")).strip()
                lyrics = str(item.get("lyrics", "")).strip()

                # Empty rows would later produce an invalid loss,
                # so remove them before training begins.
                if not prompt or not lyrics:
                    print(
                        f"Skipping invalid dataset row {line_number}: "
                        "prompt or lyrics are missing."
                    )
                    continue

                self.data.append(
                    {
                        "prompt": prompt,
                        "lyrics": lyrics,
                    }
                )

        if not self.data:
            raise ValueError("No valid training examples were found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MaskingCollateFn:
    """
    The 'collate_fn' intercepts the raw data right before it goes to the GPU.
    This applies the smart truncation and dynamic diffusion masking.
    """

    def __init__(
        self,
        tokenizer,
        max_len,
        mask_epsilon=1e-3,
        full_mask_probability=0.10,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_epsilon = mask_epsilon
        self.full_mask_probability = full_mask_probability

        # Cache special token IDs in memory to avoid slow string lookups during training
        self.sep_id = tokenizer.convert_tokens_to_ids(config.SEP_TOKEN)
        self.mask_id = tokenizer.convert_tokens_to_ids(config.MASK_TOKEN)
        self.pad_id = tokenizer.pad_token_id

        if self.sep_id is None or self.mask_id is None:
            raise ValueError("Could not find the custom [SEP] or [MASK] token.")

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        attention_masks_list = []
        timesteps_list = []
        loss_weights_list = []
        target_lengths_list = []

        for item in batch:
            # Step 1: Convert text into integers (IDs) without adding default special tokens yet
            prompt_tokens = self.tokenizer.encode(
                item["prompt"],
                add_special_tokens=False,
            )

            lyric_tokens = self.tokenizer.encode(
                item["lyrics"],
                add_special_tokens=False,
            )

            # Step 2: Build the prompt and lyric boundary.
            # We do not add EOS because Qwen uses the same ID for EOS and PAD.
            # This prototype generates fixed-length lyric sections.
            prefix = prompt_tokens + [self.sep_id]

            # Step 3: Smart Truncation
            # Calculate exactly how much lyric space remains after the prompt and [SEP].
            available_lyric_space = self.max_len - len(prefix)

            if available_lyric_space <= 0:
                raise ValueError(
                    "A prompt is longer than MAX_SEQ_LEN. "
                    "Shorten the prompt or increase MAX_SEQ_LEN."
                )

            # Chop off the end of the lyrics if they exceed the remaining space.
            lyric_tokens = lyric_tokens[:available_lyric_space]

            if not lyric_tokens:
                raise ValueError(
                    "A training row has no usable lyric tokens."
                )

            # Step 4: Assemble the clean sequence safely using list concatenation
            clean_sequence = prefix + lyric_tokens
            sequence_length = len(clean_sequence)

            # Step 5: Pad sequence with pad_token up to max_len
            seq_tensor = torch.full(
                (self.max_len,),
                fill_value=self.pad_id,
                dtype=torch.long,
            )

            seq_tensor[:sequence_length] = torch.tensor(
                clean_sequence,
                dtype=torch.long,
            )

            # We build attention from sequence length instead of checking token IDs.
            # This matters because Qwen's PAD and EOS tokens share the same ID.
            attention_mask = torch.zeros(
                self.max_len,
                dtype=torch.long,
            )

            attention_mask[:sequence_length] = 1

            lyric_start = len(prefix)
            lyric_end = sequence_length
            lyric_length = lyric_end - lyric_start

            # Step 6: Roll the diffusion timestep (t).
            # Most examples get a random mask percentage.
            # 10% are fully masked so training includes the exact state
            # the model sees at the beginning of generation.
            if random.random() < self.full_mask_probability:
                mask_ratio = 1.0
            else:
                raw_t = random.random()

                mask_ratio = (
                    (1.0 - self.mask_epsilon) * raw_t
                    + self.mask_epsilon
                )

            timesteps_list.append(mask_ratio)

            # Step 7: Initialize the labels tensor.
            # PyTorch's CrossEntropyLoss completely ignores any target labeled as -100.
            labels = torch.full(
                (self.max_len,),
                fill_value=-100,
                dtype=torch.long,
            )

            # Step 8: Choose every lyric token independently.
            # This differs from selecting an exact number of positions:
            # it matches the usual masked diffusion corruption process.
            should_mask = torch.rand(lyric_length) < mask_ratio

            # Every item must have at least one masked token to predict.
            if not should_mask.any():
                random_position = torch.randint(
                    low=0,
                    high=lyric_length,
                    size=(1,),
                )

                should_mask[random_position] = True

            lyric_positions = torch.arange(
                lyric_start,
                lyric_end,
                dtype=torch.long,
            )

            mask_positions = lyric_positions[should_mask]

            # Copy the original, correct token IDs into the -100 label tensor
            # ONLY at masked spots.
            labels[mask_positions] = seq_tensor[mask_positions]

            # Replace the original token in the input sequence with the [MASK] ID.
            seq_tensor[mask_positions] = self.mask_id

            # Step 9: Weight masked-token losses by 1 / mask_ratio.
            # This prevents high-noise and low-noise examples from being scored unfairly.
            loss_weights = torch.zeros(
                self.max_len,
                dtype=torch.float32,
            )

            loss_weights[mask_positions] = 1.0 / mask_ratio

            input_ids_list.append(seq_tensor)
            labels_list.append(labels)
            attention_masks_list.append(attention_mask)
            loss_weights_list.append(loss_weights)
            target_lengths_list.append(lyric_length)

        return {
            "input_ids": torch.stack(input_ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attention_masks_list),
            "timesteps": torch.tensor(
                timesteps_list,
                dtype=torch.float32,
            ),
            "loss_weights": torch.stack(loss_weights_list),
            "target_lengths": torch.tensor(
                target_lengths_list,
                dtype=torch.float32,
            ),
        }


# --- Quick Local Test ---
if __name__ == "__main__":
    print("1. Loading Tokenizer and Dataset...")

    tokenizer = setup_tokenizer()
    dataset = LyricDiffusionDataset(config.TRAIN_DATA)

    collate_fn = MaskingCollateFn(
        tokenizer,
        config.MAX_SEQ_LEN,
    )

    print("2. Firing up the DataLoader engine...")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
    )

    print("3. Pulling exactly ONE batch through the masking engine...\n")

    batch = next(iter(dataloader))

    print(
        f"Input IDs shape: {batch['input_ids'].shape} "
        "--> (Batch Size, Sequence Length)"
    )

    print(f"Labels shape:    {batch['labels'].shape}")
    print(f"Timesteps (t):   {batch['timesteps']}\n")

    print("--- Let's see what the GPU actually reads for Song 1 ---")

    decoded = tokenizer.decode(batch["input_ids"][0])
    print(decoded)