"""Construct clean AR inputs and prefix + corrupted block denoising inputs."""
import torch


class HybridCollateFn:
    def __init__(self, tokenizer, max_len, config, sep_token="[SEP]", mask_token="[MASK]"):
        self.tokenizer, self.max_len, self.config = tokenizer, max_len, config
        self.sep_id = tokenizer.convert_tokens_to_ids(sep_token)
        self.mask_id = tokenizer.convert_tokens_to_ids(mask_token)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("Tokenizer requires a padding token")

    def __call__(self, examples):
        # Collation intentionally runs on CPU for DataLoader compatibility.
        # Training moves every returned tensor to the selected device.
        device = torch.device("cpu")
        rows = []
        for example in examples:
            tok = self.tokenizer
            bos = [] if tok.bos_token_id is None else [tok.bos_token_id]
            prompt = tok.encode(example["prompt"], add_special_tokens=False)
            lyrics = tok.encode(example["lyrics"], add_special_tokens=False)
            if not lyrics:
                raise ValueError("Empty lyrics: no training targets")
            # Prompt is fixed conditioning. Reserve at least one lyric token.
            prompt_budget = self.max_len - len(bos) - 2
            if prompt_budget < 0:
                raise ValueError("max_len is too small for prompt separator and lyrics")
            context = bos + prompt[:prompt_budget] + [self.sep_id]
            lyrics = lyrics[:self.max_len - len(context)]
            clean = torch.tensor(context + lyrics, dtype=torch.long, device=device)
            n = len(lyrics)
            c = (int(torch.randint(n, (), device=device))
                 if self.config.prefix_policy == "uniform" else 0)
            prefix_end = len(context) + c - 1
            end = min(clean.numel(), prefix_end + 1 + self.config.max_target_tokens)
            tau = torch.empty((), device=device).uniform_(self.config.noise_min, self.config.noise_max)
            positions = torch.arange(prefix_end + 1, end, device=device)
            selected = torch.rand(positions.numel(), device=device) < tau
            if not selected.any():
                selected[torch.randint(positions.numel(), (), device=device)] = True
            masked = positions[selected]

            ar_input = torch.full((self.max_len,), self.pad_id, dtype=torch.long, device=device)
            ar_input[:clean.numel()] = clean
            ar_attention = torch.arange(self.max_len, device=device) < clean.numel()
            ar_labels = torch.full_like(ar_input, self.config.ignore_index)
            # Lyric-only AR objective, conditioned on prompt/SEP. No EOS target
            # added here, matching this project's existing lyric-only task.
            ar_labels[len(context):clean.numel()] = clean[len(context):]
            den_input = ar_input.clone()
            den_input[end:] = self.pad_id  # No context beyond the selected block.
            den_input[masked] = self.mask_id
            den_labels = torch.full_like(ar_input, self.config.ignore_index)
            den_labels[masked] = clean[masked]
            rows.append(dict(
                ar_input_ids=ar_input, ar_labels=ar_labels,
                ar_attention_mask=ar_attention.long(), input_ids=den_input,
                labels=den_labels,
                attention_mask=(torch.arange(self.max_len, device=device) < end).long(),
                timesteps=tau, prefix_end=torch.tensor(prefix_end, device=device),
            ))
        return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
