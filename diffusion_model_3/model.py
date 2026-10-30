import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
import math

class TimestepEmbedder(nn.Module):
    """
    Converts a single floating-point timestep 't' (e.g., 0.75)
    into a dense vector matching the Transformer's hidden dimension.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # We project the scalar 't' into a 256-dimensional sinusoidal wave
        self.sinusoidal_dim = 256

        # We pass that wave through a 2-layer neural network (MLP)
        # to expand it to the model's actual hidden dimension (e.g., 896 for Qwen-0.5B)
        self.mlp = nn.Sequential(
            nn.Linear(self.sinusoidal_dim, hidden_dim),
            nn.SiLU(), # Standard activation function for diffusion architectures
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Initialize the final linear layer to output near-zero values initially.
        # This prevents the random MLP weights from destroying the pre-trained Qwen embeddings.
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)


    def forward(self, timesteps):
        """
        Takes a batch of timesteps (shape: [Batch_size])
        and returns a vector (shape: [Batch_size, Hidden_dim])
        """
        # Create sinusoidal embeddings based on the Attention is All You Need math
        half_dim = self.sinusoidal_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)

        # Multiply timestep 't' by frequencies
        emb = timesteps[:, None] * emb[None, :]

        # Apply Sin and Cos to create the wave representation
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        # Match the MLP parameter dtype.
        emb = emb.to(self.mlp[0].weight.dtype)

        # Pass the wave through the MLP to scale it up to the Transformer's hidden size
        return self.mlp(emb)


class BidirectionalDiffusionTransformer(nn.Module):
    """
    The wrapper that turns a standard autoregressive LLM into a bidirectional text diffusion model.
    """
    def __init__(self, model_name, tokenizer_len, dtype=torch.float32):
        super().__init__()

        # 1. Download the raw base model
        # The caller chooses dtype; CPU defaults to float32.
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation="sdpa",
        )

        # 2. Keep the pretrained config; each forward constructs its attention mask.
        # Attention mode is selected explicitly per forward pass.

        # 3. Resize the model's vocabulary matrix to accept our structural and [MASK] tokens
        self.base_model.resize_token_embeddings(tokenizer_len)

        # 4. Fetch the hidden dimension size dynamically
        self.hidden_dim = self.base_model.config.hidden_size

        # 5. Instantiate our Timestep MLP and ensure it uses the same dtype as the model
        self.time_embedder = TimestepEmbedder(self.hidden_dim).to(dtype=dtype)

    def forward(self, input_ids, attention_mask, timesteps=None, mode="denoising"):
        """Shared weights, explicitly isolated causal or bidirectional attention."""
        if mode not in {"ar", "denoising"}:
            raise ValueError("mode must be ar or denoising")
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        if mode == "denoising":
            if timesteps is None:
                raise ValueError("Denoising requires timesteps")
            embeddings = embeddings + self.time_embedder(timesteps).unsqueeze(1)
        # Never inject diffusion-time embeddings in the original AR exercise.
        length = input_ids.size(1)
        allowed = attention_mask.bool()[:, None, None, :].expand(-1, 1, length, -1)
        if mode == "ar":
            positions = torch.arange(length, device=input_ids.device)
            causal = positions[:, None] >= positions[None, :]
            allowed = allowed & causal[None, None, :, :]
        additive_mask = torch.zeros(allowed.shape, device=input_ids.device,
                                    dtype=embeddings.dtype)
        additive_mask.masked_fill_(~allowed, torch.finfo(embeddings.dtype).min)
        outputs = self.base_model.model(
            inputs_embeds=embeddings, attention_mask=additive_mask,
            use_cache=False, output_hidden_states=False,
        )
        return self.base_model.lm_head(outputs.last_hidden_state)
