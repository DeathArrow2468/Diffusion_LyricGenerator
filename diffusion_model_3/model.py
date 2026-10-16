# 03_diffusion_model/model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        # Cast the float32 math result to bfloat16 (matching the MLP weights)
        emb = emb.to(self.mlp[0].weight.dtype)
        
        # Pass the wave through the MLP to scale it up to the Transformer's hidden size
        return self.mlp(emb)


class BidirectionalDiffusionTransformer(nn.Module):
    """
    The wrapper that turns a standard autoregressive LLM into a bidirectional text diffusion model.
    """
    def __init__(self, model_name, tokenizer_len):
        super().__init__()
        
        # 1. Download the raw base model 
        # Using bfloat16 keeps memory low for the RTX 3060 and maximizes Tensor Core speed
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            dtype=torch.bfloat16
        )
        
        # 2. Override the config flag (Helps disable causal logic in some HF architectures)
        self.base_model.config.is_decoder = False
        
        # 3. Resize the model's vocabulary matrix to accept our structural and [MASK] tokens
        self.base_model.resize_token_embeddings(tokenizer_len)
        
        # 4. Fetch the hidden dimension size dynamically 
        self.hidden_dim = self.base_model.config.hidden_size
        
        # 5. Instantiate our Timestep MLP and ensure it uses the same dtype as the model
        self.time_embedder = TimestepEmbedder(self.hidden_dim).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask, timesteps):
        # 1. Bypass the outer forward function and grab the raw token embeddings directly
        input_embeddings = self.base_model.get_input_embeddings()(input_ids)
        
        # 2. Process the Timestep float into a dense vector
        t_emb = self.time_embedder(timesteps)
        
        # 3. Inject the Timestep into the sequence
        # input_embeddings shape: [Batch, Seq_Len, Hidden_Dim]
        # t_emb shape:            [Batch, 1, Hidden_Dim] (Broadcasts across Seq_Len)
        injected_embeddings = input_embeddings + t_emb.unsqueeze(1)
        
        # --- THE HACK: Custom 4D Attention Mask ---
        # If we just pass the 2D mask, the C++ backend will silently apply a causal triangle.
        # We must explicitly construct the 4D matrix [Batch, Heads, Seq_Len, Seq_Len]
        # and penalize padding tokens with -10000.0 so the softmax ignores them.
        batch_size, seq_length = input_ids.shape
        
        # Expand [Batch, Seq_Len] -> [Batch, 1, 1, Seq_Len]
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.base_model.dtype) 
        
        # Invert the mask: 0 for real tokens, -10000.0 for padding tokens
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 4. Pass the injected embeddings and the explicit 4D mask into the inner Transformer layers
        outputs = self.base_model.model(
            inputs_embeds=injected_embeddings,
            attention_mask=extended_attention_mask,
            use_cache=False, # Cache is strictly for autoregressive generation; it breaks diffusion
            output_hidden_states=True
        )
        
        # 5. Extract the final hidden states from the last Transformer block
        last_hidden_state = outputs.last_hidden_state
        
        # 6. Pass the states through the vocabulary head to get probability logits for every token
        logits = self.base_model.lm_head(last_hidden_state)
        
        return logits

# --- Quick Local Test ---
# Run this file directly to verify tensor mathematics and VRAM usage
if __name__ == "__main__":
    print("1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    new_tokens = config.STRUCTURAL_TOKENS + [config.SEP_TOKEN, config.MASK_TOKEN]
    tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
    
    print("2. Building the Bidirectional Diffusion Model (This will take a moment to load weights)...")
    # Move to 'cuda' to ensure the custom 4D mask operations work on the GPU
    model = BidirectionalDiffusionTransformer(config.MODEL_NAME, len(tokenizer)).to(device)
    
    print("3. Creating fake tensor data to simulate our DataLoader output...")
    batch_size = 2
    seq_len = 256
    
    # Fake tokens (random integers representing words)
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    # Fake attention mask (assuming all tokens are real/no padding)
    dummy_attention_mask = torch.ones((batch_size, seq_len)).to(device)
    # Fake timesteps (random floats simulating t)
    dummy_timesteps = torch.rand((batch_size,)).to(device)
    
    print("4. Pushing data through the bidirectional model...")
    # This executes the forward pass, proving the Timestep MLP and 4D mask integrate correctly
    logits = model(dummy_input_ids, dummy_attention_mask, dummy_timesteps)
    
    print(f"\nSuccess! Output Logits Shape: {logits.shape}")
    print(f"Expected Shape: [Batch Size ({batch_size}), Sequence Length ({seq_len}), Vocabulary Size ({len(tokenizer)})]")