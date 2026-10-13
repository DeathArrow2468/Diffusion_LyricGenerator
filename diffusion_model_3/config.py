# 03_diffusion_model/config.py

MODEL_NAME = "Qwen/Qwen2.5-0.5B" 

# These must perfectly match Stage 2 so the embedding matrix is resized correctly
STRUCTURAL_TOKENS = [
    "<VERSE_1>", "<VERSE_2>", "<VERSE_3>", "<VERSE_4>", "<VERSE_5>",
      "<VERSE_6", "<CHORUS>", "<BRIDGE>", "<OUTRO>"
]
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"