# 02_dataset_and_masking/config.py

# Model & Tokenizer
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # The 0.5B model fits perfectly in 8GB VRAM
MAX_SEQ_LEN = 256                 # Keep short for local testing; prevents OOM crashes

# Special Tokens
# The model needs to know these are atomic concepts, not random strings of letters.
STRUCTURAL_TOKENS = ["<VERSE_1>", "<VERSE_2>", "<VERSE_3>", "<VERSE_4>", "<VERSE_5>","<VERSE_6", "<CHORUS>", "<BRIDGE>", "<OUTRO>"]

# We need clear boundary tokens to separate the Prompt from the Lyrics, 
# and to corrupt the text during diffusion.
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

# Data Paths
TRAIN_DATA = r"C:\Users\Manav\OneDrive\Desktop\Diffusion_LyricGenerator\Dataset_maker_1\train_data.jsonl"