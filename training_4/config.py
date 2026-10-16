# 04_training/config.py

# Data and Model Paths
# Update these folder names if your local directories are named differently 
# (e.g., masking_tokenization_2 or diffusion_model_3)
DATASET_PATH = r"C:\Users\Faiz\Desktop\Diffusion_LyricGenerator\Dataset_maker_1\train_data.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_SEQ_LEN = 256

# Hardware Optimization for 8GB VRAM
MICRO_BATCH_SIZE = 2      # How many songs fit in VRAM at once
GRAD_ACCUM_STEPS = 8      # 2 * 8 = Effective Batch Size of 16

# Training Hyperparameters
LEARNING_RATE = 1e-4
EPOCHS = 7
CHECKPOINT_DIR = "./checkpoints"

################ MAKE STRICT PYTHON PACKAGES LATER #####################

STRUCTURAL_TOKENS = ["<VERSE_1>", "<VERSE_2>", "<VERSE_3>", "<VERSE_4>", "<VERSE_5>","<VERSE_6", "<CHORUS>", "<BRIDGE>", "<OUTRO>"]

# We need clear boundary tokens to separate the Prompt from the Lyrics, 
# and to corrupt the text during diffusion.
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"