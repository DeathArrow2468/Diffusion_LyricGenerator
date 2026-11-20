# 01_dataset_maker/config.py

OLLAMA_HOST = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

INPUT_JSONL = "Genius_Dataset/raw_data.jsonl"
OUTPUT_JSONL = "train_data.jsonl"

SYSTEM_PROMPT = """
You are a synthetic data generator for a text diffusion model. 
I will give you a song's Title, Artist, and Description. 
Write 3 diverse, natural language prompts that a user might type to request a song like this.
DO NOT include the artist's name in the prompt (we want general prompts).
If the Description is "None" or missing, infer the genre and mood strictly from the Title and Artist.
Output ONLY the 3 prompts separated by a newline character. No numbering, no markdown, no intro text.
"""