"""Generate one lyric sample with the trained model and append it to results.csv."""

import csv
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "training_4" / "checkpoints" / "best_model.pt"
RESULTS = Path(__file__).resolve().parent / "results.csv"
LYRIC_TOKENS = 128
STEPS = 32
TEMPERATURE = 0.8
TOP_K = 50
SEED = 42

OLD_FIELDS = ["Prompt", "Generated Lyrics", "Time taken", "Base Model"]
FIELDS = OLD_FIELDS + ["comments"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_result(row):
    """Add the comments column to an old CSV, then append exactly one row."""
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS.exists() and RESULTS.stat().st_size:
        with RESULTS.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            header = next(reader, None)
            if header == OLD_FIELDS:
                previous_rows = list(reader)
            elif header == FIELDS:
                previous_rows = None
            else:
                raise ValueError(f"Unexpected CSV columns in {RESULTS}: {header}")

        if previous_rows is not None:
            if any(len(old) != len(OLD_FIELDS) for old in previous_rows):
                raise ValueError("Existing CSV contains a row with an unexpected number of columns")
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="", dir=RESULTS.parent,
                    prefix="results.", suffix=".tmp", delete=False,
                ) as file:
                    temporary = Path(file.name)
                    writer = csv.writer(file)
                    writer.writerow(FIELDS)
                    writer.writerows([*old, ""] for old in previous_rows)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary, RESULTS)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    new_file = not RESULTS.exists() or RESULTS.stat().st_size == 0
    with RESULTS.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({key: " ".join(str(row[key]).splitlines()) for key in FIELDS})


def generate(model, tokenizer, cfg, prompt, device):
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    sep = tokenizer.convert_tokens_to_ids(cfg.SEP_TOKEN)
    mask = tokenizer.convert_tokens_to_ids(cfg.MASK_TOKEN)
    prefix = bos + tokenizer.encode(prompt, add_special_tokens=False) + [sep]
    available = cfg.MAX_SEQ_LEN - len(prefix) - len(eos)
    if LYRIC_TOKENS > available:
        raise ValueError(
            f"Prompt leaves room for {max(0, available)} lyric tokens; "
            f"shorten it or reduce LYRIC_TOKENS ({LYRIC_TOKENS})."
        )

    start = len(prefix)
    sequence = prefix + [mask] * LYRIC_TOKENS + eos
    sequence += [tokenizer.pad_token_id] * (cfg.MAX_SEQ_LEN - len(sequence))
    ids = torch.tensor([sequence], device=device)
    attention = (ids != tokenizer.pad_token_id).long()
    remaining = torch.ones(LYRIC_TOKENS, dtype=torch.bool, device=device)

    structure_ids = set(tokenizer.convert_tokens_to_ids(cfg.STRUCTURAL_TOKENS))
    forbidden = set(tokenizer.all_special_ids) - structure_ids
    forbidden.update((sep, mask, tokenizer.pad_token_id))
    if tokenizer.eos_token_id is not None:
        forbidden.add(tokenizer.eos_token_id)
    forbidden = sorted(token for token in forbidden if token is not None)

    steps = min(STEPS, LYRIC_TOKENS)
    with torch.inference_mode():
        for step in range(steps):
            positions = torch.where(remaining)[0]
            timestep = max(0.1, positions.numel() / LYRIC_TOKENS)
            t = torch.tensor([timestep], device=device, dtype=torch.float32)
            scores = model(ids, attention, t)[0, start + positions, :].float()
            scores[:, forbidden] = -float("inf")
            if not torch.isfinite(scores).any(dim=-1).all():
                raise RuntimeError("Model produced no valid token scores")

            target_filled = math.ceil(LYRIC_TOKENS * (step + 1) / steps)
            count = target_filled - (LYRIC_TOKENS - positions.numel())
            chosen = torch.randperm(positions.numel(), device=device)[:count]
            scores = scores[chosen]

            if TEMPERATURE == 0:
                tokens = scores.argmax(dim=-1)
            elif TOP_K > 0:
                k = min(TOP_K, scores.shape[-1] - len(forbidden))
                values, indices = (scores / TEMPERATURE).topk(k, dim=-1)
                tokens = indices.gather(1, torch.multinomial(values.softmax(-1), 1)).squeeze(1)
            else:
                tokens = torch.multinomial((scores / TEMPERATURE).softmax(-1), 1).squeeze(1)

            accepted = positions[chosen]
            ids[0, start + accepted] = tokens
            remaining[accepted] = False

    return tokenizer.decode(
        ids[0, start:start + LYRIC_TOKENS].tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def main():
    config_path = ROOT / "training_4" / "config.py"
    model_path = ROOT / "diffusion_model_3" / "model.py"
    for path in (CHECKPOINT, config_path, model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    cfg = load_module("_lyric_training_config", config_path)
    previous_config = sys.modules.get("config")
    sys.modules["config"] = cfg  # The project's model.py imports config.
    try:
        model_module = load_module("_lyric_diffusion_model", model_path)
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    print(f"Device: {device}; dtype: {dtype}; checkpoint: {CHECKPOINT}")
    print("Loading tokenizer and weights (excluded from generation timing)...")

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    tokenizer.add_special_tokens({
        "additional_special_tokens": cfg.STRUCTURAL_TOKENS + [cfg.SEP_TOKEN, cfg.MASK_TOKEN]
    })
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Training tokenizer requires a PAD or EOS token")

    model = model_module.BidirectionalDiffusionTransformer(cfg.MODEL_NAME, len(tokenizer))
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError("Checkpoint must contain model_state_dict")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    print(
        f"Checkpoint metadata: epoch={saved.get('epoch', 'unknown')}; "
        f"training loss={saved.get('loss', 'unknown')}"
    )
    del saved
    model.to(device=device, dtype=dtype).eval()

    print(f"Results append to: {RESULTS}")
    print("Enter one prompt. One result will be appended, then the program will exit.")
    try:
        prompt = input("\nPrompt: ")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        lyrics = generate(model, tokenizer, cfg, prompt, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

        print(f"\n{lyrics}\n\nGeneration: {elapsed:.3f} seconds.")
        comments = input("Comments (press Enter to leave blank): ")
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no result appended.")
        return

    append_result({
        "Prompt": prompt,
        "Generated Lyrics": lyrics,
        "Time taken": f"{elapsed:.6f}",
        "Base Model": False,
        "comments": comments,
    })
    print(f"Appended one result to {RESULTS}")


if __name__ == "__main__":
    main()
