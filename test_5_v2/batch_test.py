"""Offline, multi-prompt evaluation of one training job's diffusion checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["prompt", "output", "model_job_id"]


def read_prompts(path):
    """JSON supports multiline prompts; TXT supports one prompt per nonempty line."""
    text = Path(path).read_text(encoding="utf-8-sig")
    if Path(path).suffix.lower() == ".json":
        prompts = json.loads(text)
    else:
        prompts = [line for line in text.splitlines() if line.strip()]
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Provide a nonempty JSON array of strings or a TXT prompt list")
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Every prompt must be a nonempty string")
    return prompts


def write_results(prompts, model_job_id, path, generate_one):
    """Keep completed rows on failure; publish results.csv only when all succeed."""
    path = Path(path)
    partial = path.with_suffix(".partial.csv")
    if path.exists() or partial.exists():
        raise FileExistsError("Result file already exists; choose a new output directory")
    with partial.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for index, prompt in enumerate(prompts):
            started = time.monotonic()
            print(f"Prompt {index+1}/{len(prompts)}: generating...", flush=True)
            output = generate_one(index, prompt)
            writer.writerow(dict(prompt=prompt, output=output, model_job_id=model_job_id))
            handle.flush()
            os.fsync(handle.fileno())
            print(f"Prompt {index+1}/{len(prompts)} saved ({time.monotonic()-started:.1f}s)", flush=True)
    os.replace(partial, path)


def generate(model, tokenizer, cfg, prompt, device, options):
    import torch
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    sep = tokenizer.convert_tokens_to_ids(cfg.SEP_TOKEN)
    mask = tokenizer.convert_tokens_to_ids(cfg.MASK_TOKEN)
    prefix = bos + tokenizer.encode(prompt, add_special_tokens=False) + [sep]
    available = cfg.MAX_SEQ_LEN - len(prefix) - len(eos)
    if options.lyric_tokens > available:
        raise ValueError(
            f"Prompt leaves room for {max(0, available)} lyric tokens; "
            f"shorten it or reduce options.lyric_tokens ({options.lyric_tokens})."
        )

    start = len(prefix)
    sequence = prefix + [mask] * options.lyric_tokens + eos
    sequence += [tokenizer.pad_token_id] * (cfg.MAX_SEQ_LEN - len(sequence))
    ids = torch.tensor([sequence], device=device)
    attention = torch.zeros_like(ids)
    attention[:, :len(prefix) + options.lyric_tokens + len(eos)] = 1
    remaining = torch.ones(options.lyric_tokens, dtype=torch.bool, device=device)

    structure_ids = set(tokenizer.convert_tokens_to_ids(cfg.STRUCTURAL_TOKENS))
    forbidden = set(tokenizer.all_special_ids) - structure_ids
    forbidden.update((sep, mask, tokenizer.pad_token_id))
    if tokenizer.eos_token_id is not None:
        forbidden.add(tokenizer.eos_token_id)
    forbidden = sorted(token for token in forbidden if token is not None)

    steps = min(options.steps, options.lyric_tokens)
    with torch.inference_mode():
        for step in range(steps):
            positions = torch.where(remaining)[0]
            timestep = max(0.1, positions.numel() / options.lyric_tokens)
            t = torch.tensor([timestep], device=device, dtype=torch.float32)
            scores = model(ids, attention, t)[0, start + positions, :].float()
            scores[:, forbidden] = -float("inf")
            if not torch.isfinite(scores).any(dim=-1).all():
                raise RuntimeError("Model produced no valid token scores")

            target_filled = math.ceil(options.lyric_tokens * (step + 1) / steps)
            count = target_filled - (options.lyric_tokens - positions.numel())
            chosen = torch.randperm(positions.numel(), device=device)[:count]
            scores = scores[chosen]

            if options.temperature == 0:
                tokens = scores.argmax(dim=-1)
            elif options.top_k > 0:
                k = min(options.top_k, scores.shape[-1] - len(forbidden))
                values, indices = (scores / options.temperature).topk(k, dim=-1)
                tokens = indices.gather(1, torch.multinomial(values.softmax(-1), 1)).squeeze(1)
            else:
                tokens = torch.multinomial((scores / options.temperature).softmax(-1), 1).squeeze(1)

            accepted = positions[chosen]
            ids[0, start + accepted] = tokens
            remaining[accepted] = False

    return tokenizer.decode(
        ids[0, start:start + options.lyric_tokens].tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-job-id", required=True, help="Training job ID, e.g. 7639")
    parser.add_argument("--models-root", type=Path, default=ROOT / "models")
    parser.add_argument("--prompts", type=Path, default=ROOT / "test_5/prompts.json")
    parser.add_argument("--base-model", type=Path, default=ROOT / "pretrained/Qwen2.5-0.5B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lyric-tokens", type=int, default=128)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.model_job_id):
        parser.error("model-job-id must be a numeric training job ID")
    if args.lyric_tokens < 1 or args.steps < 1 or args.top_k < 0:
        parser.error("lyric-tokens and steps must be positive; top-k must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    prompts = read_prompts(args.prompts)
    model_dir = args.models_root.resolve() / f"run-{args.model_job_id}"
    checkpoint = model_dir / "best_model.pt"
    config_file = model_dir / "run_config.json"
    for path in (checkpoint, config_file, model_dir / "tokenizer_config.json",
                 args.base_model / "config.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
    # Do not use today's training config for yesterday's checkpoint.
    cfg = SimpleNamespace(**json.loads(config_file.read_text(encoding="utf-8")))
    for key in ("STRUCTURAL_TOKENS", "SEP_TOKEN", "MASK_TOKEN", "MAX_SEQ_LEN"):
        if not hasattr(cfg, key):
            raise ValueError(f"Saved run configuration is missing {key}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.csv"
    if result_path.exists() or result_path.with_suffix(".partial.csv").exists():
        raise FileExistsError("Output already contains results; use a new output directory")
    (args.output_dir / "prompts.json").write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")

    # Set offline mode before importing Hugging Face libraries.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(ROOT))
    from diffusion_model_3.model import BidirectionalDiffusionTransformer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing to silently test on CPU")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Saved tokenizer has no pad token")
    vocab = tokenizer.get_vocab()
    if any(token not in vocab for token in cfg.STRUCTURAL_TOKENS + [cfg.SEP_TOKEN, cfg.MASK_TOKEN]):
        raise ValueError("Saved tokenizer is missing training special tokens")

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or "model_state_dict" not in saved:
        raise ValueError("Checkpoint has no model_state_dict")
    if saved.get("tokenizer_length") != len(tokenizer):
        raise ValueError("Checkpoint and saved tokenizer vocabulary sizes differ")
    cfg.MAX_SEQ_LEN = int(saved.get("max_seq_len", cfg.MAX_SEQ_LEN))
    # Check every prompt before allocating the model on GPU.
    for index, prompt in enumerate(prompts, 1):
        used = len(tokenizer.encode(prompt, add_special_tokens=False)) + 1
        used += int(tokenizer.bos_token_id is not None) + int(tokenizer.eos_token_id is not None)
        available = cfg.MAX_SEQ_LEN - used
        if args.lyric_tokens > available:
            raise ValueError(f"Prompt {index} allows {max(available, 0)} lyric tokens, "
                             f"but requested {args.lyric_tokens}. Shorten it or lower --lyric-tokens.")

    print(f"Loading model from training job {args.model_job_id}: {checkpoint}", flush=True)
    model = BidirectionalDiffusionTransformer(str(args.base_model.resolve()), len(tokenizer))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    metadata = dict(vars(args), checkpoint=str(checkpoint), checkpoint_epoch=saved.get("epoch"),
                    checkpoint_loss=saved.get("loss"), checkpoint_bytes=checkpoint.stat().st_size,
                    max_seq_len=cfg.MAX_SEQ_LEN, dtype=str(dtype),
                    test_job_id=os.environ.get("SLURM_JOB_ID"),
                    torch_version=torch.__version__, prompt_count=len(prompts))
    (args.output_dir / "test_config.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    del saved
    model.to(device=device, dtype=dtype).eval()

    def generate_one(index, prompt):
        torch.manual_seed(args.seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + index)
        return generate(model, tokenizer, cfg, prompt, device, args)

    write_results(prompts, args.model_job_id, result_path, generate_one)
    print(f"SUCCESS: saved {len(prompts)} results to {result_path}", flush=True)


if __name__ == "__main__":
    main()
