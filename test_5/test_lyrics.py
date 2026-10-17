"""Test one lyric prompt per run and append the result to test_5/results.csv."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


FIELDS = ["Prompt", "Generated Lyrics", "Time taken", "Base Model"]

LEGACY_FIELDS = [
    "timestamp_utc", "prompt", "generated_lyrics", "generation_seconds",
    "checkpoint", "lyric_tokens", "denoising_steps", "temperature",
    "top_k", "seed", "device", "dtype", "first_generation",
]


def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def single_line(value):
    return " ".join(str(value).splitlines())

def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    if exists:
        with path.open("r", encoding="utf-8-sig", newline="") as old_file:
            reader = csv.DictReader(old_file)
            old_fields = reader.fieldnames
            legacy_rows = (
                list(reader) if old_fields == LEGACY_FIELDS else None
            )

        if old_fields == LEGACY_FIELDS:
            backup = path.with_name(path.name + ".backup")
            if not backup.exists():
                shutil.copy2(path, backup)

            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=path.parent,
                    prefix=path.name + ".",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    writer = csv.DictWriter(
                        temporary,
                        fieldnames=FIELDS,
                    )
                    writer.writeheader()

                    for old in legacy_rows:
                        writer.writerow({
                            "Prompt": old["prompt"],
                            "Generated Lyrics": old["generated_lyrics"],
                            "Time taken": old["generation_seconds"],
                            "Base Model": old["checkpoint"].startswith(
                                "BASE:"
                            ),
                        })

                    temporary.flush()
                    os.fsync(temporary.fileno())

                os.replace(temporary_path, path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        elif old_fields != FIELDS:
            raise ValueError(
                f"Unrecognized CSV header in {path}; "
                "choose a different --output path."
            )

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not exists:
            writer.writeheader()

        writer.writerow({
            **row,
            "Prompt": single_line(row["Prompt"]),
            "Generated Lyrics": single_line(row["Generated Lyrics"]),
        })
        f.flush()
        os.fsync(f.fileno())


def generate(model, tokenizer, cfg, prompt, args, device):
    import torch

    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    )
    bos = (
        [] if tokenizer.bos_token_id is None
        else [tokenizer.bos_token_id]
    )
    eos = (
        [] if tokenizer.eos_token_id is None
        else [tokenizer.eos_token_id]
    )

    sep_id = tokenizer.convert_tokens_to_ids(cfg.SEP_TOKEN)
    mask_id = tokenizer.convert_tokens_to_ids(cfg.MASK_TOKEN)

    prefix = bos + prompt_ids + [sep_id]
    available = cfg.MAX_SEQ_LEN - len(prefix) - len(eos)

    if args.lyric_tokens > available:
        raise ValueError(
            f"Prompt leaves room for {max(0, available)} lyric tokens, "
            f"but --lyric-tokens is {args.lyric_tokens}. "
            "Shorten the prompt or lower the length."
        )

    n = args.lyric_tokens
    start = len(prefix)

    sequence = prefix + [mask_id] * n + eos
    sequence += [tokenizer.pad_token_id] * (
        cfg.MAX_SEQ_LEN - len(sequence)
    )

    ids = torch.tensor(
        [sequence],
        device=device,
        dtype=torch.long,
    )
    attention = (ids != tokenizer.pad_token_id).long()
    remaining = torch.ones(
        n,
        dtype=torch.bool,
        device=device,
    )

    structure_ids = set(
        tokenizer.convert_tokens_to_ids(cfg.STRUCTURAL_TOKENS)
    )
    forbidden = set(tokenizer.all_special_ids) - structure_ids
    forbidden.update([
        sep_id,
        mask_id,
        tokenizer.pad_token_id,
    ])

    if tokenizer.eos_token_id is not None:
        forbidden.add(tokenizer.eos_token_id)

    forbidden = sorted(
        x for x in forbidden if x is not None
    )

    steps = min(args.steps, n)
    diagnostics = []

    with torch.inference_mode():
        for step in range(steps):
            positions = torch.where(remaining)[0]
            t = max(0.1, positions.numel() / n)

            times = torch.tensor(
                [t],
                device=device,
                dtype=torch.float32,
            )

            all_logits = model(ids, attention, times)
            scores = all_logits[
                0,
                start + positions,
                :,
            ].float()
            del all_logits

            scores[:, forbidden] = -float("inf")

            allowed_scores = scores.clone()
            allowed_scores[:, forbidden] = 0

            if not torch.isfinite(allowed_scores).all():
                raise RuntimeError(
                    "Model produced invalid logits; "
                    "check checkpoint compatibility."
                )

            del allowed_scores

            if args.diagnostics and step == 0:
                first_probs = scores[0].softmax(-1)
                values, indices = first_probs.topk(
                    min(
                        5,
                        scores.shape[-1] - len(forbidden),
                    )
                )

                guesses = scores.argmax(-1).tolist()
                dominant_id, dominant_count = Counter(
                    guesses
                ).most_common(1)[0]

                diagnostics.append(
                    f"First pass: "
                    f"{dominant_count}/{len(guesses)} "
                    "masked positions have "
                    f"{tokenizer.decode([dominant_id])!r} "
                    "as their greedy prediction."
                )

                predictions = ", ".join(
                    f"{tokenizer.decode([token_id])!r}: "
                    f"{probability:.2%}"
                    for token_id, probability in zip(
                        indices.tolist(),
                        values.tolist(),
                    )
                )
                diagnostics.append(
                    "First masked position, "
                    "allowed-vocabulary probabilities: "
                    + predictions
                )
                del first_probs

            target_filled = math.ceil(
                n * (step + 1) / steps
            )
            count = target_filled - (
                n - positions.numel()
            )

            chosen = torch.randperm(
                positions.numel(),
                device=device,
            )[:count]

            accepted = positions[chosen]
            scores = scores[chosen]

            if args.temperature == 0:
                candidates = scores.argmax(dim=-1)
            else:
                sample_scores = (
                    scores / args.temperature
                )

                if args.top_k > 0:
                    k = min(
                        args.top_k,
                        scores.shape[-1] - len(forbidden),
                    )

                    values, indices = (
                        sample_scores.topk(k, dim=-1)
                    )
                    sampled = torch.multinomial(
                        values.softmax(-1),
                        1,
                    )
                    candidates = indices.gather(
                        1,
                        sampled,
                    ).squeeze(1)
                else:
                    candidates = torch.multinomial(
                        sample_scores.softmax(-1),
                        1,
                    ).squeeze(1)

            ids[0, start + accepted] = candidates
            remaining[accepted] = False

    generated = ids[
        0,
        start:start + n,
    ].tolist()

    lyrics = tokenizer.decode(
        generated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    if args.diagnostics:
        token_id, count = Counter(
            generated
        ).most_common(1)[0]

        diagnostics.append(
            f"Final output: "
            f"{len(set(generated))}/{n} unique token IDs; "
            f"most frequent token "
            f"{tokenizer.decode([token_id])!r} "
            f"occurs {count}/{n} times."
        )

    return lyrics, steps, diagnostics


def reconstruction_check(
    model,
    tokenizer,
    cfg,
    root,
    device,
    seed,
):
    import torch
    import torch.nn.functional as F

    data_path = Path(cfg.DATASET_PATH)

    if not data_path.is_file():
        data_path = (
            root / "Dataset_maker_1/train_data.jsonl"
        )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Reconstruction data not found: {data_path}"
        )

    bos = (
        [] if tokenizer.bos_token_id is None
        else [tokenizer.bos_token_id]
    )
    eos = (
        [] if tokenizer.eos_token_id is None
        else [tokenizer.eos_token_id]
    )

    sep_id = tokenizer.convert_tokens_to_ids(
        cfg.SEP_TOKEN
    )
    mask_id = tokenizer.convert_tokens_to_ids(
        cfg.MASK_TOKEN
    )

    example = None

    with data_path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            prefix = (
                bos
                + tokenizer.encode(
                    item["prompt"],
                    add_special_tokens=False,
                )
                + [sep_id]
            )

            available = (
                cfg.MAX_SEQ_LEN
                - len(prefix)
                - len(eos)
            )

            if available < 16:
                continue

            lyrics = tokenizer.encode(
                item["lyrics"],
                add_special_tokens=False,
            )[:available]

            excluded = {
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            }

            allowed = [
                i
                for i, token in enumerate(lyrics)
                if token not in excluded
            ]

            if len(allowed) >= 16:
                example = (
                    item,
                    prefix,
                    lyrics,
                    allowed,
                    line_number,
                )
                break

    if example is None:
        raise ValueError(
            "No training example with at least "
            "16 maskable lyric tokens found."
        )

    (
        item,
        prefix,
        lyrics,
        allowed,
        line_number,
    ) = example

    sequence = prefix + lyrics + eos
    sequence += [tokenizer.pad_token_id] * (
        cfg.MAX_SEQ_LEN - len(sequence)
    )

    clean = torch.tensor(
        [sequence],
        dtype=torch.long,
        device=device,
    )
    attention = (
        clean != tokenizer.pad_token_id
    ).long()

    positions = (
        torch.tensor(
            allowed,
            dtype=torch.long,
            device=device,
        )
        + len(prefix)
    )

    rng = torch.Generator(
        device=device
    ).manual_seed(seed)

    order = torch.randperm(
        positions.numel(),
        device=device,
        generator=rng,
    )

    print(
        f"\nReconstruction probe: training data "
        f"line {line_number}; "
        f"{len(allowed)} maskable tokens."
    )
    print("Training prompt:", item["prompt"])
    print(
        "Reference excerpt:",
        repr(
            tokenizer.decode(
                lyrics[:40],
                skip_special_tokens=False,
            )
        ),
    )

    with torch.inference_mode():
        for fraction in (0.25, 0.5, 1.0):
            count = max(
                1,
                int(positions.numel() * fraction),
            )
            selected = positions[order[:count]]

            corrupted = clean.clone()
            corrupted[0, selected] = mask_id

            t = torch.tensor(
                [fraction],
                dtype=torch.float32,
                device=device,
            )

            all_logits = model(
                corrupted,
                attention,
                t,
            )
            scores = all_logits[
                0,
                selected,
            ].float()
            del all_logits

            targets = clean[0, selected]
            loss = F.cross_entropy(
                scores,
                targets,
            ).item()

            predictions = scores.argmax(-1)
            accuracy = (
                (predictions == targets)
                .float()
                .mean()
                .item()
            )

            token_id, frequency = Counter(
                predictions.tolist()
            ).most_common(1)[0]

            print(
                f"Mask {fraction:.0%}: "
                f"cross-entropy={loss:.4f}; "
                f"exact token accuracy={accuracy:.2%}; "
                f"most common prediction="
                f"{tokenizer.decode([token_id])!r} "
                f"({frequency}/{count})."
            )

            if fraction == 1.0:
                rebuilt = clean.clone()
                rebuilt[0, selected] = predictions

                excerpt = rebuilt[
                    0,
                    len(prefix):len(prefix)
                    + min(40, len(lyrics)),
                ].tolist()

                print(
                    "100% masked, one-pass greedy excerpt:",
                    repr(
                        tokenizer.decode(
                            excerpt,
                            skip_special_tokens=False,
                        )
                    ),
                )

    print(
        "These probes are outside the generation timer "
        "and are not appended as lyric results.\n"
    )


def run_base_model(cfg, args, output):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    chosen_device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device

    if (
        chosen_device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable in this Python environment."
        )

    device = torch.device(chosen_device)

    dtype = (
        torch.bfloat16
        if (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        else torch.float32
    )

    print(
        f"BASE MODEL: {cfg.MODEL_NAME}; "
        f"device={device}; dtype={dtype}"
    )
    print(
        "Loading the original causal model. "
        "Diffusion checkpoint weights are not loaded."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "The base tokenizer needs a PAD or EOS token."
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        dtype=dtype,
    ).to(device).eval()

    print(
        "Base Qwen continues text; "
        "it is not the instruction-tuned model."
    )

    try:
        prompt = (
            args.prompt
            if args.prompt is not None
            else input("\nPrompt: ")
        )
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no result appended.")
        return

    if not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    prefix_length = inputs["input_ids"].shape[1]

    context_limit = getattr(
        model.config,
        "max_position_embeddings",
        None,
    )

    if (
        context_limit is not None
        and prefix_length + args.lyric_tokens
        > context_limit
    ):
        raise ValueError(
            "Prompt plus requested output exceeds "
            f"model context limit {context_limit}."
        )

    generation_options = dict(
        max_new_tokens=args.lyric_tokens,
        do_sample=args.temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=False,
    )

    if args.temperature > 0:
        generation_options.update(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=1.0,
        )

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **generation_options,
        )

    new_ids = generated[
        0,
        prefix_length:,
    ].tolist()

    lyrics = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start_time

    row = {
        "Prompt": prompt,
        "Generated Lyrics": lyrics,
        "Time taken": f"{elapsed:.6f}",
        "Base Model": True,
    }

    if args.diagnostics:
        print(
            f"Base autoregressive generation: "
            f"{len(new_ids)} new tokens "
            "(including any ending special token)."
        )

    try:
        append_result(output, row)
    except (OSError, ValueError):
        print(
            f"CSV save failed. Generated text:\n"
            f"{lyrics}\nSeconds: {elapsed:.6f}"
        )
        raise

    print(
        f"\n{lyrics}\n\n"
        f"Generation: {elapsed:.3f} seconds."
    )
    print(
        f"Appended one BASE result to {output}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "training_4/checkpoints_2_7_1e-4/"
            "diffusion_epoch_7.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help=(
            "CSV path relative to test_5, "
            "or an absolute path."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Supply one prompt here, or omit "
            "to enter one when asked."
        ),
    )
    parser.add_argument(
        "--lyric-tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help=(
            "Default 0.8 samples tokens; "
            "0 uses greedy predictions."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help=(
            "Used for sampling; "
            "0 disables it."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Print model predictions "
            "and token repetition."
        ),
    )
    parser.add_argument(
        "--reconstruction-check",
        action="store_true",
        help=(
            "Probe masked-token recovery "
            "on one training example."
        ),
    )
    parser.add_argument(
        "--base-model",
        action="store_true",
        help=(
            "Use original Qwen with "
            "autoregressive generation."
        ),
    )

    args = parser.parse_args()

    if (
        args.base_model
        and args.reconstruction_check
    ):
        parser.error(
            "--reconstruction-check applies to "
            "the diffusion model; omit it "
            "for --base-model."
        )

    if (
        args.lyric_tokens < 1
        or args.steps < 1
        or args.top_k < 0
    ):
        parser.error(
            "Length/steps must be positive; "
            "top-k must be nonnegative."
        )

    if (
        not math.isfinite(args.temperature)
        or args.temperature < 0
    ):
        parser.error(
            "Temperature must be finite "
            "and nonnegative."
        )

    test_dir = Path(__file__).resolve().parent
    root = test_dir.parent

    checkpoint = (
        root / args.checkpoint
    ).resolve()
    output = (
        test_dir / args.output
    ).resolve()

    if (
        not args.base_model
        and not checkpoint.is_file()
    ):
        parser.error(
            f"Checkpoint not found: "
            f"{checkpoint}. "
            "Use --checkpoint to choose one."
        )

    config_path = root / "training_4/config.py"
    model_path = root / "diffusion_model_3/model.py"

    if (
        not config_path.is_file()
        or (
            not args.base_model
            and not model_path.is_file()
        )
    ):
        parser.error(
            "Place test_lyrics.py inside "
            "the project's test_5 folder. "
            "training_4 and diffusion_model_3 "
            "should be sibling folders."
        )

    import torch
    from transformers import AutoTokenizer

    cfg = read_module(
        "_lyric_training_config",
        config_path,
    )

    if args.base_model:
        run_base_model(cfg, args, output)
        return

    previous_config = sys.modules.get("config")
    sys.modules["config"] = cfg

    try:
        module = read_module(
            "_lyric_diffusion_model",
            model_path,
        )
    finally:
        if previous_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous_config

    chosen_device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    ) if args.device == "auto" else args.device

    if (
        chosen_device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA is unavailable in this "
            "Python environment."
        )

    device = torch.device(chosen_device)

    dtype = (
        torch.bfloat16
        if (
            device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        else torch.float32
    )

    print(
        f"Device: {device}; "
        f"dtype: {dtype}; "
        f"checkpoint: {checkpoint}"
    )
    print(
        "Loading tokenizer and weights "
        "(excluded from generation timing)..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.MODEL_NAME
    )

    tokenizer.add_special_tokens({
        "additional_special_tokens": (
            cfg.STRUCTURAL_TOKENS
            + [cfg.SEP_TOKEN, cfg.MASK_TOKEN]
        )
    })

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Training tokenizer requires "
            "a PAD or EOS token."
        )

    model = (
        module.BidirectionalDiffusionTransformer(
            cfg.MODEL_NAME,
            len(tokenizer),
        )
    )

    saved = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )

    if (
        not isinstance(saved, dict)
        or "model_state_dict" not in saved
    ):
        raise ValueError(
            "Expected a training checkpoint "
            "containing model_state_dict."
        )

    model.load_state_dict(
        saved["model_state_dict"],
        strict=True,
    )

    print(
        f"Checkpoint metadata: "
        f"epoch={saved.get('epoch', 'unknown')}; "
        f"training loss="
        f"{saved.get('loss', 'unknown')}"
    )

    del saved
    gc.collect()

    model.to(
        device=device,
        dtype=dtype,
    ).eval()

    print(
        "Sampler: random position selection, "
        "then token sampling."
    )

    if args.reconstruction_check:
        reconstruction_check(
            model,
            tokenizer,
            cfg,
            root,
            device,
            args.seed,
        )

    print(f"Results append to: {output}")
    print(
        "Enter one prompt. "
        "One result will be appended, "
        "then the program will exit."
    )

    try:
        prompt = (
            args.prompt
            if args.prompt is not None
            else input("\nPrompt: ")
        )
    except (EOFError, KeyboardInterrupt):
        print(
            "\nCancelled; no result appended."
        )
        return

    if not prompt.strip():
        parser.error(
            "Prompt must not be empty."
        )

    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(
            args.seed
        )
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    lyrics, steps, diagnostics = generate(
        model,
        tokenizer,
        cfg,
        prompt,
        args,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = (
        time.perf_counter()
        - start_time
    )

    for message in diagnostics:
        print(message)

    if not any(
        character.isalnum()
        for character in lyrics
    ):
        print(
            "WARNING: output contains no "
            "letters or digits. This failed "
            "generation will still be logged."
        )

    row = {
        "Prompt": prompt,
        "Generated Lyrics": lyrics,
        "Time taken": f"{elapsed:.6f}",
        "Base Model": False,
    }

    try:
        append_result(output, row)
    except (OSError, ValueError):
        print(
            f"\nCSV save failed. "
            f"Generated lyrics:\n{lyrics}\n"
            f"Seconds: {elapsed:.6f}"
        )
        raise

    print(
        f"\n{lyrics}\n\n"
        f"Generation: {elapsed:.3f} seconds."
    )
    print(
        f"Appended one result to {output}"
    )


if __name__ == "__main__":
    main()