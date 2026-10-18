"""Model integration, kept separate from HTTP and the frontend.

DemoEngine serves clearly labelled samples. DiffusionEngine loads the user's
existing model.py and checkpoint once, then runs the masked-token sampler.
It does not modify training files or silently substitute a different model.
"""

import importlib.util
import math
import os
from pathlib import Path
import secrets
import sys
import threading
from typing import Callable

from schemas import GenerateRequest, stage_for


class GenerationCancelled(Exception):
    pass


ProgressCallback = Callable[[dict], None]


class DemoEngine:
    mode = "demo"

    def generate(self, request: GenerateRequest, report: ProgressCallback, cancelled: threading.Event) -> dict:
        for step in range(1, 25):
            # Event.wait makes preview cancellation responsive, without busy polling.
            if cancelled.wait(0.16):
                raise GenerationCancelled()
            percent = step / 24 * 100
            report({"type": "progress", "progress": percent, "stage": stage_for(percent), "step": step, "steps": 24})

        title = "Somewhere after midnight"
        lyrics = "[Verse]\nThe dashboard paints your hands in green\nWe miss the turn at Evergreen\nYou say the long way sounds like home\nSo I let the engine set the tone\n\n[Chorus]\nLet the city fall behind us\nLet the morning take its time\nWe have nothing left to run from\nAnd the whole night left to find"
        if request.mood == "Melancholic":
            title = "The things a pocket keeps"
            lyrics = "[Verse]\nA faded ticket, folded twice\nA winter coat that smells like rain\nI thought that I had paid the price\nOf never saying your name again\n\n[Chorus]\nSome things stay without a sound\nA little weight I carry round\nI can let you go and keep\nThe things a pocket keeps"
        if request.length == "short":
            lyrics = lyrics[lyrics.index("[Chorus]"):]
        return {"title": title, "lyrics": lyrics, "demo": True}


def load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required repository file is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DiffusionEngine:
    mode = "model"

    def __init__(self):
        # Imports are deliberately delayed: the API and UI demo need no PyTorch.
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        repo_value = os.environ.get("LYRICWALA_REPO", "").strip()
        checkpoint_value = os.environ.get("LYRICWALA_CHECKPOINT", "").strip()
        if not repo_value or not checkpoint_value:
            raise ValueError("Model mode requires LYRICWALA_REPO and LYRICWALA_CHECKPOINT in .env.")
        repo = Path(repo_value).expanduser().resolve()
        checkpoint = Path(checkpoint_value).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        self.cfg = load_module("_lyricwala_training_config", repo / "training_4/config.py")

        # The existing wrapper imports a bare `config`. Scope that compatibility
        # bridge to module import; do not permanently overwrite another config.
        previous = sys.modules.get("config")
        sys.modules["config"] = self.cfg
        try:
            model_module = load_module("_lyricwala_diffusion_model", repo / "diffusion_model_3/model.py")
        finally:
            if previous is None:
                sys.modules.pop("config", None)
            else:
                sys.modules["config"] = previous

        tokenizer_path = Path(os.environ.get("LYRICWALA_TOKENIZER") or checkpoint.parent).expanduser()
        if (tokenizer_path / "tokenizer_config.json").is_file():
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
        elif os.environ.get("LYRICWALA_ALLOW_LEGACY_TOKENIZER", "false").lower() == "true":
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.MODEL_NAME)
            self.tokenizer.add_special_tokens({"additional_special_tokens": self.cfg.STRUCTURAL_TOKENS + [self.cfg.SEP_TOKEN, self.cfg.MASK_TOKEN]})
        else:
            raise FileNotFoundError("Saved tokenizer not found. Set LYRICWALA_TOKENIZER to the tokenizer folder saved during training. See .env.example for old checkpoints.")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The training tokenizer must have a padding token.")

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(saved, dict) or "model_state_dict" not in saved:
            raise ValueError("Expected a training checkpoint containing model_state_dict.")
        if saved.get("tokenizer_length", len(self.tokenizer)) != len(self.tokenizer):
            raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match.")
        if saved.get("model_name", self.cfg.MODEL_NAME) != self.cfg.MODEL_NAME:
            raise ValueError("Checkpoint model_name does not match training_4/config.py.")
        self.max_length = int(saved.get("max_seq_len", self.cfg.MAX_SEQ_LEN))
        requested_device = os.environ.get("LYRICWALA_DEVICE", "auto")
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise ValueError("LYRICWALA_DEVICE must be auto, cpu, or cuda.")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.model = model_module.BidirectionalDiffusionTransformer(self.cfg.MODEL_NAME, len(self.tokenizer))
        self.model.load_state_dict(saved["model_state_dict"], strict=True)
        del saved
        # Inference can store weights in BF16: there are no small training updates to preserve.
        self.model.to(device=self.device, dtype=dtype).eval()
        self.steps = int(os.environ.get("LYRICWALA_STEPS", "32"))
        self.min_t = float(os.environ.get("LYRICWALA_MIN_TIMESTEP", "0.001"))
        if not 1 <= self.steps <= 256 or not 0 < self.min_t <= 1:
            raise ValueError("Invalid diffusion steps or minimum timestep.")
        vocabulary = self.tokenizer.get_vocab()
        if self.cfg.MASK_TOKEN not in vocabulary or self.cfg.SEP_TOKEN not in vocabulary:
            raise ValueError("The saved tokenizer is missing the trained MASK or SEP token.")
        self.mask_id = vocabulary[self.cfg.MASK_TOKEN]
        self.sep_id = vocabulary[self.cfg.SEP_TOKEN]
        structures = {vocabulary[s] for s in self.cfg.STRUCTURAL_TOKENS if s in vocabulary}
        forbidden = set(self.tokenizer.all_special_ids) - structures
        forbidden.update([self.sep_id, self.mask_id, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id])
        self.forbidden = sorted(i for i in forbidden if i is not None)

    def generate(self, request: GenerateRequest, report: ProgressCallback, cancelled: threading.Event) -> dict:
        torch = self.torch
        prompt_ids = self.tokenizer.encode(request.model_prompt(), add_special_tokens=False)
        # Matches the updated collator from this project: prompt + SEP + lyric area.
        # The original Qwen tokenizer has no BOS; EOS is not generated by this sampler.
        prefix = prompt_ids + [self.sep_id]
        available = self.max_length - len(prefix)
        desired = {"short": 64, "standard": 128, "long": 176}[request.length]
        if available < desired:
            raise ValueError(f"This idea is too long for the model's context window. Shorten it or choose a shorter draft (room for {max(0, available)} lyric tokens; requested {desired}).")
        n = desired
        start = len(prefix)
        sequence = prefix + [self.mask_id] * n
        attention = torch.zeros((1, self.max_length), device=self.device, dtype=torch.long)
        attention[0, :len(sequence)] = 1
        sequence += [self.tokenizer.pad_token_id] * (self.max_length - len(sequence))
        ids = torch.tensor([sequence], device=self.device, dtype=torch.long)
        remaining = torch.ones(n, device=self.device, dtype=torch.bool)
        # One local generator per request avoids changing global training/test RNG state.
        seed = secrets.randbelow(2**31)
        rng = torch.Generator(device=self.device).manual_seed(seed)
        temperature = 0.4 + request.creativity / 100 * 0.8
        steps = min(self.steps, n)
        report({"type": "progress", "progress": 0, "stage": "Finding the feeling", "step": 0, "steps": steps})

        with torch.inference_mode():
            for step in range(steps):
                if cancelled.is_set():
                    raise GenerationCancelled()
                positions = torch.where(remaining)[0]
                timestep = max(self.min_t, positions.numel() / n)
                times = torch.tensor([timestep], device=self.device, dtype=torch.float32)
                all_logits = self.model(ids, attention, times)
                target_filled = math.ceil(n * (step + 1) / steps)
                count = target_filled - (n - positions.numel())
                chosen = torch.randperm(positions.numel(), device=self.device, generator=rng)[:count]
                accepted = positions[chosen]
                scores = all_logits[0, start + accepted, :].float()
                del all_logits
                if not torch.isfinite(scores).all():
                    raise RuntimeError("The model produced invalid predictions. Check checkpoint compatibility.")
                scores[:, self.forbidden] = -float("inf")
                top_k = min(50, scores.shape[-1] - len(self.forbidden))
                values, indices = (scores / temperature).topk(top_k, dim=-1)
                sampled = torch.multinomial(values.softmax(-1), 1, generator=rng)
                ids[0, start + accepted] = indices.gather(1, sampled).squeeze(1)
                remaining[accepted] = False
                # Synchronize before reporting progress, so progress tracks completed GPU work.
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                percent = (step + 1) / steps * 100
                report({"type": "progress", "progress": percent, "stage": stage_for(percent), "step": step + 1, "steps": steps})

        if cancelled.is_set():
            raise GenerationCancelled()
        lyrics = self.tokenizer.decode(ids[0, start:start+n].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False).strip()
        if not lyrics or not any(c.isalnum() for c in lyrics):
            raise RuntimeError("The model did not produce readable lyrics for this take. Try another idea or check the checkpoint.")
        return {"title": f"A {request.mood.lower()} little feeling", "lyrics": lyrics, "demo": False, "seed": seed, "lyric_tokens": n, "steps": steps}
