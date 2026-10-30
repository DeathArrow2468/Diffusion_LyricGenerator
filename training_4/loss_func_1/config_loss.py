"""Experimental settings for specification sections 1–8 and 10 (no third loss)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HybridLossConfig:
    alpha: float = 0.7          # AR share; denoising share is 1 - alpha.
    rho: float = 0.1            # Uniform share of denoising coefficient mass.
    half_life_start: float = 16.0  # Tokens; illustrative, not tuned.
    half_life_end: float = 16.0    # Equal endpoints disable the curriculum.
    max_target_tokens: int = 256   # H_max; clipped to remaining lyric length.
    prefix_policy: str = "uniform"  # "uniform" lyric prefix or "prompt_only".
    noise_min: float = 0.1
    noise_max: float = 1.0
    # m(tau)=tau. If Bernoulli masking selects nothing, mask one random target.
    # This fallback changes the sampling distribution; it is not resampling.
    ignore_index: int = -100

    def __post_init__(self):
        for name in ("alpha", "rho"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("half_life_start", "half_life_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.max_target_tokens, int) or self.max_target_tokens < 1:
            raise ValueError("max_target_tokens must be a positive integer")
        if self.prefix_policy not in {"uniform", "prompt_only"}:
            raise ValueError("Unknown prefix_policy")
        if not 0 <= self.noise_min <= self.noise_max <= 1:
            raise ValueError("Require 0 <= noise_min <= noise_max <= 1")
        if self.ignore_index >= 0:
            raise ValueError("ignore_index must be negative")


LOSS_CONFIG = HybridLossConfig()
