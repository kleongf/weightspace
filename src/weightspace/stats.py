from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class Verdict:
    best_control: int
    difference: Interval

    @property
    def passed(self) -> bool:
        return self.difference.low > 0


def paired_bootstrap(a: Sequence[float], b: Sequence[float], seed: int = 0) -> Interval:
    """95% interval for mean(a - b), resampling examples so each pair stays together."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.ndim != 1 or d.size == 0:
        raise ValueError("need two equal-length, non-empty score lists from the same examples")
    means = d[np.random.default_rng(seed).integers(0, d.size, (RESAMPLES, d.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return Interval(float(d.mean()), float(low), float(high))


def verdict(patch: Sequence[float], controls: Sequence[Sequence[float]], seed: int = 0) -> Verdict:
    """Compare a Patch's per-example scores with its best-scoring Random-edit control."""
    best = int(np.argmax([np.mean(c) for c in controls]))
    return Verdict(best, paired_bootstrap(patch, controls[best], seed))
