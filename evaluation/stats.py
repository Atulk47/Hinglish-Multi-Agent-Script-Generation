"""evaluation/stats.py  -  the small statistics toolbox used by every evaluation table: Wilson intervals for rates, bootstrap intervals
(resampling the UNIT that matters, e.g. stories, not lines), Cohen's kappa, and a paired bootstrap for differences between systems."""
from __future__ import annotations
import math
import random
from typing import Callable, Dict, List, Sequence, Tuple


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """(rate, low, high) of a binomial proportion with the Wilson score interval (good for small n and rates near 0 / 1)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def fmt_rate(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{p:.1%} [{lo:.0%}-{hi:.0%}] (n={n})"


def bootstrap_ci(units: Sequence, stat: Callable[[Sequence], float], n_boot: int = 4000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """(point estimate, low, high): resample `units` with replacement (the units are STORIES for system-level claims) and recompute `stat`."""
    units = list(units)
    if not units:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    est = stat(units)
    vals = []
    for _ in range(n_boot):
        s = [units[rng.randrange(len(units))] for _ in units]
        try:
            v = stat(s)
        except ZeroDivisionError:
            continue
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return (est, float("nan"), float("nan"))
    return (est, vals[int(alpha / 2 * len(vals))], vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))])


def paired_bootstrap_diff(a: Dict[str, float], b: Dict[str, float], n_boot: int = 4000, seed: int = 0) -> Tuple[float, float, float, float]:
    """Mean paired difference (a - b) over the stories both have, its 95% interval, and the share of resamples in which a <= b (a one-sided p-like value)."""
    keys = [k for k in a if k in b and a[k] == a[k] and b[k] == b[k]]
    if not keys:
        return (float("nan"),) * 4
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    means = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return (sum(d) / len(d), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)], sum(1 for m in means if m <= 0) / n_boot)


def cohen_kappa(x: Sequence[bool], y: Sequence[bool]) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_ci(x: Sequence[bool], y: Sequence[bool], n_boot: int = 3000, seed: int = 0) -> Tuple[float, float, float]:
    pairs = list(zip(x, y))
    return bootstrap_ci(pairs, lambda s: cohen_kappa([a for a, _ in s], [b for _, b in s]), n_boot, seed)
