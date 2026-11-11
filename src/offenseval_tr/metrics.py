"""Macro-F1, bootstrap intervals, paired bootstrap and threshold selection."""
from __future__ import annotations

import numpy as np

SEED = 42


def _as_bool(values):
    values = np.asarray(values)
    if values.dtype.kind in "OUS":
        return values == "OFF"
    return values.astype(bool)


def macro_f1(y, p) -> float:
    """Binary macro-F1 for labels ("OFF"/"NOT") or boolean masks."""
    y = _as_bool(y)
    p = _as_bool(p)
    tp = (y & p).sum(); fp = (~y & p).sum(); fn = (y & ~p).sum(); tn = (~y & ~p).sum()
    f_off = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f_not = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return float((f_off + f_not) / 2)


def macro_f1_rows(y, p):
    """Vectorised macro-F1 over leading rows; y and p are boolean arrays."""
    tp = (y & p).sum(-1); fp = (~y & p).sum(-1); fn = (y & ~p).sum(-1); tn = (~y & ~p).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_off = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
        f_not = np.nan_to_num(2 * tn / (2 * tn + fn + fp))
    return (f_off + f_not) / 2


def bootstrap_ci(y, p, n: int = 10_000, seed: int = SEED, chunk: int = 500):
    """95% percentile interval for macro-F1, resampling rows with replacement."""
    y = _as_bool(y); p = _as_bool(p)
    rng = np.random.default_rng(seed)
    scores = []
    for start in range(0, n, chunk):
        idx = rng.integers(0, len(y), (min(chunk, n - start), len(y)))
        scores.append(macro_f1_rows(y[idx], p[idx]))
    return np.percentile(np.concatenate(scores), [2.5, 97.5])


def paired_bootstrap(y, p_a, p_b, n: int = 10_000, seed: int = SEED):
    """One-sided 95% bootstrap for macro-F1(A) - macro-F1(B) on paired rows.

    Returns (mean_gap, lower_bound, win_rate). A gap is called real only if the
    lower bound is above zero and the observed gap is at least 0.02.
    """
    y = _as_bool(y); p_a = _as_bool(p_a); p_b = _as_bool(p_b)
    rng = np.random.default_rng(seed)
    gaps = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y), len(y))
        gaps[i] = macro_f1(y[idx], p_a[idx]) - macro_f1(y[idx], p_b[idx])
    return float(macro_f1(y, p_a) - macro_f1(y, p_b)), float(np.percentile(gaps, 5)), float((gaps > 0).mean())


def best_threshold(y, scores) -> tuple[float, float]:
    """Threshold maximising macro-F1; candidates are midpoints between scores."""
    y = _as_bool(y)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(s)
    s, y = s[order], y[order]
    candidates = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
    best_t, best_f = 0.0, -1.0
    for t in candidates:
        f = macro_f1(y, s > t)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, float(best_f)
