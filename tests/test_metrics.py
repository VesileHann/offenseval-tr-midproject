import numpy as np

from offenseval_tr.metrics import best_threshold, bootstrap_ci, macro_f1, paired_bootstrap, paired_bootstrap_runs


def test_macro_f1_perfect_and_majority():
    y = ["OFF", "NOT", "OFF", "NOT"]
    assert macro_f1(y, y) == 1.0
    assert macro_f1(y, ["NOT"] * 4) == 1 / 3


def test_macro_f1_known_value():
    y = ["OFF", "OFF", "NOT", "NOT"]
    p = ["OFF", "NOT", "OFF", "NOT"]
    # OFF: tp=1, fp=1, fn=1 -> 0.5; NOT: tp=1, fp=1, fn=1 -> 0.5
    assert macro_f1(y, p) == 0.5


def test_threshold_recovers_separable_scores():
    y = ["OFF"] * 10 + ["NOT"] * 10
    scores = np.r_[np.linspace(2, 4, 10), np.linspace(-4, -2, 10)]
    threshold, f1 = best_threshold(y, scores)
    assert -2 < threshold < 2
    assert f1 == 1.0


def test_bootstrap_and_paired_shapes():
    y = ["OFF", "NOT"] * 20
    p = ["OFF", "NOT"] * 20
    lo, hi = bootstrap_ci(y, p, n=200)
    assert 0.99 <= lo <= hi <= 1.0
    mean_gap, lower, wins = paired_bootstrap(y, p, ["NOT"] * 40, n=200)
    assert mean_gap > 0.4 and lower > 0 and wins > 0.9


def test_paired_bootstrap_runs_matches_single_run_case():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.2
    good = [np.where(rng.random(300) < 0.9, y, ~y) for _ in range(3)]
    bad = [rng.random(300) < 0.2 for _ in range(3)]
    gap, lower, wins, (lo, hi) = paired_bootstrap_runs(y, good, bad, n=500)
    assert gap > 0.2 and lower > 0 and wins > 0.99 and lo <= gap <= hi
    gap0, lower0, _, _ = paired_bootstrap_runs(y, good, good, n=200)
    assert gap0 == 0 and lower0 == 0
