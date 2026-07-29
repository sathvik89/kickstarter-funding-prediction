"""Metric helpers and threshold selection."""

import numpy as np
import pytest

from src import evaluate as ev


@pytest.fixture
def perfect():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.05, 0.1, 0.9, 0.95])
    return y, proba


def test_score_is_perfect_on_separable_data(perfect):
    y, proba = perfect
    s = ev.score(y, proba)
    assert s["accuracy"] == 1.0
    assert s["f1"] == 1.0
    assert s["roc_auc"] == 1.0


def test_score_respects_the_threshold(perfect):
    y, proba = perfect
    # Threshold above every probability predicts all-negative: recall 0.
    assert ev.score(y, proba, threshold=0.99)["recall"] == 0.0
    # Below every probability predicts all-positive: recall 1, precision 0.5.
    s = ev.score(y, proba, threshold=0.01)
    assert s["recall"] == 1.0
    assert s["precision"] == 0.5


def test_score_does_not_divide_by_zero_when_nothing_is_predicted_positive(perfect):
    y, proba = perfect
    assert ev.score(y, proba, threshold=0.999)["precision"] == 0.0


def test_ranking_metrics_ignore_the_threshold(perfect):
    y, proba = perfect
    a = ev.score(y, proba, threshold=0.2)
    b = ev.score(y, proba, threshold=0.8)
    assert a["roc_auc"] == b["roc_auc"]
    assert a["average_precision"] == b["average_precision"]


def test_tune_threshold_finds_the_f1_maximum():
    rng = np.random.RandomState(0)
    y = rng.binomial(1, 0.4, 3000)
    # Probabilities correlated with y but deliberately shifted low, so the
    # F1-optimal threshold sits well under 0.5.
    proba = np.clip(0.15 + 0.3 * y + rng.normal(0, 0.1, 3000), 0.001, 0.999)

    chosen = ev.tune_threshold(y, proba)
    grid = ev.threshold_grid()
    from sklearn.metrics import f1_score
    best = max(f1_score(y, (proba >= t).astype(int)) for t in grid)
    assert f1_score(y, (proba >= chosen).astype(int)) == pytest.approx(best)
    assert chosen < 0.5, "on a low-shifted score, the F1 optimum is below 0.5"


def test_tune_threshold_stays_inside_the_grid():
    rng = np.random.RandomState(1)
    y = rng.binomial(1, 0.5, 500)
    proba = rng.uniform(size=500)
    chosen = ev.tune_threshold(y, proba)
    assert ev.threshold_grid()[0] <= chosen <= ev.threshold_grid()[-1]


@pytest.mark.parametrize("rate,expected", [(0.4, 0.6), (0.6, 0.6), (0.5, 0.5)])
def test_majority_baseline_always_takes_the_larger_class(rate, expected):
    y = np.zeros(1000, dtype=int)
    y[: int(rate * 1000)] = 1
    assert ev.majority_baseline(y) == pytest.approx(expected, abs=1e-9)


def test_majority_baseline_matches_this_dataset(fused):
    """The honesty anchor: 0.596, quoted throughout the docs."""
    assert ev.majority_baseline(fused.y_test) == pytest.approx(0.596, abs=0.002)
