"""Metrics and threshold selection.

The one rule enforced here: thresholds and hyperparameters are chosen on the
validation split, never on test. `tune_threshold` only ever sees validation
probabilities; `score` is what gets pointed at test, once, at the end.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config as cfg


def threshold_grid() -> np.ndarray:
    return np.arange(
        cfg.THRESHOLD_GRID_START, cfg.THRESHOLD_GRID_STOP, cfg.THRESHOLD_GRID_STEP
    )


def score(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Full metric set at a given decision threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
    }


def tune_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Threshold maximising F1 on the validation split."""
    grid = threshold_grid()
    f1s = [f1_score(y_val, (proba_val >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the more common class - the honesty check."""
    rate = float(np.mean(y))
    return max(rate, 1.0 - rate)
