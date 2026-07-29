"""Guards on the results themselves.

These are regression tests on the claims the README and the live site make. If a
future change quietly drops the model below its published numbers, this fails
instead of the website misleading people.
"""

import json

import pytest

from src import config as cfg
from src import evaluate as ev
from tests.conftest import requires_artifacts

pytestmark = requires_artifacts


@pytest.fixture(scope="module")
def meta():
    return json.loads((cfg.V2 / "v2_meta.json").read_text())


@pytest.fixture(scope="module")
def results(meta):
    return {r["model"]: r for r in meta["results"]}


def test_linear_model_holds_its_published_numbers(results):
    r = results["logreg_l2_fused_v2"]
    assert r["accuracy"] >= 0.695
    assert r["f1"] >= 0.589
    assert r["roc_auc"] >= 0.753


def test_boosting_holds_its_published_numbers(results):
    r = results["hgb_tab_plus_textsvd"]
    assert r["accuracy"] >= 0.705, "README claims 0.707"
    assert r["f1"] >= 0.611
    assert r["roc_auc"] >= 0.770


def test_both_models_beat_the_majority_baseline(results, meta):
    baseline = meta["majority_baseline_accuracy"]
    for name in ("logreg_l2_fused_v2", "hgb_tab_plus_textsvd"):
        assert results[name]["accuracy"] > baseline + 0.08, (
            f"{name} must clear the majority baseline by a real margin"
        )


def test_phase9_improves_on_the_phase8_baseline(results):
    phase8 = json.loads((cfg.PROCESSED / "eval_meta.json").read_text())["best_test_metrics"]
    for name in ("logreg_l2_fused_v2", "hgb_tab_plus_textsvd"):
        assert results[name]["f1"] > phase8["f1"]
        assert results[name]["accuracy"] > phase8["accuracy"]


def test_boosting_beats_the_linear_model_on_ranking(results):
    """The stated reason boosting ships as a ceiling check."""
    lin, tree = results["logreg_l2_fused_v2"], results["hgb_tab_plus_textsvd"]
    assert tree["roc_auc"] > lin["roc_auc"]
    assert tree["average_precision"] > lin["average_precision"]


def test_clears_the_published_launch_time_benchmark(results):
    """Springer 2019 reports 69.8% using launch-time-only features."""
    assert results["hgb_tab_plus_textsvd"]["accuracy"] > 0.698


def test_hand_crafted_interactions_did_not_earn_their_place():
    """Documented finding - keep it honest if someone re-enables the block."""
    path = cfg.V2 / "v2_interaction_ablation.csv"
    if not path.exists():
        pytest.skip("run `python -m src.ablations` first")
    import pandas as pd
    table = pd.read_csv(path)
    baseline = float(table.iloc[0]["f1"])
    best_gain = float(table["f1"].max()) - baseline
    assert best_gain < 0.005, (
        "interactions now help more than documented - update the docs and "
        "reconsider USE_INTERACTIONS"
    )
    assert cfg.USE_INTERACTIONS is False


def test_threshold_choice_was_made_on_validation(meta):
    assert "validation" in meta["selection_policy"].lower()
    assert meta["best_threshold_by_val_f1"] < 0.5, (
        "the F1-optimal threshold on this imbalanced task sits below 0.5"
    )


def test_val_tuned_threshold_trades_accuracy_for_recall(results):
    """Documented trade - if this inverts, the docs are wrong."""
    at_half = results["logreg_l2_fused_v2"]
    tuned = results["logreg_l2_fused_v2 @val-thr"]
    assert tuned["recall"] > at_half["recall"]
    assert tuned["f1"] > at_half["f1"]
    assert tuned["accuracy"] < at_half["accuracy"]


def test_split_sizes_are_as_documented(meta):
    assert meta["n_train"] == 211782
    assert meta["n_val"] == 52946
    assert meta["n_test"] == 66183, "test must stay the Phase 3/8 test set"


def test_majority_baseline_is_stable(fused):
    assert ev.majority_baseline(fused.y_test) == pytest.approx(0.5961, abs=1e-3)
