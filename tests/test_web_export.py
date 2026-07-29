"""The exported browser model must agree with sklearn.

If this drifts, the live demo quietly starts telling visitors the wrong thing
while every number in the repo stays correct. That is the worst kind of bug, so
it gets its own test rather than living only inside the export script.
"""

import json

import joblib
import numpy as np
import pytest

from src import config as cfg
from src import export_web
from tests.conftest import requires_artifacts

pytestmark = requires_artifacts


@pytest.fixture(scope="module")
def payload():
    return export_web.build_payload()


class TestPayloadShape:
    def test_every_coefficient_is_accounted_for(self, payload):
        model = joblib.load(cfg.V2 / "v2_logreg_l2.joblib")
        exported = (len(payload["vocab"]) + len(payload["numeric"])
                    + sum(len(v) for v in payload["categorical"].values()))
        assert exported == model.coef_.shape[1]

    def test_tfidf_settings_are_pinned(self, payload):
        """These are what the JavaScript mirrors; changing them breaks parity."""
        assert payload["tfidf"] == {"norm": "l2", "smooth_idf": True,
                                    "sublinear_tf": False, "min_token_length": 2}

    def test_declares_both_thresholds(self, payload):
        assert payload["thresholds"]["default"] == 0.5
        assert 0.15 <= payload["thresholds"]["f1_optimal"] <= 0.85

    def test_stopwords_are_shipped(self, payload):
        # The browser cannot import sklearn, so the list has to travel with it.
        assert len(payload["stopwords"]) > 300
        assert "the" in payload["stopwords"]

    def test_metrics_quote_the_committed_numbers(self, payload):
        m = payload["metrics"]
        assert m["logreg_at_0.5"]["accuracy"] == pytest.approx(0.6963, abs=1e-3)
        assert m["boosting_at_0.5"]["accuracy"] == pytest.approx(0.7069, abs=1e-3)
        assert m["majority_baseline_accuracy"] == pytest.approx(0.5961, abs=1e-3)

    def test_serialises_to_json_and_stays_small(self, payload):
        blob = json.dumps(payload, separators=(",", ":"))
        assert len(blob) / 1024 < 200, "payload should stay light enough to ship"
        assert json.loads(blob) == payload


class TestParityWithSklearn:
    def test_matches_predict_proba_on_real_rows(self, payload):
        max_diff = export_web.verify_parity(payload, n=300, tol=1e-6)
        assert max_diff < 1e-6

    def test_unknown_category_contributes_nothing(self, payload):
        """`handle_unknown="ignore"` means an unseen level adds 0 to the log-odds."""
        row = {"name": "handmade oak desk lamp", "log_goal": 9.0,
               "duration_days": 30.0, "name_len": 21.0, "name_words": 4.0,
               "category": "Product Design", "main_category": "Design",
               "currency": "USD", "country": "US",
               "launch_year": "2017", "launch_month": "4", "launch_dow": "1"}
        unknown = {**row, "country": "Atlantis"}

        p_known = export_web.score_from_payload(payload, [row])[0]
        p_unknown = export_web.score_from_payload(payload, [unknown])[0]

        us_coef = payload["categorical"]["country"]["US"]
        # Removing a known level's contribution shifts the log-odds by exactly
        # that coefficient, so the two probabilities must differ accordingly.
        assert p_known != p_unknown
        logit = lambda p: np.log(p / (1 - p))
        assert logit(p_known) - logit(p_unknown) == pytest.approx(us_coef, abs=1e-5)

    def test_empty_title_does_not_produce_nan(self, payload):
        """An all-stopword title has zero L2 norm - the guard must hold."""
        row = {"name": "the and of", "log_goal": 8.5, "duration_days": 30.0,
               "name_len": 10.0, "name_words": 3.0,
               "category": "Product Design", "main_category": "Design",
               "currency": "USD", "country": "US",
               "launch_year": "2017", "launch_month": "4", "launch_dow": "1"}
        p = export_web.score_from_payload(payload, [row])[0]
        assert np.isfinite(p) and 0.0 < p < 1.0

    def test_higher_goal_lowers_the_probability(self, payload):
        """Sanity check on direction - a bigger ask should not help."""
        base = {"name": "handmade oak desk lamp", "duration_days": 30.0,
                "name_len": 21.0, "name_words": 4.0, "category": "Product Design",
                "main_category": "Design", "currency": "USD", "country": "US",
                "launch_year": "2017", "launch_month": "4", "launch_dow": "1"}
        cheap = export_web.score_from_payload(payload, [{**base, "log_goal": 7.0}])[0]
        pricey = export_web.score_from_payload(payload, [{**base, "log_goal": 13.0}])[0]
        assert pricey < cheap


def test_shipped_json_is_current(payload):
    """docs/model.json must match what the code produces right now."""
    path = cfg.ROOT / "docs" / "model.json"
    if not path.exists():
        pytest.skip("docs/model.json not generated yet - run `python -m src.export_web`")
    on_disk = json.loads(path.read_text())
    assert on_disk == json.loads(json.dumps(payload)), (
        "docs/model.json is stale - re-run `python -m src.export_web`"
    )
