"""Feature engineering correctness, including the bug that actually bit."""

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.features import add_engineered_columns, build_interaction_block
from src.text_cleaner import clean_text


@pytest.fixture
def toy():
    return pd.DataFrame({
        "usd_goal_real": [0.0, 5000.0, 166_361_390.7],
        "launched": ["2011-04-01 12:00:00", "2015-12-25 08:30:00", "not a date"],
        "name_len": [10, 20, np.nan],
        "name_words": [2, 4, np.nan],
        "main_category": ["Music", "Technology", "Music"],
        "target": [1, 0, 1],
    })


class TestTextCleaner:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("The Songs of Adelaide & Abullah") == "songs adelaide abullah"

    def test_drops_digits_entirely(self):
        # TOKEN_RE keeps [a-z] only, so "3D" contributes "d" (dropped later as
        # a single character by the vectoriser's token pattern).
        assert clean_text("Project 2024") == "project"

    def test_handles_empty_and_non_string_input(self):
        assert clean_text("") == ""
        # str(None) -> "none", which is itself in sklearn's stopword list, so a
        # null title collapses to empty rather than injecting a junk token.
        assert clean_text(None) == ""

    def test_is_idempotent(self):
        once = clean_text("Where is Hank?")
        assert clean_text(once) == once


class TestEngineeredColumns:
    def test_log_goal_matches_log1p(self, toy):
        out = add_engineered_columns(toy)
        np.testing.assert_allclose(out["log_goal"], np.log1p(toy["usd_goal_real"]))

    def test_log_goal_tames_the_skew(self, frame):
        """The stated reason this column exists: skew 83 -> about zero."""
        raw = frame["usd_goal_real"].skew()
        logged = frame["log_goal"].skew()
        assert raw > 50, f"expected heavy skew in raw goal, got {raw}"
        assert abs(logged) < 1, f"log1p should tame the skew, got {logged}"

    def test_negative_goals_are_clipped_not_nan(self):
        out = add_engineered_columns(pd.DataFrame({
            "usd_goal_real": [-5.0], "launched": ["2015-01-01"],
            "name_len": [3], "name_words": [1],
        }))
        assert np.isfinite(out["log_goal"]).all()

    def test_date_parts_are_levels_not_numbers(self, toy):
        out = add_engineered_columns(toy)
        for col in ("launch_year", "launch_month", "launch_dow"):
            assert out[col].dtype == object or str(out[col].dtype) == "string", (
                f"{col} must be categorical - as a number, 'December is 12x January'"
            )

    def test_unparseable_dates_become_unknown_not_nan(self, toy):
        out = add_engineered_columns(toy)
        assert out["launch_year"].notna().all()
        assert "2011" in out["launch_year"].iloc[0]

    def test_missing_title_lengths_become_zero(self, toy):
        out = add_engineered_columns(toy)
        assert out["name_len"].iloc[2] == 0
        assert out["name_words"].notna().all()


class TestInteractionBlock:
    def test_places_value_in_the_matching_level_column(self, toy):
        df = add_engineered_columns(toy)
        levels = np.array(["Music", "Technology"])
        block, names = build_interaction_block(df, "log_goal", "main_category", levels)

        assert block.shape == (3, 2)
        assert names == ["inter__log_goal_x_main_category_Music",
                         "inter__log_goal_x_main_category_Technology"]
        dense = block.toarray()
        # row 0 is Music with log_goal 0 -> the only nonzero would be 0 anyway
        assert dense[1, 1] == pytest.approx(df["log_goal"].iloc[1], rel=1e-5)
        assert dense[1, 0] == 0.0

    def test_unknown_levels_contribute_nothing(self, toy):
        df = add_engineered_columns(toy)
        block, _ = build_interaction_block(df, "log_goal", "main_category",
                                          np.array(["Games"]))
        assert block.nnz == 0

    def test_survives_more_than_127_levels(self, frame):
        """Regression test for a real int8 overflow.

        Category codes come back as int8; `code * n_bins` silently wrapped
        negative past level 127, which raised
        `ValueError: negative axis 1 index` on the 159-level column.
        """
        levels = np.array(sorted(frame["category"].astype(str).unique()))
        assert len(levels) > 127, "need a high-cardinality column to test this"

        sample = frame.head(2000)
        block, names = build_interaction_block(sample, "log_goal", "category", levels)
        assert block.shape == (2000, len(levels))
        assert len(names) == len(levels)
        assert block.min() >= 0


def test_title_length_columns_measure_the_raw_title(frame):
    """Pins the semantics the browser demo reimplements in JavaScript.

    Phase 1 defined these as `name.str.len()` and `name.str.split().str.len()`
    - i.e. the *raw* title, not the cleaned one. The web predictor derives them
    from the title box the same way. If this definition ever changes, the live
    demo silently starts feeding the model different numbers than it trained on,
    and no parity test would catch it (both sides would agree, and both would be
    wrong).
    """
    sample = frame.head(500)
    expected_len = sample["name"].astype(str).str.len()
    expected_words = sample["name"].astype(str).str.split().str.len()

    np.testing.assert_array_equal(sample["name_len"].to_numpy(), expected_len.to_numpy())
    np.testing.assert_array_equal(sample["name_words"].to_numpy(), expected_words.to_numpy())


def test_config_declares_the_engineered_columns():
    """Guards against adding a feature to features.py and forgetting config."""
    assert "log_goal" in cfg.NUMERIC_COLS
    assert "usd_goal_real" not in cfg.NUMERIC_COLS, "raw goal was replaced, not kept"
    for col in ("launch_year", "launch_month", "launch_dow"):
        assert col in cfg.CATEGORICAL_COLS
