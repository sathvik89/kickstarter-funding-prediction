"""The tests that matter most: the whole project's credibility rests on these.

Every headline number is only meaningful if (a) no outcome column reached the
model and (b) nothing fitted on train ever saw validation or test. Those are
easy to break silently with a one-line edit months from now, so they are
asserted rather than trusted.
"""

import numpy as np

from src import config as cfg

# Locked in docs/overview.md. `pledged`, `backers` and the usd_pledged variants
# are all known only *after* a campaign runs - including any of them turns the
# task into reading the answer off the feature matrix.
LEAKAGE_COLUMNS = ["pledged", "backers", "usd pledged", "usd_pledged_real",
                   "state", "target"]


def test_no_leakage_column_is_configured_as_a_feature():
    configured = set(cfg.NUMERIC_COLS) | set(cfg.CATEGORICAL_COLS)
    assert configured.isdisjoint(LEAKAGE_COLUMNS)


def test_every_tabular_feature_traces_to_a_configured_column(fused):
    """No column may reach the matrix unless config declares it.

    Deliberately *not* a substring scan over all feature names: TF-IDF tokens
    come from campaign titles, so a title containing "United States" legitimately
    produces a `text__state` feature. Substring-matching flags that as leakage
    and is a false positive. The real guarantee is that the non-text blocks
    contain nothing but the configured columns.
    """
    allowed = set(fused.numeric_cols) | set(fused.categorical_cols)

    for name in map(str, fused.feature_names):
        if name.startswith("text__"):
            continue  # title-derived; covered by the vocabulary test below
        body = name.split("__", 1)[1] if "__" in name else name
        assert any(body == col or body.startswith(col + "_") for col in allowed), (
            f"feature {name!r} does not trace to any configured column"
        )


def test_text_features_only_ever_come_from_the_title(fused):
    """Every text feature must be a token of the fitted title vocabulary."""
    vocab = set(fused.vectorizer.vocabulary_)
    text_names = [str(n) for n in fused.feature_names if str(n).startswith("text__")]
    assert len(text_names) == fused.n_text
    assert all(n[len("text__"):] in vocab for n in text_names)


def test_feature_count_matches_declared_blocks(fused):
    assert fused.n_text + fused.n_tab + fused.n_inter == fused.X_train.shape[1]
    assert len(fused.feature_names) == fused.X_train.shape[1]


def test_splits_are_disjoint(splits):
    train, val, test = splits
    ids = [set(f["ID"]) for f in (train, val, test)]
    assert ids[0].isdisjoint(ids[1]), "train/val overlap"
    assert ids[0].isdisjoint(ids[2]), "train/test overlap"
    assert ids[1].isdisjoint(ids[2]), "val/test overlap"


def test_splits_cover_every_row_exactly_once(splits, frame):
    train, val, test = splits
    assert len(train) + len(val) + len(test) == len(frame)


def test_vectorizer_vocabulary_comes_from_train_only(fused, splits):
    """A token only present in val/test must not be in the vocabulary.

    This is the check that catches a future refactor that fits the vectoriser
    before splitting - the classic multi-modal leakage mistake.
    """
    train, val, test = splits
    vocab = set(fused.vectorizer.vocabulary_)

    train_tokens = set()
    for text in train["name_clean"]:
        train_tokens.update(t for t in str(text).split() if len(t) >= 2)

    assert vocab <= train_tokens, (
        f"{len(vocab - train_tokens)} vocabulary tokens never appear in train"
    )


def test_scaler_statistics_come_from_train_only(fused, splits):
    train, _, _ = splits
    scaler = fused.tabular.named_transformers_["num"]
    expected = train[fused.numeric_cols].to_numpy(dtype=float).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-9)


def test_transformers_never_refit_on_val_or_test(fused):
    """Row counts prove the fit used the train block only."""
    scaler = fused.tabular.named_transformers_["num"]
    assert scaler.n_samples_seen_ == fused.X_train.shape[0]
