"""Export the L2 logistic regression to JSON so the browser can score it.

Why this works at all: a logistic regression is arithmetic, not a runtime.

    p = sigmoid(b + sum_i w_i * x_i)

So the model *is* its coefficients. Ship the vocabulary, the IDF weights, the
scaler statistics and the category levels, and ~60 lines of JavaScript can
reproduce sklearn's `predict_proba` exactly - no Python, no server, no hosting.

The linear form is also why the demo can explain itself: each token's
contribution to the log-odds is literally `coef * tfidf`, so the page can rank
what drove a prediction. A tree cannot do that, which is the main reason the
in-browser model is the logistic regression rather than the (more accurate)
boosting model.

Everything sklearn does that JS must mirror is pinned here and verified by
`verify_parity()`, which fails loudly rather than shipping a subtly wrong demo:

* `clean_text` - lowercase, letters only, drop sklearn's English stopwords
* `token_pattern=(?u)\\b\\w\\w+\\b` - so single-character tokens are dropped
* `tf * idf`, then **L2-normalise the row** (`norm="l2"`)
* `smooth_idf=True`, `sublinear_tf=False`
* StandardScaler: `(x - mean) / scale`
* OneHotEncoder with `handle_unknown="ignore"` - an unseen level contributes 0
"""

import json

import joblib
import numpy as np

from . import config as cfg
from . import evaluate as ev
from . import fusion_pipeline as fp
from .text_cleaner import STOPWORDS, clean_text

OUT_PATH = cfg.ROOT / "docs" / "model.json"


def _load():
    model = joblib.load(cfg.V2 / "v2_logreg_l2.joblib")
    vec = joblib.load(cfg.V2 / "v2_tfidf_vectorizer.joblib")
    tab = joblib.load(cfg.V2 / "v2_tabular_preprocessor.joblib")
    meta = json.loads((cfg.V2 / "v2_meta.json").read_text())
    return model, vec, tab, meta


def build_payload() -> dict:
    model, vec, tab, meta = _load()
    coef = model.coef_.ravel()
    scaler = tab.named_transformers_["num"]
    ohe = tab.named_transformers_["cat"]

    n_text = len(vec.vocabulary_)
    n_num = len(cfg.NUMERIC_COLS)

    # text block: token -> [idf, coefficient]
    vocab = {
        token: [round(float(vec.idf_[i]), 6), round(float(coef[i]), 6)]
        for token, i in vec.vocabulary_.items()
    }

    # numeric block, in ColumnTransformer order
    numeric = [
        {
            "name": name,
            "mean": float(scaler.mean_[j]),
            "scale": float(scaler.scale_[j]),
            "coef": round(float(coef[n_text + j]), 6),
        }
        for j, name in enumerate(cfg.NUMERIC_COLS)
    ]

    # categorical block: one coefficient per level, in encoder order
    categorical, offset = {}, n_text + n_num
    for col, levels in zip(cfg.CATEGORICAL_COLS, ohe.categories_):
        categorical[col] = {
            str(level): round(float(coef[offset + k]), 6)
            for k, level in enumerate(levels)
        }
        offset += len(levels)
    assert offset == len(coef), f"coefficient accounting off: {offset} vs {len(coef)}"

    # UX support: the demo needs `category` scoped to its parent `main_category`,
    # otherwise a visitor can pick Music + "Product Design" and score a
    # combination that does not exist in reality. The model treats the two
    # columns independently, so this constraint lives in the payload, not the model.
    frame = fp.load_frame()
    pairs = (frame[["main_category", "category"]].astype(str)
             .drop_duplicates().sort_values(["main_category", "category"]))
    category_tree: dict[str, list[str]] = {}
    for main, sub in pairs.itertuples(index=False):
        category_tree.setdefault(main, []).append(sub)

    # The dropdown stays alphabetical (easy to scan), but the *initial* pick is
    # the most common subcategory of each parent - otherwise "Design" opens on
    # "Architecture", which is a strange default for a desk lamp.
    modal = (frame.groupby(["main_category", "category"]).size()
             .reset_index(name="n").sort_values("n", ascending=False)
             .drop_duplicates("main_category"))
    default_category = {str(r.main_category): str(r.category) for r in modal.itertuples()}

    # Sensible starting values so the form is never in an absurd state.
    defaults = {
        "goal_usd": float(np.median(frame["usd_goal_real"])),
        "duration_days": float(np.median(frame["duration_days"])),
        "main_category": "Design",
        "currency": "USD",
        "country": "US",
        "launch_year": str(int(frame["launch_year"].astype(int).max())),
        "launch_month": "4",
    }

    results = {r["model"]: r for r in meta["results"]}
    return {
        "category_tree": category_tree,
        "default_category": default_category,
        "defaults": defaults,
        # Pinned so the browser derives these the way training did:
        # name_len = len(raw title), name_words = whitespace word count.
        "title_feature_semantics": {"name_len": "character count of the raw title",
                                    "name_words": "whitespace-separated word count"},
        "schema": 1,
        "generated_by": "python -m src.export_web",
        "model": "LogisticRegression(l2, liblinear)",
        "intercept": round(float(model.intercept_[0]), 6),
        "tfidf": {"norm": "l2", "smooth_idf": True, "sublinear_tf": False,
                  "min_token_length": 2},
        "stopwords": sorted(STOPWORDS),
        "vocab": vocab,
        "numeric": numeric,
        "categorical": categorical,
        "thresholds": {
            "default": 0.5,
            "f1_optimal": meta["best_threshold_by_val_f1"],
        },
        "metrics": {
            "logreg_at_0.5": {k: results["logreg_l2_fused_v2"][k]
                              for k in ("accuracy", "precision", "recall", "f1",
                                        "roc_auc", "average_precision")},
            "logreg_at_f1_threshold": {k: results["logreg_l2_fused_v2 @val-thr"][k]
                                       for k in ("accuracy", "precision", "recall", "f1")},
            "boosting_at_0.5": {k: results["hgb_tab_plus_textsvd"][k]
                                for k in ("accuracy", "precision", "recall", "f1",
                                          "roc_auc", "average_precision")},
            "majority_baseline_accuracy": meta["majority_baseline_accuracy"],
            "n_train": meta["n_train"], "n_val": meta["n_val"], "n_test": meta["n_test"],
            "n_features": meta["n_features"],
        },
    }


# --------------------------------------------------------------------------
# Parity: reimplement scoring from the payload alone, the way the JS will, and
# require it to match sklearn. If this drifts, the demo lies to visitors.
# --------------------------------------------------------------------------

def score_from_payload(payload: dict, rows) -> np.ndarray:
    """Score records using ONLY the exported JSON - no sklearn objects.

    Deliberately written the way the JavaScript is written, so that a mismatch
    here means a mismatch in the browser.
    """
    vocab = payload["vocab"]
    stops = set(payload["stopwords"])
    min_len = payload["tfidf"]["min_token_length"]
    out = np.empty(len(rows))

    for r, row in enumerate(rows):
        z = payload["intercept"]

        # --- text: clean -> tokens -> tf*idf -> L2 normalise -----------------
        cleaned = clean_text(row["name"])
        tokens = [t for t in cleaned.split() if len(t) >= min_len and t not in stops]
        counts: dict[str, int] = {}
        for t in tokens:
            if t in vocab:
                counts[t] = counts.get(t, 0) + 1
        weights = {t: c * vocab[t][0] for t, c in counts.items()}
        norm = np.sqrt(sum(w * w for w in weights.values()))
        if norm > 0:
            for t, w in weights.items():
                z += (w / norm) * vocab[t][1]

        # --- numeric: standardise, then dot ---------------------------------
        for spec in payload["numeric"]:
            value = float(row[spec["name"]])
            z += ((value - spec["mean"]) / spec["scale"]) * spec["coef"]

        # --- categorical: one-hot, unknown level contributes nothing --------
        for col, levels in payload["categorical"].items():
            z += levels.get(str(row[col]), 0.0)

        out[r] = 1.0 / (1.0 + np.exp(-z))

    return out


def verify_parity(payload: dict, n: int = 400, tol: float = 1e-6) -> float:
    """Compare payload-only scoring against sklearn on real test rows."""
    model, *_ = _load()
    data = fp.build()
    test = data.frames["test"]

    idx = np.random.RandomState(cfg.RANDOM_STATE).choice(len(test), n, replace=False)
    sample = test.iloc[idx]

    expected = model.predict_proba(data.X_test[idx])[:, 1]
    rows = sample.to_dict("records")
    actual = score_from_payload(payload, rows)

    max_diff = float(np.max(np.abs(expected - actual)))
    if max_diff > tol:
        worst = int(np.argmax(np.abs(expected - actual)))
        raise AssertionError(
            f"parity FAILED: max |diff| = {max_diff:.3e} > {tol:.1e}\n"
            f"  worst row: name={sample.iloc[worst]['name']!r}\n"
            f"  sklearn={expected[worst]:.8f} payload={actual[worst]:.8f}"
        )
    return max_diff


def main() -> dict:
    payload = build_payload()
    print(f"vocab {len(payload['vocab']):,} tokens | "
          f"{len(payload['numeric'])} numeric | "
          f"{sum(len(v) for v in payload['categorical'].values())} category levels")

    print("verifying parity against sklearn...")
    max_diff = verify_parity(payload)
    print(f"  max |sklearn - payload| = {max_diff:.3e}  OK")

    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"wrote {OUT_PATH} ({size_kb:.0f} KB, ~{size_kb / 3.5:.0f} KB gzipped)")
    return payload


if __name__ == "__main__":
    main()
