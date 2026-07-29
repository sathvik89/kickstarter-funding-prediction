"""Phase 9 training entrypoint.

Run from the repo root:

    python -m src.train

Trains three things and writes everything under `data/processed/v2/`:

1. **L2 logistic regression** on the widened fused matrix, with `C` chosen on
   validation. Stays the interpretable headline model - coefficient plots are a
   stated deliverable, so the linear model does not get retired.
2. **The same model without the interaction block**, to show what the goal x
   category columns are actually worth.
3. **HistGradientBoosting** on dense tabular + a TruncatedSVD compression of the
   TF-IDF block. This is a *ceiling check*: it measures how much the linear
   assumption costs, it is not a replacement for the headline model.

Decision thresholds come from validation in every case.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

from . import config as cfg
from . import evaluate as ev
from . import fusion_pipeline as fp
from . import report


def fit_logreg(data: fp.FusedData, C: float, **kw) -> LogisticRegression:
    model = LogisticRegression(
        C=C, solver="liblinear", max_iter=2000, random_state=cfg.RANDOM_STATE, **kw
    )
    model.fit(data.X_train, data.y_train)
    return model


def sweep_C(data: fp.FusedData) -> tuple[pd.DataFrame, float]:
    """Pick C on validation. Phase 8 picked it on test; this is the fix."""
    rows = []
    for C in cfg.C_GRID:
        model = fit_logreg(data, C)
        proba_val = model.predict_proba(data.X_val)[:, 1]
        rows.append({"C": C, "split": "val", **ev.score(data.y_val, proba_val)})
    table = pd.DataFrame(rows)
    best_C = float(table.loc[table["f1"].idxmax(), "C"])
    return table, best_C


def build_dense(data: fp.FusedData) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Dense view for the tree model: ordinal categoricals + SVD-compressed text.

    Trees cannot use a 2,500-column sparse TF-IDF block usefully, and
    HistGradientBoosting does not accept sparse input at all. SVD is how text
    gets into the tree without abandoning the sparse fusion design upstream.
    """
    cols_num = data.numeric_cols
    cols_cat = data.categorical_cols
    tr, va, te = data.frames["train"], data.frames["val"], data.frames["test"]

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    oe.fit(tr[cols_cat].astype(str))

    text_tr = data.X_train[:, : data.n_text]
    svd = TruncatedSVD(n_components=cfg.SVD_COMPONENTS, random_state=cfg.RANDOM_STATE)
    svd.fit(text_tr)

    def pack(frame: pd.DataFrame, X: sparse.csr_matrix) -> np.ndarray:
        return np.hstack(
            [
                frame[cols_num].to_numpy(dtype=float),
                oe.transform(frame[cols_cat].astype(str)),
                svd.transform(X[:, : data.n_text]),
            ]
        )

    mask = [False] * len(cols_num) + [True] * len(cols_cat) + [False] * cfg.SVD_COMPONENTS
    return pack(tr, data.X_train), pack(va, data.X_val), pack(te, data.X_test), mask


def main() -> dict:
    cfg.V2.mkdir(parents=True, exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("building fused matrices (fit on train only)...")
    data = fp.build()
    print(
        f"  train {data.X_train.shape} | val {data.X_val.shape} | test {data.X_test.shape}\n"
        f"  blocks: text={data.n_text} tabular={data.n_tab} interaction={data.n_inter}"
    )

    results: list[dict] = []

    # --- 1. headline linear model ------------------------------------------
    print("\nsweeping C on validation...")
    c_table, best_C = sweep_C(data)
    print(c_table.to_string(index=False))
    print(f"  best C by validation F1: {best_C}")

    model = fit_logreg(data, best_C)
    proba_val = model.predict_proba(data.X_val)[:, 1]
    proba_test = model.predict_proba(data.X_test)[:, 1]
    thr = ev.tune_threshold(data.y_val, proba_val)
    print(f"  best threshold by validation F1: {thr:.2f}")

    results.append({"model": "logreg_l2_fused_v2", **ev.score(data.y_test, proba_test, 0.5)})
    results.append({"model": "logreg_l2_fused_v2 @val-thr", **ev.score(data.y_test, proba_test, thr)})

    # --- 2. would hand-crafted interactions have helped? --------------------
    # Reported here rather than assumed. Full sweep in `src/ablations.py`.
    print("\nchecking the hand-crafted interaction block...")
    data_int = fp.build(with_interactions=True)
    m_int = fit_logreg(data_int, best_C)
    p_int = m_int.predict_proba(data_int.X_test)[:, 1]
    results.append({"model": "logreg_l2_plus_goal_x_category", **ev.score(data_int.y_test, p_int, 0.5)})

    # --- 3. non-linear ceiling check ---------------------------------------
    print("\nfitting the gradient boosting ceiling check...")
    Atr, Ava, Ate, cat_mask = build_dense(data)
    hgb = HistGradientBoostingClassifier(
        random_state=cfg.RANDOM_STATE,
        max_iter=400,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        categorical_features=cat_mask,
    )
    hgb.fit(Atr, data.y_train)
    hgb_val = hgb.predict_proba(Ava)[:, 1]
    hgb_test = hgb.predict_proba(Ate)[:, 1]
    hgb_thr = ev.tune_threshold(data.y_val, hgb_val)
    results.append({"model": "hgb_tab_plus_textsvd", **ev.score(data.y_test, hgb_test, 0.5)})
    results.append({"model": "hgb_tab_plus_textsvd @val-thr", **ev.score(data.y_test, hgb_test, hgb_thr)})

    # --- persist ------------------------------------------------------------
    table = pd.DataFrame(results)
    table.to_csv(cfg.V2 / "v2_model_comparison.csv", index=False)
    c_table.to_csv(cfg.V2 / "v2_validation_C_sweep.csv", index=False)

    coefs = pd.DataFrame({"feature": data.feature_names, "coef": model.coef_.ravel()})
    coefs["block"] = np.select(
        [
            coefs["feature"].str.startswith("text__"),
            coefs["feature"].str.startswith("inter__"),
        ],
        ["text", "interaction"],
        default="tabular",
    )
    coefs.to_csv(cfg.V2 / "v2_coefficients.csv", index=False)

    # Predicted probabilities are persisted so report figures can be redrawn
    # without refitting anything (`python -m src.report`).
    np.savez_compressed(
        cfg.V2 / "v2_probabilities.npz",
        logreg_val=proba_val, logreg_test=proba_test,
        hgb_val=hgb_val, hgb_test=hgb_test,
        y_val=data.y_val, y_test=data.y_test,
    )

    joblib.dump(model, cfg.V2 / "v2_logreg_l2.joblib")
    joblib.dump(hgb, cfg.V2 / "v2_hgb.joblib")
    joblib.dump(data.vectorizer, cfg.V2 / "v2_tfidf_vectorizer.joblib")
    joblib.dump(data.tabular, cfg.V2 / "v2_tabular_preprocessor.joblib")
    sparse.save_npz(cfg.V2 / "v2_X_train_fused.npz", data.X_train)
    sparse.save_npz(cfg.V2 / "v2_X_val_fused.npz", data.X_val)
    sparse.save_npz(cfg.V2 / "v2_X_test_fused.npz", data.X_test)
    np.save(cfg.V2 / "v2_y_train.npy", data.y_train)
    np.save(cfg.V2 / "v2_y_val.npy", data.y_val)
    np.save(cfg.V2 / "v2_y_test.npy", data.y_test)

    meta = {
        "n_train": int(data.X_train.shape[0]),
        "n_val": int(data.X_val.shape[0]),
        "n_test": int(data.X_test.shape[0]),
        "n_features": int(data.X_train.shape[1]),
        "blocks": {"text": data.n_text, "tabular": data.n_tab, "interaction": data.n_inter},
        "numeric_cols": cfg.NUMERIC_COLS,
        "categorical_cols": cfg.CATEGORICAL_COLS,
        "best_C_by_val_f1": best_C,
        "best_threshold_by_val_f1": round(thr, 3),
        "hgb_threshold_by_val_f1": round(hgb_thr, 3),
        "majority_baseline_accuracy": round(ev.majority_baseline(data.y_test), 4),
        "selection_policy": "C and thresholds chosen on validation; test touched once",
        "results": results,
    }
    with open(cfg.V2 / "v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # --- figures -------------------------------------------------------------
    print("\nrendering figures...")
    baseline = _phase8_baseline(data.y_test)
    figures = [
        report.model_comparison(table, baseline),
        report.pr_curves(
            data.y_test,
            {"L2 logistic regression": proba_test, "Gradient boosting": hgb_test},
        ),
        report.threshold_selection(data.y_val, proba_val, data.y_test, proba_test, thr),
        report.success_rate_by_year(data.frames["train"]),
        report.top_coefficients(coefs),
    ]
    for csv_name, fn in (
        ("v2_feature_ablation.csv", lambda t: report.feature_ablation(t)),
        ("v2_interaction_ablation.csv",
         lambda t: report.interaction_vs_tree(t, float(table.loc[table["model"] == "hgb_tab_plus_textsvd", "f1"].iloc[0]))),
    ):
        path = cfg.V2 / csv_name
        if path.exists():
            figures.append(fn(pd.read_csv(path)))
        else:
            print(f"  (skipped {csv_name} figure - run `python -m src.ablations` first)")

    meta["figures"] = [str(Path(f).name) for f in figures]
    with open(cfg.V2 / "v2_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 92)
    print(table.to_string(index=False))
    print("=" * 92)
    print(f"majority baseline accuracy: {meta['majority_baseline_accuracy']}")
    print(f"artifacts -> {cfg.V2}")
    print(f"figures   -> {cfg.FIG_DIR}")
    return meta


def _phase8_baseline(y_test) -> dict:
    """Phase 8's best test metrics, for the side-by-side comparison figure.

    Read from the committed artifact rather than retyped, so the comparison can
    never drift from what Phase 8 actually reported.
    """
    with open(cfg.PROCESSED / "eval_meta.json", encoding="utf-8") as f:
        best = json.load(f)["best_test_metrics"]
    return {**best, "majority": ev.majority_baseline(y_test)}


if __name__ == "__main__":
    main()
