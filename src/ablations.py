"""Reproducible answer to "why does a tree beat the linear model here?".

The tempting explanation is a missing interaction: a $10k goal is routine for
Music and brutal for Technology, and `w_goal + w_technology` cannot express
that. This module tests that explanation properly and finds it does not hold -
so the conclusion in the docs is backed by a table anyone can regenerate rather
than by an appeal to intuition.

Variants tested, all on top of the same v2 fused matrix with the same C:

* H1 - log_goal crossed with main_category (15 levels) and with category (159)
* H2 - log_goal binned into quantiles, letting the linear model bend freely
* H3 - the full (goal bin x main_category) cell grid, the most expressive
       hand-crafted form short of a tree
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

from . import config as cfg
from . import evaluate as ev
from . import fusion_pipeline as fp
from .features import build_interaction_block


def _fit_score(data: fp.FusedData, Xtr, Xte, C: float) -> dict:
    model = LogisticRegression(
        C=C, solver="liblinear", max_iter=3000, random_state=cfg.RANDOM_STATE
    )
    model.fit(Xtr, data.y_train)
    return ev.score(data.y_test, model.predict_proba(Xte)[:, 1])


def _goal_bin_cross(frame: pd.DataFrame, binner, levels: np.ndarray) -> sparse.csr_matrix:
    """One column per (goal quantile bin, main_category) cell."""
    n_bins = len(binner.bin_edges_[0]) - 1
    bins = binner.transform(frame[["log_goal"]]).ravel().astype(np.int32)
    # int32 matters: category codes are int8 and `code * n_bins` silently
    # overflows to negative indices for the later levels.
    codes = pd.Categorical(
        frame[cfg.INTERACTION_WITH].astype(str), categories=list(levels)
    ).codes.astype(np.int32)
    keep = codes >= 0
    keys = codes * n_bins + bins
    return sparse.csr_matrix(
        (np.ones(int(keep.sum()), np.float32), (np.where(keep)[0], keys[keep])),
        shape=(len(frame), len(levels) * n_bins),
        dtype=np.float32,
    )


def run(C: float = 10.0) -> pd.DataFrame:
    """Return a test-metric table across the hand-crafted interaction variants."""
    data = fp.build(with_interactions=False)
    tr, te = data.frames["train"], data.frames["test"]
    ohe = data.tabular.named_transformers_["cat"]

    rows = [{"variant": "v2 fused (no hand-crafted interactions)",
             "n_extra_cols": 0, **_fit_score(data, data.X_train, data.X_test, C)}]

    # H1 - explicit per-level slopes on log_goal
    for col in ("main_category", "category"):
        levels = ohe.categories_[cfg.CATEGORICAL_COLS.index(col)]
        btr, _ = build_interaction_block(tr, cfg.INTERACTION_BASE, col, levels)
        bte, _ = build_interaction_block(te, cfg.INTERACTION_BASE, col, levels)
        rows.append({
            "variant": f"H1  log_goal x {col} ({len(levels)} levels)",
            "n_extra_cols": btr.shape[1],
            **_fit_score(data,
                         sparse.hstack([data.X_train, btr], format="csr"),
                         sparse.hstack([data.X_test, bte], format="csr"), C),
        })

    # H2 - let the goal curve bend
    for n_bins in (10, 25, 50):
        kb = KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile",
                              subsample=None).fit(tr[["log_goal"]])
        rows.append({
            "variant": f"H2  log_goal binned ({n_bins} quantiles)",
            "n_extra_cols": kb.transform(tr[["log_goal"]]).shape[1],
            **_fit_score(data,
                         sparse.hstack([data.X_train, kb.transform(tr[["log_goal"]])], format="csr"),
                         sparse.hstack([data.X_test, kb.transform(te[["log_goal"]])], format="csr"), C),
        })

    # H3 - the full cell grid
    kb = KBinsDiscretizer(n_bins=cfg.GOAL_BIN_COUNT, encode="ordinal",
                          strategy="quantile", subsample=None).fit(tr[["log_goal"]])
    levels = ohe.categories_[cfg.CATEGORICAL_COLS.index(cfg.INTERACTION_WITH)]
    ctr, cte = _goal_bin_cross(tr, kb, levels), _goal_bin_cross(te, kb, levels)
    rows.append({
        "variant": "H3  (goal bin x main_category) cells",
        "n_extra_cols": ctr.shape[1],
        **_fit_score(data,
                     sparse.hstack([data.X_train, ctr], format="csr"),
                     sparse.hstack([data.X_test, cte], format="csr"), C),
    })

    table = pd.DataFrame(rows)
    cfg.V2.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.V2 / "v2_interaction_ablation.csv", index=False)
    return table


#: Feature groups added on top of the Phase 4 tabular set, in the order the
#: docs present them. `add_one_in` measures each group alone; the final row is
#: everything together, which is what the shipped v2 pipeline uses.
FEATURE_GROUPS = {
    "phase4 baseline (raw goal, duration)": ([], []),
    "+ log1p(goal) replaces raw goal": (["log_goal"], []),
    "+ title length (name_len, name_words)": (["name_len", "name_words"], []),
    "+ launch year / month / dow": ([], ["launch_year", "launch_month", "launch_dow"]),
}

PHASE4_NUM = ["usd_goal_real", "duration_days"]
PHASE4_CAT = ["category", "main_category", "currency", "country"]


def feature_ablation(C: float = 10.0) -> pd.DataFrame:
    """Add one engineered feature group at a time, then all of them.

    Every row refits the vectoriser and scaler on train only and reports test
    metrics, so the deltas are attributable rather than cumulative guesses.
    """
    frame = fp.load_frame()
    rows = []

    for label, (extra_num, extra_cat) in FEATURE_GROUPS.items():
        num = list(PHASE4_NUM)
        if "log_goal" in extra_num:
            num = ["log_goal", "duration_days"]  # replaces, does not add
            extra_num = [c for c in extra_num if c != "log_goal"]
        num += extra_num
        data = fp.build(with_interactions=False, numeric_cols=num,
                        categorical_cols=PHASE4_CAT + extra_cat, frame=frame)
        rows.append({"features": label, "n_features": int(data.X_train.shape[1]),
                     **_fit_score(data, data.X_train, data.X_test, C)})

    data = fp.build(with_interactions=False, frame=frame)
    rows.append({"features": "all engineered (shipped v2)",
                 "n_features": int(data.X_train.shape[1]),
                 **_fit_score(data, data.X_train, data.X_test, C)})

    # text-free row: how much is the TF-IDF block actually carrying?
    data_nt = fp.build(with_interactions=False, with_text=False, frame=frame)
    rows.append({"features": "all engineered, NO text block",
                 "n_features": int(data_nt.X_train.shape[1]),
                 **_fit_score(data_nt, data_nt.X_train, data_nt.X_test, C)})

    table = pd.DataFrame(rows)
    cfg.V2.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.V2 / "v2_feature_ablation.csv", index=False)
    return table


if __name__ == "__main__":
    print("--- feature groups ---")
    print(feature_ablation().to_string(index=False))
    print("\n--- hand-crafted interactions ---")
    print(run().to_string(index=False))
