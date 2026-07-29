"""Phase 9 report figures.

Palette note
------------
Phases 1-8 used `#2f5d50` / `#8c6b3f`. As a *categorical pair* that combination
fails two hard accessibility gates: both hues sit below the chroma floor (they
read as gray) and their normal-vision separation is dE 14.8, under the 15 floor -
i.e. readers with full colour vision struggle to tell the two series apart. The
Phase 9 figures therefore adopt a validated three-slot palette (all-pairs CVD
dE 9.2, normal-vision dE 24.0). Single-series figures use slot 1 only.

Slot 3 (aqua) sits below 3:1 contrast on a light surface, so every chart that
uses it also carries visible direct value labels - that is the relief rule, not
decoration.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config as cfg
from . import evaluate as ev

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3
NEUTRAL = "#8a8a85"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#ffffff"


def _style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#d9d9d4",
        "axes.labelcolor": INK_SOFT,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#ececE7",
        "grid.linewidth": 0.8,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def _save(fig, name: str) -> str:
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return str(path)


def model_comparison(table: pd.DataFrame, baseline: dict, name="09_model_comparison.png") -> str:
    """Phase 8 baseline vs v2 linear vs v2 tree, at threshold 0.5.

    One axis: every metric is a 0-1 rate, so they share a scale honestly.
    """
    _style()
    metrics = ["accuracy", "f1", "roc_auc", "average_precision"]
    labels = ["Accuracy", "F1", "ROC-AUC", "Avg precision"]

    lin = table[table["model"] == "logreg_l2_fused_v2"].iloc[0]
    tree = table[table["model"] == "hgb_tab_plus_textsvd"].iloc[0]
    series = [
        ("Phase 8 baseline (L2 logreg)", [baseline[m] for m in metrics], SERIES[0]),
        ("Phase 9 L2 logreg", [lin[m] for m in metrics], SERIES[1]),
        ("Phase 9 gradient boosting", [tree[m] for m in metrics], SERIES[2]),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(metrics))
    width = 0.24
    for i, (label, vals, color) in enumerate(series):
        pos = x + (i - 1) * (width + 0.015)  # surface gap between adjacent bars
        ax.bar(pos, vals, width, label=label, color=color, edgecolor=SURFACE, linewidth=1.5)
        for px, v in zip(pos, vals):
            ax.text(px, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8, color=INK_SOFT)

    # The reference line goes in the legend rather than as an inline label: the
    # line crosses the F1 group, so any inline placement collides with a value.
    ax.axhline(baseline["majority"], color=NEUTRAL, linestyle="--", linewidth=1.2,
               label=f"majority-class accuracy ({baseline['majority']:.3f})")

    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("score (threshold 0.50)")
    ax.set_title("Phase 9 lifts every metric over the Phase 8 baseline")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    return _save(fig, name)


def feature_ablation(table: pd.DataFrame, name="09_feature_ablation.png") -> str:
    """Which engineered feature groups actually paid, one at a time."""
    _style()
    df = table.iloc[::-1].reset_index(drop=True)
    shipped = df["features"].str.contains("shipped")

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    colors = [SERIES[1] if s else SERIES[0] for s in shipped]
    ax.barh(df["features"], df["f1"], height=0.55, color=colors,
            edgecolor=SURFACE, linewidth=1.5)
    for y, v in enumerate(df["f1"]):
        ax.text(v + 0.004, y, f"{v:.4f}", va="center", fontsize=8.5, color=INK_SOFT)

    base = float(table.iloc[0]["f1"])
    ax.axvline(base, color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax.set_xlim(0.5, max(df["f1"]) + 0.03)
    ax.set_xlabel("test F1")
    ax.set_title("Engineered features, added one group at a time\n"
                 "(dashed line = Phase 4 feature set rebuilt on the same split)",
                 fontsize=11)
    return _save(fig, name)


def interaction_vs_tree(table: pd.DataFrame, tree_f1: float,
                        name="09_interaction_vs_tree.png") -> str:
    """The figure that killed the interaction hypothesis.

    Every hand-crafted way of giving a linear model the goal x category effect
    lands in a flat band well short of the tree, so the tree's advantage is
    diffuse non-linearity rather than one nameable missing term.
    """
    _style()
    df = table.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.barh(df["variant"], df["f1"], height=0.55, color=SERIES[0],
            edgecolor=SURFACE, linewidth=1.5)
    for y, v in enumerate(df["f1"]):
        ax.text(v + 0.0008, y, f"{v:.4f}", va="center", fontsize=8.5, color=INK_SOFT)

    ax.axvline(tree_f1, color=SERIES[1], linestyle="--", linewidth=2)
    ax.text(tree_f1 - 0.0012, -0.42, f"gradient boosting  {tree_f1:.4f}",
            ha="right", va="center", fontsize=9, color=INK_SOFT)

    ax.set_xlim(0.585, tree_f1 + 0.004)
    ax.set_xlabel("test F1")
    ax.set_title("Hand-crafted interactions do not close the gap to a tree",
                 fontsize=11)
    return _save(fig, name)


def pr_curves(y_test, proba_map: dict, name="09_pr_curves.png") -> str:
    """Precision-recall, the threshold-independent view."""
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 5))
    for (label, proba), color in zip(proba_map.items(), SERIES):
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = ev.score(y_test, proba)["average_precision"]
        ax.plot(recall, precision, linewidth=2, color=color, label=f"{label} (AP {ap:.3f})")

    prevalence = float(np.mean(y_test))
    ax.axhline(prevalence, color=NEUTRAL, linestyle="--", linewidth=1.2)
    ax.text(0.02, prevalence + 0.012, f"no-skill baseline {prevalence:.3f}",
            fontsize=8, color=INK_SOFT)

    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0.3, 1.0)
    ax.set_title("Precision-recall on the held-out test set")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, name)


def threshold_selection(y_val, proba_val, y_test, proba_test, chosen: float,
                        name="09_threshold_selection.png") -> str:
    """Proof the validation-chosen threshold transfers to test.

    This is the figure that answers "how did you pick 0.30?" - the whole point of
    adding a validation split instead of tuning on test like Phase 8 did.
    """
    _style()
    from sklearn.metrics import f1_score

    grid = ev.threshold_grid()
    val_f1 = [f1_score(y_val, (proba_val >= t).astype(int)) for t in grid]
    test_f1 = [f1_score(y_test, (proba_test >= t).astype(int)) for t in grid]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(grid, val_f1, linewidth=2, color=SERIES[0], label="validation F1")
    ax.plot(grid, test_f1, linewidth=2, color=SERIES[1], label="test F1")

    peak = test_f1[int(np.argmin(np.abs(grid - chosen)))]
    ax.scatter([chosen], [peak], s=90, color=SERIES[1],
               edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.axvline(chosen, color=NEUTRAL, linestyle="--", linewidth=1.2)
    # Annotate beside the marker, not down in the legend's corner.
    ax.annotate(f"chosen on validation: {chosen:.2f}",
                xy=(chosen, peak), xytext=(chosen + 0.06, peak - 0.055),
                fontsize=9, color=INK_SOFT,
                arrowprops=dict(arrowstyle="-", color=NEUTRAL, linewidth=1))

    ax.set_xlabel("decision threshold")
    ax.set_ylabel("F1")
    ax.set_title("The validation-selected threshold lands on the test optimum")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, name)


def success_rate_by_year(frame: pd.DataFrame, name="09_success_rate_by_launch_year.png") -> str:
    """Why launch date earns a place in the feature set."""
    _style()
    by_year = (frame.assign(year=frame["launch_year"].astype(str))
               .groupby("year")["target"].agg(["mean", "size"])
               .reset_index().sort_values("year"))

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(by_year["year"], by_year["mean"], width=0.62, color=SERIES[0],
           edgecolor=SURFACE, linewidth=1.5)

    hi, lo = by_year["mean"].idxmax(), by_year["mean"].idxmin()
    for i in (hi, lo):
        row = by_year.loc[i]
        ax.text(row["year"], row["mean"] + 0.008, f"{row['mean']:.3f}",
                ha="center", fontsize=9, color=INK_SOFT)

    ax.set_ylabel("success rate")
    ax.set_xlabel("launch year")
    ax.set_ylim(0, 0.60)
    spread = 100 * (by_year["mean"].max() - by_year["mean"].min())
    ax.set_title(f"Success rate swings {spread:.1f} points by launch year - "
                 "signal Phase 4 discarded", fontsize=11)
    return _save(fig, name)


def top_coefficients(coefs: pd.DataFrame, k: int = 15,
                     name="09_top_coefficients.png") -> str:
    """Keeps the interpretability deliverable alive on the widened matrix.

    Polarity, so the two colours are a diverging pair (warm/cool), not identity.
    """
    _style()
    ranked = coefs.sort_values("coef")
    bottom, top = ranked.head(k), ranked.tail(k)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, part, color, title in (
        (axes[0], top.iloc[::-1], SERIES[0], f"Top {k} pushing toward success"),
        (axes[1], bottom, SERIES[1], f"Top {k} pushing toward failure"),
    ):
        names = part["feature"].str.replace("^(text|num|cat|inter)__", "", regex=True)
        ax.barh(names, part["coef"], height=0.6, color=color,
                edgecolor=SURFACE, linewidth=1.2)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("coefficient")
        ax.invert_yaxis()
        ax.tick_params(labelsize=8.5)

    fig.suptitle("L2 logistic regression coefficients on the Phase 9 fused matrix",
                 fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def regenerate_all() -> list[str]:
    """Redraw every Phase 9 figure from saved artifacts - no model refitting.

    Run `python -m src.train` once, then this as often as the figures need
    tweaking.
    """
    import json

    from . import fusion_pipeline as fp

    table = pd.read_csv(cfg.V2 / "v2_model_comparison.csv")
    coefs = pd.read_csv(cfg.V2 / "v2_coefficients.csv")
    probs = np.load(cfg.V2 / "v2_probabilities.npz")
    with open(cfg.V2 / "v2_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    with open(cfg.PROCESSED / "eval_meta.json", encoding="utf-8") as f:
        phase8 = json.load(f)["best_test_metrics"]

    y_val, y_test = probs["y_val"], probs["y_test"]
    baseline = {**phase8, "majority": ev.majority_baseline(y_test)}
    tree_f1 = float(table.loc[table["model"] == "hgb_tab_plus_textsvd", "f1"].iloc[0])

    out = [
        model_comparison(table, baseline),
        pr_curves(y_test, {"L2 logistic regression": probs["logreg_test"],
                           "Gradient boosting": probs["hgb_test"]}),
        threshold_selection(y_val, probs["logreg_val"], y_test, probs["logreg_test"],
                            meta["best_threshold_by_val_f1"]),
        top_coefficients(coefs),
    ]
    for csv_name, fn in (("v2_feature_ablation.csv", feature_ablation),
                         ("v2_interaction_ablation.csv",
                          lambda t: interaction_vs_tree(t, tree_f1))):
        path = cfg.V2 / csv_name
        if path.exists():
            out.append(fn(pd.read_csv(path)))

    train_df, _, _ = fp.three_way_split(fp.load_frame())
    out.append(success_rate_by_year(train_df))
    return out


if __name__ == "__main__":
    for path in regenerate_all():
        print("wrote", path)
