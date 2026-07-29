# Phase 9 — feature engineering, honest validation, and a non-linear ceiling check

Notebook: [`notebooks/09_improved_pipeline.ipynb`](../notebooks/09_improved_pipeline.ipynb)
Modules: [`src/`](../src) · Artifacts: `data/processed/v2/` · Figures: `reports/figures/09_*.png`

Phases 1–8 stay frozen — they are the record of how the fusion pipeline was built.
Phase 9 is the promoted version of that logic plus the changes below.

## Headline

| | Phase 8 | Phase 9 (L2 logreg) | Phase 9 (boosting) |
|---|---|---|---|
| Accuracy | 0.6898 | 0.6963 | **0.7069** |
| F1 | 0.5747 | 0.5905 | **0.6126** |
| ROC-AUC | 0.7404 | 0.7543 | **0.7718** |
| Avg precision | 0.6473 | 0.6605 | **0.6800** |

Majority-class accuracy is 0.5961. All numbers are test-set, threshold 0.50, no leakage
columns, and — new in Phase 9 — nothing selected on test.

At the validation-selected threshold (0.31) the linear model reaches **F1 0.6528** and
boosting reaches **F1 0.6662**, trading precision for recall (~0.82 recall). That is the
right trade only if a missed fundable campaign costs more than a false alarm.

## What changed and why

### 1. `log1p(goal)` — the biggest single win (+0.009 F1)

`usd_goal_real` has skew **83.1**: median \$5,000, 99th percentile \$300,000, max
\$166M. Standardising that hands a linear model a coefficient on a variable whose useful
range is the first 0.2% of its span. `log1p` brings skew to **−0.14**.

External corroboration: published work finds the negative effect of a higher goal
["does not appear until Log(Goal) exceeds 7"](https://www.sciencedirect.com/science/article/pii/S2667277424000033)
— a genuine non-linearity, not a scaling nicety.

### 2. Launch date parts (+0.005 F1)

Phase 4 derived `duration_days` from `launched`/`deadline` and discarded the rest. In
this dataset success rate by launch year runs:

| 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---|---|---|---|---|---|---|---|
| 0.491 | 0.480 | **0.506** | 0.465 | 0.472 | 0.356 | **0.321** | 0.381 | 0.425 |

A 19-point spread, matching the
[documented platform-wide decline](https://medium.com/@aldgadra/what-made-kickstarter-success-rate-so-low-f5dc575005f9)
after 2014. `launch_year`, `launch_month` and `launch_dow` are one-hot encoded as levels,
not fed in as numbers.

### 3. Title length (+0.006 F1)

`name_len` and `name_words` were computed back in Phase 1 and sat unused in
`ks_binary_base.csv` — Phase 4's `numeric_cols` was only `usd_goal_real` and
`duration_days`. Every comparable study uses name or blurb length.

### 4. Validation split — the methodology fix

Phase 8 selected `best_C` by **test** F1, and the roughbook selected the decision
threshold by **test** F1. Both are selection-on-test: the reported number is then
optimistic by an unknown amount and there is no answer to "how did you choose 0.32?"

Phase 9 keeps `test_size`/`random_state` identical to Phase 3 — so the test set is the
same rows and the comparison stays fair — then carves 20% of the old train block into
validation. `C` and the threshold come from validation; test is touched once.

The roughbook's threshold finding **survived** this: F1 0.6424 tuned on test versus
0.6412 tuned on validation. It was a real effect.

## The interaction hypothesis, and why it was wrong

The obvious explanation for a tree beating a linear model here: a \$10k goal is routine
for Music and brutal for Technology, and `w_goal + w_technology` cannot say that.

That explanation does not survive measurement. Six hand-crafted variants, all on the same
fused matrix with the same `C` (reproduce with `python -m src.ablations`):

| Variant | Extra cols | Test F1 | Δ |
|---|---|---|---|
| v2 fused, no hand-crafted interactions | 0 | 0.5905 | — |
| `log_goal × main_category` (15 levels) | 15 | 0.5910 | +0.0005 |
| `log_goal × category` (159 levels) | 159 | 0.5912 | +0.0007 |
| `log_goal` binned, 10 quantiles | 10 | 0.5917 | +0.0012 |
| `log_goal` binned, 50 quantiles | 43 | 0.5927 | +0.0022 |
| `(goal bin × main_category)` cell grid | 150 | 0.5939 | +0.0034 |
| **gradient boosting** | — | **0.6126** | **+0.0221** |

Even the full cell grid — the most expressive hand-crafted form short of a tree —
recovers about 15% of the gap. **The tree's advantage is diffuse non-linearity across
many features, not one nameable missing term.**

Two consequences:

1. The interaction columns are **not** in the shipped model (`USE_INTERACTIONS = False`).
   They did not earn their place; the code and the ablation stay so the decision is
   reproducible rather than asserted.
2. Inventing more interaction features on this dataset is not worth the time. Use a tree
   for non-linearity and keep the linear model for interpretation.

## Why boosting ships *alongside*, not instead

Gradient boosting on the tabular block **alone, with no TF-IDF at all** already beats L2
logistic regression on the full 2,743-column fused matrix — the linear assumption was
costing more than the entire text modality contributes.

It still does not replace the headline model:

- Coefficient plots are a stated deliverable in [`docs/overview.md`](overview.md) and a
  tree does not produce them.
- Sparse fusion via `hstack` is the point of the project. Boosting cannot consume sparse
  input, so text reaches it through `TruncatedSVD(120)` — the fusion design upstream is
  untouched.

So both ship: logistic regression for *why*, boosting for *how well*.

## Where this lands against published work

| Source | Reported | Comparable? |
|---|---|---|
| [Springer 2019, *Kickstarter at Launch Time*](https://link.springer.com/chapter/10.1007/978-3-030-29516-5_39) (XGBoost) | 69.8% | **Yes** — launch-time features only |
| This project, Phase 8 | 69.0% | baseline |
| **This project, Phase 9 (boosting)** | **70.7%** | — |
| [Same-dataset LightGBM](https://github.com/srishtis/Kaggle-Kickstarter-Project-Status-Prediction) | 70.3% | No — `avg_pledge_per_backer` is leakage-derived |
| Assorted blogs / repos | 86–94% | No — use `backers`, i.e. the outcome |
| [UTwente 2024 thesis](https://essay.utwente.nl/fileshare/file/101151/41TScIT_submission_175%20(1).pdf) (Random Forest) | 88.3% | No — in-sample; its own CV figure is ~77% on a rebalanced set |
| [Stanford CS229, *Plead or Pitch?*](https://cs229.stanford.edu/proj2015/239_report.pdf) | F1 0.79 | No — full descriptions + LIWC |

Phase 9 clears the only leakage-free launch-time benchmark found. The 86–94% figures are
not a target; they are a different problem, one where the answer is already in the
features.

## Ceiling — what will not work

- **More TF-IDF tuning.** Bigrams and a bigger vocabulary have almost nothing to work
  with: the 2018 Kaggle CSV has no `blurb` or description, only a ~4-word title. Every
  study that beats us on text uses full descriptions. This is a dataset limit, not a
  code limit.
- **More hand-crafted interactions.** Measured above.
- **Heavier regularization.** The Phase 8 sweep was already flat with a ~1-point
  train/test gap.

Genuinely promising, if the scope ever widens: richer campaign text from a source that
has it, creator history and seasonality (leakage-safe only), and probability calibration
so thresholds mean something.

## Reproducing

```bash
python -m src.train       # fits, evaluates, writes data/processed/v2/ + figures
python -m src.ablations   # feature + interaction ablation tables
python -m src.report      # redraws figures from saved artifacts, no refitting
```

## A note on the figures

Phases 1–8 used `#2f5d50` / `#8c6b3f`. As a *categorical pair* that combination fails two
accessibility gates: both hues fall below the chroma floor (they read as gray) and their
normal-vision separation is ΔE 14.8, under the 15 floor — full-colour-vision readers
struggle to tell the two series apart. The Phase 9 figures use a validated three-slot
palette instead (all-pairs CVD ΔE 9.2, normal-vision ΔE 24.0), with visible value labels
wherever a low-contrast slot appears.
