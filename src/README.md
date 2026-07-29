# `src/` — the canonical pipeline

Notebooks 01–08 are the frozen record of how each stage was originally validated.
These modules are the promoted, parameterised version of that logic, plus the
Phase 9 feature and methodology work. When the notebooks and these modules
disagree, **these modules are what actually produced the published numbers.**

Full background in [`../PROJECT.md`](../PROJECT.md).

## Reading order

If you are new to the code, read it in this order — each file builds on the last:

| # | File | ~Lines | What it does |
|---|---|---|---|
| 1 | `config.py` | 61 | Every path, constant and feature group. **Start here** — it is the map of what the model is allowed to see. |
| 2 | `text_cleaner.py` | 20 | `clean_text()`. Lowercase → letters only → drop stopwords. Promoted verbatim from Phase 2. |
| 3 | `features.py` | 68 | The engineered columns: `log_goal`, launch-date parts, title lengths, and interaction blocks. |
| 4 | `fusion_pipeline.py` | 172 | The heart of it. Three-way split, fit-on-train-only, sparse `hstack`. |
| 5 | `evaluate.py` | 51 | Metrics, threshold tuning, majority baseline. Enforces one rule: thresholds come from validation, never test. |
| 6 | `train.py` | 247 | The entrypoint. `python -m src.train` fits everything and writes artifacts + figures. |
| 7 | `ablations.py` | 163 | Answers "what was each feature worth?" and "why does a tree beat the linear model?" with tables instead of opinions. |
| 8 | `report.py` | 309 | The seven Phase 9 figures. `python -m src.report` redraws them without refitting. |
| 9 | `export_web.py` | 241 | Serialises the logistic regression to JSON so the browser can score it, and verifies the result against scikit-learn. |

## Commands

```bash
python -m src.train        # fit, select on validation, evaluate once on test, save everything
python -m src.ablations    # the two ablation tables
python -m src.export_web   # rebuild docs/model.json (raises if parity breaks)
python -m src.report       # redraw figures from saved artifacts — fast, no refitting
```

Run them from the repository root, not from inside `src/`.

## The three things not to break

**1. Fit on train only.** `fusion_pipeline.build()` fits the vectoriser and the
scaler on the train block and only *transforms* validation and test. If you ever
move a `fit` call above the split, the model will look better and be wrong.
`tests/test_leakage.py` will catch it.

**2. The excluded columns.** `pledged`, `backers`, `usd pledged`,
`usd_pledged_real` are outcomes, not inputs. Adding any of them pushes accuracy
into the 90s and makes the model useless, because a campaign that has not launched
has no backers.

**3. Sparsity.** Never call `.toarray()` on the fused matrix. It is 99.5% zeros —
densifying turns 35 MB into 4.6 GB.

## Where the numbers live

`python -m src.train` writes everything to `data/processed/v2/`. The file to read
first is `v2_meta.json`: it holds every headline metric, the split sizes, and the
selection policy in one place.

## A note on `USE_INTERACTIONS = False`

That flag in `config.py` is a **measured decision, not an oversight**. The
hypothesis was that a linear model loses to a tree because it cannot express
"this goal is ambitious *for this category*". Six hand-crafted variants were
tested, up to a full 150-column `(goal bin × category)` grid, and the best
recovered only ~15% of the gap. Run `python -m src.ablations` to regenerate the
table. Don't re-litigate it without reading that first.
