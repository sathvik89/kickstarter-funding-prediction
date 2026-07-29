# Predictive Sales Analytics Engine

Multi-modal feature fusion on Kickstarter campaign data: clean campaign titles, build TF-IDF text features, preprocess tabular fields, fuse sparse + dense matrices, and train a regularized logistic regression model to predict funding success.

**🔗 [Live site + in-browser predictor](https://sathvik89.github.io/kickstarter-funding-prediction/)** — the trained model runs client-side, no server involved.

## Problem

Predict a **binary funding outcome** (`successful` vs `failed`) from:

- **Text:** campaign `name`
- **Tabular:** pre-campaign metadata (goal, category, country, duration, etc.)

## Results

Test set, no leakage columns, nothing selected on test. Majority-class accuracy is 0.596.

| | Phase 8 baseline | Phase 9 (L2 logreg) | Phase 9 (boosting) |
|---|---|---|---|
| Accuracy | 0.690 | 0.696 | **0.707** |
| F1 | 0.575 | 0.591 | **0.613** |
| ROC-AUC | 0.740 | 0.754 | **0.772** |
| Avg precision | 0.647 | 0.660 | **0.680** |

For context, the best leakage-free published benchmark using launch-time-only features is
**69.8%** ([Springer 2019](https://link.springer.com/chapter/10.1007/978-3-030-29516-5_39)).
Higher figures in the wild (86–94%) generally include `backers` or `pledged` — the outcome
itself. See [Phase 9 notes](docs/phase9-improvements.md) for the full comparison.

## Dataset

- **Source:** Kaggle Kickstarter Projects (2018)
- **Local path:** `data/raw/kickstarter_2018.csv`
- **Size:** ~379k rows, 15 columns

## Project decisions

| Topic | Decision |
|--------|----------|
| Target | `successful` vs `failed` only |
| Text field | `name` |
| Leakage | Exclude outcome columns (`pledged`, `backers`, `usd pledged`, `usd_pledged_real`) |
| Workflow | Notebooks first, then refactor into `src/` modules |
| Environment | Python venv |

## Repository layout

```text
data/
  raw/           # original CSV (do not edit)
  interim/       # cleaned / intermediate tables
  processed/     # model-ready matrices / artifacts
notebooks/       # exploratory + pipeline notebooks (primary for now)
src/             # reusable modules (added after notebook validation)
configs/         # parameters / experiment settings
docs/            # project documentation
reports/
  figures/       # plots for evaluation and report
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=ml-summer --display-name="ML Summer Project"
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest                    # 66 tests, ~6s
```

The suite exists because the headline numbers are only meaningful if two properties hold, and both are easy to break silently:

- **No leakage.** No outcome column may be configured as a feature, and every non-text column in the fused matrix must trace back to a declared config entry.
- **Fit on train only.** The TF-IDF vocabulary must be a subset of training-block tokens and the scaler's row count must equal the training block's — which catches fitting before splitting.

It also guards **browser parity**: the model exported to `docs/model.json` is re-scored from the JSON alone and must match scikit-learn, and the live page's JavaScript is executed in Node and compared too. Without those, the live demo could quietly mislead visitors while every number in the repo stayed correct.

## Live predictor

A logistic regression is just `p = σ(b + Σ wᵢxᵢ)`, so the model *is* its coefficients. `python -m src.export_web` serializes the vocabulary, IDF weights, coefficients and scaler stats to a ~24 KB (gzipped) JSON file, and the page scores it in JavaScript — free hosting on GitHub Pages, no cold starts, and the linear form lets it show which words and fields drove each prediction.

The demo runs the logistic regression (0.696) rather than the boosting model (0.707): it is 20× lighter to ship and, unlike a tree, it can explain itself. Both numbers are labelled on the page.

## Development approach

1. Explore and validate each step in `notebooks/`
2. Keep commits phase-aligned (see `docs/phases.md`)
3. After the pipeline is stable, move logic into `src/` (`text_cleaner.py`, `fusion_pipeline.py`, `train.py`)

## Documentation

- [Project overview](docs/overview.md)
- [Phases & commits](docs/phases.md)
- [Data notes](docs/data.md)
- [EDA findings (what the figures say)](docs/eda-findings.md)
- [Text cleaning notes](docs/text-cleaning.md)
- [TF-IDF notes](docs/tfidf.md)
- [Tabular preprocessing notes](docs/tabular.md)
- [Fusion plan (sparse vs dense)](docs/fusion-plan.md)
- [Feature fusion notes](docs/fusion.md)
- [Modeling notes (L2 logistic regression)](docs/modeling.md)
- [Evaluation notes](docs/evaluation.md)
- [Metric improvement thinking (roughbook)](docs/metric-improvement-notes.md)
- **[Phase 9: features, honest validation, boosting ceiling check](docs/phase9-improvements.md)**
- **[Concept learning guide (HTML)](docs/learning-guide.html)** — TF-IDF, fusion, regularization, metrics, and more

## Stack

Python 3.10+, NumPy, Pandas, SciPy, Scikit-Learn, Matplotlib, Seaborn, NLTK, Jupyter
