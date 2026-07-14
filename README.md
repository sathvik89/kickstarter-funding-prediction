# Predictive Sales Analytics Engine

Multi-modal feature fusion on Kickstarter campaign data: clean campaign titles, build TF-IDF text features, preprocess tabular fields, fuse sparse + dense matrices, and train a regularized logistic regression model to predict funding success.

## Problem

Predict a **binary funding outcome** (`successful` vs `failed`) from:

- **Text:** campaign `name`
- **Tabular:** pre-campaign metadata (goal, category, country, duration, etc.)

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
- **[Concept learning guide (HTML)](docs/learning-guide.html)** — TF-IDF, fusion, regularization, metrics, and more

## Stack

Python 3.10+, NumPy, Pandas, SciPy, Scikit-Learn, Matplotlib, Seaborn, NLTK, Jupyter
