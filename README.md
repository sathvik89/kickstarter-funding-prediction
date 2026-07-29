# Kickstarter Funding Predictor

Predicts whether a Kickstarter campaign will hit its funding goal, using **only information that existed before the campaign launched**.

### ▶ [Try the live model](https://sathvik89.github.io/kickstarter-funding-prediction/) · 📖 [Read the full project guide](PROJECT.md)

The trained model runs entirely in your browser — no server, no waiting. Type in a campaign idea and it scores it and shows which words and fields drove the answer.

---

## What this project is really about

The prediction task is the vehicle. The subject is **multi-modal feature fusion** — combining inputs with genuinely different shapes into one matrix a single model can learn from.

|  | Shape | Character |
|---|---|---|
| **Text** — the campaign title, via TF-IDF | 2,500 columns | very wide, 99.5% zeros |
| **Tabular** — goal, category, country, duration, launch date | 243 columns | narrow, dense, mixed types |

You cannot paste those together carelessly. Three things go wrong, and all three are handled here:

- **Memory** — densifying the fused matrix turns 35 MB into 4.6 GB. It stays sparse throughout.
- **Row alignment** — if row *i* of the text block is a different campaign than row *i* of the tabular block, you train on mismatched labels and get a model that looks fine and is nonsense.
- **Leakage** — building the vocabulary before splitting leaks test data into training. Every transformer is fitted on the train block alone, and a test asserts it.

## Why the accuracy is 70% and not 90%

Plenty of Kickstarter models online report 86–94%. Nearly all of them feed in `backers` or `pledged` — the *outcome*. A campaign with 400 backers obviously succeeded; that is reading the answer, not predicting it, and it is useless in practice because an unlaunched campaign has no backers.

This project bans those columns. Under the same honest rules, the best published result is **69.8%**, and always guessing "failed" gets **59.6%**.

**A lower number under honest rules beats a higher number under dishonest ones.**

## Results

Test set, no leakage columns, nothing selected on test. Majority-class accuracy is 0.596.

| | Phase 8 baseline | Phase 9 (L2 logreg) | Phase 9 (boosting) |
|---|---|---|---|
| Accuracy | 0.690 | 0.696 | **0.707** |
| F1 | 0.575 | 0.591 | **0.613** |
| ROC-AUC | 0.740 | 0.754 | **0.772** |
| Avg precision | 0.647 | 0.660 | **0.680** |

Gradient boosting is the best model. The logistic regression is kept because it explains its
predictions — which is why it, not the booster, is the one in the live demo. Both numbers are
labelled on the site so nobody is misled about which they are playing with.

**Careful with thresholds:** at the F1-optimal threshold of 0.31 the logistic regression reaches
F1 0.653 and recall 0.82, but accuracy drops to 0.649 — below the majority baseline. Quote the
threshold whenever you quote a metric, and never mix the best accuracy and the best F1 from
different rows.

Full benchmark comparison in [PROJECT.md § 10](PROJECT.md#10-how-this-compares-to-published-work).

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
PROJECT.md       # the full guide — read this one
data/
  raw/           # original CSV (do not edit)
  interim/       # cleaned / intermediate tables
  processed/     # model-ready matrices / artifacts
    v2/          # Phase 9 models, metrics, ablation tables
notebooks/       # 01-08 frozen pipeline record; 09 the improvements
src/             # the canonical pipeline (see src/README.md)
tests/           # 66 tests
docs/            # GitHub Pages site + markdown documentation
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

## Run it

```bash
python -m src.train        # fit, select on validation, evaluate once on test (~2-3 min)
python -m src.ablations    # what each feature was worth, and why a tree wins
python -m src.export_web   # rebuild the browser model (verifies parity with sklearn)
python -m src.report       # redraw all seven figures — fast, no refitting
pytest                     # 66 tests, ~6s
```

To preview the site locally: `cd docs && python -m http.server`, then open
<http://localhost:8000>. Opening `docs/index.html` straight off disk will not work —
browsers block the `model.json` fetch on `file://`.

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

- **[PROJECT.md — the complete guide](PROJECT.md)** — every decision and why, all results, what failed, the test suite, a glossary, and the questions you'll get asked. Start here.
- [`src/README.md`](src/README.md) — module-by-module map of the pipeline code
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
