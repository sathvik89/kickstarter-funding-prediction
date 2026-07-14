# Phases

Work phase by phase. Validate in notebooks before promoting code to `src/`.

| Phase | Focus | Output |
|-------|--------|--------|
| 0 | Repo structure, env, docs | Folders, `README`, venv, requirements |
| 1 | EDA | Notebook: target filter, leakage audit, basic distributions |
| 2 | Text cleaning | Cleaned `name` column |
| 3 | TF-IDF | Sparse matrix `[N × 2500]` |
| 4 | Tabular preprocessing | Scaled + encoded tabular matrix |
| 5 | Matrix layout check | Dense vs sparse memory notes + merge plan |
| 6 | Feature fusion | Unified `X_fused` via `hstack` |
| 7 | Baseline model | L2 logistic regression on fused features |
| 8 | Evaluation | Metrics, PR curve, coefficient plots |
| 9 | Refactor | `text_cleaner.py`, `fusion_pipeline.py`, `train.py` |
| 10 | Final package | Extra regularization sweeps, report assets |

## Suggested commit messages

Use these after each phase (edit lightly if needed):

- **Phase 0:** `Set up project structure, docs, and Python environment`
- **Phase 1:** `Add EDA notebook and lock target plus feature exclusions`
- **Phase 2:** `Clean campaign titles and strip stopwords`
- **Phase 3:** `Build TF-IDF text features with a 2500-term limit`
- **Phase 4:** `Scale numeric fields and encode categorical metadata`
- **Phase 5:** `Document sparse versus dense fusion plan`
- **Phase 6:** `Fuse text and tabular features with sparse hstack`
- **Phase 7:** `Train L2-regularized logistic regression on fused features`
- **Phase 8:** `Evaluate model performance and plot feature coefficients`
- **Phase 9:** `Refactor validated notebook logic into source modules`
- **Phase 10:** `Tighten evaluation sweeps and package final project assets`
