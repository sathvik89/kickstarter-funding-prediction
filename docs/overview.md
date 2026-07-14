# Project overview

## Goal

Build an integrated ML pipeline that merges unstructured text with structured tabular features and predicts whether a Kickstarter campaign will be funded.

## Why this project

Business systems often score outcomes from mixed inputs: free-text messaging plus rigid numeric/categorical fields. This project practices that pattern with:

1. Text cleaning and stopword removal
2. TF-IDF vectorization (capped at 2,500 features)
3. Scaling + one-hot encoding for tabular columns
4. Sparse/dense fusion via `scipy.sparse.hstack`
5. Regularized logistic regression (start simple with L2, then extend)
6. Coefficient analysis and Precision–Recall / F1 evaluation

## Scope locks

- **Positive class:** `state == successful`
- **Negative class:** `state == failed`
- **Dropped states:** `canceled`, `live`, `suspended`, `undefined`
- **Text modality:** `name` (2018 CSV has no long description field)
- **Excluded (leakage):** `pledged`, `backers`, `usd pledged`, `usd_pledged_real`

Likely model inputs (finalized during EDA):

- Text: `name`
- Numeric: `usd_goal_real` (or `goal`), campaign duration from `launched` → `deadline`
- Categorical: `category`, `main_category`, `currency`, `country`

## Success criteria (handbook)

- Working fusion pipeline with no train/test leakage
- Regularized classifier on the fused matrix
- Top text-token and tabular coefficient plots
- Metrics table (Precision, Recall, Accuracy, F1) across regularization strengths
- Clean repo + markdown docs; report / slides / demo video later
