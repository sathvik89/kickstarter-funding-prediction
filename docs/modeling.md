# Modeling notes — L2 logistic regression

Companion to `notebooks/07_train_l2_logreg.ipynb`.

## Goal

Train a regularized binary classifier on the fused matrix so text tokens cannot freely dominate tabular signals.

## Model setup

| Setting | Value |
|---------|--------|
| Model | `sklearn.linear_model.LogisticRegression` |
| Penalty | L2 |
| `C` | 1.0 (inverse strength; start simple) |
| Solver | `liblinear` (good for binary L2 + sparse input) |
| Features | `X_train_fused.npz` / `X_test_fused.npz` |

## Why L2 first

- Fused width is 2,713 features (2,500 text + 213 tabular)
- L2 shrinks large weights smoothly without forcing exact zeros
- Matches the handbook baseline before ElasticNet / C sweeps

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `model_logreg_l2.joblib` | Fitted model |
| `model_l2_metrics.csv` | Train/test Accuracy, Precision, Recall, F1, ROC-AUC |
| `model_l2_coefficients.csv` | Per-feature coefficients + modality tag |
| `model_l2_meta.json` | Training config + test metrics snapshot |

## Figure

`reports/figures/07_mean_abs_coef_by_modality.png` — average absolute weight for text vs tabular features.

## Next phase

Full evaluation: Precision–Recall curve, regularization strength comparison, top 15 ± text tokens and tabular coefficient plots.
