# Tabular preprocessing notes

Companion to `notebooks/04_tabular_preprocess.ipynb`.

## Goal

Build a numeric tabular matrix aligned with the Phase-3 TF-IDF rows.

## Features

| Kind | Columns | Transform |
|------|---------|-----------|
| Numeric | `usd_goal_real`, `duration_days` | `StandardScaler` |
| Categorical | `category`, `main_category`, `currency`, `country` | `OneHotEncoder(handle_unknown="ignore")` |

## Leakage control

1. Reload `split_ids.csv` and rebuild train/test in that exact ID order
2. Assert targets match `y_train.npy` / `y_test.npy`
3. `fit` the `ColumnTransformer` on train only; `transform` test

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `X_tab_train.npz` / `X_tab_test.npz` | Sparse tabular matrices |
| `tab_feature_names.csv` | Scaled + one-hot feature names |
| `tabular_preprocessor.joblib` | Fitted transformer |
| `tabular_meta.json` | Shape / density metadata |

## Figures

| File | Reading |
|------|---------|
| `04_scaled_numeric_distributions.png` | Train z-scores for goal and duration |
| `04_categorical_cardinality.png` | How many one-hot levels each categorical adds |

## Next phase

Compare memory/layout of sparse text vs tabular matrices and plan `scipy.sparse.hstack` fusion.
