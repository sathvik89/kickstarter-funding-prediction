# Data notes

## File

| Item | Value |
|------|--------|
| File | `data/raw/kickstarter_2018.csv` |
| Rows | ~378,661 |
| Columns | 15 |
| Approx size | ~58 MB |

## Columns

| Column | Role |
|--------|------|
| `ID` | Identifier (not a feature) |
| `name` | Text feature |
| `category` | Categorical feature |
| `main_category` | Categorical feature |
| `currency` | Categorical feature |
| `deadline` | Used to derive duration |
| `launched` | Used to derive duration |
| `goal` | Numeric (prefer `usd_goal_real` when consistent) |
| `usd_goal_real` | Numeric feature (goal in USD) |
| `country` | Categorical feature |
| `state` | Target source |
| `pledged` | Leakage — exclude |
| `backers` | Leakage — exclude |
| `usd pledged` | Leakage — exclude |
| `usd_pledged_real` | Leakage — exclude |

## Target construction

Keep rows where `state` is `successful` or `failed`.

Map:

- `successful` → 1
- `failed` → 0

## Folder convention

| Path | Purpose |
|------|---------|
| `data/raw/` | Untouched source data |
| `data/interim/` | Cleaned tables after EDA / text cleaning |
| `data/processed/` | Matrices and train/test splits ready for modeling |

## Interim outputs

| File | From | Description |
|------|------|-------------|
| `data/interim/ks_binary_base.csv` | Phase 1 EDA | `successful`/`failed` only, leakage columns removed, with `target`, `duration_days`, and basic name stats |
| `data/interim/ks_text_cleaned.csv` | Phase 2 text cleaning | Same base plus `name_clean` / `name_clean_tokens`; empty cleaned titles removed |

## Processed outputs

| File | From | Description |
|------|------|-------------|
| `data/processed/X_text_*.npz` | Phase 3 TF-IDF | Sparse text matrices (`max_features=2500`), fit on train only |
| `data/processed/split_ids.csv` | Phase 3 | Shared train/test IDs for later tabular alignment |
| `data/processed/X_tab_*.npz` | Phase 4 tabular | Scaled numerics + one-hot categoricals, same row order as text |
| `data/processed/fusion_plan.json` | Phase 5 | Locked `hstack` merge plan and expected fused shapes |
| `data/processed/X_*_fused.npz` | Phase 6 fusion | Unified sparse matrices `[N × 2713]` (text + tabular) |
| `data/processed/model_logreg_l2.joblib` | Phase 7 model | L2 logistic regression on fused features |
