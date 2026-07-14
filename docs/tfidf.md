# TF-IDF notes

Companion to `notebooks/03_tfidf.ipynb`.

## Goal

Turn `name_clean` into a fixed-width sparse text matrix for later fusion with tabular features.

## Setup

| Setting | Value |
|---------|--------|
| Input | `data/interim/ks_text_cleaned.csv` |
| Vectorizer | `sklearn.TfidfVectorizer` |
| `max_features` | 2,500 |
| `ngram_range` | `(1, 1)` |
| `min_df` | 2 |
| Split | 80/20 stratified by `target` |
| Random state | 42 |

## Leakage control

1. Split train/test first
2. `fit_transform` on train titles only
3. `transform` on test titles only

The same `split_ids.csv` is reused in later phases so tabular rows stay aligned with text rows.

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `X_text_train.npz` | Sparse TF-IDF train matrix |
| `X_text_test.npz` | Sparse TF-IDF test matrix |
| `y_train.npy` / `y_test.npy` | Binary targets |
| `tfidf_feature_names.csv` | Token vocabulary (2,500) |
| `tfidf_vectorizer.joblib` | Fitted vectorizer |
| `split_ids.csv` | `ID` + `train`/`test` label |
| `tfidf_meta.json` | Shapes, density, split settings |

## Figures

| File | Reading |
|------|---------|
| `reports/figures/03_top_idf_tokens.png` | Rare / campaign-specific tokens (high IDF) |
| `reports/figures/03_top_mean_tfidf_tokens.png` | Tokens with the strongest average presence on train |

## Next phase

Preprocess tabular columns on the **same train/test IDs**, then compare dense tabular memory with this sparse text matrix.
