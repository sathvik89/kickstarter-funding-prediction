# Feature fusion notes

Companion to `notebooks/06_feature_fusion.ipynb`.

## Goal

Merge text and tabular modalities into one model-ready matrix:

```text
X_fused = [ X_tfidf | X_tabular ]
```

## Implementation

```python
from scipy.sparse import hstack

X_train_fused = hstack([X_text_train, X_tab_train], format="csr")
X_test_fused  = hstack([X_text_test,  X_tab_test],  format="csr")
```

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `X_train_fused.npz` | Fused train matrix `[264728 × 2713]` |
| `X_test_fused.npz` | Fused test matrix `[66183 × 2713]` |
| `fused_feature_names.csv` | `text__*` then `tab__*` names |
| `fusion_meta.json` | Shapes, density, memory |

## Feature order

1. TF-IDF tokens (`text__...`) — 2,500 columns  
2. Tabular scaled + one-hot (`tab__...`) — 213 columns  

This order matters later when slicing coefficients by modality.

## Figure

`reports/figures/06_fused_feature_width.png` — text dominates width; regularization will balance influence.

## Next phase

Train L2 logistic regression on the fused matrices (`penalty="l2"`).
