# Sparse vs dense layouts & fusion plan

Companion to `notebooks/05_sparse_dense_fusion_plan.ipynb`.

## Why this phase exists

Before merging modalities, we need a clear answer to:

1. How different are the text and tabular matrices?
2. What breaks if we concatenate the wrong way?
3. What exact merge will Phase 6 run?

## Observed contrast (train)

| Matrix | Shape | Density | Sparse CSR | Dense float32 (est.) |
|--------|-------|---------|------------|----------------------|
| TF-IDF text | `[N × 2500]` | ~0.10% | ~6–7 MB | ~2.6 GB |
| Tabular | `[N × 213]` | ~2.8% | ~20 MB | ~225 MB |

Text sparsity is the dominant memory constraint.

## What fails

| Approach | Problem |
|----------|---------|
| `np.hstack(sparse, sparse)` | Wrong API / type mismatch for SciPy sparse objects |
| `X_text.toarray()` then concatenate | Works mathematically, wastes gigabytes of RAM |
| Merging without shared row order | Silent sample misalignment |

## Locked Phase-6 plan

```text
X_train_fused = scipy.sparse.hstack([X_text_train, X_tab_train], format="csr")
X_test_fused  = scipy.sparse.hstack([X_text_test,  X_tab_test],  format="csr")
```

Expected shapes:

- train: `[264728 × 2713]`
- test: `[66183 × 2713]`

Feature-name order: TF-IDF tokens first, then tabular feature names.

Machine-readable copy: `data/processed/fusion_plan.json`

## Figure

`reports/figures/05_sparse_vs_dense_memory.png` — density and log-scale memory for train text vs tabular.
