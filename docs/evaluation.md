# Evaluation notes

Companion to `notebooks/08_evaluation.ipynb`.

## Goal

Validate the fused L2 model the way the handbook asks:

1. Metrics across regularization strengths
2. Precision–Recall curves
3. Top text-token and tabular coefficient plots

## Sweep setup

| Setting | Value |
|---------|--------|
| Penalty | L2 |
| `C` grid | 0.01, 0.1, 1.0, 10.0 |
| Selection rule (saved best model) | highest **test F1** |
| Threshold for class metrics | 0.5 |

Smaller `C` means stronger regularization.

## Metrics tracked

Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision (AP)

## Outputs (`data/processed/`)

| File | Description |
|------|-------------|
| `eval_regularization_metrics.csv` | Train + test metrics for every `C` |
| `eval_test_metrics_by_C.csv` | Test-only comparison table |
| `eval_best_coefficients.csv` | Full coefficient table for best-F1 model |
| `eval_top_positive_text.csv` / `eval_top_negative_text.csv` | Top 15 text tokens |
| `eval_top_tabular.csv` | Top 15 tabular features by `|coef|` |
| `model_logreg_l2_best.joblib` | Best-F1 L2 model from the sweep |
| `eval_meta.json` | Best `C` + summary |

## Figures

| File | Reading |
|------|---------|
| `08_regularization_sweep_metrics.png` | Test Precision / Recall / F1 vs `C` |
| `08_precision_recall_curves.png` | PR curves for each `C` |
| `08_top_text_coefficients.png` | Top + / − title tokens |
| `08_top_tabular_coefficients.png` | Strongest tabular weights |

## How to read business tradeoffs

- **Higher precision:** fewer false “will succeed” calls  
- **Higher recall:** fewer missed true successes  
- Use the PR curve when the 0.5 threshold is not your operating point
