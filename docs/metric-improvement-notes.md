# Metric improvement thinking (roughbook)

> **Status: superseded by Phase 9.** Everything below was written before the
> experiments were promoted. The backlog items marked "medium" were measured and
> either shipped or rejected on evidence — see
> [`phase9-improvements.md`](phase9-improvements.md) for the outcome:
>
> - `log1p` on goal → **shipped**, largest single win (+0.009 F1)
> - Richer tabular features (launch date, title length) → **shipped** (+0.011 F1 combined)
> - Tabular boosting as a ceiling check → **shipped**, best model at 0.707 accuracy
> - Threshold tuning → **shipped**, now selected on validation rather than test
> - ElasticNet / L1 → **rejected**, a wash
> - TF-IDF bigrams / larger vocab → **rejected**, the dataset has no text to spare
> - "Richer tabular interactions" → **rejected on measurement**; six hand-crafted
>   variants recover ~15% of the linear/tree gap, so this line of work is closed
>
> This file is kept as the record of the reasoning that pointed at the right places.

This is analysis only. Main Phase 1–8 code stays unchanged.

Experiment notebook: `notebooks/roughbook_metric_improvements.ipynb`  
Outputs: `data/processed/roughbook/`, `reports/figures/roughbook/`

## Are these metrics the most important?

For this project, **yes — with the right priority**:

| Priority | Metric | Why |
|----------|--------|-----|
| 1 | F1 | Balances Precision and Recall; handbook cares about both |
| 1 | Average Precision / PR curve | Ranking quality; threshold-independent view |
| 2 | Precision & Recall (separately) | Map to business cost of false alarms vs misses |
| 3 | ROC-AUC | Solid secondary ranking metric |
| 4 | Accuracy | Sanity check only |

### Accuracy reality check

- Predict all `failed` → Accuracy ≈ **0.596**
- Our fused model → Accuracy ≈ **0.690**
- That is a real gain (~+9 pts), but Accuracy alone hides that Recall sits near ~0.52 at threshold 0.5

So: improve Accuracy when possible, but **do not optimize Accuracy alone**.

## What the roughbook already shows

1. **Fusion helps:** fused > tabular-only > text-only  
2. **Text is the weak modality** (short `name` only)  
3. **`class_weight='balanced'`** ↑ Recall/F1, usually ↓ Accuracy/Precision  
4. **Threshold tuning** can raise F1 a lot without retraining (trade Accuracy)

## How to improve (AIML order)

### Quick wins (no main-pipeline rewrite)
- Tune decision threshold on validation for F1 or a custom cost
- Optionally enable class weights if Recall matters more
- Always report Accuracy next to F1/AP so gains are honest

### Medium experiments (promote later if they win in roughbook)
- `log1p` on goal before scaling
- TF-IDF bigrams / slightly larger vocab
- ElasticNet / L1 to sparsify noisy tokens
- Richer tabular interactions
- Tabular boosting model as a ceiling check

### High-impact data lifts
- Longer campaign text (if obtainable cleanly)
- Creator history / seasonality (leakage-safe only)
- Probability calibration for better thresholds

### Hard constraints
- No leakage columns (`pledged`, `backers`, …)
- Keep sparse fusion design
- Don’t expect 90%+ Accuracy on this feature set — the signal ceiling is real

## Recommended project stance

Ship the handbook pipeline as-is for deliverables, and treat this roughbook as the place to chase metric gains. Promote only experiments that clearly improve **test F1/AP** without leakage.
