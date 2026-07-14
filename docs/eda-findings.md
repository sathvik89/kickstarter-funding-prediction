# EDA findings — what the figures say

Companion notes for `notebooks/01_eda.ipynb` and the plots in `reports/figures/`.

## Dataset cut

| Item | Value |
|------|--------|
| Raw rows | ~378,661 |
| After binary filter + light quality filters | ~331,672 |
| Target mix | ~60% failed (`0`) / ~40% successful (`1`) |

We keep only `successful` and `failed`. Other states are dropped so the label matches a clear funding outcome.

## Figure 1 — State distribution (raw)

**File:** `reports/figures/01_state_distribution_raw.png`

Most campaigns finish as failed (~52%) or successful (~35%). Smaller slices are canceled, undefined, live, or suspended.

**Takeaway:** those smaller states are noisy for this project’s binary goal, so they leave the modeling set.

## Figure 2 — Numeric feature shapes

**File:** `reports/figures/01_numeric_distributions.png`

| Panel | What you see | Why it matters |
|-------|----------------|----------------|
| `log1p(usd_goal_real)` | Strong right skew; most goals are modest, some are extreme | Scaling (and optionally log transforms later) will matter |
| `duration_days` | Mass near ~30 days | Duration is a real pre-launch feature, usually in a narrow band |
| `name` length | Short titles (roughly 20–50 characters common) | Text model will use sparse title tokens, not long descriptions |

## Figure 3 — Success rates by category and country

**File:** `reports/figures/01_categorical_success_rates.png`

**Main categories (volume-aware view):**

- Higher success examples: Dance, Theater, Comics, Music
- Lower success examples: Technology, Journalism, Food, Crafts
- Large volume categories (Film & Video, Publishing, Games) sit nearer the middle

**Countries (top 10 by volume):**

- US dominates row count
- Success rate still varies across countries (for example, higher in US/GB than in lower-volume markets like IT in this slice)

**Takeaway:** categorical metadata carries useful prior signal and belongs in the tabular branch before fusion.

## Leakage (not plotted as a main figure, but critical)

Columns like `backers`, `pledged`, `usd pledged`, and `usd_pledged_real` correlate with success because they are campaign outcomes. They are excluded so the model answers: *given launch-time information, will this campaign fund?*

## Modeling implications

1. Evaluate with **Precision / Recall / F1** (and PR curves), not accuracy alone.
2. Use **`name`** as the text modality.
3. Use pre-launch tabular fields: goal, duration, category fields, currency, country.
4. Next step: clean titles (Phase 2), then TF-IDF + tabular preprocessing + sparse fusion.
