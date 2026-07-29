# The Kickstarter Funding Predictor — complete project reference

**Live site:** https://sathvik89.github.io/kickstarter-funding-prediction/
**Repo:** https://github.com/sathvik89/kickstarter-funding-prediction

This is the document to read when you come back to this project in six months and
need to remember not just *what* it does but *why* every decision was made. It is
written to be read start to finish, but the table of contents below is there for
when you only need one answer.

Every number in this file comes from a committed artifact. Nothing is from memory,
and nothing is rounded in a flattering direction.

---

## Contents

1. [What this project actually is](#1-what-this-project-actually-is)
2. [Results, stated honestly](#2-results-stated-honestly)
3. [Quickstart — run it yourself](#3-quickstart--run-it-yourself)
4. [The problem in detail](#4-the-problem-in-detail)
5. [The data](#5-the-data)
6. [How the pipeline works, stage by stage](#6-how-the-pipeline-works-stage-by-stage)
7. [The two models, and why both exist](#7-the-two-models-and-why-both-exist)
8. [What made the numbers move](#8-what-made-the-numbers-move)
9. [What failed — the most valuable section](#9-what-failed--the-most-valuable-section)
10. [How this compares to published work](#10-how-this-compares-to-published-work)
11. [The live site and the in-browser model](#11-the-live-site-and-the-in-browser-model)
12. [The test suite](#12-the-test-suite)
13. [Repository map](#13-repository-map)
14. [Reproducing everything from scratch](#14-reproducing-everything-from-scratch)
15. [Questions you might get asked](#15-questions-you-might-get-asked)
16. [Known limitations](#16-known-limitations)
17. [Glossary](#17-glossary)
18. [Project history](#18-project-history)

---

## 1. What this project actually is

### The one-sentence version

It predicts whether a Kickstarter campaign will hit its funding goal, using only
information that existed **before the campaign launched**.

### The honest version

The prediction task is the vehicle. The actual subject of the project is
**multi-modal feature fusion** — the problem of combining inputs that have
genuinely different shapes into one matrix a single model can learn from.

A campaign gives you two kinds of information:

- **Text** — the campaign title. Turned into TF-IDF features, this becomes a
  matrix that is *very wide* (2,500 columns) and *almost entirely zeros* (a
  four-word title touches maybe four of those columns).
- **Tabular** — funding goal, category, country, currency, duration, launch date.
  Narrow, dense, and mixed: some numeric, some categorical.

Those two things cannot simply be pasted together without thought. Densify the
text block and you blow up memory. Get the row order wrong between the two blocks
and you silently train on mismatched labels — a bug that produces a *plausible*
model, which is the worst kind. Fit your vectoriser before splitting and you leak
test information into training.

This project handles all three properly, and the discipline is the point.

### Why the accuracy number is lower than you might expect

**70.7%.** That is deliberate and it is the honest ceiling for this problem.

You can find dozens of Kickstarter models online reporting 86–94%. Nearly all of
them use `backers` or `pledged` as features. Those are the *outcome*. A campaign
with 400 backers succeeded — you have not predicted anything, you have read the
answer. See [section 10](#10-how-this-compares-to-published-work).

This project bans those columns. The best published result that plays by the same
rules is **69.8%**. We are above it.

---

## 2. Results, stated honestly

All figures are on a held-out test set of **66,183 campaigns**, never used for any
model or hyperparameter choice.

### Headline table

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg precision |
|---|---|---|---|---|---|---|---|
| Majority baseline (always "failed") | — | 0.5961 | — | — | — | — | — |
| Phase 8 baseline, L2 logistic regression | 0.50 | 0.6898 | 0.6440 | 0.5189 | 0.5747 | 0.7404 | 0.6473 |
| Phase 9 L2 logistic regression | 0.50 | 0.6963 | 0.6483 | 0.5421 | 0.5905 | 0.7543 | 0.6605 |
| Phase 9 L2 logistic regression | 0.31 | 0.6490 | 0.5436 | 0.8169 | 0.6528 | 0.7543 | 0.6605 |
| **Phase 9 gradient boosting** | **0.50** | **0.7069** | **0.6571** | **0.5736** | **0.6126** | **0.7718** | **0.6800** |
| Phase 9 gradient boosting | 0.35 | 0.6788 | 0.5740 | 0.7937 | 0.6662 | 0.7718 | 0.6800 |

Source: `data/processed/v2/v2_meta.json`.

### The one thing to be careful about

Notice that the same model appears twice with different thresholds, and that the
low-threshold rows have **better F1 but worse accuracy**.

That is not a contradiction. Lowering the decision threshold makes the model say
"funded" more often: it catches more of the real successes (recall 0.54 → 0.82)
at the cost of more false alarms (precision 0.65 → 0.54). F1 improves because it
balances the two; accuracy drops because accuracy rewards the majority class and
"failed" is the majority here.

**So: never quote the best accuracy and the best F1 from two different rows.**
If someone asks for accuracy, the answer is 0.707 at threshold 0.50. If they ask
for F1, it is 0.666 at threshold 0.35. Say the threshold out loud both times.

Which threshold is *correct* depends on cost, not statistics. If missing a
fundable campaign is expensive and a false alarm is cheap, use the low threshold.
That is a business decision the model cannot make for you.

### Is 0.707 actually good?

Three reference points:

- **0.5961** — always guessing "failed". We beat this by 11 points.
- **0.6898** — the project's own earlier baseline. We beat this by 1.7 points.
- **0.698** — the best published leakage-free result. We beat this by 0.9 points.

And it is wrong about three times in ten. That is a real limit, not a bug — see
[section 16](#16-known-limitations).

---

## 3. Quickstart — run it yourself

```bash
# 1 · environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt      # pytest, for the test suite

# 2 · check everything is sound (~6 seconds, 66 tests)
pytest

# 3 · train, evaluate, write artifacts and figures (~2-3 minutes)
python -m src.train

# 4 · the ablation tables — what helped, and what didn't
python -m src.ablations

# 5 · rebuild the browser model (verifies it matches scikit-learn)
python -m src.export_web

# 6 · redraw the figures without refitting anything (fast)
python -m src.report
```

### To preview the website locally

```bash
cd docs
python -m http.server
# then open http://localhost:8000
```

Opening `docs/index.html` directly from disk will **not** work — the browser
blocks the `model.json` fetch on `file://` URLs. The page detects this and tells
you so, rather than failing silently.

### To play with the model without any of the above

Just open the [live site](https://sathvik89.github.io/kickstarter-funding-prediction/)
and scroll to "Try the model". It runs in your browser.

---

## 4. The problem in detail

### Target definition

Binary classification:

- **Positive class (1):** `state == "successful"`
- **Negative class (0):** `state == "failed"`
- **Dropped entirely:** `canceled`, `live`, `suspended`, `undefined`

Why drop them? A `canceled` campaign may have been on track to succeed or doomed;
its label is genuinely unknown. `live` campaigns have no outcome yet. Including
either would mean training on labels that do not mean what they claim to. Roughly
48,000 rows are dropped this way.

Class balance after filtering: **40.4% successful, 59.6% failed.** Mildly
imbalanced — enough to matter for threshold choice, not enough to need
resampling.

### The leakage rule, and why it is the most important decision in the project

These columns are **permanently excluded** from every feature matrix:

| Column | Why it is banned |
|---|---|
| `pledged` | Money raised. Known only after the campaign runs. |
| `usd pledged` | Same, currency-converted. |
| `usd_pledged_real` | Same. |
| `backers` | Number of supporters. Known only after the campaign runs. |
| `state` | The label itself. |

The test that enforces this is `tests/test_leakage.py`. It does not just check a
list — it asserts that **every non-text column in the fused matrix traces back to
a declared config entry**, so a new column cannot sneak in.

If you include `backers`, accuracy jumps into the 90s and the model becomes
worthless: you cannot know the backer count of a campaign that has not launched.
The whole point is scoring a campaign *at launch time*, when the creator can still
change the title, the goal, or the timing.

**This is the thing to defend if anyone challenges the project.** A lower number
under honest rules beats a higher number under dishonest ones.

---

## 5. The data

- **Source:** Kaggle — "Kickstarter Projects" (2018 release, `ks-projects-201801.csv`)
- **Local path:** `data/raw/kickstarter_2018.csv` (58 MB, never edited)
- **Raw size:** ~379,000 rows, 15 columns
- **After filtering to successful/failed:** 331,672 rows
- **After dropping titles with no usable words:** **330,911 rows** — these are the
  modelled rows

That last drop is 761 rows whose titles consisted entirely of stopwords,
punctuation, or non-Latin characters, leaving nothing for TF-IDF to work with.

### Columns actually used

| Column | Type | Notes |
|---|---|---|
| `name` | text | The campaign title. Averages **4.21 usable words** after cleaning. |
| `usd_goal_real` | numeric | Funding goal in USD. Log-transformed — see below. |
| `duration_days` | numeric | Derived: `deadline − launched`. |
| `name_len` | numeric | Character count of the **raw** title. |
| `name_words` | numeric | Word count of the **raw** title. |
| `category` | categorical | 159 subcategories (e.g. "Product Design"). |
| `main_category` | categorical | 15 parents (e.g. "Design"). |
| `currency` | categorical | 14 levels. |
| `country` | categorical | 23 levels. |
| `launch_year` | categorical | 9 levels, 2009–2017. |
| `launch_month` | categorical | 12 levels. |
| `launch_dow` | categorical | 7 levels (day of week). |

### The two facts about this data that drove the biggest improvements

**1. The funding goal is savagely skewed.**

| Statistic | Value |
|---|---|
| Minimum | $0.00 |
| 25th percentile | $2,000 |
| Median | $5,000 |
| 75th percentile | $15,000 |
| 99th percentile | $300,000 |
| Maximum | $166,361,391 |
| **Skew (raw)** | **83.9** |
| **Skew (after `log1p`)** | **−0.14** |

A skew of 84 means that if you standardise this column and hand it to a linear
model, the coefficient describes a variable whose useful range is the first ~0.2%
of its span. Everything interesting is compressed against zero. The log transform
fixes this and was the single biggest win in the project.

*(You may see 83.1 quoted in `docs/phase9-improvements.md`. That figure is
computed before the 761 empty-title rows are dropped; 83.9 is on the final
modelling table. Both are correct for their row set.)*

**2. Success rate depends heavily on when a campaign launched.**

| 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 |
|---|---|---|---|---|---|---|---|---|
| 0.491 | 0.480 | **0.506** | 0.465 | 0.472 | 0.356 | **0.321** | 0.381 | 0.425 |

A **19-point spread**. This matches Kickstarter's documented platform-wide decline
after 2014, when campaign volume grew much faster than the backer pool. The
earlier pipeline derived `duration_days` from the launch date and threw the rest
away, which discarded all of this.

---

## 6. How the pipeline works, stage by stage

The canonical implementation is `src/`. Notebooks 01–08 are the frozen record of
how each stage was originally validated; notebook 09 documents the improvements.

### Stage 1 — Clean the title

`src/text_cleaner.py`

```python
def clean_text(text: str) -> str:
    text = str(text).lower()             # "Oak Desk LAMP" -> "oak desk lamp"
    text = TOKEN_RE.sub(" ", text)       # strip everything that is not a-z
    return " ".join(t for t in text.split()
                    if t and t not in STOPWORDS)   # drop "the", "of", "and", ...
```

`"The Songs of Adelaide & Abullah"` → `"songs adelaide abullah"`

Three notes worth remembering:

- Digits are stripped along with punctuation, so `"Project 2024"` → `"project"`.
- `clean_text(None)` returns `""`, not `"none"` — because `str(None)` is `"none"`
  and `"none"` happens to be in sklearn's stopword list. Lucky, but tested.
- The function is idempotent: cleaning already-clean text changes nothing.

### Stage 2 — Split the data three ways

`src/fusion_pipeline.py :: three_way_split`

```
330,911 rows
├── 264,728 (80%)  ─── split again ───┬── 211,782  train
│                                     └──  52,946  validation
└──  66,183 (20%)                         66,183   test
```

Stratified on the target at both steps, so all three splits carry the same 40.4%
success rate.

**The critical detail:** `test_size=0.20` and `random_state=42` are *unchanged*
from the original Phase 3 split. That means the test set is literally the same
66,183 rows the earlier baseline reported on, so every before/after comparison in
this document is apples-to-apples. Validation was carved out of the old training
block, not out of test.

### Stage 3 — TF-IDF the text

`TfidfVectorizer(max_features=2500, ngram_range=(1,1), min_df=2)`

Fitted on **train only**, then used to transform validation and test. Settings
that matter:

- `max_features=2500` — keeps the 2,500 most informative terms.
- `min_df=2` — a term must appear in at least 2 documents, which drops typos and
  one-off proper nouns.
- `norm="l2"` — each row is normalised to unit length, so a long title does not
  outweigh a short one purely by having more words.
- `smooth_idf=True`, `sublinear_tf=False` — sklearn defaults, kept.
- Tokens must be **at least 2 characters** (sklearn's default `token_pattern`),
  so single letters left behind by digit-stripping disappear.

Those five settings are pinned in `src/export_web.py` because the browser
implementation has to mirror them exactly.

### Stage 4 — Preprocess the tabular block

`ColumnTransformer` with two branches:

- `StandardScaler` on `log_goal`, `duration_days`, `name_len`, `name_words`
- `OneHotEncoder(handle_unknown="ignore")` on the seven categorical columns

`handle_unknown="ignore"` means a level never seen in training contributes zero
rather than raising — important for the live demo, where someone can type a
country the model has never seen.

Result: **243 columns** (4 scaled numeric + 239 one-hot levels).

Date parts are kept as **strings**, deliberately. As numbers, a model would infer
that "December (12) is twelve times January (1)", which is nonsense.

### Stage 5 — Fuse

```python
X = scipy.sparse.hstack([text_block, tabular_block], format="csr")
```

| | Value |
|---|---|
| Shape (train) | 211,782 × 2,743 |
| Non-zero entries | 2,873,955 |
| Density | **0.495%** |
| Size as sparse | **35.3 MB** |
| Size if densified | **4,647 MB** |

That last row is why this is the centrepiece of the project. Calling `.toarray()`
turns a 35 MB matrix into 4.6 GB — a **132× blow-up** — for a matrix that is
99.5% zeros. `scipy.sparse.hstack` keeps it sparse throughout.

### Stage 6 — Train and select

- Sweep `C ∈ {0.01, 0.1, 1.0, 10.0}` and pick the best by **validation** F1 → `C = 10.0`
- Tune the decision threshold on **validation** → `0.31`
- Evaluate on test **once**

Validation C sweep (`v2_validation_C_sweep.csv`):

| C | Val accuracy | Val F1 | Val ROC-AUC |
|---|---|---|---|
| 0.01 | 0.6835 | 0.5584 | 0.7395 |
| 0.10 | 0.6935 | 0.5807 | 0.7518 |
| 1.00 | 0.6951 | 0.5890 | 0.7524 |
| **10.00** | 0.6946 | **0.5898** | 0.7517 |

Note how flat that is above `C = 0.1`. That flatness is itself a finding — see
[section 9](#9-what-failed--the-most-valuable-section).

---

## 7. The two models, and why both exist

### L2 logistic regression — the interpretable one

`LogisticRegression(C=10.0, solver="liblinear", max_iter=2000)`

Test: accuracy 0.6963, F1 0.5905, ROC-AUC 0.7543.

It ships for three reasons:

1. **Coefficient plots are a project deliverable.** You can read straight off it
   which words and which categories push toward funding.
2. **It is the model in the live demo**, because a linear model's contribution per
   feature is exactly `coefficient × value` — so the page can show its reasoning.
3. **It does not overfit here.** Train/test F1 gap is about one point.

### Gradient boosting — the accuracy ceiling check

`HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, early_stopping=True)`
— stopped at **217 iterations**, 13,237 tree nodes.

Test: accuracy 0.7069, F1 0.6126, ROC-AUC 0.7718.

Text reaches it through `TruncatedSVD(120)` — a 120-dimensional compression of
the 2,500-column TF-IDF block. Two reasons: `HistGradientBoosting` cannot accept
sparse input at all, and a tree cannot usefully split on 2,500 mostly-zero columns.

**It is framed as a ceiling check, not a replacement.** Its job is to measure what
the linear assumption costs. The answer is uncomfortable and worth stating plainly:

> Gradient boosting on the tabular block **alone, with no TF-IDF whatsoever**,
> already beats L2 logistic regression on the full 2,743-column fused matrix.

The linear assumption was costing more than the entire text modality contributes.
That is the single most interesting modelling fact in the project.

---

## 8. What made the numbers move

Each row below refits the vectoriser and scaler on the training block alone and
reports test metrics, so the deltas are attributable rather than guessed.
Source: `data/processed/v2/v2_feature_ablation.csv`.

| Feature set | Columns | Accuracy | F1 | ΔF1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (raw goal, duration) | 2,713 | 0.6877 | 0.5726 | — | 0.7384 |
| `log1p(goal)` replaces raw goal | 2,713 | 0.6925 | 0.5819 | **+0.0093** | 0.7449 |
| + title length | 2,715 | 0.6903 | 0.5782 | +0.0056 | 0.7433 |
| + launch year / month / day-of-week | 2,741 | 0.6898 | 0.5779 | +0.0053 | 0.7428 |
| **All engineered (shipped)** | **2,743** | **0.6963** | **0.5905** | **+0.0179** | **0.7543** |
| All engineered, **no text block** | 243 | 0.6796 | 0.5598 | −0.0128 | 0.7327 |

### Reading this table

- **`log1p(goal)` is the biggest single win**, and it cost nothing — same number of
  columns, one function call. This is the lesson to carry forward: check your
  distributions before you reach for a fancier model.
- **The groups stack.** Individually they are worth +0.009, +0.006, +0.005; together
  they are worth +0.018, which is slightly more than the sum. They are capturing
  partly independent signal.
- **The text block earns its place.** The last row removes TF-IDF entirely and F1
  drops about 3 points. Four words of title carry real information — just not a
  lot.
- **Two of the three were free.** `name_len` and `name_words` had been computed in
  Phase 1 and were sitting unused in `ks_binary_base.csv`; the launch date parts
  came from one `to_datetime` call on a column already in the file.

---

## 9. What failed — the most valuable section

A project that only reports what worked is a sales pitch. These are the things
that did not, and each one saves future-you time.

### The interaction hypothesis — wrong, and measured to prove it

**The reasoning:** boosting beats logistic regression by ~2 F1 points. The obvious
explanation is a missing *interaction*. A $10,000 goal is routine for a Music
project and brutal for a Technology one. A linear model computes
`w_goal + w_technology` — it adds the two effects and cannot express "this goal is
ambitious *for this category*". A tree can, by splitting on category then on goal.

**So we built it.** Six ways, on the same matrix with the same `C`
(`data/processed/v2/v2_interaction_ablation.csv`):

| Variant | Extra columns | Test F1 | ΔF1 |
|---|---|---|---|
| No hand-crafted interactions | 0 | 0.5905 | — |
| `log_goal × main_category` (15 levels) | 15 | 0.5910 | +0.0005 |
| `log_goal × category` (159 levels) | 159 | 0.5912 | +0.0007 |
| `log_goal` binned, 10 quantiles | 10 | 0.5917 | +0.0012 |
| `log_goal` binned, 50 quantiles | 43 | 0.5927 | +0.0022 |
| Full `(goal bin × main_category)` grid | 150 | 0.5939 | +0.0034 |
| **Gradient boosting** | — | **0.6126** | **+0.0221** |

**The verdict:** even the full cell grid — the most expressive hand-crafted form
short of an actual tree — recovers about **15%** of the gap.

So the tree's advantage is **not one nameable missing term**. It is diffuse
non-linearity spread thinly across many features: exactly the thing hand
engineering cannot reach and a tree finds for free.

**Two consequences:**

1. The interaction columns are **not** in the shipped model
   (`USE_INTERACTIONS = False` in `src/config.py`). The code and the ablation stay
   so the decision is reproducible rather than asserted.
2. Inventing more interaction features on this dataset is closed as a line of work.
   If you want non-linearity here, use a tree.

There is even a test (`test_hand_crafted_interactions_did_not_earn_their_place`)
that fails if this ever stops being true, so the documentation cannot quietly rot.

### Regularization was never the lever

The earlier C sweep was flat above `C = 0.1` with a train/test F1 gap of about one
point. That is the signature of **underfitting**, not overfitting. An underfitting
model does not improve by being penalised harder. Time spent tuning `C` here was
time spent on the wrong axis.

### More TF-IDF tuning will not help

Bigrams, a larger vocabulary, sublinear TF — all marginal. The reason is the data,
not the code: this dataset has **no campaign description**, only a title averaging
4.21 usable words. Every study that beats us on text uses full project
descriptions plus psycholinguistic features. You cannot tune your way out of a
missing column.

### Chasing 90% accuracy

Reachable only by including `backers` or `pledged`. That is not a better model,
it is a different and much easier problem — one with no practical use, since a
campaign that has not launched has no backers.

---

## 10. How this compares to published work

Comparing accuracy on this task is mostly an exercise in auditing what went into
someone else's feature matrix.

| Source | Reported | Fair comparison? |
|---|---|---|
| [Springer 2019, *Kickstarter at Launch Time*](https://link.springer.com/chapter/10.1007/978-3-030-29516-5_39) (XGBoost) | 69.8% | ✅ **Yes** — launch-time features only. The real benchmark. |
| This project, earlier baseline | 69.0% | ✅ Yes |
| **This project, gradient boosting** | **70.7%** | ✅ Yes |
| [Same-dataset LightGBM](https://github.com/srishtis/Kaggle-Kickstarter-Project-Status-Prediction) | 70.3% | ❌ No — top features are `avg_pledge_per_backer` and a category/year success rate built from pledged amounts. |
| Assorted blogs and repos | 86–94% | ❌ No — use `backers`. Feature-importance charts in these often rank "number of backers" first, which gives the game away. |
| [UTwente 2024 thesis](https://essay.utwente.nl/fileshare/file/101151/41TScIT_submission_175%20(1).pdf) (Random Forest) | 88.3% | ❌ No — that is its in-sample score. Its own text quotes 77.01% cross-validated (and swaps the two labels), and captions a figure "In-sample ROC Curve". Also on a rebalanced 50/50 dataset, so its lift over baseline is not comparable to ours over 59.6%. |
| [Stanford CS229, *Plead or Pitch?*](https://cs229.stanford.edu/proj2015/239_report.pdf) | F1 0.79 | ❌ No — uses full descriptions + risks sections + LIWC psycholinguistic features. Honest work, different inputs. |

**The takeaway to remember:** we clear the only leakage-free launch-time benchmark
found. The 86–94% figures are not a target and should not be treated as one.

Worth noting what the Stanford paper *does* transfer: it also found L2 logistic
regression with balanced class weights competitive with an RBF SVM, and that the
SVM overfitted while the logistic regression did not. Same conclusion we reached
independently.

---

## 11. The live site and the in-browser model

**https://sathvik89.github.io/kickstarter-funding-prediction/**

GitHub Pages serves from the `docs/` directory. No build step, no CI, no server.

### How a real model runs with no backend

A logistic regression is arithmetic, not a runtime:

```
p = σ(b + Σ wᵢxᵢ)
```

The model *is* its coefficients. So `python -m src.export_web` writes
`docs/model.json` (~82 KB raw, ~24 KB gzipped) containing:

- `vocab` — every token mapped to `[idf, coefficient]`
- `numeric` — each column's scaler mean, scale, and coefficient
- `categorical` — a coefficient per level, per column
- `stopwords` — sklearn's list, since the browser cannot import sklearn
- `intercept`, both thresholds, and the metrics shown on the page
- `category_tree` and `default_category` — so the subcategory dropdown is scoped
  to its parent and opens on a sensible default

About 60 lines of JavaScript then reproduce `predict_proba`. Free hosting, no cold
starts, nothing to keep awake, and no data leaves the visitor's browser.

### Why the demo runs logistic regression, not the better model

Three reasons, in order of importance:

1. **It can explain itself.** Each token's contribution to the log-odds is exactly
   `coefficient × tfidf`, so the page can rank what drove a prediction. A tree
   cannot do this.
2. **Size.** The booster would need 217 trees (~580 KB) *plus* the SVD matrix
   (120 × 2,500 floats ≈ 1.2 MB) — roughly 20× the payload.
3. Categorical splits in `HistGradientBoosting` use bitsets, which is considerably
   more fiddly to reimplement correctly in JavaScript.

The 0.707 boosting figure is reported on the page and labelled as the best model;
the demo is labelled as the logistic regression. **Both numbers are stated so
nobody is misled about which one they are playing with.**

### The three-layer parity chain

A live demo that quietly disagrees with the model is the worst possible bug:
every number in the repo would still be correct while the website misled people.
So agreement is enforced three times over:

1. **`export_web.verify_parity()`** — re-scores real test rows using *only* the
   JSON and diffs against sklearn's `predict_proba`. Currently **5.9 × 10⁻⁷**,
   which is just the 6-decimal rounding on the exported coefficients. It raises
   rather than warns.
2. **`tests/test_js_parity.py` + `tests/js_parity.mjs`** — extracts the actual
   `<script>` block from `docs/index.html`, runs it in Node against Python, and
   requires exact agreement across nine edge cases. It tests the *shipped* code,
   not a copy, because a copy would drift and still pass.
3. **`test_shipped_json_is_current`** — fails if `docs/model.json` is stale
   relative to the code.

### The four rules the JavaScript must mirror

Break any of these and the demo silently diverges:

1. Clean the title identically — lowercase, letters only, drop sklearn stopwords
2. Keep tokens of length ≥ 2 (sklearn's default `token_pattern`)
3. Compute `tf × idf`, then **L2-normalise the row**
4. Unknown categorical levels contribute **0** (`handle_unknown="ignore"`)

### One gap parity tests structurally cannot catch

`name_len` and `name_words` are derived in JavaScript from the title box. If the
*training* definition changed, both sides would agree with each other and both
would be wrong — no parity test could detect it.

So `test_title_length_columns_measure_the_raw_title` pins the definition:
character count and whitespace word count of the **raw** title, not the cleaned
one. This is the subtlest correctness issue in the project.

---

## 12. The test suite

```bash
pytest      # 66 tests, ~6 seconds
```

Fast on purpose. A suite that takes ten minutes does not get run.

| File | Tests | What it guards |
|---|---|---|
| `test_leakage.py` | 8 | No outcome column reaches the model; splits are disjoint and complete; transformers fitted on train only |
| `test_features.py` | 15 | Text cleaning, `log1p`, date parsing, title-length semantics, interaction block correctness |
| `test_evaluate.py` | 10 | Metric helpers, threshold tuning finds the true optimum, majority baseline |
| `test_web_export.py` | 11 | Payload completeness, pinned TF-IDF settings, parity with sklearn, unknown-level handling, empty-title guard |
| `test_model_quality.py` | 11 | Floors under every published number, including the negative result |
| `test_js_parity.py` | 10 | The page's real JavaScript agrees with Python |

### The two tests that matter most

**Leakage.** `test_every_tabular_feature_traces_to_a_configured_column` asserts
every non-text column traces back to a config entry.

Worth knowing: the first version of this test was a substring scan over all
feature names, and it **failed** — because a campaign titled something containing
"United States" legitimately produces a `text__state` TF-IDF feature, which a
substring scan flags as the banned `state` column. That was a false positive in
the test, not leakage in the model. The fix was to make the check narrower and
stronger: text features are validated separately, against the fitted vocabulary.

**Fit-on-train-only.** Checked two ways:

- `test_vectorizer_vocabulary_comes_from_train_only` — every vocabulary token must
  actually appear in the train block
- `test_transformers_never_refit_on_val_or_test` — the scaler's `n_samples_seen_`
  must equal the train row count

Either would catch vectorising before splitting, which is *the* classic
multi-modal leakage mistake.

### One test exists purely as a regression guard

`test_survives_more_than_127_levels`. During development, category codes came back
as **int8**, so `code * n_bins` silently overflowed to negative array indices past
level 127 and crashed with `ValueError: negative axis 1 index` on the 159-level
`category` column. The test builds an interaction block on that column and asserts
the result stays non-negative.

### Tests as documentation

`test_model_quality.py` puts floors under the published numbers, so a change that
degrades the model fails the build instead of making the README wrong. That
includes the negative result: if hand-crafted interactions ever start helping more
than documented, a test fails and tells you to update the write-up.

---

## 13. Repository map

```
├── PROJECT.md                  ← this document
├── README.md                   quick orientation + results
├── pytest.ini                  test configuration
├── requirements.txt            runtime dependencies
├── requirements-dev.txt        pytest, for the suite
│
├── data/
│   ├── raw/                    the original CSV — never edited
│   ├── interim/                cleaned intermediate tables
│   └── processed/
│       ├── (phase 3-8 artifacts)
│       └── v2/                 Phase 9 artifacts: models, metrics, ablations
│
├── notebooks/
│   ├── 01_eda.ipynb            target lock, leakage audit, distributions
│   ├── 02_text_cleaning.ipynb  title cleaning
│   ├── 03_tfidf.ipynb          split + TF-IDF
│   ├── 04_tabular_preprocess.ipynb   scaling + one-hot
│   ├── 05_sparse_dense_fusion_plan.ipynb   memory analysis
│   ├── 06_feature_fusion.ipynb sparse hstack
│   ├── 07_train_l2_logreg.ipynb baseline model
│   ├── 08_evaluation.ipynb     sweeps, PR curves, coefficients
│   ├── 09_improved_pipeline.ipynb   Phase 9: features, validation, ceiling check
│   └── roughbook_metric_improvements.ipynb   exploratory experiments
│
├── src/                        ← the canonical implementation
│   ├── config.py               all paths, constants, feature groups
│   ├── text_cleaner.py         clean_text()
│   ├── features.py             log_goal, date parts, interaction blocks
│   ├── fusion_pipeline.py      three-way split, fit-on-train, sparse fusion
│   ├── evaluate.py             metrics, threshold tuning, majority baseline
│   ├── train.py                the entrypoint: fit, select, evaluate, save
│   ├── ablations.py            feature + interaction ablation tables
│   ├── report.py               the seven Phase 9 figures
│   └── export_web.py           browser model export + parity verification
│
├── tests/                      66 tests
│   ├── conftest.py             session fixtures
│   ├── test_leakage.py         the credibility tests
│   ├── test_features.py        feature engineering correctness
│   ├── test_evaluate.py        metric helpers
│   ├── test_web_export.py      payload + sklearn parity
│   ├── test_js_parity.py       page JavaScript vs Python
│   ├── test_model_quality.py   floors under published numbers
│   └── js_parity.mjs           Node harness for the page script
│
├── docs/                       ← GitHub Pages serves from here
│   ├── index.html              the live site + in-browser predictor
│   ├── model.json              the exported model (generated)
│   ├── leadership-report.html  the original leadership report
│   ├── learning-guide.html     concept explainer
│   ├── figures/                figures the site displays
│   ├── PROJECT (see root)      
│   ├── phase9-improvements.md  Phase 9 technical write-up
│   ├── overview.md · phases.md · data.md · eda-findings.md
│   ├── text-cleaning.md · tfidf.md · tabular.md
│   ├── fusion-plan.md · fusion.md · modeling.md · evaluation.md
│   └── metric-improvement-notes.md   the roughbook reasoning (superseded)
│
└── reports/figures/            all generated figures
```

### Key artifacts in `data/processed/v2/`

| File | Contents |
|---|---|
| `v2_meta.json` | Every headline number, split sizes, selection policy |
| `v2_model_comparison.csv` | The five model/threshold rows |
| `v2_feature_ablation.csv` | What each feature group was worth |
| `v2_interaction_ablation.csv` | The six failed interaction variants |
| `v2_validation_C_sweep.csv` | C selection, on validation |
| `v2_coefficients.csv` | Every coefficient, tagged by block |
| `v2_probabilities.npz` | Saved predictions, so figures redraw without refitting |
| `v2_logreg_l2.joblib` / `v2_hgb.joblib` | The fitted models |

---

## 14. Reproducing everything from scratch

```bash
# from a clean checkout
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# notebooks 01-08 regenerate the original artifacts (run in order)
jupyter lab

# or go straight to the current pipeline:
python -m src.train        # ~2-3 min: artifacts under data/processed/v2/ + figures
python -m src.ablations    # ~3-4 min: the two ablation tables
python -m src.export_web   # instant: docs/model.json, verifies parity
python -m src.report       # instant: redraws all seven figures from artifacts
pytest                     # ~6s: 66 tests
```

Everything is seeded with `random_state=42`. Re-running produces identical
numbers.

### If a number in this document disagrees with a fresh run

Trust the fresh run and update the document. Most likely causes: a library version
bump changed a default, or a feature group in `src/config.py` was edited. Start by
checking `git log -p src/config.py`.

---

## 15. Questions you might get asked

**"Only 70%? That seems low."**
It is high for this problem under honest rules. The best published launch-time
result is 69.8%. Everything above ~85% you will find online uses `backers` or
`pledged`, which are outcomes — those models cannot score a campaign that has not
launched yet. Majority-class guessing gets 59.6%, so there are 11 real points of
signal here.

**"Why not just use the boosting model everywhere?"**
It is the reported best model. It is not in the browser demo because it cannot
explain individual predictions and would need ~20× the payload. Both numbers are
on the site.

**"How do you know there is no leakage?"**
Three ways: the excluded columns are named in config, `tests/test_leakage.py`
asserts every non-text feature traces to a declared column, and the vectoriser
and scaler are verified to have been fitted on the train block only.

**"How did you pick the threshold 0.31?"**
On a validation split carved out of training data — never on test. The figure
`09_threshold_selection.png` shows the validation and test F1 curves are nearly
coincident and that the validation-chosen threshold lands on the test optimum,
which is the evidence it generalised rather than got lucky.

**"Why is accuracy lower at the tuned threshold?"**
Because the threshold optimises F1, not accuracy. It trades precision for recall
(0.54 → 0.82). Accuracy falls because accuracy rewards predicting the majority
class. Which is right depends on whether a missed opportunity costs more than a
false alarm.

**"What did you learn that surprised you?"**
That the intuitive explanation for the linear/tree gap was wrong. I assumed it was
a missing goal × category interaction, built it six different ways including a
full 150-column cell grid, and recovered only 15% of the gap. The advantage is
diffuse non-linearity, not one nameable term — and that closed off a whole
direction of work.

**"Why keep the logistic regression at all?"**
Interpretability is a deliverable, not a nicety. It tells you which words and
categories move the outcome, it powers the live demo's explanations, and it does
not overfit. Boosting measures the ceiling; logistic regression explains the
floor.

**"What is the single most important line in the codebase?"**
`sparse.hstack([text_block, tabular_block], format="csr")`. Everything upstream
exists to make that call correct and everything downstream depends on it. Getting
it wrong — densifying, or misaligning rows — produces a model that looks fine and
is not.

---

## 16. Known limitations

**Be upfront about these.** They are the honest boundaries of the work.

1. **The text signal is capped by the dataset.** A ~4-word title is all there is.
   The 2018 Kaggle CSV has no `blurb` and no description. This single fact
   explains most of the gap to papers reporting F1 0.79.
2. **The data ends in 2017.** The model describes 2009–2017 Kickstarter. Platform
   dynamics have changed; treat predictions as historical description, not advice.
3. **Launch year is a feature, which limits forward use.** A 2026 campaign gets
   an unseen level contributing zero. Fine for the study, awkward for deployment —
   a production version would need a recency-aware or trend-based encoding.
4. **It is wrong ~3 times in 10.** At threshold 0.50, precision is 0.657 and
   recall 0.574. It is a decision *aid*, not an oracle.
5. **The demo runs the second-best model.** Documented on the page, but true.
6. **Country and currency are nearly collinear.** Their coefficients split the
   effect between them (in the demo you will often see `currency: USD +0.609`
   sitting next to `country: US −0.520`). Individually they are hard to interpret;
   together they are fine. Worth knowing before over-reading a single coefficient.
7. **No probability calibration.** Predicted probabilities rank well but are not
   guaranteed to mean "62% of campaigns like this succeed". Calibration would be
   the right next step if the numbers needed to be read literally.

### What would actually move the needle

In descending order of expected value:

1. **Richer campaign text** — descriptions or blurbs, from a source that has them.
   This is the big one.
2. **Creator history** — first-time vs repeat creator, previous success rate.
   Leakage-safe if taken strictly from before the launch date.
3. **Probability calibration** — so thresholds and probabilities mean something.
4. **Reward-tier structure** — count and pricing of pledge levels.

Not worth revisiting: TF-IDF tuning, hand-crafted interactions, regularization
sweeps. All measured, all closed.

---

## 17. Glossary

**TF-IDF** — Term Frequency × Inverse Document Frequency. Scores a word high if it
appears often in *this* title but rarely across all titles, so common words are
downweighted without a manual stoplist.

**Sparse matrix** — a matrix stored as only its non-zero entries. Ours is 99.5%
zeros; storing it densely would cost 4.6 GB instead of 35 MB.

**`hstack`** — horizontal stack. Glues matrices side by side, so row *i* of the
result is row *i* of the text block followed by row *i* of the tabular block.
Row alignment is everything.

**L2 regularization** — a penalty on the sum of squared coefficients, which
discourages any single feature from dominating. `C` is its *inverse* strength:
large `C` means weak penalty.

**One-hot encoding** — turning a categorical column into one binary column per
level, so the model does not assume the levels are ordered.

**StandardScaler** — rescales a numeric column to mean 0, standard deviation 1.

**Precision** — of the campaigns we predicted would fund, what fraction did.

**Recall** — of the campaigns that actually funded, what fraction we caught.

**F1** — the harmonic mean of precision and recall; punishes lopsidedness.

**ROC-AUC** — probability that a randomly chosen success is ranked above a
randomly chosen failure. Threshold-independent. 0.5 is random.

**Average precision** — area under the precision-recall curve. Better than ROC-AUC
when the positive class is the minority and the one you care about.

**Decision threshold** — the probability cutoff for saying "funded". Default 0.5;
moving it trades precision against recall without retraining.

**Majority baseline** — always predict the more common class. The floor any real
model must clear. Ours is 0.5961.

**Leakage** — using information at training time that would not exist at
prediction time. Here: `backers`, `pledged`.

**Stratified split** — splitting so every part keeps the same class balance.

**Underfitting** — the model is too simple to capture the signal. Symptom: train
and test scores are both mediocre and close together. That was our situation, and
it is why regularization was the wrong lever.

**TruncatedSVD** — dimensionality reduction that works on sparse matrices. Used to
squeeze 2,500 TF-IDF columns into 120 dense ones so a tree can consume them.

---

## 18. Project history

| Phase | What happened |
|---|---|
| 0 | Repo structure, environment, docs |
| 1 | EDA; target locked to successful/failed; leakage audit |
| 2 | Title cleaning — lowercase, letters only, stopwords removed |
| 3 | Stratified 80/20 split; TF-IDF fitted on train only |
| 4 | Numeric scaling + categorical one-hot encoding |
| 5 | Sparse vs dense memory analysis; explicit fusion plan |
| 6 | `scipy.sparse.hstack` → the fused matrix |
| 7 | L2 logistic regression baseline |
| 8 | C sweep, PR curves, coefficient plots — **0.690 accuracy** |
| — | Roughbook: exploratory experiments, threshold tuning, ablations |
| 9 | Refactor into `src/`; engineered features; validation split; boosting ceiling check; test suite; live site — **0.707 accuracy** |
| 10 | Remaining: slides, demo video, final packaging |

Phases 1–8 stay frozen in the notebooks as the record of how each stage was
validated. Phase 9 is the promoted, parameterised version plus the improvements.

---

*Built by Sathvik, with Dhanush, Harsha and Karthik.*
