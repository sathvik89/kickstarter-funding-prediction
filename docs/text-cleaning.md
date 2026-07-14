# Text cleaning notes

Companion to `notebooks/02_text_cleaning.ipynb`.

## Goal

Turn raw campaign titles (`name`) into a stable text field for TF-IDF.

## Cleaning rules

| Step | Action |
|------|--------|
| 1 | Lowercase the title |
| 2 | Replace non-alphabetic characters with spaces |
| 3 | Split on whitespace |
| 4 | Drop English stopwords (`sklearn` `ENGLISH_STOP_WORDS`) |
| 5 | Join remaining tokens into `name_clean` |

Example:

| Raw | Cleaned |
|-----|---------|
| `The Songs of Adelaide & Abullah` | `songs adelaide abullah` |
| `Where is Hank?` | `hank` |

## Outputs

| Item | Path / column |
|------|----------------|
| Interim table | `data/interim/ks_text_cleaned.csv` |
| Cleaned text | `name_clean` |
| Token-count QA | `name_clean_tokens` |
| Figure | `reports/figures/02_title_token_counts.png` |

Rows that become empty after cleaning are dropped from the interim export.

## What the figure shows

Raw titles are already short. After stopword and punctuation removal, token counts shift lower, but most campaigns keep a few content words — enough for a capped TF-IDF matrix in Phase 3.

## Next phase

Build a TF-IDF matrix from `name_clean` with `max_features=2500`.
