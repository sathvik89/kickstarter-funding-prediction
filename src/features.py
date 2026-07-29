"""Phase 9 tabular feature engineering.

Three things Phase 4 left on the table, all measured in
`docs/phase9-improvements.md`:

1. `log_goal` - raw `usd_goal_real` has skew 83.1 (median $5k, max $166M).
   Standardising that gives a linear model a coefficient on a variable whose
   useful range is the first 0.2% of its span. `log1p` brings skew to -0.14.
2. Launch date parts - Phase 4 derived `duration_days` and threw the rest away.
   Success rate in this dataset runs 50.6% (2011) to 32.1% (2015), an 18.5pt
   spread the model could not see.
3. `name_len` / `name_words` - already computed back in Phase 1 and sitting
   unused in `ks_binary_base.csv`.
"""

import numpy as np
import pandas as pd
from scipy import sparse


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with the Phase 9 tabular columns attached."""
    out = df.copy()

    out["log_goal"] = np.log1p(out["usd_goal_real"].clip(lower=0))

    launched = pd.to_datetime(out["launched"], errors="coerce")
    # Kept as strings so the one-hot encoder treats them as levels, not as
    # numbers where "December is 12x January" would be nonsense.
    out["launch_year"] = launched.dt.year.astype("string").fillna("unknown")
    out["launch_month"] = launched.dt.month.astype("string").fillna("unknown")
    out["launch_dow"] = launched.dt.dayofweek.astype("string").fillna("unknown")

    for col in ("name_len", "name_words"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


def build_interaction_block(
    df: pd.DataFrame, base: str, with_col: str, levels: np.ndarray
) -> tuple[sparse.csr_matrix, list[str]]:
    """One column per level of `with_col`, holding `base` where that level is active.

    This is the cheap way to give a *linear* model the goal x category effect
    that a tree finds on its own: a $10k goal is routine for Music and brutal
    for Technology, but `w_goal + w_technology` cannot say that.

    `levels` comes from the encoder fitted on train, so train/val/test always
    produce the same columns in the same order. Zero off-level, so it stays sparse.
    """
    values = df[base].to_numpy(dtype=np.float32)
    col_of = df[with_col].astype(str).to_numpy()

    rows, cols, data = [], [], []
    level_index = {lvl: j for j, lvl in enumerate(levels)}
    for i, (lvl, val) in enumerate(zip(col_of, values)):
        j = level_index.get(lvl)
        if j is not None and val != 0.0:
            rows.append(i)
            cols.append(j)
            data.append(val)

    block = sparse.csr_matrix(
        (data, (rows, cols)), shape=(len(df), len(levels)), dtype=np.float32
    )
    names = [f"inter__{base}_x_{with_col}_{lvl}" for lvl in levels]
    return block, names
