"""Phase 9 fusion pipeline: three-way split, fit on train only, sparse hstack.

What changed from Phase 3/4/6
-----------------------------
* **Three-way split.** Phase 8 picked `best_C` on the test set and the roughbook
  picked the decision threshold on the test set. Both are selection-on-test. Here
  test is split off with the *same* `random_state`/`test_size` as Phase 3 (so it is
  literally the same rows and the metrics stay comparable), then validation is
  carved out of the old train block. C and the threshold are chosen on validation.
* **Vectoriser and scaler are refit on the smaller train block.** They never see
  validation or test rows, so the vocabulary differs slightly from Phase 3's -
  that is the correct behaviour, not a regression.
* **Sparse hstack is unchanged.** It is the point of the project. The engineered
  columns join the existing text and tabular blocks; nothing is densified.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as cfg
from .features import add_engineered_columns, build_interaction_block


@dataclass
class FusedData:
    """Everything the training step needs, with names kept alongside matrices."""

    X_train: sparse.csr_matrix
    X_val: sparse.csr_matrix
    X_test: sparse.csr_matrix
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: np.ndarray
    frames: dict[str, pd.DataFrame]
    vectorizer: TfidfVectorizer
    tabular: ColumnTransformer
    n_text: int
    n_tab: int
    n_inter: int
    numeric_cols: list[str]
    categorical_cols: list[str]


def load_frame() -> pd.DataFrame:
    """Cleaned interim table plus the Phase 9 engineered columns."""
    df = pd.read_csv(cfg.CLEANED_PATH)
    df["name_clean"] = df["name_clean"].fillna("").astype(str)
    return add_engineered_columns(df)


def three_way_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce the Phase 3 test set exactly, then carve validation out of train."""
    train_full, test_df = train_test_split(
        df,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df["target"],
    )
    train_df, val_df = train_test_split(
        train_full,
        test_size=cfg.VAL_SIZE_OF_TRAIN,
        random_state=cfg.RANDOM_STATE,
        stratify=train_full["target"],
    )
    return train_df, val_df, test_df


def build(
    with_interactions: bool = cfg.USE_INTERACTIONS,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    with_text: bool = True,
    frame: pd.DataFrame | None = None,
) -> FusedData:
    """Fit every transformer on train, transform all three splits, fuse sparsely.

    The column overrides exist so `ablations.py` can rebuild the exact same
    pipeline with one feature group removed and attribute the lift honestly.
    Pass `frame` to reuse an already-loaded table across many ablation runs.
    """
    numeric_cols = list(cfg.NUMERIC_COLS if numeric_cols is None else numeric_cols)
    categorical_cols = list(cfg.CATEGORICAL_COLS if categorical_cols is None else categorical_cols)

    df = load_frame() if frame is None else frame
    train_df, val_df, test_df = three_way_split(df)

    # --- text block ---------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=cfg.MAX_FEATURES,
        ngram_range=(1, 1),
        min_df=cfg.MIN_DF,
        dtype=np.float32,
    )
    if with_text:
        text_blocks = {
            "train": vectorizer.fit_transform(train_df["name_clean"]),
            "val": vectorizer.transform(val_df["name_clean"]),
            "test": vectorizer.transform(test_df["name_clean"]),
        }
        text_names = [f"text__{t}" for t in vectorizer.get_feature_names_out()]
    else:
        text_blocks = {
            name: sparse.csr_matrix((len(frame_), 0), dtype=np.float32)
            for name, frame_ in (("train", train_df), ("val", val_df), ("test", test_df))
        }
        text_names = []

    # --- tabular block ------------------------------------------------------
    tabular = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    cols = numeric_cols + categorical_cols
    tab_blocks = {
        "train": tabular.fit_transform(train_df[cols]),
        "val": tabular.transform(val_df[cols]),
        "test": tabular.transform(test_df[cols]),
    }
    tab_blocks = {k: sparse.csr_matrix(v) if not sparse.issparse(v) else v.tocsr()
                  for k, v in tab_blocks.items()}
    tab_names = list(tabular.get_feature_names_out())

    # --- interaction block --------------------------------------------------
    inter_blocks: dict[str, sparse.csr_matrix] = {}
    inter_names: list[str] = []
    if with_interactions:
        ohe = tabular.named_transformers_["cat"]
        levels = ohe.categories_[categorical_cols.index(cfg.INTERACTION_WITH)]
        for split_name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
            block, inter_names = build_interaction_block(
                frame, cfg.INTERACTION_BASE, cfg.INTERACTION_WITH, levels
            )
            inter_blocks[split_name] = block

    # --- fuse ---------------------------------------------------------------
    def fuse(split_name: str) -> sparse.csr_matrix:
        parts = [text_blocks[split_name], tab_blocks[split_name]]
        if with_interactions:
            parts.append(inter_blocks[split_name])
        return sparse.hstack(parts, format="csr")

    feature_names = np.array(text_names + tab_names + inter_names, dtype=object)

    return FusedData(
        X_train=fuse("train"),
        X_val=fuse("val"),
        X_test=fuse("test"),
        y_train=train_df["target"].to_numpy(dtype=np.int8),
        y_val=val_df["target"].to_numpy(dtype=np.int8),
        y_test=test_df["target"].to_numpy(dtype=np.int8),
        feature_names=feature_names,
        frames={"train": train_df, "val": val_df, "test": test_df},
        vectorizer=vectorizer,
        tabular=tabular,
        n_text=len(text_names),
        n_tab=len(tab_names),
        n_inter=len(inter_names),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
