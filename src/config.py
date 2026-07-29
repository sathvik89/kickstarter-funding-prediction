"""Shared paths and constants for the Phase 9 pipeline.

Phases 1-8 live in the notebooks and stay frozen. Everything here is the
promoted, parameterised version of that logic plus the Phase 9 feature work.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
V2 = PROCESSED / "v2"
FIG_DIR = ROOT / "reports" / "figures"

CLEANED_PATH = INTERIM / "ks_text_cleaned.csv"

RANDOM_STATE = 42

# Test fraction is unchanged from Phase 3 so the v2 test set is the *same rows*
# as the Phase 8 test set. Validation is then carved out of the old train block,
# which is what lets us tune C and the decision threshold without touching test.
TEST_SIZE = 0.20
VAL_SIZE_OF_TRAIN = 0.20

MAX_FEATURES = 2500
MIN_DF = 2

# --- tabular feature groups -------------------------------------------------
# Phase 4 used raw usd_goal_real (skew 83.1) and dropped the launch date. Both
# cost us real signal; see docs/phase9-improvements.md for the measurements.
NUMERIC_COLS = ["log_goal", "duration_days", "name_len", "name_words"]
CATEGORICAL_COLS = [
    "category",
    "main_category",
    "currency",
    "country",
    "launch_year",
    "launch_month",
    "launch_dow",
]

# Hand-crafted interactions are OFF by default, and that is a measured decision,
# not an oversight. The hypothesis was that a linear model loses to a tree because
# it cannot say "this goal is ambitious *for this category*". Crossing log_goal
# with main_category (and with the 159-level category, and with binned goal, and
# all of those together) buys at most +0.003 F1, while the tree gains +0.022. The
# tree's advantage is diffuse non-linearity, not one nameable interaction, so the
# columns do not earn their place in the shipped model. Flip the flag on
# `fusion_pipeline.build()` to reproduce the ablation.
USE_INTERACTIONS = False
INTERACTION_BASE = "log_goal"
INTERACTION_WITH = "main_category"
GOAL_BIN_COUNT = 10

C_GRID = [0.01, 0.1, 1.0, 10.0]
THRESHOLD_GRID_START = 0.15
THRESHOLD_GRID_STOP = 0.85
THRESHOLD_GRID_STEP = 0.01

SVD_COMPONENTS = 120
