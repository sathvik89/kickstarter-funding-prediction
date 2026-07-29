"""Shared fixtures.

The fused pipeline is built once per session: it reads a 58 MB CSV and fits a
vectoriser over ~212k documents, so rebuilding it per test would make the suite
too slow to actually run.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import fusion_pipeline as fp  # noqa: E402


def _artifacts_present() -> bool:
    return (cfg.V2 / "v2_logreg_l2.joblib").exists()


requires_artifacts = pytest.mark.skipif(
    not _artifacts_present(),
    reason="fitted artifacts missing - run `python -m src.train` first",
)


@pytest.fixture(scope="session")
def frame():
    return fp.load_frame()


@pytest.fixture(scope="session")
def splits(frame):
    return fp.three_way_split(frame)


@pytest.fixture(scope="session")
def fused():
    return fp.build()
