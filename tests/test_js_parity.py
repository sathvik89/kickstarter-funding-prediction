"""Does the browser predictor agree with scikit-learn?

`test_web_export.py` proves the exported JSON reproduces sklearn. This closes
the last gap: that the JavaScript on the page implements the same arithmetic.
Without it, the live demo could drift from the model while every Python number
in the repo stayed correct - the failure mode that would embarrass us publicly.

Skipped when node is unavailable, so the suite still runs anywhere.
"""

import json
import shutil
import subprocess

import pytest

from src import config as cfg
from src import export_web
from tests.conftest import requires_artifacts

pytestmark = requires_artifacts

NODE = shutil.which("node")

# Chosen to exercise the parts most likely to break: unknown levels, empty
# titles, extreme goals, and the L2-normalisation guard.
CASES = [
    {"label": "typical design project",
     "title": "Handmade Oak Desk Lamp", "goal": 5000, "duration": 30,
     "category": "Product Design", "main_category": "Design",
     "country": "US", "currency": "USD", "year": "2017", "month": "4"},
    {"label": "all-stopword title (zero L2 norm)",
     "title": "the and of it", "goal": 5000, "duration": 30,
     "category": "Product Design", "main_category": "Design",
     "country": "US", "currency": "USD", "year": "2017", "month": "4"},
    {"label": "empty title",
     "title": "", "goal": 1200, "duration": 45,
     "category": "Rock", "main_category": "Music",
     "country": "GB", "currency": "GBP", "year": "2014", "month": "11"},
    {"label": "repeated token (tf > 1)",
     "title": "game game game board game", "goal": 25000, "duration": 30,
     "category": "Tabletop Games", "main_category": "Games",
     "country": "US", "currency": "USD", "year": "2016", "month": "9"},
    {"label": "huge goal",
     "title": "Ambitious Technology Platform", "goal": 5_000_000, "duration": 60,
     "category": "Hardware", "main_category": "Technology",
     "country": "US", "currency": "USD", "year": "2015", "month": "7"},
    {"label": "tiny goal",
     "title": "Tiny Zine About Cats", "goal": 1, "duration": 7,
     "category": "Zines", "main_category": "Publishing",
     "country": "CA", "currency": "CAD", "year": "2013", "month": "2"},
    {"label": "unknown country level",
     "title": "Handmade Oak Desk Lamp", "goal": 5000, "duration": 30,
     "category": "Product Design", "main_category": "Design",
     "country": "Atlantis", "currency": "USD", "year": "2017", "month": "4"},
    {"label": "unseen launch year",
     "title": "Handmade Oak Desk Lamp", "goal": 5000, "duration": 30,
     "category": "Product Design", "main_category": "Design",
     "country": "US", "currency": "USD", "year": "2099", "month": "4"},
    {"label": "punctuation and digits stripped",
     "title": "3D-Printed  Robot!!! (v2)", "goal": 9000, "duration": 21,
     "category": "Robots", "main_category": "Technology",
     "country": "US", "currency": "USD", "year": "2016", "month": "6"},
]


def _python_probability(payload: dict, case: dict) -> float:
    """Score a case through the Python payload-only implementation."""
    import numpy as np

    row = {
        "name": case["title"],
        "log_goal": float(np.log1p(max(0, case["goal"]))),
        "duration_days": float(case["duration"]),
        "name_len": float(len(case["title"])),
        "name_words": float(len([w for w in case["title"].split() if w])),
        "category": case["category"],
        "main_category": case["main_category"],
        "currency": case["currency"],
        "country": case["country"],
        "launch_year": case["year"],
        "launch_month": case["month"],
        "launch_dow": "1",
    }
    return float(export_web.score_from_payload(payload, [row])[0])


@pytest.fixture(scope="module")
def payload():
    return export_web.build_payload()


@pytest.mark.parametrize("case", CASES, ids=[c["label"] for c in CASES])
def test_python_scoring_is_finite_and_bounded(payload, case):
    p = _python_probability(payload, case)
    assert 0.0 < p < 1.0, f"probability out of range for {case['label']}"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_page_javascript_matches_python(payload, tmp_path):
    """Execute the real <script> block from docs/index.html and compare."""
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([
        {"inputs": {k: c[k] for k in
                    ("title", "goal", "duration", "category", "main_category",
                     "country", "currency", "year", "month")},
         "expected": _python_probability(payload, c)}
        for c in CASES
    ]))

    result = subprocess.run(
        [NODE, str(cfg.ROOT / "tests" / "js_parity.mjs"), str(cases_file)],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, (
        f"browser predictor disagrees with the model\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "JS parity OK" in result.stdout
