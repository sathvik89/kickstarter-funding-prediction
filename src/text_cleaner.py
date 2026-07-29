"""Title cleaning, promoted verbatim from the Phase 2 notebook.

Lowercase, strip everything that is not a letter, drop sklearn's English
stopword list. Kept identical on purpose so v2 text features stay comparable
to the Phase 3 vocabulary.
"""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
TOKEN_RE = re.compile(r"[^a-z]+")


def clean_text(text: str) -> str:
    """Lowercase, keep letters only, drop stopwords."""
    text = str(text).lower()
    text = TOKEN_RE.sub(" ", text)
    return " ".join(tok for tok in text.split() if tok and tok not in STOPWORDS)
