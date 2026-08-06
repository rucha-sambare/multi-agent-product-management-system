"""One text-normalisation contract shared by model training and inference."""

from __future__ import annotations

import pandas as pd


def normalize_review_text(values: pd.Series) -> pd.Series:
    """Preserve the text form used by the persisted TF-IDF vocabulary."""
    return values.fillna("").astype(str).str.lower().str.strip()
