"""Deterministic sentiment, deduplication, and priority scoring."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


POSITIVE = {"good", "great", "excellent", "love", "best", "amazing", "useful", "easy"}
NEGATIVE = {"bad", "worst", "hate", "crash", "broken", "slow", "issue", "problem", "bug"}
CATEGORY_WEIGHT = {
    "Bug": 4.0,
    "Performance Issue": 3.5,
    "Complaint": 3.0,
    "UI Issue": 2.5,
    "Feature Request": 2.0,
    "Praise": 0.5,
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    sentiments = []
    scores = []
    for text in result["review_text"].fillna(""):
        words = _tokens(text)
        score = len(words & POSITIVE) - len(words & NEGATIVE)
        scores.append(score)
        sentiments.append("Positive" if score > 0 else "Negative" if score < 0 else "Neutral")
    result["sentiment_score"] = scores
    result["sentiment"] = sentiments
    return result


def add_duplicate_groups(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    result = df.copy()
    texts = result["review_text"].fillna("").astype(str)
    if len(result) < 2 or texts.str.strip().eq("").all():
        result["duplicate_group"] = np.arange(len(result))
        result["duplicate_count"] = 1
        result["dedupe_method"] = "not_required"
        return result

    try:
        from sentence_transformers import SentenceTransformer

        matrix = SentenceTransformer("all-MiniLM-L6-v2").encode(
            texts.tolist(), normalize_embeddings=True, show_progress_bar=False
        )
        result["dedupe_method"] = "all-MiniLM-L6-v2"
    except (ImportError, OSError):
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
        result["dedupe_method"] = "tfidf_fallback"
    neighbors = NearestNeighbors(metric="cosine", n_neighbors=min(5, len(result)))
    neighbors.fit(matrix)
    distances, indices = neighbors.kneighbors(matrix)
    parent = list(range(len(result)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        for distance, other in zip(row_distances[1:], row_indices[1:]):
            if 1.0 - distance >= threshold:
                union(row, int(other))

    roots = [find(i) for i in range(len(result))]
    normalized = {root: n for n, root in enumerate(dict.fromkeys(roots), 1)}
    result["duplicate_group"] = [normalized[root] for root in roots]
    result["duplicate_count"] = result.groupby("duplicate_group")["duplicate_group"].transform("size")
    return result


def add_priority(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    rating = pd.to_numeric(result.get("rating", 3), errors="coerce").fillna(3).clip(1, 5)
    thumbs = pd.to_numeric(result.get("thumbs_up", 0), errors="coerce").fillna(0).clip(lower=0)
    duplicates = pd.to_numeric(result.get("duplicate_count", 1), errors="coerce").fillna(1)
    category = result["category"].map(CATEGORY_WEIGHT).fillna(1.0)
    result["priority_score"] = (
        category + (5 - rating) * 0.75 + np.log1p(thumbs) * 0.5 + np.log1p(duplicates) * 0.75
    ).round(3)
    result["priority"] = pd.cut(
        result["priority_score"],
        bins=[-np.inf, 4.0, 6.5, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return result


def build_feature_ranking(df: pd.DataFrame, limit: int = 10) -> list[dict]:
    features = df[df["category"].eq("Feature Request")].copy()
    if features.empty:
        return []
    grouped = (
        features.groupby("duplicate_group", as_index=False)
        .agg(
            request=("review_text", "first"),
            reach=("review_text", "size"),
            average_priority=("priority_score", "mean"),
            confidence=("prediction_confidence", "mean"),
            evidence_weight=("prediction_confidence", "sum"),
        )
    )
    grouped["rice_score"] = (
        # Every request is visible as reach; uncertain labels have proportionally
        # less influence until a human reviewer confirms them.
        grouped["evidence_weight"] * grouped["average_priority"] * grouped["confidence"]
    ).round(2)
    return grouped.sort_values("rice_score", ascending=False).head(limit).to_dict("records")
