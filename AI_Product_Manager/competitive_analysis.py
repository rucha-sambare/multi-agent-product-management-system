"""Review-led competitor comparison.

RAG is useful for citing product descriptions and release notes, but it cannot
infer market position from a small document set. This module uses the same
sampled Play Store review evidence for the product and its competitors.
"""

from __future__ import annotations

import re

import pandas as pd

ISSUE_CATEGORIES = {"Bug", "Performance Issue", "Complaint", "UI Issue"}
# Normalised product topics. Each topic can be evidenced by several 2-4 word
# phrases/variants, but is reported once with unique-review support.
TOPIC_PATTERNS = {
    "Advertisements": r"\b(ad|ads|advertisement|advertisements|too many ads|ad break)\b",
    "Subscription pricing": r"\b(subscription|premium (price|cost)|pricing|too expensive|monthly (fee|price))\b",
    "Playback performance": r"\b(buffering|buffer|loading (slow|issue)|slow loading|playback (issue|problem)|video (not )?play)\b",
    "Offline downloads": r"\b(download (fail|issue|problem)|offline (download|mode|view)|cannot download)\b",
    "Login and account access": r"\b(log ?in|sign ?in|cannot access|account (locked|suspended|problem))\b",
    "Payment and billing": r"\b(payment|billing|charged|refund|transaction)\b",
    "Crashes and stability": r"\b(crash|crashes|crashing|keeps stopping|force close)\b",
    "Search and discovery": r"\b(search (feature|function|result|bar)|cannot find|find (video|music|content))\b",
    "Recommendations": r"\b(recommendation|recommend(ed|ing)|suggestion|algorithm)\b",
    "UI and navigation": r"\b(user interface|ui |navigation|layout|design|hard to (use|navigate))\b",
    "Notifications": r"\b(notification|notifications|alert|alerts)\b",
    "Casting and TV": r"\b(cast(ing)?|chromecast|smart tv|tv app)\b",
    "Background playback": r"\b(background play|play in background|screen off)\b",
    "Audio synchronization": r"\b(audio.*(sync|delay)|sound.*(sync|delay)|out of sync)\b",
}


def _rate(value: float) -> float:
    return round(float(value) * 100, 1)


def _valid_text_mask(df: pd.DataFrame) -> pd.Series:
    text = df.get("review_text", pd.Series("", index=df.index)).fillna("").astype(str)
    return text.str.contains(r"[A-Za-z]", regex=True)


def _top_themes(df: pd.DataFrame, positive: bool, limit: int = 5) -> list[dict]:
    if df.empty:
        return []
    subset = df[(df["sentiment"].eq("Positive") | df["category"].eq("Praise")) if positive else df["category"].isin(ISSUE_CATEGORIES)]
    support = {theme: 0 for theme in TOPIC_PATTERNS}
    for text in subset.get("review_text", pd.Series(dtype=str)).fillna(""):
        text = str(text).casefold()
        for theme, pattern in TOPIC_PATTERNS.items():
            if re.search(pattern, text):
                support[theme] += 1
    # A topic needs at least three distinct reviews and at least 1% of the
    # relevant review subset; this prevents one-off vocabulary becoming insight.
    minimum = max(3, int(len(subset) * 0.01 + 0.999))
    return [
        {"theme": theme, "reviews": count, "percentage": _rate(count / len(subset))}
        for theme, count in sorted(support.items(), key=lambda item: item[1], reverse=True)
        if count >= minimum
    ][:limit]


def profile_reviews(name: str, df: pd.DataFrame, metadata: dict | None = None) -> dict:
    """Use all reviews; prevent uncertain category predictions dominating claims."""
    if df.empty:
        return {"name": name, "sample_size": 0, "available": False, "metadata": metadata or {}}
    result = df.copy()
    text = result.get("review_text", pd.Series("", index=result.index)).fillna("").astype(str)
    # Text-only quality flag: empty, emoji-only, very short, or no recognised
    # lexical content are not useful for category/theme inference.
    valid = _valid_text_mask(result) & text.str.findall(r"[A-Za-z]{2,}").str.len().ge(2)
    confidence = pd.to_numeric(result.get("prediction_confidence", 0), errors="coerce").fillna(0)
    category = result.get("category", pd.Series("", index=result.index)).fillna("")
    rating = pd.to_numeric(result.get("rating", 0), errors="coerce")
    sentiment = result.get("sentiment", pd.Series("Neutral", index=result.index)).fillna("Neutral")
    usable = valid.sum()
    denominator = confidence[valid].sum()
    weighted = lambda mask: float(confidence[mask & valid].sum() / denominator) if denominator else 0.0
    queue_count = int(result.get("needs_human_review", pd.Series(False, index=result.index)).fillna(False).sum())
    return {
        "name": name, "available": True, "sample_size": int(len(result)),
        "metadata": metadata or {},
        "interpretable_reviews": int(usable), "uninterpretable_reviews": int(len(result) - usable),
        "queued_reviews": queue_count,
        "average_rating": round(float(rating.dropna().mean()) if rating.notna().any() else 0.0, 2),
        "positive_rate": _rate(float((sentiment[valid] == "Positive").mean()) if usable else 0.0),
        "negative_rate": _rate(float((sentiment[valid] == "Negative").mean()) if usable else 0.0),
        "praise_rate": _rate(weighted(category.eq("Praise"))),
        "issue_rate": _rate(weighted(category.isin(ISSUE_CATEGORIES))),
        "confidence_coverage": _rate(float(confidence[valid].mean()) if usable else 0.0),
        "praise_themes": _top_themes(result, positive=True), "issue_themes": _top_themes(result, positive=False),
        "category_rates": {
            label: _rate(weighted(category.eq(label)))
            for label in ("Feature Request", "Bug", "Performance Issue", "UI Issue", "Complaint")
        },
    }


def build_competitive_analysis(
    app_name: str,
    own_reviews: pd.DataFrame,
    competitor_reviews: dict[str, pd.DataFrame],
    skipped: list[dict] | None = None,
    metadata_by_app: dict[str, dict] | None = None,
) -> dict:
    """Compare sampled customer voice and return report-ready, traceable findings."""
    metadata_by_app = metadata_by_app or {}
    own = profile_reviews(app_name, own_reviews, metadata_by_app.get(app_name))
    competitors = [profile_reviews(name, frame, metadata_by_app.get(name)) for name, frame in competitor_reviews.items()]
    available = [item for item in competitors if item["available"]]
    comparisons = []
    for item in available:
        advantages, gaps = [], []
        for key, label, higher_is_better in (("average_rating", "average rating", True), ("positive_rate", "positive-feedback rate", True), ("praise_rate", "confidence-weighted praise rate", True), ("issue_rate", "confidence-weighted issue rate", False)):
            delta = own[key] - item[key]
            is_advantage = delta > 0 if higher_is_better else delta < 0
            if abs(delta) >= (0.05 if key == "average_rating" else 1.0):
                (advantages if is_advantage else gaps).append(label)
        comparisons.append({"competitor": item["name"], "our_advantages": advantages, "our_gaps": gaps})
    pool = [own, *available]
    rank = (
        sorted(pool, key=lambda item: (item["average_rating"], item["positive_rate"]), reverse=True).index(own) + 1
        if available else None
    )
    return {"method": "matched Play Store review samples; category rates are confidence-weighted", "our_product": own, "competitors": competitors, "comparisons": comparisons, "market_position": {"rank": rank, "out_of": len(pool), "basis": "average rating, then positive-feedback rate; not a market-share ranking"}, "skipped_competitors": skipped or []}
