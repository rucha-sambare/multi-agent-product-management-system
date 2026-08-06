import pandas as pd

from AI_Product_Manager.competitive_analysis import build_competitive_analysis


def _reviews(ratings, categories, sentiment, confidence, queued=False):
    return pd.DataFrame({
        "review_text": ["great editing tools", "crashes after update"],
        "rating": ratings,
        "category": categories,
        "sentiment": sentiment,
        "prediction_confidence": confidence,
        "needs_human_review": [queued, False],
    })


def test_competitive_analysis_uses_queued_reviews_without_overclaiming_categories():
    ours = _reviews([5, 1], ["Praise", "Bug"], ["Positive", "Negative"], [0.2, 1.0], queued=True)
    rival = _reviews([4, 4], ["Praise", "Praise"], ["Positive", "Positive"], [1.0, 1.0])
    result = build_competitive_analysis("Ours", ours, {"Rival": rival})
    assert result["our_product"]["sample_size"] == 2
    assert result["our_product"]["queued_reviews"] == 1
    assert result["our_product"]["praise_rate"] < 50  # uncertain praise is down-weighted
    assert result["competitors"][0]["praise_rate"] == 100
    assert result["market_position"]["out_of"] == 2
