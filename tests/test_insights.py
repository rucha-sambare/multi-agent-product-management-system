import pandas as pd

from AI_Product_Manager.feedback_analysis_mgr.insights import (
    add_duplicate_groups,
    add_priority,
    add_sentiment,
    build_feature_ranking,
)


def sample():
    return pd.DataFrame(
        {
            "review_text": [
                "Please add dark mode",
                "Please add dark mode",
                "App crashes and is worst",
                "Great useful app",
            ],
            "rating": [4, 3, 1, 5],
            "thumbs_up": [5, 2, 20, 1],
            "category": ["Feature Request", "Feature Request", "Bug", "Praise"],
            "prediction_confidence": [0.98, 0.97, 0.99, 0.99],
        }
    )


def test_sentiment_and_priority_are_deterministic():
    result = add_priority(add_duplicate_groups(add_sentiment(sample())))
    assert result.loc[2, "sentiment"] == "Negative"
    assert result.loc[3, "sentiment"] == "Positive"
    assert result.loc[2, "priority_score"] > result.loc[3, "priority_score"]
    assert result.loc[0, "duplicate_count"] == 2


def test_feature_ranking_groups_duplicates():
    result = add_priority(add_duplicate_groups(add_sentiment(sample())))
    ranking = build_feature_ranking(result)
    assert len(ranking) == 1
    assert ranking[0]["reach"] == 2

