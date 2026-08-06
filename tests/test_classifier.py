import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from AI_Product_Manager.feedback_analysis_mgr.feedback_classifier import (
    FeedbackClassifierAgent,
)


class FakeModel:
    def predict(self, features):
        return np.zeros(features.shape[0], dtype=int)

    def predict_proba(self, features):
        return np.tile([0.98, 0.02], (features.shape[0], 1))


def test_classifier_returns_complete_dataframe_with_confidence():
    vectorizer = TfidfVectorizer().fit(["bug crash", "great app"])
    encoder = LabelEncoder().fit(["Bug", "Praise"])
    agent = FeedbackClassifierAgent(FakeModel(), vectorizer, encoder)
    source = pd.DataFrame(
        {"review_text": ["Bug crash"], "rating": [1], "app_id": ["example"]}
    )
    result = agent.classify(source)
    assert list(result["category"]) == ["Bug"]
    assert result.loc[0, "prediction_confidence"] == 0.98
    assert result.loc[0, "app_id"] == "example"
    assert not bool(result.loc[0, "needs_human_review"])

