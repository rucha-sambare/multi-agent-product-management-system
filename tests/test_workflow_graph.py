import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from AI_Product_Manager.feedback_analysis_mgr.feedback_classifier import FeedbackClassifierAgent
from AI_Product_Manager.workflow_graph import analyze_with_graph


class FakeModel:
    def predict(self, features):
        return np.zeros(features.shape[0], dtype=int)

    def predict_proba(self, features):
        return np.tile([0.98, 0.02], (features.shape[0], 1))


def test_langgraph_analysis_path():
    vectorizer = TfidfVectorizer().fit(["bug crash", "great app"])
    encoder = LabelEncoder().fit(["Bug", "Praise"])
    classifier = FeedbackClassifierAgent(FakeModel(), vectorizer, encoder)
    source = pd.DataFrame(
        {"review_text": ["bug crash"], "rating": [1], "thumbs_up": [4]}
    )
    result = analyze_with_graph(source, classifier)
    assert result.loc[0, "category"] == "Bug"
    assert result.loc[0, "priority_score"] > 0
    assert "duplicate_group" in result

