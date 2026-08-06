import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from AI_Product_Manager.feedback_analysis_mgr.feedback_classifier import FeedbackClassifierAgent
from AI_Product_Manager.ml.preprocessing import normalize_review_text


class _Model:
    n_features_in_ = 2

    def predict(self, features):
        return np.ones(features.shape[0], dtype=int)

    def predict_proba(self, features):
        return np.tile([0.45, 0.55], (features.shape[0], 1))


def test_training_and_inference_share_normalisation_and_route_ambiguous_reviews():
    assert normalize_review_text(pd.Series(["  GREAT App "]))[0] == "great app"
    vectorizer = TfidfVectorizer().fit(["great app"])
    agent = FeedbackClassifierAgent(_Model(), vectorizer, LabelEncoder().fit(["Bug", "Praise"]))
    result = agent.classify(pd.DataFrame({"review_text": ["great app", "日本語 בלבד"]}))
    assert result["prediction_confidence"].tolist() == [0.55, 0.55]
    assert result["review_route"].tolist() == ["llm_ambiguity", "translation_or_human"]
