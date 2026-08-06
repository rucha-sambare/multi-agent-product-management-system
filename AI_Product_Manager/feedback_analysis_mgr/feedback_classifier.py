"""Batched review classification using the existing TF-IDF/XGBoost artifacts."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from AI_Product_Manager.ml.preprocessing import normalize_review_text

try:
    from AI_Product_Manager.config import settings
except ImportError:  # Supports running from inside the package directory.
    from config import settings


class FeedbackClassifierAgent:
    def __init__(self, model=None, vectorizer=None, encoder=None):
        try:
            self.model = model or joblib.load(
                settings.model_dir / "xgboost_feedback_model.pkl"
            )
            self.vectorizer = vectorizer or joblib.load(
                settings.model_dir / "tfidf_vectorizer.pkl"
            )
            self.encoder = encoder or joblib.load(
                settings.model_dir / "label_encoder.pkl"
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "A model dependency is missing. Run: pip install -r requirements.txt"
            ) from exc
        # Pickles created by newer XGBoost releases can omit this deprecated
        # attribute, while 1.x's sklearn wrapper still reads it.
        if not hasattr(self.model, "use_label_encoder"):
            self.model.use_label_encoder = False

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        if "review_text" not in df:
            raise ValueError("Input must contain a 'review_text' column.")

        result = df.copy()
        result["clean_text"] = normalize_review_text(result["review_text"])
        features = self.vectorizer.transform(result["clean_text"])
        expected_features = getattr(self.model, "n_features_in_", features.shape[1])
        if expected_features != features.shape[1]:
            raise RuntimeError(
                "Model/vectorizer feature mismatch: retrain and save both artifacts together."
            )
        probabilities = None
        if hasattr(self.model, "_Booster"):
            # Use the stable Booster interface for legacy/new sklearn-wrapper
            # pickle compatibility.
            from xgboost import DMatrix

            raw = np.asarray(self.model._Booster.predict(DMatrix(features)))
            if raw.ndim == 2:
                probabilities = raw
                predictions = raw.argmax(axis=1)
            else:
                predictions = raw.astype(int)
        else:
            predictions = np.asarray(self.model.predict(features), dtype=int)
        result["category"] = self.encoder.inverse_transform(predictions)

        if probabilities is None and hasattr(self.model, "predict_proba"):
            probabilities = np.asarray(self.model.predict_proba(features))
        if probabilities is not None:
            result["prediction_confidence"] = probabilities.max(axis=1)
        else:
            result["prediction_confidence"] = 1.0

        low_confidence = result["prediction_confidence"] < settings.classification_threshold
        # A zero-vector review is outside the English TF-IDF model vocabulary
        # (often another language, emoji-only, or meaningless text). It needs
        # translation/human triage, not an LLM category guess.
        result["vocabulary_hits"] = np.asarray(features.getnnz(axis=1)).ravel()
        result["review_route"] = np.where(
            low_confidence & result["vocabulary_hits"].eq(0),
            "translation_or_human",
            np.where(low_confidence, "llm_ambiguity", "model_confident"),
        )
        result["needs_human_review"] = low_confidence
        result["requires_llm_review"] = result["review_route"].eq("llm_ambiguity")
        result["classification_source"] = "xgboost"
        return result
