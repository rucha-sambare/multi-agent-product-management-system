"""Reproducible TF-IDF + XGBoost training command."""

from __future__ import annotations

import argparse
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from AI_Product_Manager.config import settings
from AI_Product_Manager.ml.preprocessing import normalize_review_text


def train(dataset):
    df = pd.read_csv(dataset)
    required = {"review_text", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain {sorted(required)}")
    clean = df.dropna(subset=["review_text", "category"]).copy()
    x_train, x_test, y_train, y_test = train_test_split(
        normalize_review_text(clean["review_text"]),
        clean["category"].astype(str),
        test_size=0.2,
        random_state=42,
        stratify=clean["category"],
    )
    encoder = LabelEncoder()
    encoded_train = encoder.fit_transform(y_train)
    encoded_test = encoder.transform(y_test)
    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    train_features = vectorizer.fit_transform(x_train)
    test_features = vectorizer.transform(x_test)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train_features, encoded_train)
    predictions = model.predict(test_features)
    metrics = classification_report(
        encoded_test,
        predictions,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    probabilities = model.predict_proba(test_features)
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    calibration = {}
    for threshold in (0.50, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95):
        retained = confidence >= threshold
        calibration[str(threshold)] = {
            "coverage": round(float(retained.mean()), 4),
            "accuracy_when_retained": round(
                float(accuracy_score(encoded_test[retained], predicted[retained]))
                if retained.any() else 0.0,
                4,
            ),
        }
    metrics["confidence_diagnostics"] = {
        "minimum": round(float(confidence.min()), 4),
        "maximum": round(float(confidence.max()), 4),
        "mean": round(float(confidence.mean()), 4),
        "threshold_calibration": calibration,
    }
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_dir / "xgboost_feedback_model.pkl")
    joblib.dump(vectorizer, settings.model_dir / "tfidf_vectorizer.pkl")
    joblib.dump(encoder, settings.model_dir / "label_encoder.pkl")
    (settings.model_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(settings.data_dir / "final_labeled_feedback.csv"),
    )
    args = parser.parse_args()
    metrics = train(args.dataset)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
