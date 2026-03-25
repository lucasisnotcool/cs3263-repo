from __future__ import annotations

import joblib

from .dataset_loader import LABEL_TEXT_BY_ID
from .preprocess import SentimentPreprocessor


class SentimentPredictor:
    def __init__(self, model_path: str, feature_builder_path: str):
        self.model = joblib.load(model_path)
        self.feature_builder = joblib.load(feature_builder_path)
        self.preprocessor = SentimentPreprocessor()

    def predict_one(self, text: str) -> dict:
        cleaned_text = self.preprocessor.transform_texts([text])
        x = self.feature_builder.transform(cleaned_text)
        probs = self.model.predict_proba(x)[0]
        class_to_prob = {
            int(label): float(probs[idx]) for idx, label in enumerate(self.model.classes_)
        }
        predicted_label = max(class_to_prob, key=class_to_prob.get)

        return {
            "negative_probability": class_to_prob.get(0, 0.0),
            "positive_probability": class_to_prob.get(1, 0.0),
            "predicted_label": predicted_label,
            "predicted_label_text": LABEL_TEXT_BY_ID.get(predicted_label, str(predicted_label)),
        }
