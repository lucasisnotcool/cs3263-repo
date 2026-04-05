from __future__ import annotations

from collections.abc import Sequence

import joblib

from .dataset_loader import LABEL_TEXT_BY_ID
from .preprocess import SentimentPreprocessor


class SentimentPredictor:
    def __init__(self, model_path: str, feature_builder_path: str):
        loaded_model = joblib.load(model_path)
        if isinstance(loaded_model, dict) and "model" in loaded_model:
            self.model = loaded_model["model"]
            self.model_name = loaded_model.get("model_name")
        else:
            self.model = loaded_model
            self.model_name = None
        self.feature_builder = joblib.load(feature_builder_path)
        self.preprocessor = SentimentPreprocessor()

    def predict_one(self, text: str) -> dict:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Sequence[str]) -> list[dict]:
        if not texts:
            raise ValueError("texts must contain at least one review.")

        cleaned_text = self.preprocessor.transform_texts(texts)
        x = self.feature_builder.transform(cleaned_text)
        probabilities = self.model.predict_proba(x)

        predictions: list[dict] = []
        for probs in probabilities:
            class_to_prob = {
                int(label): float(probs[idx])
                for idx, label in enumerate(self.model.classes_)
            }
            predicted_label = max(class_to_prob, key=class_to_prob.get)
            predictions.append(
                {
                    "negative_probability": class_to_prob.get(0, 0.0),
                    "positive_probability": class_to_prob.get(1, 0.0),
                    "predicted_label": predicted_label,
                    "predicted_label_text": LABEL_TEXT_BY_ID.get(
                        predicted_label, str(predicted_label)
                    ),
                }
            )

        return predictions
