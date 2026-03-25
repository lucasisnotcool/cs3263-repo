from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from .preprocess import HelpfulnessPreprocessor


class HelpfulnessPredictor:
    def __init__(
        self,
        model_path: str,
        feature_builder_path: str,
        preprocessor: HelpfulnessPreprocessor | None = None,
    ):
        loaded_model = joblib.load(model_path)
        if isinstance(loaded_model, dict) and "model" in loaded_model:
            self.model = loaded_model["model"]
            self.threshold = float(loaded_model.get("threshold", 0.5))
        else:
            self.model = loaded_model
            self.threshold = 0.5

        self.feature_builder = joblib.load(feature_builder_path)
        self.preprocessor = preprocessor or HelpfulnessPreprocessor()

    def predict_one(
        self,
        *,
        title: str,
        text: str,
        rating: float = 0.0,
        verified_purchase: bool = False,
    ) -> dict[str, Any]:
        frame = pd.DataFrame(
            [
                {
                    "title": title,
                    "text": text,
                    "rating": rating,
                    "verified_purchase": verified_purchase,
                }
            ]
        )
        transformed = self.preprocessor.transform(frame)
        x = self.feature_builder.transform(transformed)
        probabilities = self.model.predict_proba(x)[0]
        class_to_index = {
            int(label): index for index, label in enumerate(self.model.classes_)
        }
        usefulness_probability = float(probabilities[class_to_index[1]])
        return {
            "usefulness_probability": usefulness_probability,
            "is_useful": usefulness_probability >= self.threshold,
        }
