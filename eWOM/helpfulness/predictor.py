from __future__ import annotations

from collections.abc import Sequence
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
            self.model_name = loaded_model.get("model_name")
            self.threshold = float(loaded_model.get("threshold", 0.5))
        else:
            self.model = loaded_model
            self.model_name = None
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
        return self.predict_many(
            titles=[title],
            texts=[text],
            ratings=[rating],
            verified_purchases=[verified_purchase],
        )[0]

    def predict_many(
        self,
        *,
        titles: Sequence[str],
        texts: Sequence[str],
        ratings: Sequence[float] | None = None,
        verified_purchases: Sequence[bool] | None = None,
    ) -> list[dict[str, Any]]:
        if len(titles) != len(texts):
            raise ValueError("titles and texts must have the same length.")

        review_count = len(texts)
        if review_count == 0:
            raise ValueError("texts must contain at least one review.")

        if ratings is None:
            ratings = [0.0] * review_count
        if verified_purchases is None:
            verified_purchases = [False] * review_count
        if len(ratings) != review_count:
            raise ValueError("ratings and texts must have the same length.")
        if len(verified_purchases) != review_count:
            raise ValueError("verified_purchases and texts must have the same length.")

        frame = pd.DataFrame(
            {
                "title": list(titles),
                "text": list(texts),
                "rating": list(ratings),
                "verified_purchase": list(verified_purchases),
            }
        )
        transformed = self.preprocessor.transform(frame)
        x = self.feature_builder.transform(transformed)
        probabilities = self.model.predict_proba(x)
        class_to_index = {
            int(label): index for index, label in enumerate(self.model.classes_)
        }

        return [
            {
                "usefulness_probability": float(
                    probability_row[class_to_index[1]]
                ),
                "is_useful": float(probability_row[class_to_index[1]]) >= self.threshold,
            }
            for probability_row in probabilities
        ]
