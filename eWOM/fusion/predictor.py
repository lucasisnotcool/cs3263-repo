from __future__ import annotations

from collections.abc import Sequence

from eWOM.deception import DeceptionPredictor
from eWOM.helpfulness.predictor import HelpfulnessPredictor
from eWOM.helpfulness.preprocess import HelpfulnessPreprocessor
from eWOM.sentiment_analysis.predictor import SentimentPredictor

from .scorer import EWOMFusionConfig, EWOMFusionScorer


class EWOMFusionPredictor:
    def __init__(
        self,
        *,
        helpfulness_model_path: str,
        helpfulness_feature_builder_path: str,
        sentiment_model_path: str,
        sentiment_feature_builder_path: str,
        fusion_config: EWOMFusionConfig | None = None,
    ):
        self.helpfulness_predictor = HelpfulnessPredictor(
            helpfulness_model_path,
            helpfulness_feature_builder_path,
            HelpfulnessPreprocessor(),
        )
        self.sentiment_predictor = SentimentPredictor(
            sentiment_model_path,
            sentiment_feature_builder_path,
        )
        self.deception_predictor = DeceptionPredictor()
        self.scorer = EWOMFusionScorer(fusion_config)

    def predict_one(
        self,
        *,
        title: str,
        text: str,
        rating: float = 0.0,
        verified_purchase: bool = False,
    ) -> dict:
        review_prediction = self.predict_many(
            review_texts=[text],
            titles=[title],
            ratings=[rating],
            verified_purchases=[verified_purchase],
        )["reviews"][0]
        return {
            "helpfulness": review_prediction["helpfulness"],
            "sentiment": review_prediction["sentiment"],
            "deception": review_prediction["deception"],
            "fusion": review_prediction["fusion"],
        }

    def predict_many(
        self,
        *,
        review_texts: Sequence[str],
        titles: Sequence[str] | None = None,
        ratings: Sequence[float] | None = None,
        verified_purchases: Sequence[bool] | None = None,
    ) -> dict:
        review_count = len(review_texts)
        if review_count == 0:
            raise ValueError("review_texts must contain at least one review.")

        titles = [""] * review_count if titles is None else list(titles)
        ratings = [0.0] * review_count if ratings is None else list(ratings)
        verified_purchases = (
            [False] * review_count
            if verified_purchases is None
            else list(verified_purchases)
        )

        helpfulness_predictions = self.helpfulness_predictor.predict_many(
            titles=titles,
            texts=review_texts,
            ratings=ratings,
            verified_purchases=verified_purchases,
        )
        sentiment_predictions = self.sentiment_predictor.predict_many(review_texts)
        deception_predictions = self.deception_predictor.predict_many(
            review_texts,
            titles=titles,
        )

        reviews = []
        review_scores = []
        for text, helpfulness_prediction, sentiment_prediction, deception_prediction in zip(
            review_texts,
            helpfulness_predictions,
            sentiment_predictions,
            deception_predictions,
        ):
            fusion_prediction = self.scorer.score(
                usefulness_probability=helpfulness_prediction["usefulness_probability"],
                positive_probability=sentiment_prediction["positive_probability"],
                negative_probability=sentiment_prediction["negative_probability"],
                deception_probability=deception_prediction["deception_probability"],
            )
            reviews.append(
                {
                    "text": text,
                    "helpfulness": helpfulness_prediction,
                    "sentiment": sentiment_prediction,
                    "deception": deception_prediction,
                    "fusion": fusion_prediction,
                }
            )
            review_scores.append(fusion_prediction)

        aggregate_prediction = self.scorer.aggregate(review_scores)

        return {
            "review_count": review_count,
            "reviews": reviews,
            "aggregate": aggregate_prediction,
        }
