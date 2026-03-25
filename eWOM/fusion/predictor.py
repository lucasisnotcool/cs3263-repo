from __future__ import annotations

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
        self.scorer = EWOMFusionScorer(fusion_config)

    def predict_one(
        self,
        *,
        title: str,
        text: str,
        rating: float = 0.0,
        verified_purchase: bool = False,
    ) -> dict:
        helpfulness_prediction = self.helpfulness_predictor.predict_one(
            title=title,
            text=text,
            rating=rating,
            verified_purchase=verified_purchase,
        )
        sentiment_prediction = self.sentiment_predictor.predict_one(text)
        fusion_prediction = self.scorer.score(
            usefulness_probability=helpfulness_prediction["usefulness_probability"],
            positive_probability=sentiment_prediction["positive_probability"],
            negative_probability=sentiment_prediction["negative_probability"],
        )

        return {
            "helpfulness": helpfulness_prediction,
            "sentiment": sentiment_prediction,
            "fusion": fusion_prediction,
        }
