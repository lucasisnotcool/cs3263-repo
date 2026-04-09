from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from eWOM.fusion import EWOMFusionPredictor, EWOMFusionScorer


class EWOMFusionScorerTests(unittest.TestCase):
    def test_deception_probability_downweights_sentiment(self) -> None:
        scorer = EWOMFusionScorer()

        baseline = scorer.score(
            usefulness_probability=0.9,
            positive_probability=0.9,
            negative_probability=0.1,
        )
        with_deception = scorer.score(
            usefulness_probability=0.9,
            positive_probability=0.9,
            negative_probability=0.1,
            deception_probability=0.8,
        )

        self.assertAlmostEqual(with_deception["deception_weight"], 0.2)
        self.assertLess(with_deception["informative_gate"], baseline["informative_gate"])
        self.assertLess(
            with_deception["ewom_magnitude_0_to_100"],
            baseline["ewom_magnitude_0_to_100"],
        )

    def test_aggregate_legacy_scores_defaults_to_full_authenticity(self) -> None:
        scorer = EWOMFusionScorer()

        aggregate = scorer.aggregate(
            [
                {
                    "usefulness_probability": 0.7,
                    "helpfulness_gate": 0.8,
                    "positive_probability": 0.9,
                    "negative_probability": 0.1,
                    "sentiment_polarity": 0.8,
                    "sentiment_strength": 0.8,
                }
            ]
        )

        self.assertIsNone(aggregate["mean_deception_probability"])
        self.assertEqual(aggregate["mean_deception_weight"], 1.0)
        self.assertEqual(aggregate["mean_informative_gate"], 0.8)

    def test_aggregate_gate_uses_effective_informative_support(self) -> None:
        scorer = EWOMFusionScorer()
        low_informative_reviews = [
            {
                "usefulness_probability": 0.1,
                "helpfulness_gate": 0.1,
                "deception_weight": 0.5,
                "informative_gate": 0.05,
                "positive_probability": 0.9,
                "negative_probability": 0.1,
                "sentiment_polarity": 0.8,
                "sentiment_strength": 0.8,
            }
            for _ in range(20)
        ]

        aggregate = scorer.aggregate(low_informative_reviews)
        expected_support = 1.0
        expected_gate = 1.0 - math.exp(
            -expected_support / scorer.config.review_set_gate_scale
        )

        self.assertEqual(aggregate["review_count"], 20)
        self.assertAlmostEqual(aggregate["informative_review_weight"], expected_support)
        self.assertAlmostEqual(aggregate["review_set_gate"], expected_gate)
        self.assertLess(aggregate["review_set_gate"], 0.3)


class EWOMFusionPredictorTests(unittest.TestCase):
    @patch("eWOM.fusion.predictor.DeceptionPredictor")
    @patch("eWOM.fusion.predictor.SentimentPredictor")
    @patch("eWOM.fusion.predictor.HelpfulnessPredictor")
    def test_predict_many_includes_deception_stream(
        self,
        helpfulness_predictor_mock,
        sentiment_predictor_mock,
        deception_predictor_mock,
    ) -> None:
        helpfulness_predictor_mock.return_value.predict_many.return_value = [
            {
                "usefulness_probability": 0.9,
                "is_useful": True,
            }
        ]
        sentiment_predictor_mock.return_value.predict_many.return_value = [
            {
                "negative_probability": 0.2,
                "positive_probability": 0.8,
                "predicted_label": 1,
                "predicted_label_text": "positive",
            }
        ]
        deception_predictor_mock.return_value.predict_many.return_value = [
            {
                "status": "ok",
                "source": "experiment_trust_fake_reviews",
                "deception_probability": 0.25,
                "authenticity_probability": 0.75,
                "trust_probability": 0.75,
                "is_deceptive": False,
                "graph_uncertainty_entropy": 0.4,
                "overall_confidence": 0.8,
                "error": None,
            }
        ]

        predictor = EWOMFusionPredictor(
            helpfulness_model_path="helpfulness-model.joblib",
            helpfulness_feature_builder_path="helpfulness-feature-builder.joblib",
            sentiment_model_path="sentiment-model.joblib",
            sentiment_feature_builder_path="sentiment-feature-builder.joblib",
        )

        result = predictor.predict_many(
            review_texts=["Battery lasts all day."],
            titles=["Solid battery life"],
            ratings=[5.0],
            verified_purchases=[True],
        )

        review = result["reviews"][0]
        self.assertEqual(review["deception"]["status"], "ok")
        self.assertEqual(review["fusion"]["deception_probability"], 0.25)
        self.assertEqual(review["fusion"]["deception_weight"], 0.75)
        self.assertIn("mean_deception_probability", result["aggregate"])

    @patch("eWOM.fusion.predictor.DeceptionPredictor")
    @patch("eWOM.fusion.predictor.SentimentPredictor")
    @patch("eWOM.fusion.predictor.HelpfulnessPredictor")
    def test_predict_many_falls_back_when_deception_is_unavailable(
        self,
        helpfulness_predictor_mock,
        sentiment_predictor_mock,
        deception_predictor_mock,
    ) -> None:
        helpfulness_predictor_mock.return_value.predict_many.return_value = [
            {
                "usefulness_probability": 0.85,
                "is_useful": True,
            }
        ]
        sentiment_predictor_mock.return_value.predict_many.return_value = [
            {
                "negative_probability": 0.3,
                "positive_probability": 0.7,
                "predicted_label": 1,
                "predicted_label_text": "positive",
            }
        ]
        deception_predictor_mock.return_value.predict_many.return_value = [
            {
                "status": "unavailable",
                "source": "experiment_trust_fake_reviews",
                "deception_probability": None,
                "authenticity_probability": None,
                "trust_probability": None,
                "is_deceptive": None,
                "graph_uncertainty_entropy": None,
                "overall_confidence": None,
                "error": {
                    "type": "EnvironmentValidationError",
                    "message": "missing artifacts",
                },
            }
        ]

        predictor = EWOMFusionPredictor(
            helpfulness_model_path="helpfulness-model.joblib",
            helpfulness_feature_builder_path="helpfulness-feature-builder.joblib",
            sentiment_model_path="sentiment-model.joblib",
            sentiment_feature_builder_path="sentiment-feature-builder.joblib",
        )

        result = predictor.predict_many(review_texts=["Works well."])

        review = result["reviews"][0]
        self.assertEqual(review["deception"]["status"], "unavailable")
        self.assertIsNone(review["fusion"]["deception_probability"])
        self.assertEqual(review["fusion"]["deception_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
