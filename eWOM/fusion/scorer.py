from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)

    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


@dataclass(frozen=True)
class EWOMFusionConfig:
    helpfulness_gate_center: float = 0.5
    helpfulness_gate_sharpness: float = 8.0
    review_set_gate_scale: float = 3.0


class EWOMFusionScorer:
    """
    Fuses late-stage helpfulness and sentiment probabilities into an explainable
    eWOM score.

    The helpfulness branch acts as a soft gate: reviews predicted as unhelpful
    contribute little even when sentiment is strong. The sentiment branch
    supplies the sign and strength of the final score.
    """

    def __init__(self, config: EWOMFusionConfig | None = None):
        self.config = config or EWOMFusionConfig()

    def score(
        self,
        *,
        usefulness_probability: float,
        positive_probability: float,
        negative_probability: float,
    ) -> dict[str, float]:
        usefulness_probability = _clamp_probability(usefulness_probability)
        positive_probability = max(0.0, float(positive_probability))
        negative_probability = max(0.0, float(negative_probability))

        total_sentiment_probability = positive_probability + negative_probability
        if total_sentiment_probability > 0:
            positive_probability /= total_sentiment_probability
            negative_probability /= total_sentiment_probability
        else:
            positive_probability = 0.5
            negative_probability = 0.5

        positive_probability = _clamp_probability(positive_probability)
        negative_probability = _clamp_probability(negative_probability)

        helpfulness_gate = _sigmoid(
            self.config.helpfulness_gate_sharpness
            * (usefulness_probability - self.config.helpfulness_gate_center)
        )
        sentiment_polarity = positive_probability - negative_probability
        sentiment_strength = abs(sentiment_polarity)
        signed_ewom_score = helpfulness_gate * sentiment_polarity
        magnitude_ewom_score = helpfulness_gate * sentiment_strength

        return {
            "usefulness_probability": usefulness_probability,
            "helpfulness_gate": helpfulness_gate,
            "positive_probability": positive_probability,
            "negative_probability": negative_probability,
            "sentiment_polarity": sentiment_polarity,
            "sentiment_strength": sentiment_strength,
            "signed_ewom_score": signed_ewom_score,
            "magnitude_ewom_score": magnitude_ewom_score,
            "ewom_score_0_to_100": 50.0 * (signed_ewom_score + 1.0),
            "ewom_magnitude_0_to_100": 100.0 * magnitude_ewom_score,
        }

    def aggregate(
        self,
        review_scores: Sequence[Mapping[str, float]],
    ) -> dict[str, float]:
        if not review_scores:
            raise ValueError("review_scores must contain at least one review.")

        normalized_scores = [self._normalize_review_score(score) for score in review_scores]
        review_count = len(normalized_scores)
        informative_review_weight = sum(
            score["helpfulness_gate"] for score in normalized_scores
        )

        mean_usefulness_probability = sum(
            score["usefulness_probability"] for score in normalized_scores
        ) / review_count
        mean_helpfulness_gate = informative_review_weight / review_count

        if informative_review_weight > 0:
            weighted_positive_probability = (
                sum(
                    score["helpfulness_gate"] * score["positive_probability"]
                    for score in normalized_scores
                )
                / informative_review_weight
            )
            weighted_negative_probability = (
                sum(
                    score["helpfulness_gate"] * score["negative_probability"]
                    for score in normalized_scores
                )
                / informative_review_weight
            )
            weighted_sentiment_polarity = (
                sum(
                    score["helpfulness_gate"] * score["sentiment_polarity"]
                    for score in normalized_scores
                )
                / informative_review_weight
            )
            weighted_sentiment_strength = (
                sum(
                    score["helpfulness_gate"] * score["sentiment_strength"]
                    for score in normalized_scores
                )
                / informative_review_weight
            )
        else:
            weighted_positive_probability = 0.5
            weighted_negative_probability = 0.5
            weighted_sentiment_polarity = 0.0
            weighted_sentiment_strength = 0.0

        review_set_gate_scale = max(float(self.config.review_set_gate_scale), 1e-9)
        review_set_gate = 1.0 - math.exp(-review_count / review_set_gate_scale)
        final_signed_ewom_score = review_set_gate * weighted_sentiment_polarity
        final_magnitude_ewom_score = review_set_gate * weighted_sentiment_strength

        return {
            "review_count": review_count,
            "informative_review_weight": informative_review_weight,
            "mean_usefulness_probability": mean_usefulness_probability,
            "mean_helpfulness_gate": mean_helpfulness_gate,
            "weighted_positive_probability": weighted_positive_probability,
            "weighted_negative_probability": weighted_negative_probability,
            "weighted_sentiment_polarity": weighted_sentiment_polarity,
            "weighted_sentiment_strength": weighted_sentiment_strength,
            "review_set_gate": review_set_gate,
            "final_signed_ewom_score": final_signed_ewom_score,
            "final_magnitude_ewom_score": final_magnitude_ewom_score,
            "final_ewom_score_0_to_100": 50.0 * (final_signed_ewom_score + 1.0),
            "final_ewom_magnitude_0_to_100": 100.0 * final_magnitude_ewom_score,
        }

    def _normalize_review_score(self, review_score: Mapping[str, float]) -> dict[str, float]:
        return {
            "usefulness_probability": _clamp_probability(
                review_score["usefulness_probability"]
            ),
            "helpfulness_gate": _clamp_probability(review_score["helpfulness_gate"]),
            "positive_probability": _clamp_probability(
                review_score["positive_probability"]
            ),
            "negative_probability": _clamp_probability(
                review_score["negative_probability"]
            ),
            "sentiment_polarity": max(
                -1.0,
                min(1.0, float(review_score["sentiment_polarity"])),
            ),
            "sentiment_strength": _clamp_probability(
                review_score["sentiment_strength"]
            ),
        }
