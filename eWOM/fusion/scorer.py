from __future__ import annotations

import math
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
