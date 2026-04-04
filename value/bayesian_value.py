from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

from .bayes import DiscreteBayesNode, DiscreteBayesianNetwork


THREE_STATES = ("low", "medium", "high")
POLARITY_STATES = ("negative", "mixed", "positive")
STRENGTH_STATES = ("weak", "medium", "strong")
PRICE_STATES = ("much_pricier", "pricier", "fair", "cheaper", "much_cheaper")
TARGET_STATES = ("no", "yes")


@dataclass(frozen=True)
class BayesianValueInput:
    trust_probability: float | None = None
    ewom_score_0_to_100: float | None = None
    ewom_magnitude_0_to_100: float | None = None
    average_rating: float | None = None
    rating_count: float | None = None
    verified_purchase_rate: float | None = None
    price: float | None = None
    peer_price: float | None = None
    warranty_months: float | None = None
    return_window_days: float | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BayesianValueInput":
        return cls(
            trust_probability=_to_float(payload.get("trust_probability")),
            ewom_score_0_to_100=_to_float(payload.get("ewom_score_0_to_100")),
            ewom_magnitude_0_to_100=_to_float(payload.get("ewom_magnitude_0_to_100")),
            average_rating=_to_float(payload.get("average_rating")),
            rating_count=_to_float(payload.get("rating_count")),
            verified_purchase_rate=_to_float(payload.get("verified_purchase_rate")),
            price=_to_float(payload.get("price")),
            peer_price=_to_float(payload.get("peer_price")),
            warranty_months=_to_float(payload.get("warranty_months")),
            return_window_days=_to_float(payload.get("return_window_days")),
        )


def score_good_value_probability(
    raw_input: BayesianValueInput | Mapping[str, Any],
    *,
    network: DiscreteBayesianNetwork | None = None,
) -> dict[str, Any]:
    resolved_input = (
        raw_input if isinstance(raw_input, BayesianValueInput) else BayesianValueInput.from_mapping(raw_input)
    )
    evidence, derived = build_value_evidence(resolved_input)
    resolved_network = network or default_bayesian_value_network()

    good_value_posterior = resolved_network.posterior("GoodValueForMoney", evidence)
    component_nodes = (
        "Trustworthiness",
        "ReviewEvidence",
        "ProductQuality",
        "ServiceSupport",
    )
    component_posteriors = {
        node_name: resolved_network.posterior(node_name, evidence)
        for node_name in component_nodes
    }

    return {
        "good_value_probability": good_value_posterior["yes"],
        "posterior": good_value_posterior,
        "component_posteriors": component_posteriors,
        "most_likely_component_states": {
            node_name: max(posterior, key=posterior.get)
            for node_name, posterior in component_posteriors.items()
        },
        "evidence": evidence,
        "derived_metrics": derived,
    }


def build_value_evidence(raw_input: BayesianValueInput) -> tuple[dict[str, str], dict[str, float | None]]:
    evidence: dict[str, str] = {}

    trust_signal = _bucket_trust_probability(raw_input.trust_probability)
    if trust_signal is not None:
        evidence["TrustSignal"] = trust_signal

    review_polarity = _bucket_review_polarity(raw_input.ewom_score_0_to_100)
    if review_polarity is not None:
        evidence["ReviewPolarity"] = review_polarity

    review_strength = _bucket_review_strength(raw_input.ewom_magnitude_0_to_100)
    if review_strength is not None:
        evidence["ReviewStrength"] = review_strength

    rating_signal = _bucket_average_rating(raw_input.average_rating)
    if rating_signal is not None:
        evidence["RatingSignal"] = rating_signal

    review_volume = _bucket_review_volume(raw_input.rating_count)
    if review_volume is not None:
        evidence["ReviewVolume"] = review_volume

    verified_signal = _bucket_verified_rate(raw_input.verified_purchase_rate)
    if verified_signal is not None:
        evidence["VerifiedSignal"] = verified_signal

    relative_price_bucket, price_gap_vs_peer = _bucket_relative_price(
        raw_input.price,
        raw_input.peer_price,
    )
    if relative_price_bucket is not None:
        evidence["RelativePriceBucket"] = relative_price_bucket

    warranty_signal = _bucket_warranty(raw_input.warranty_months)
    if warranty_signal is not None:
        evidence["WarrantySignal"] = warranty_signal

    return_signal = _bucket_return_window(raw_input.return_window_days)
    if return_signal is not None:
        evidence["ReturnSignal"] = return_signal

    derived = {
        "price_gap_vs_peer": price_gap_vs_peer,
        "price": raw_input.price,
        "peer_price": raw_input.peer_price,
    }
    return evidence, derived


@lru_cache(maxsize=1)
def default_bayesian_value_network() -> DiscreteBayesianNetwork:
    nodes = [
        _root_node(
            "TrustSignal",
            THREE_STATES,
            {"low": 0.25, "medium": 0.45, "high": 0.30},
        ),
        _root_node(
            "ReviewPolarity",
            POLARITY_STATES,
            {"negative": 0.20, "mixed": 0.35, "positive": 0.45},
        ),
        _root_node(
            "ReviewStrength",
            STRENGTH_STATES,
            {"weak": 0.30, "medium": 0.45, "strong": 0.25},
        ),
        _root_node(
            "RatingSignal",
            THREE_STATES,
            {"low": 0.20, "medium": 0.35, "high": 0.45},
        ),
        _root_node(
            "ReviewVolume",
            THREE_STATES,
            {"low": 0.30, "medium": 0.40, "high": 0.30},
        ),
        _root_node(
            "VerifiedSignal",
            THREE_STATES,
            {"low": 0.20, "medium": 0.35, "high": 0.45},
        ),
        _root_node(
            "RelativePriceBucket",
            PRICE_STATES,
            {
                "much_pricier": 0.10,
                "pricier": 0.20,
                "fair": 0.40,
                "cheaper": 0.20,
                "much_cheaper": 0.10,
            },
        ),
        _root_node(
            "WarrantySignal",
            THREE_STATES,
            {"low": 0.25, "medium": 0.50, "high": 0.25},
        ),
        _root_node(
            "ReturnSignal",
            THREE_STATES,
            {"low": 0.20, "medium": 0.50, "high": 0.30},
        ),
        DiscreteBayesNode(
            name="Trustworthiness",
            states=THREE_STATES,
            parents=("TrustSignal",),
            cpt={
                ("low",): {"low": 0.75, "medium": 0.20, "high": 0.05},
                ("medium",): {"low": 0.20, "medium": 0.60, "high": 0.20},
                ("high",): {"low": 0.05, "medium": 0.25, "high": 0.70},
            },
        ),
        DiscreteBayesNode(
            name="ReviewEvidence",
            states=THREE_STATES,
            parents=("ReviewPolarity", "ReviewStrength", "ReviewVolume"),
            cpt=_build_review_evidence_cpt(),
        ),
        DiscreteBayesNode(
            name="ServiceSupport",
            states=THREE_STATES,
            parents=("WarrantySignal", "ReturnSignal"),
            cpt=_build_service_support_cpt(),
        ),
        DiscreteBayesNode(
            name="ProductQuality",
            states=THREE_STATES,
            parents=("Trustworthiness", "ReviewEvidence", "RatingSignal", "VerifiedSignal"),
            cpt=_build_product_quality_cpt(),
        ),
        DiscreteBayesNode(
            name="GoodValueForMoney",
            states=TARGET_STATES,
            parents=("ProductQuality", "RelativePriceBucket", "ServiceSupport"),
            cpt=_build_good_value_cpt(),
        ),
    ]
    return DiscreteBayesianNetwork(nodes)


def _build_review_evidence_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    polarity_score = {"negative": -1.0, "mixed": 0.0, "positive": 1.0}
    strength_scale = {"weak": 0.35, "medium": 0.70, "strong": 1.00}
    volume_bonus = {"low": 0.00, "medium": 0.20, "high": 0.35}

    for polarity in POLARITY_STATES:
        for strength in STRENGTH_STATES:
            for volume in THREE_STATES:
                score = 1.0
                score += polarity_score[polarity] * (0.70 + 0.30 * strength_scale[strength])
                score += volume_bonus[volume]
                cpt[(polarity, strength, volume)] = _three_state_distribution(score)
    return cpt


def _build_service_support_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    ordinal = {"low": 0.0, "medium": 1.0, "high": 2.0}

    for warranty in THREE_STATES:
        for returns in THREE_STATES:
            score = 0.55 * ordinal[warranty] + 0.45 * ordinal[returns]
            cpt[(warranty, returns)] = _three_state_distribution(score)
    return cpt


def _build_product_quality_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    centered = {"low": -1.0, "medium": 0.0, "high": 1.0}

    for trustworthiness in THREE_STATES:
        for review_evidence in THREE_STATES:
            for rating_signal in THREE_STATES:
                for verified_signal in THREE_STATES:
                    score = 1.0
                    score += 0.35 * centered[trustworthiness]
                    score += 0.40 * centered[review_evidence]
                    score += 0.50 * centered[rating_signal]
                    score += 0.15 * centered[verified_signal]
                    cpt[(
                        trustworthiness,
                        review_evidence,
                        rating_signal,
                        verified_signal,
                    )] = _three_state_distribution(score)
    return cpt


def _build_good_value_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    quality_weight = {"low": -1.2, "medium": 0.0, "high": 1.2}
    price_weight = {
        "much_pricier": -1.4,
        "pricier": -0.7,
        "fair": 0.0,
        "cheaper": 0.8,
        "much_cheaper": 1.4,
    }
    service_weight = {"low": -0.35, "medium": 0.0, "high": 0.35}

    for product_quality in THREE_STATES:
        for relative_price in PRICE_STATES:
            for service_support in THREE_STATES:
                logit = -0.10
                logit += 1.60 * quality_weight[product_quality]
                logit += 1.20 * price_weight[relative_price]
                logit += 0.40 * service_weight[service_support]
                p_yes = _sigmoid(logit)
                cpt[(product_quality, relative_price, service_support)] = {
                    "no": 1.0 - p_yes,
                    "yes": p_yes,
                }
    return cpt


def _root_node(
    name: str,
    states: tuple[str, ...],
    prior: dict[str, float],
) -> DiscreteBayesNode:
    return DiscreteBayesNode(name=name, states=states, parents=(), cpt={(): prior})


def _three_state_distribution(score: float) -> dict[str, float]:
    clipped_score = _clamp(score, 0.0, 2.0)
    weights = {
        "low": math.exp(-1.40 * (clipped_score - 0.0) ** 2),
        "medium": math.exp(-1.40 * (clipped_score - 1.0) ** 2),
        "high": math.exp(-1.40 * (clipped_score - 2.0) ** 2),
    }
    return _normalize(weights)


def _bucket_trust_probability(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 0.40:
        return "low"
    if value < 0.70:
        return "medium"
    return "high"


def _bucket_review_polarity(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 42.0:
        return "negative"
    if value <= 58.0:
        return "mixed"
    return "positive"


def _bucket_review_strength(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 25.0:
        return "weak"
    if value < 60.0:
        return "medium"
    return "strong"


def _bucket_average_rating(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 3.6:
        return "low"
    if value < 4.3:
        return "medium"
    return "high"


def _bucket_review_volume(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 20.0:
        return "low"
    if value < 200.0:
        return "medium"
    return "high"


def _bucket_verified_rate(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 0.50:
        return "low"
    if value < 0.80:
        return "medium"
    return "high"


def _bucket_relative_price(
    price: float | None,
    peer_price: float | None,
) -> tuple[str | None, float | None]:
    if price is None or peer_price is None or peer_price <= 0.0:
        return None, None

    price_gap = (peer_price - price) / peer_price
    if price_gap > 0.20:
        return "much_cheaper", price_gap
    if price_gap > 0.05:
        return "cheaper", price_gap
    if price_gap >= -0.05:
        return "fair", price_gap
    if price_gap >= -0.20:
        return "pricier", price_gap
    return "much_pricier", price_gap


def _bucket_warranty(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= 0.0:
        return "low"
    if value <= 12.0:
        return "medium"
    return "high"


def _bucket_return_window(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= 7.0:
        return "low"
    if value <= 30.0:
        return "medium"
    return "high"


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0.0:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}
    return {key: value / total for key, value in weights.items()}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)
