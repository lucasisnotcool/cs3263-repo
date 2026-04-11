from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any, Mapping

from .bayes import DiscreteBayesNode, DiscreteBayesianNetwork


THREE_STATES = ("low", "medium", "high")
TRUST_SIGNAL_STATES = ("very_low", "low", "medium", "high", "very_high")
POLARITY_STATES = (
    "very_negative",
    "negative",
    "mixed",
    "positive",
    "very_positive",
)
STRENGTH_STATES = ("weak", "medium", "strong")
PRICE_STATES = (
    "extreme_pricier",
    "much_pricier",
    "pricier",
    "fair",
    "cheaper",
    "much_cheaper",
    "extreme_cheaper",
)
TARGET_STATES = ("no", "yes")

# Final intended emphasis:
# - price context should be the strongest signal when it is trustworthy
# - trust and review evidence should matter similarly
# - all other auxiliary fields should only provide a light nudge
REVIEW_EVIDENCE_WEIGHTS = {
    "polarity": 0.80,
    "strength": 0.12,
    "volume": 0.08,
}
PRODUCT_QUALITY_WEIGHTS = {
    "trustworthiness": 0.4166666666666667,
    "review_evidence": 0.4166666666666667,
    "rating_signal": 0.12,
    "verified_signal": 0.04666666666666667,
}
GOOD_VALUE_WEIGHTS = {
    "product_quality": 0.60,
    "relative_price": 0.35,
    "service_support": 0.05,
}


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


def extract_ewom_bayesian_signals(ewom_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(ewom_payload, Mapping):
        raise TypeError("ewom_payload must be a mapping.")

    aggregate = ewom_payload.get("aggregate")
    if isinstance(aggregate, Mapping):
        return {
            "source_type": "review_set",
            "review_count": _to_int(ewom_payload.get("review_count")),
            "trust_probability": _resolve_review_set_trust_probability(
                ewom_payload=ewom_payload,
                aggregate=aggregate,
            ),
            "ewom_score_0_to_100": _to_float(
                aggregate.get("final_ewom_score_0_to_100")
            ),
            "ewom_magnitude_0_to_100": _to_float(
                aggregate.get("final_ewom_magnitude_0_to_100")
            ),
        }

    fusion = ewom_payload.get("fusion")
    if isinstance(fusion, Mapping):
        return {
            "source_type": "single_review",
            "review_count": 1,
            "trust_probability": _resolve_single_review_trust_probability(ewom_payload),
            "ewom_score_0_to_100": _to_float(fusion.get("ewom_score_0_to_100")),
            "ewom_magnitude_0_to_100": _to_float(
                fusion.get("ewom_magnitude_0_to_100")
            ),
        }

    raise ValueError(
        "ewom_payload must contain either an 'aggregate' review-set result or a "
        "'fusion' single-review result."
    )


def fuse_ewom_result_into_bayesian_input(
    base_input: BayesianValueInput | Mapping[str, Any] | None,
    ewom_payload: Mapping[str, Any],
    *,
    include_trust_probability: bool = True,
) -> tuple[BayesianValueInput, dict[str, Any]]:
    resolved_base_input = (
        BayesianValueInput()
        if base_input is None
        else (
            base_input
            if isinstance(base_input, BayesianValueInput)
            else BayesianValueInput.from_mapping(base_input)
        )
    )
    ewom_signals = extract_ewom_bayesian_signals(ewom_payload)

    fused_input = BayesianValueInput(
        trust_probability=(
            _prefer_non_null(
                ewom_signals.get("trust_probability"),
                resolved_base_input.trust_probability,
            )
            if include_trust_probability
            else resolved_base_input.trust_probability
        ),
        ewom_score_0_to_100=_prefer_non_null(
            ewom_signals.get("ewom_score_0_to_100"),
            resolved_base_input.ewom_score_0_to_100,
        ),
        ewom_magnitude_0_to_100=_prefer_non_null(
            ewom_signals.get("ewom_magnitude_0_to_100"),
            resolved_base_input.ewom_magnitude_0_to_100,
        ),
        average_rating=resolved_base_input.average_rating,
        rating_count=resolved_base_input.rating_count,
        verified_purchase_rate=resolved_base_input.verified_purchase_rate,
        price=resolved_base_input.price,
        peer_price=resolved_base_input.peer_price,
        warranty_months=resolved_base_input.warranty_months,
        return_window_days=resolved_base_input.return_window_days,
    )
    return fused_input, ewom_signals


def score_good_value_probability(
    raw_input: BayesianValueInput | Mapping[str, Any],
    *,
    network: DiscreteBayesianNetwork | None = None,
    default_relative_price_bucket: str | None = None,
) -> dict[str, Any]:
    resolved_input = (
        raw_input if isinstance(raw_input, BayesianValueInput) else BayesianValueInput.from_mapping(raw_input)
    )
    evidence, derived = build_value_evidence(
        resolved_input,
        default_relative_price_bucket=default_relative_price_bucket,
    )
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


def build_value_evidence(
    raw_input: BayesianValueInput,
    *,
    default_relative_price_bucket: str | None = None,
) -> tuple[dict[str, str], dict[str, float | None | str]]:
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
    relative_price_bucket_source: str | None = None
    if relative_price_bucket is not None:
        evidence["RelativePriceBucket"] = relative_price_bucket
        relative_price_bucket_source = "peer_price"
    elif default_relative_price_bucket in PRICE_STATES:
        evidence["RelativePriceBucket"] = str(default_relative_price_bucket)
        relative_price_bucket = str(default_relative_price_bucket)
        relative_price_bucket_source = "default"

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
        "relative_price_bucket": relative_price_bucket,
        "relative_price_bucket_source": relative_price_bucket_source,
    }
    return evidence, derived


@lru_cache(maxsize=1)
def default_bayesian_value_network() -> DiscreteBayesianNetwork:
    nodes = [
        _root_node(
            "TrustSignal",
            TRUST_SIGNAL_STATES,
            {
                "very_low": 0.08,
                "low": 0.17,
                "medium": 0.30,
                "high": 0.27,
                "very_high": 0.18,
            },
        ),
        _root_node(
            "ReviewPolarity",
            POLARITY_STATES,
            {
                "very_negative": 0.08,
                "negative": 0.16,
                "mixed": 0.30,
                "positive": 0.26,
                "very_positive": 0.20,
            },
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
                "extreme_pricier": 0.04,
                "much_pricier": 0.08,
                "pricier": 0.18,
                "fair": 0.40,
                "cheaper": 0.18,
                "much_cheaper": 0.08,
                "extreme_cheaper": 0.04,
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
            cpt=_build_trustworthiness_cpt(),
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


def _build_trustworthiness_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    signal_score = {
        "very_low": 0.00,
        "low": 0.25,
        "medium": 0.50,
        "high": 0.75,
        "very_high": 1.00,
    }

    for trust_signal in TRUST_SIGNAL_STATES:
        cpt[(trust_signal,)] = _three_state_distribution(
            2.0 * signal_score[trust_signal]
        )
    return cpt


def _build_review_evidence_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    polarity_score = {
        "very_negative": 0.00,
        "negative": 0.25,
        "mixed": 0.50,
        "positive": 0.75,
        "very_positive": 1.00,
    }
    strength_scale = {"weak": 0.00, "medium": 0.50, "strong": 1.00}
    volume_bonus = {"low": 0.00, "medium": 0.50, "high": 1.00}

    for polarity in POLARITY_STATES:
        for strength in STRENGTH_STATES:
            for volume in THREE_STATES:
                combined_score = (
                    REVIEW_EVIDENCE_WEIGHTS["polarity"] * polarity_score[polarity]
                    + REVIEW_EVIDENCE_WEIGHTS["strength"] * strength_scale[strength]
                    + REVIEW_EVIDENCE_WEIGHTS["volume"] * volume_bonus[volume]
                )
                cpt[(polarity, strength, volume)] = _three_state_distribution(
                    2.0 * combined_score
                )
    return cpt


def _build_service_support_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    ordinal = {"low": 0.0, "medium": 0.5, "high": 1.0}

    for warranty in THREE_STATES:
        for returns in THREE_STATES:
            score = (0.55 * ordinal[warranty]) + (0.45 * ordinal[returns])
            cpt[(warranty, returns)] = _three_state_distribution(2.0 * score)
    return cpt


def _build_product_quality_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    normalized = {"low": 0.0, "medium": 0.5, "high": 1.0}

    for trustworthiness in THREE_STATES:
        for review_evidence in THREE_STATES:
            for rating_signal in THREE_STATES:
                for verified_signal in THREE_STATES:
                    score = (
                        PRODUCT_QUALITY_WEIGHTS["trustworthiness"]
                        * normalized[trustworthiness]
                        + PRODUCT_QUALITY_WEIGHTS["review_evidence"]
                        * normalized[review_evidence]
                        + PRODUCT_QUALITY_WEIGHTS["rating_signal"]
                        * normalized[rating_signal]
                        + PRODUCT_QUALITY_WEIGHTS["verified_signal"]
                        * normalized[verified_signal]
                    )
                    cpt[(
                        trustworthiness,
                        review_evidence,
                        rating_signal,
                        verified_signal,
                    )] = _three_state_distribution(2.0 * score)
    return cpt


def _build_good_value_cpt() -> dict[tuple[str, ...], dict[str, float]]:
    cpt: dict[tuple[str, ...], dict[str, float]] = {}
    quality_weight = {"low": 0.0, "medium": 0.5, "high": 1.0}
    price_weight = {
        "extreme_pricier": 0.00,
        "much_pricier": 0.15,
        "pricier": 0.35,
        "fair": 0.50,
        "cheaper": 0.65,
        "much_cheaper": 0.82,
        "extreme_cheaper": 1.00,
    }
    service_weight = {"low": 0.0, "medium": 0.5, "high": 1.0}

    for product_quality in THREE_STATES:
        for relative_price in PRICE_STATES:
            for service_support in THREE_STATES:
                combined_score = (
                    GOOD_VALUE_WEIGHTS["product_quality"]
                    * quality_weight[product_quality]
                    + GOOD_VALUE_WEIGHTS["relative_price"]
                    * price_weight[relative_price]
                    + GOOD_VALUE_WEIGHTS["service_support"]
                    * service_weight[service_support]
                )
                p_yes = _sigmoid((combined_score - 0.50) * 5.0)
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
    if value < 0.20:
        return "very_low"
    if value < 0.40:
        return "low"
    if value < 0.60:
        return "medium"
    if value < 0.82:
        return "high"
    return "very_high"


def _bucket_review_polarity(value: float | None) -> str | None:
    if value is None:
        return None
    if value < 20.0:
        return "very_negative"
    if value < 40.0:
        return "negative"
    if value < 60.0:
        return "mixed"
    if value < 80.0:
        return "positive"
    return "very_positive"


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
    if price_gap > 0.40:
        return "extreme_cheaper", price_gap
    if price_gap > 0.20:
        return "much_cheaper", price_gap
    if price_gap > 0.05:
        return "cheaper", price_gap
    if price_gap >= -0.05:
        return "fair", price_gap
    if price_gap >= -0.20:
        return "pricier", price_gap
    if price_gap >= -0.40:
        return "much_pricier", price_gap
    return "extreme_pricier", price_gap


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


def _to_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric >= 0 else None


def _resolve_review_set_trust_probability(
    *,
    ewom_payload: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> float | None:
    mean_deception_probability = _to_float(aggregate.get("mean_deception_probability"))
    if mean_deception_probability is not None:
        return _clamp(1.0 - mean_deception_probability, 0.0, 1.0)

    reviews = ewom_payload.get("reviews")
    if not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes)):
        return None

    trust_probabilities: list[float] = []
    for review in reviews:
        if not isinstance(review, Mapping):
            continue
        trust_probability = _resolve_single_review_trust_probability(review)
        if trust_probability is not None:
            trust_probabilities.append(trust_probability)

    if not trust_probabilities:
        return None
    return sum(trust_probabilities) / len(trust_probabilities)


def _resolve_single_review_trust_probability(payload: Mapping[str, Any]) -> float | None:
    deception = payload.get("deception")
    if not isinstance(deception, Mapping):
        return None

    trust_probability = _to_float(deception.get("trust_probability"))
    if trust_probability is not None:
        return _clamp(trust_probability, 0.0, 1.0)

    deception_probability = _to_float(deception.get("deception_probability"))
    if deception_probability is None:
        return None
    return _clamp(1.0 - deception_probability, 0.0, 1.0)


def _prefer_non_null(primary: float | None, fallback: float | None) -> float | None:
    return primary if primary is not None else fallback


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
