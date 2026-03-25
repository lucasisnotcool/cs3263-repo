from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


VALUE_COMPARISON_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["listing_a", "listing_b"],
    "properties": {
        "listing_a": {"$ref": "#/$defs/listing"},
        "listing_b": {"$ref": "#/$defs/listing"},
        "required_spec_keys": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
    "$defs": {
        "listing": {
            "type": "object",
            "additionalProperties": False,
            "required": ["url"],
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "platform": {"type": "string", "default": ""},
                "title": {"type": "string", "default": ""},
                "currency": {"type": "string", "default": ""},
                "base_price": {"type": "number"},
                "shipping_fee": {"type": "number", "default": 0.0},
                "platform_discount": {"type": "number", "default": 0.0},
                "seller_discount": {"type": "number", "default": 0.0},
                "voucher_discount": {"type": "number", "default": 0.0},
                "total_price": {"type": "number"},
                "delivery_days": {"type": "number"},
                "warranty_months": {"type": "number"},
                "return_window_days": {"type": "number"},
                "specs": {
                    "type": "object",
                    "default": {},
                    "additionalProperties": {
                        "type": ["string", "number", "boolean", "null"]
                    },
                },
            },
        }
    },
}

VALUE_COMPARISON_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "value_score_a",
        "value_score_b",
        "confidence",
        "verdict",
        "reasons",
        "evidence",
        "missing_fields",
    ],
    "properties": {
        "value_score_a": {"type": "number"},
        "value_score_b": {"type": "number"},
        "confidence": {"type": "number"},
        "verdict": {
            "type": "string",
            "enum": ["better_A", "better_B", "tie", "insufficient_evidence"],
        },
        "reasons": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "object"},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass(frozen=True)
class ValueAgentConfig:
    # Fixed defaults for v1 (de-biasing oriented, no user personalization).
    weight_cost: float = 0.60
    weight_spec: float = 0.25
    weight_service: float = 0.15
    tie_margin: float = 3.0
    min_confidence_for_decision: float = 0.45
    low_comparability_threshold: float = 0.60


def compare_listings(
    payload: Mapping[str, Any],
    config: ValueAgentConfig | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping with listing_a and listing_b")

    cfg = config or ValueAgentConfig()
    missing_fields: list[str] = []

    listing_a = _normalize_listing(payload.get("listing_a"), "a", missing_fields)
    listing_b = _normalize_listing(payload.get("listing_b"), "b", missing_fields)

    required_spec_keys_raw = payload.get("required_spec_keys", [])
    required_spec_keys = _normalize_required_spec_keys(required_spec_keys_raw)

    cost = _score_cost(listing_a, listing_b, missing_fields)
    spec = _score_spec(listing_a, listing_b, required_spec_keys, missing_fields)
    service = _score_service(listing_a, listing_b, missing_fields)

    value_score_a = (
        cfg.weight_cost * cost["score_a"]
        + cfg.weight_spec * spec["score_a"]
        + cfg.weight_service * service["score_a"]
    )
    value_score_b = (
        cfg.weight_cost * cost["score_b"]
        + cfg.weight_spec * spec["score_b"]
        + cfg.weight_service * service["score_b"]
    )

    comparability_score = _comparability_score(
        listing_a=listing_a,
        listing_b=listing_b,
        required_spec_keys=required_spec_keys,
    )
    confidence = _confidence_score(
        missing_fields=missing_fields,
        comparability_score=comparability_score,
        spec_coverage=spec["coverage_avg"],
        currency_mismatch=_currency_mismatch(listing_a, listing_b),
        low_comparability_threshold=cfg.low_comparability_threshold,
    )
    verdict = _resolve_verdict(value_score_a, value_score_b, confidence, cfg)

    reasons = _build_reasons(
        listing_a=listing_a,
        listing_b=listing_b,
        cost=cost,
        spec=spec,
        service=service,
        missing_fields=missing_fields,
        confidence=confidence,
        comparability_score=comparability_score,
        config=cfg,
        verdict=verdict,
    )

    return {
        "value_score_a": round(value_score_a, 2),
        "value_score_b": round(value_score_b, 2),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "reasons": reasons,
        "evidence": {
            "a": {
                "url": listing_a["url"],
                "platform": listing_a["platform"],
                "title": listing_a["title"],
                "currency": listing_a["currency"],
                "total_price": listing_a["total_price"],
                "component_scores": {
                    "cost": round(cost["score_a"], 2),
                    "spec": round(spec["score_a"], 2),
                    "service": round(service["score_a"], 2),
                },
            },
            "b": {
                "url": listing_b["url"],
                "platform": listing_b["platform"],
                "title": listing_b["title"],
                "currency": listing_b["currency"],
                "total_price": listing_b["total_price"],
                "component_scores": {
                    "cost": round(cost["score_b"], 2),
                    "spec": round(spec["score_b"], 2),
                    "service": round(service["score_b"], 2),
                },
            },
            "comparability_score": round(comparability_score, 3),
            "weights": {
                "cost": cfg.weight_cost,
                "spec": cfg.weight_spec,
                "service": cfg.weight_service,
            },
        },
        "missing_fields": sorted(set(missing_fields)),
    }


def _normalize_listing(
    raw_listing: Any,
    side: str,
    missing_fields: list[str],
) -> dict[str, Any]:
    if not isinstance(raw_listing, Mapping):
        raise TypeError(f"listing_{side} must be an object")

    url = _safe_text(raw_listing.get("url"))
    if not url:
        raise ValueError(f"listing_{side}.url is required")

    listing: dict[str, Any] = {
        "url": url,
        "platform": _safe_text(raw_listing.get("platform")),
        "title": _safe_text(raw_listing.get("title")),
        "currency": _safe_text(raw_listing.get("currency")).upper(),
        "base_price": _to_float(raw_listing.get("base_price")),
        "shipping_fee": _to_float(raw_listing.get("shipping_fee")),
        "platform_discount": _to_float(raw_listing.get("platform_discount")),
        "seller_discount": _to_float(raw_listing.get("seller_discount")),
        "voucher_discount": _to_float(raw_listing.get("voucher_discount")),
        "total_price": _to_float(raw_listing.get("total_price")),
        "delivery_days": _to_float(raw_listing.get("delivery_days")),
        "warranty_months": _to_float(raw_listing.get("warranty_months")),
        "return_window_days": _to_float(raw_listing.get("return_window_days")),
        "specs": _normalize_specs(raw_listing.get("specs")),
        "estimated_total_price": False,
    }

    if listing["total_price"] is None and listing["base_price"] is not None:
        shipping_fee = listing["shipping_fee"] or 0.0
        platform_discount = listing["platform_discount"] or 0.0
        seller_discount = listing["seller_discount"] or 0.0
        voucher_discount = listing["voucher_discount"] or 0.0
        listing["total_price"] = max(
            0.0,
            listing["base_price"]
            + shipping_fee
            - platform_discount
            - seller_discount
            - voucher_discount,
        )
        listing["estimated_total_price"] = True

    if listing["total_price"] is None:
        missing_fields.append(f"{side}.total_price")
    if not listing["specs"]:
        missing_fields.append(f"{side}.specs")

    return listing


def _normalize_specs(raw_specs: Any) -> dict[str, Any]:
    if not isinstance(raw_specs, Mapping):
        return {}
    specs: dict[str, Any] = {}
    for raw_key, raw_value in raw_specs.items():
        key = _safe_text(raw_key)
        if key:
            specs[key] = raw_value
    return specs


def _normalize_required_spec_keys(raw_keys: Any) -> list[str]:
    if raw_keys is None:
        return []
    if not isinstance(raw_keys, list):
        raise TypeError("required_spec_keys must be a list")

    keys: list[str] = []
    for raw_key in raw_keys:
        key = _safe_text(raw_key)
        if key and key not in keys:
            keys.append(key)
    return keys


def _score_cost(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    missing_fields: list[str],
) -> dict[str, float]:
    price_a = _to_float(listing_a.get("total_price"))
    price_b = _to_float(listing_b.get("total_price"))
    score_a, score_b = _pairwise_relative(
        a=price_a, b=price_b, higher_is_better=False
    )

    if listing_a.get("estimated_total_price"):
        missing_fields.append("a.total_price_estimated")
    if listing_b.get("estimated_total_price"):
        missing_fields.append("b.total_price_estimated")

    return {"score_a": score_a, "score_b": score_b}


def _score_spec(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    required_spec_keys: list[str],
    missing_fields: list[str],
) -> dict[str, float]:
    specs_a = listing_a.get("specs", {})
    specs_b = listing_b.get("specs", {})
    if not required_spec_keys:
        keys = sorted(set(specs_a.keys()) | set(specs_b.keys()))
    else:
        keys = required_spec_keys

    if not keys:
        return {"score_a": 50.0, "score_b": 50.0, "coverage_avg": 0.0}

    present_a = 0
    present_b = 0
    advantage_a = 0.0
    advantage_b = 0.0
    compared = 0

    for key in keys:
        has_a = key in specs_a and specs_a[key] not in {None, ""}
        has_b = key in specs_b and specs_b[key] not in {None, ""}

        if has_a:
            present_a += 1
        else:
            missing_fields.append(f"a.specs.{key}")
        if has_b:
            present_b += 1
        else:
            missing_fields.append(f"b.specs.{key}")

        if not (has_a and has_b):
            continue

        a_value = specs_a[key]
        b_value = specs_b[key]
        a_score, b_score = _compare_spec_values(a_value, b_value)
        advantage_a += a_score
        advantage_b += b_score
        compared += 1

    coverage_a = present_a / len(keys)
    coverage_b = present_b / len(keys)
    coverage_avg = (coverage_a + coverage_b) / 2.0

    if compared == 0:
        advantage_a = 50.0
        advantage_b = 50.0
    else:
        advantage_a /= compared
        advantage_b /= compared

    # Emphasize coverage first, then pairwise advantage.
    score_a = 100.0 * (0.7 * coverage_a + 0.3 * (advantage_a / 100.0))
    score_b = 100.0 * (0.7 * coverage_b + 0.3 * (advantage_b / 100.0))

    return {
        "score_a": _clamp(score_a, 0.0, 100.0),
        "score_b": _clamp(score_b, 0.0, 100.0),
        "coverage_avg": coverage_avg,
    }


def _score_service(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    missing_fields: list[str],
) -> dict[str, float]:
    metrics = [
        ("delivery_days", False),
        ("warranty_months", True),
        ("return_window_days", True),
    ]

    score_a = 0.0
    score_b = 0.0
    for metric, higher_is_better in metrics:
        a_value = _to_float(listing_a.get(metric))
        b_value = _to_float(listing_b.get(metric))
        pair_a, pair_b = _pairwise_relative(
            a=a_value, b=b_value, higher_is_better=higher_is_better
        )
        score_a += pair_a
        score_b += pair_b

        if a_value is None:
            missing_fields.append(f"a.{metric}")
        if b_value is None:
            missing_fields.append(f"b.{metric}")

    return {"score_a": score_a / 3.0, "score_b": score_b / 3.0}


def _pairwise_relative(
    a: float | None,
    b: float | None,
    higher_is_better: bool,
) -> tuple[float, float]:
    if a is None and b is None:
        return 50.0, 50.0
    if a is None:
        return 45.0, 55.0
    if b is None:
        return 55.0, 45.0

    if higher_is_better:
        max_value = max(a, b, 1e-9)
        return (
            _clamp(100.0 * a / max_value, 0.0, 100.0),
            _clamp(100.0 * b / max_value, 0.0, 100.0),
        )

    a = max(a, 1e-9)
    b = max(b, 1e-9)
    min_value = min(a, b)
    return (
        _clamp(100.0 * min_value / a, 0.0, 100.0),
        _clamp(100.0 * min_value / b, 0.0, 100.0),
    )


def _compare_spec_values(a_value: Any, b_value: Any) -> tuple[float, float]:
    a_num = _to_float(a_value)
    b_num = _to_float(b_value)

    # For numeric specs in v1, "higher is better" is the default assumption.
    if a_num is not None and b_num is not None:
        return _pairwise_relative(a_num, b_num, higher_is_better=True)

    a_text = _normalize_token_text(_safe_text(a_value))
    b_text = _normalize_token_text(_safe_text(b_value))
    if a_text == b_text and a_text:
        return 100.0, 100.0
    return 50.0, 50.0


def _confidence_score(
    *,
    missing_fields: list[str],
    comparability_score: float,
    spec_coverage: float,
    currency_mismatch: bool,
    low_comparability_threshold: float,
) -> float:
    confidence = 1.0
    confidence -= min(0.55, 0.03 * len(set(missing_fields)))

    if "a.total_price" in missing_fields:
        confidence -= 0.12
    if "b.total_price" in missing_fields:
        confidence -= 0.12
    if "a.total_price_estimated" in missing_fields:
        confidence -= 0.06
    if "b.total_price_estimated" in missing_fields:
        confidence -= 0.06

    if currency_mismatch:
        confidence -= 0.20

    if comparability_score < low_comparability_threshold:
        confidence -= (
            0.30
            * (low_comparability_threshold - comparability_score)
            / max(low_comparability_threshold, 1e-9)
        )

    if spec_coverage < 0.80:
        confidence -= 0.20 * (0.80 - spec_coverage) / 0.80

    return _clamp(confidence, 0.0, 1.0)


def _resolve_verdict(
    value_score_a: float,
    value_score_b: float,
    confidence: float,
    config: ValueAgentConfig,
) -> str:
    if confidence < config.min_confidence_for_decision:
        return "insufficient_evidence"

    delta = value_score_a - value_score_b
    if abs(delta) <= config.tie_margin:
        return "tie"
    return "better_A" if delta > 0 else "better_B"


def _build_reasons(
    *,
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    cost: Mapping[str, float],
    spec: Mapping[str, float],
    service: Mapping[str, float],
    missing_fields: list[str],
    confidence: float,
    comparability_score: float,
    config: ValueAgentConfig,
    verdict: str,
) -> list[str]:
    reasons: list[str] = []

    price_a = listing_a.get("total_price")
    price_b = listing_b.get("total_price")
    if isinstance(price_a, (int, float)) and isinstance(price_b, (int, float)):
        if price_a < price_b:
            saving = 100.0 * (price_b - price_a) / max(price_b, 1e-9)
            reasons.append(f"Listing A is cheaper by about {saving:.1f}% on total price.")
        elif price_b < price_a:
            saving = 100.0 * (price_a - price_b) / max(price_a, 1e-9)
            reasons.append(f"Listing B is cheaper by about {saving:.1f}% on total price.")
        else:
            reasons.append("Both listings have the same total price.")

    spec_gap = spec["score_a"] - spec["score_b"]
    if abs(spec_gap) >= 2.0:
        reasons.append(
            "Specification coverage/value is stronger for listing "
            + ("A." if spec_gap > 0 else "B.")
        )

    service_gap = service["score_a"] - service["score_b"]
    if abs(service_gap) >= 2.0:
        reasons.append(
            "Service terms (delivery/returns/warranty) are better for listing "
            + ("A." if service_gap > 0 else "B.")
        )

    unique_missing = sorted(set(missing_fields))
    if unique_missing:
        sample = ", ".join(unique_missing[:4])
        if len(unique_missing) > 4:
            sample += ", ..."
        reasons.append(f"Confidence reduced due to missing data: {sample}.")

    if comparability_score < config.low_comparability_threshold:
        reasons.append(
            f"Low comparability score ({comparability_score:.2f}) may indicate product mismatch."
        )

    if verdict == "insufficient_evidence":
        reasons.append(
            f"Confidence ({confidence:.2f}) is below threshold ({config.min_confidence_for_decision:.2f})."
        )

    if not reasons:
        reasons.append("Listings are very close on available value signals.")

    return reasons[:5]


def _comparability_score(
    *,
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    required_spec_keys: list[str],
) -> float:
    title_similarity = _jaccard_similarity(
        _title_tokens(_safe_text(listing_a.get("title"))),
        _title_tokens(_safe_text(listing_b.get("title"))),
    )

    specs_a = listing_a.get("specs", {})
    specs_b = listing_b.get("specs", {})

    if required_spec_keys:
        keys = [key for key in required_spec_keys if key in specs_a or key in specs_b]
    else:
        keys = sorted(set(specs_a.keys()) | set(specs_b.keys()))

    if not keys:
        return title_similarity

    matched = 0
    compared = 0
    for key in keys:
        if key not in specs_a or key not in specs_b:
            continue
        compared += 1
        if _normalize_token_text(_safe_text(specs_a[key])) == _normalize_token_text(
            _safe_text(specs_b[key])
        ):
            matched += 1

    spec_match_ratio = matched / compared if compared > 0 else 0.5
    return _clamp(0.5 * title_similarity + 0.5 * spec_match_ratio, 0.0, 1.0)


def _title_tokens(title: str) -> set[str]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "new",
        "original",
        "official",
        "authentic",
        "sale",
        "free",
        "shipping",
    }
    tokens = _normalize_token_text(title).split(" ")
    return {
        token
        for token in tokens
        if token and token not in stop_words and len(token) > 1
    }


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.5
    union = a | b
    if not union:
        return 0.5
    return len(a & b) / len(union)


def _currency_mismatch(listing_a: Mapping[str, Any], listing_b: Mapping[str, Any]) -> bool:
    currency_a = _safe_text(listing_a.get("currency")).upper()
    currency_b = _safe_text(listing_b.get("currency")).upper()
    if not currency_a or not currency_b:
        return False
    return currency_a != currency_b


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_token_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(number) or math.isnan(number):
        return None
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
