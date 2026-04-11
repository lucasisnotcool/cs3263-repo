from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from core.entities.candidate import Candidate
from experiment_trust_fake_reviews import DeployConfig, TrustFakeReviewsDeployPipeline


DEFAULT_SOURCE = "experiment_trust_fake_reviews"


def build_candidate_listing_payload(
    candidate: Candidate | Mapping[str, Any],
) -> dict[str, Any]:
    title = _clean_text(_candidate_value(candidate, "title"))
    description = _build_listing_description(candidate)
    bullet_points = _build_listing_bullet_points(candidate)
    product_id = _clean_text(
        _candidate_value(candidate, "product_id")
        or _candidate_value(candidate, "legacy_item_id")
        or _candidate_value(candidate, "source_url")
    )

    if not any([title, bullet_points, description]):
        raise ValueError("Candidate is missing listing content for trust scoring.")

    return {
        "product_id": product_id or None,
        "title": title or None,
        "bullet_points": bullet_points or None,
        "description": description or None,
    }


def build_listing_trust_runtime_status(
    config: DeployConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or DeployConfig()
    pipeline = _get_trust_pipeline(resolved_config)
    environment = pipeline.validate_environment()
    ollama = environment.get("ollama", {}) if isinstance(environment, Mapping) else {}
    return {
        "ready": bool(environment.get("ok", False)),
        "provider": "ollama",
        "binary_path": ollama.get("binary_path"),
        "service_reachable": bool(ollama.get("service_reachable", False)),
        "cli_reachable": bool(ollama.get("cli_reachable", False)),
        "api_models": list(ollama.get("api_models", [])),
        "cli_models": list(ollama.get("cli_models", [])),
        "available_models": list(ollama.get("available_models", [])),
        "model_required": ollama.get("model_required"),
        "model_present": bool(ollama.get("model_present", False)),
        "api_error": ollama.get("api_error"),
        "cli_error": ollama.get("cli_error"),
        "errors": list(environment.get("errors", [])),
        "warnings": list(environment.get("warnings", [])),
    }


def score_candidate_listing_trust(
    candidate: Candidate | Mapping[str, Any],
    *,
    config: DeployConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or DeployConfig()
    payload = build_candidate_listing_payload(candidate)
    pipeline = _get_trust_pipeline(resolved_config)
    result = pipeline.run([payload], raise_on_environment_error=False)
    environment = result.get("environment", {})

    if not environment.get("ok", False) and not result.get("results"):
        return _build_environment_error(environment, payload, resolved_config)

    rows = result.get("results", [])
    row = rows[0] if isinstance(rows, list) and rows else None
    if not isinstance(row, Mapping):
        return _build_error_prediction(
            payload,
            resolved_config,
            error_type="MissingPredictionError",
            message="Listing trust pipeline did not return a result row.",
        )

    return _normalize_row(row, payload, resolved_config)


@lru_cache(maxsize=4)
def _get_trust_pipeline(config: DeployConfig) -> TrustFakeReviewsDeployPipeline:
    return TrustFakeReviewsDeployPipeline(config=config)


def _normalize_row(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    config: DeployConfig,
) -> dict[str, Any]:
    if row.get("status") != "ok":
        error = row.get("error") or {}
        return _build_error_prediction(
            payload,
            config,
            error_type=str(error.get("type") or "ListingTrustError"),
            message=str(error.get("message") or "Unknown listing trust error."),
        )

    scores = row.get("scores") or {}
    labels = row.get("labels") or {}
    graph_trust_probability = _to_optional_float(
        scores.get("phase_b_truth_likelihood_graph")
    )
    logistic_trust_probability = _to_optional_float(
        scores.get("phase_b_truth_likelihood_logistic")
    )
    graph_deception_probability = _to_optional_float(scores.get("trust_risk_index_graph"))
    logistic_deception_probability = _to_optional_float(
        scores.get("trust_risk_index_logistic")
    )

    if graph_deception_probability is None and graph_trust_probability is not None:
        graph_deception_probability = max(0.0, min(1.0, 1.0 - graph_trust_probability))
    if graph_trust_probability is None and graph_deception_probability is not None:
        graph_trust_probability = max(0.0, min(1.0, 1.0 - graph_deception_probability))

    if logistic_deception_probability is None and logistic_trust_probability is not None:
        logistic_deception_probability = max(
            0.0, min(1.0, 1.0 - logistic_trust_probability)
        )
    if logistic_trust_probability is None and logistic_deception_probability is not None:
        logistic_trust_probability = max(
            0.0, min(1.0, 1.0 - logistic_deception_probability)
        )

    # The graph head is the deploy pipeline's primary risk score, but it is fed by
    # discretized label buckets and can flatten small listing-to-listing differences.
    # Prefer the logistic head for the consumer-facing trust probability while
    # preserving the graph head explicitly for audit/debug use.
    trust_probability = (
        logistic_trust_probability
        if logistic_trust_probability is not None
        else graph_trust_probability
    )
    deception_probability = (
        logistic_deception_probability
        if logistic_deception_probability is not None
        else graph_deception_probability
    )
    if deception_probability is None and trust_probability is not None:
        deception_probability = max(0.0, min(1.0, 1.0 - trust_probability))
    if trust_probability is None and deception_probability is not None:
        trust_probability = max(0.0, min(1.0, 1.0 - deception_probability))

    return {
        "status": "ok",
        "source": DEFAULT_SOURCE,
        "provider": "ollama",
        "model": config.ollama_model,
        "input": dict(row.get("input") or payload),
        "trust_probability": trust_probability,
        "listing_trust_probability": trust_probability,
        "trust_probability_graph": graph_trust_probability,
        "trust_probability_logistic": logistic_trust_probability,
        "deception_probability": deception_probability,
        "title_deception_score": deception_probability,
        "deception_probability_graph": graph_deception_probability,
        "deception_probability_logistic": logistic_deception_probability,
        "authenticity_probability": trust_probability,
        "is_deceptive": None
        if deception_probability is None
        else deception_probability >= 0.5,
        "score_head": (
            "logistic"
            if logistic_trust_probability is not None
            else ("graph" if graph_trust_probability is not None else None)
        ),
        "graph_uncertainty_entropy": _to_optional_float(
            scores.get("graph_uncertainty_entropy")
        ),
        "overall_confidence": _to_optional_float(labels.get("overall_confidence")),
        "labels": dict(labels) if isinstance(labels, Mapping) else None,
        "error": None,
    }


def _build_environment_error(
    environment: Mapping[str, Any],
    payload: Mapping[str, Any],
    config: DeployConfig,
) -> dict[str, Any]:
    message = "; ".join(str(error) for error in environment.get("errors", []))
    if not message:
        message = "Listing trust pipeline environment is unavailable."
    return {
        "status": "unavailable",
        "source": DEFAULT_SOURCE,
        "provider": "ollama",
        "model": config.ollama_model,
        "input": dict(payload),
        "trust_probability": None,
        "listing_trust_probability": None,
        "trust_probability_graph": None,
        "trust_probability_logistic": None,
        "deception_probability": None,
        "title_deception_score": None,
        "deception_probability_graph": None,
        "deception_probability_logistic": None,
        "authenticity_probability": None,
        "is_deceptive": None,
        "score_head": None,
        "graph_uncertainty_entropy": None,
        "overall_confidence": None,
        "labels": None,
        "error": {
            "type": "EnvironmentValidationError",
            "message": message,
        },
    }


def _build_error_prediction(
    payload: Mapping[str, Any],
    config: DeployConfig,
    *,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "status": "error",
        "source": DEFAULT_SOURCE,
        "provider": "ollama",
        "model": config.ollama_model,
        "input": dict(payload),
        "trust_probability": None,
        "listing_trust_probability": None,
        "trust_probability_graph": None,
        "trust_probability_logistic": None,
        "deception_probability": None,
        "title_deception_score": None,
        "deception_probability_graph": None,
        "deception_probability_logistic": None,
        "authenticity_probability": None,
        "is_deceptive": None,
        "score_head": None,
        "graph_uncertainty_entropy": None,
        "overall_confidence": None,
        "labels": None,
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def _build_listing_bullet_points(candidate: Candidate | Mapping[str, Any]) -> str:
    raw_bullets = _candidate_value(candidate, "listing_bullet_points")
    if isinstance(raw_bullets, list):
        cleaned = [_clean_text(value) for value in raw_bullets]
        bullet_text = "; ".join(value for value in cleaned if value)
        if bullet_text:
            return bullet_text

    item_specifics = _candidate_value(candidate, "item_specifics")
    if isinstance(item_specifics, Mapping):
        parts: list[str] = []
        for name, raw_value in item_specifics.items():
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            cleaned_values = [_clean_text(value) for value in values]
            cleaned_values = [value for value in cleaned_values if value]
            if not cleaned_values:
                continue
            parts.append(f"{_clean_text(name)}: {', '.join(cleaned_values)}")
            if len(parts) >= 8:
                break
        return "; ".join(parts)

    return ""


def _build_listing_description(candidate: Candidate | Mapping[str, Any]) -> str:
    description = _clean_text(_candidate_value(candidate, "listing_description"))
    if description:
        return description

    parts: list[str] = []
    condition = _clean_text(_candidate_value(candidate, "condition"))
    if condition:
        parts.append(f"Condition: {condition}")

    returns_info = _candidate_value(candidate, "returns")
    if isinstance(returns_info, Mapping):
        return_period = returns_info.get("returnPeriod")
        if isinstance(return_period, Mapping):
            value = _clean_text(return_period.get("value"))
            unit = _clean_text(return_period.get("unit"))
            if value:
                parts.append(
                    f"Return window: {value} {unit.lower() if unit else 'days'}"
                )

    return " ".join(parts)


def _candidate_value(candidate: Candidate | Mapping[str, Any], key: str) -> Any:
    if isinstance(candidate, Candidate):
        return getattr(candidate, key)
    if isinstance(candidate, Mapping):
        return candidate.get(key)
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _to_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, numeric))
