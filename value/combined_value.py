from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .bayesian_value import BayesianValueInput, score_good_value_probability
from .worth_buying import (
    WorthBuyingConfig,
    load_model as load_worth_buying_model,
    load_prepared_catalog,
    score_worth_buying_catalog,
)


def score_combined_value_split(
    *,
    model_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
    max_rows: int | None = None,
    probability_threshold: float = 0.50,
    min_confidence_for_prediction: float | None = None,
) -> dict[str, Any]:
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1.")

    worth_buying_bundle = load_worth_buying_model(model_path)
    worth_buying_config = WorthBuyingConfig(**worth_buying_bundle["config"])
    confidence_threshold = (
        worth_buying_config.min_confidence_for_verdict
        if min_confidence_for_prediction is None
        else float(min_confidence_for_prediction)
    )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("min_confidence_for_prediction must be between 0 and 1.")

    catalog = load_prepared_catalog(split_path, max_rows=max_rows).reset_index(drop=True)
    retrieval_scored = score_worth_buying_catalog(
        catalog,
        model_bundle=worth_buying_bundle,
    )
    combined_scored = _build_combined_predictions(
        catalog=catalog,
        retrieval_scored=retrieval_scored,
        probability_threshold=probability_threshold,
        confidence_threshold=confidence_threshold,
    )

    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_scored.to_csv(resolved_output_path, index=False)

    summary = {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "split_path": str(Path(split_path).expanduser().resolve()),
        "output_path": str(resolved_output_path) if resolved_output_path else None,
        "rows_scored": int(len(combined_scored)),
        "probability_threshold": float(probability_threshold),
        "min_confidence_for_prediction": float(confidence_threshold),
        "prediction_counts": {
            prediction: int(count)
            for prediction, count in combined_scored["combined_prediction"]
            .value_counts()
            .sort_index()
            .items()
        },
        "combined_probability_quantiles": {
            "p10": float(combined_scored["combined_good_value_probability"].quantile(0.10)),
            "p50": float(combined_scored["combined_good_value_probability"].quantile(0.50)),
            "p90": float(combined_scored["combined_good_value_probability"].quantile(0.90)),
        },
    }
    return summary


def _build_combined_predictions(
    *,
    catalog: pd.DataFrame,
    retrieval_scored: pd.DataFrame,
    probability_threshold: float,
    confidence_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, retrieval_row in retrieval_scored.iterrows():
        catalog_row_index = int(retrieval_row["catalog_row_index"])
        source_row = catalog.iloc[catalog_row_index]

        bayesian_input = BayesianValueInput(
            trust_probability=_to_optional_float(source_row.get("trust_probability")),
            ewom_score_0_to_100=_to_optional_float(source_row.get("ewom_score_0_to_100")),
            ewom_magnitude_0_to_100=_to_optional_float(source_row.get("ewom_magnitude_0_to_100")),
            average_rating=_to_optional_float(source_row.get("average_rating")),
            rating_count=_resolve_rating_count(source_row),
            verified_purchase_rate=_to_optional_float(source_row.get("verified_purchase_rate")),
            price=_to_optional_float(source_row.get("price")),
            peer_price=_to_optional_float(retrieval_row.get("peer_price")),
            warranty_months=_to_optional_float(source_row.get("warranty_months")),
            return_window_days=_to_optional_float(source_row.get("return_window_days")),
        )
        bayesian_result = score_good_value_probability(bayesian_input)
        combined_probability = float(bayesian_result["good_value_probability"])
        retrieval_confidence = float(retrieval_row["confidence_score"])
        combined_prediction = _resolve_combined_prediction(
            probability=combined_probability,
            retrieval_confidence=retrieval_confidence,
            probability_threshold=probability_threshold,
            confidence_threshold=confidence_threshold,
        )
        evidence = bayesian_result["evidence"]
        component_states = bayesian_result["most_likely_component_states"]

        rows.append(
            {
                "catalog_row_index": catalog_row_index,
                "parent_asin": source_row["parent_asin"],
                "title": source_row["title"],
                "price": _to_optional_float(source_row.get("price")),
                "peer_price": _to_optional_float(retrieval_row.get("peer_price")),
                "price_gap_vs_peer": _to_optional_float(retrieval_row.get("price_gap_vs_peer")),
                "average_rating": _to_optional_float(source_row.get("average_rating")),
                "rating_number": _to_optional_float(source_row.get("rating_number")),
                "review_count": int(source_row.get("review_count", 0) or 0),
                "verified_purchase_rate": _to_optional_float(source_row.get("verified_purchase_rate")),
                "trust_probability": _to_optional_float(source_row.get("trust_probability")),
                "ewom_score_0_to_100": _to_optional_float(source_row.get("ewom_score_0_to_100")),
                "ewom_magnitude_0_to_100": _to_optional_float(source_row.get("ewom_magnitude_0_to_100")),
                "retrieval_confidence_score": retrieval_confidence,
                "retrieval_worth_buying_score": float(retrieval_row["worth_buying_score"]),
                "retrieval_verdict": retrieval_row["verdict"],
                "combined_good_value_probability": combined_probability,
                "combined_prediction": combined_prediction,
                "bayesian_evidence_trust_signal": evidence.get("TrustSignal"),
                "bayesian_evidence_review_polarity": evidence.get("ReviewPolarity"),
                "bayesian_evidence_review_strength": evidence.get("ReviewStrength"),
                "bayesian_evidence_rating_signal": evidence.get("RatingSignal"),
                "bayesian_evidence_review_volume": evidence.get("ReviewVolume"),
                "bayesian_evidence_verified_signal": evidence.get("VerifiedSignal"),
                "bayesian_evidence_relative_price_bucket": evidence.get("RelativePriceBucket"),
                "bayesian_component_trustworthiness": component_states.get("Trustworthiness"),
                "bayesian_component_review_evidence": component_states.get("ReviewEvidence"),
                "bayesian_component_product_quality": component_states.get("ProductQuality"),
                "bayesian_component_service_support": component_states.get("ServiceSupport"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["combined_good_value_probability", "retrieval_confidence_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _resolve_rating_count(source_row: pd.Series) -> float | None:
    rating_number = _to_optional_float(source_row.get("rating_number"))
    if rating_number is not None:
        return rating_number
    review_count = _to_optional_float(source_row.get("review_count"))
    return review_count


def _resolve_combined_prediction(
    *,
    probability: float,
    retrieval_confidence: float,
    probability_threshold: float,
    confidence_threshold: float,
) -> str:
    if retrieval_confidence < confidence_threshold:
        return "insufficient_evidence"
    if probability >= probability_threshold:
        return "good_value"
    return "not_good_value"


def _to_optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric
