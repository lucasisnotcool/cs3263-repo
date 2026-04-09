from __future__ import annotations

from dataclasses import asdict
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.entities.candidate import Candidate
from eWOM.api import EWOMModelPaths, score_review_set

from .bayesian_value import (
    BayesianValueInput,
    fuse_ewom_result_into_bayesian_input,
    score_good_value_probability,
)
from .worth_buying import (
    inspect_worth_buying_catalog_neighbors,
    score_worth_buying_catalog,
)


def build_bayesian_input_from_candidate(
    candidate: Candidate | Mapping[str, Any],
    *,
    peer_price: float | None = None,
) -> BayesianValueInput:
    resolved_candidate = _coerce_candidate(candidate)
    return BayesianValueInput(
        average_rating=_to_optional_float(resolved_candidate.product_average_rating),
        rating_count=_to_optional_float(resolved_candidate.product_rating_count),
        price=resolve_candidate_total_price(resolved_candidate),
        peer_price=_to_optional_float(peer_price),
        warranty_months=_extract_warranty_months(resolved_candidate.item_specifics),
        return_window_days=_extract_return_window_days(resolved_candidate.returns),
    )


def build_worth_buying_query_row(candidate: Candidate | Mapping[str, Any]) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    average_rating = _to_optional_float(resolved_candidate.product_average_rating)
    rating_count = _to_optional_float(resolved_candidate.product_rating_count)
    review_count = int(rating_count or 0.0)
    title = str(resolved_candidate.title or "")

    return {
        "parent_asin": _resolve_candidate_identity(resolved_candidate),
        "title": title,
        "store": str(resolved_candidate.seller_id or ""),
        "main_category": "",
        "product_document": _build_candidate_product_document(resolved_candidate),
        "price": resolve_candidate_total_price(resolved_candidate),
        "average_rating": average_rating,
        "rating_number": rating_count,
        "review_count": review_count,
        "verified_purchase_rate": None,
        "helpful_vote_total": 0.0,
        "helpful_vote_avg": 0.0,
        "avg_review_rating": average_rating,
        "trust_probability": None,
        "ewom_score_0_to_100": None,
        "ewom_magnitude_0_to_100": None,
    }


def infer_candidate_market_context(
    candidate: Candidate | Mapping[str, Any],
    *,
    model_path: str | Path,
    top_k_neighbors: int | None = None,
    min_similarity: float | None = None,
) -> dict[str, Any]:
    query_row = build_worth_buying_query_row(candidate)
    scored = score_worth_buying_catalog(
        pd.DataFrame([query_row]),
        model_path=model_path,
        config_overrides={
            "top_k_neighbors": top_k_neighbors,
            "min_similarity": min_similarity,
        },
    )
    return _to_builtin(scored.iloc[0].to_dict())


def score_ebay_candidate_value(
    candidate: Candidate | Mapping[str, Any],
    *,
    peer_price: float | None = None,
    worth_buying_model_path: str | Path | None = None,
    top_k_neighbors: int | None = None,
    ewom_result: Mapping[str, Any] | None = None,
    ewom_model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    base_input = build_bayesian_input_from_candidate(
        resolved_candidate,
        peer_price=peer_price,
    )

    market_context: dict[str, Any] | None = None
    if worth_buying_model_path is not None:
        market_context = infer_candidate_market_context(
            resolved_candidate,
            model_path=worth_buying_model_path,
            top_k_neighbors=top_k_neighbors,
        )
        inferred_peer_price = _to_optional_float(market_context.get("peer_price"))
        if inferred_peer_price is not None:
            base_input = build_bayesian_input_from_candidate(
                resolved_candidate,
                peer_price=inferred_peer_price,
            )

    resolved_ewom_result: dict[str, Any] | None = None
    fused_agent_signals: dict[str, Any] | None = None
    if ewom_result is not None:
        resolved_ewom_result = _to_builtin(dict(ewom_result))
    elif resolved_candidate.seller_feedback_texts:
        resolved_ewom_result = score_review_set(
            resolved_candidate.seller_feedback_texts,
            model_paths=ewom_model_paths,
        )

    if resolved_ewom_result is not None:
        base_input, fused_agent_signals = fuse_ewom_result_into_bayesian_input(
            base_input,
            resolved_ewom_result,
        )

    bayesian_result = score_good_value_probability(base_input)
    bayesian_result["resolved_input"] = asdict(base_input)
    if fused_agent_signals is not None:
        bayesian_result["fused_agent_signals"] = fused_agent_signals

    return {
        "candidate": resolved_candidate.to_output_dict(),
        "market_context": market_context,
        "ewom_result": resolved_ewom_result,
        "bayesian_result": _to_builtin(bayesian_result),
    }


def sweep_candidate_market_context_k(
    candidate: Candidate | Mapping[str, Any],
    *,
    model_path: str | Path,
    k_values: list[int] | tuple[int, ...],
    ewom_result: Mapping[str, Any] | None = None,
    ewom_model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    normalized_k_values = _normalize_k_values(k_values)

    resolved_ewom_result: dict[str, Any] | None = None
    if ewom_result is not None:
        resolved_ewom_result = _to_builtin(dict(ewom_result))
    elif resolved_candidate.seller_feedback_texts:
        resolved_ewom_result = score_review_set(
            resolved_candidate.seller_feedback_texts,
            model_paths=ewom_model_paths,
        )

    sweep_rows: list[dict[str, Any]] = []
    query_row = build_worth_buying_query_row(resolved_candidate)
    for k in normalized_k_values:
        diagnostics = inspect_worth_buying_catalog_neighbors(
            pd.DataFrame([query_row]),
            model_path=model_path,
            config_overrides={"top_k_neighbors": int(k)},
        )[0]
        peer_price = _to_optional_float(diagnostics.get("peer_price"))
        good_value_probability = None
        prediction = None
        if peer_price is not None and resolved_ewom_result is not None:
            bayesian_result = score_ebay_candidate_value(
                resolved_candidate,
                peer_price=peer_price,
                ewom_result=resolved_ewom_result,
            )["bayesian_result"]
            good_value_probability = _to_optional_float(
                bayesian_result.get("good_value_probability")
            )
            prediction = _resolve_probability_label(
                good_value_probability,
                threshold=0.50,
            )

        sweep_rows.append(
            {
                "k": int(k),
                "peer_price": peer_price,
                "neighbor_count": int(diagnostics.get("neighbor_count", 0) or 0),
                "average_neighbor_similarity": _to_optional_float(
                    diagnostics.get("average_neighbor_similarity")
                ),
                "neighbor_prices": [
                    _to_optional_float(neighbor.get("price"))
                    for neighbor in diagnostics.get("neighbors", [])
                    if _to_optional_float(neighbor.get("price")) is not None
                ],
                "good_value_probability": good_value_probability,
                "prediction": prediction,
            }
        )

    return {
        "candidate": resolved_candidate.to_output_dict(),
        "ewom_result": resolved_ewom_result,
        "k_sweep": sweep_rows,
    }


def summarize_candidate_market_context_k_sweep(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    candidate = result.get("candidate")
    rows = result.get("k_sweep")
    if not isinstance(candidate, Mapping) or not isinstance(rows, list):
        raise ValueError("result must contain 'candidate' and 'k_sweep'.")

    return {
        "source_url": candidate.get("source_url"),
        "title": candidate.get("title"),
        "k_sweep": [
            {
                "k": row.get("k"),
                "peer_price": row.get("peer_price"),
                "neighbor_count": row.get("neighbor_count"),
                "average_neighbor_similarity": row.get("average_neighbor_similarity"),
                "good_value_probability": row.get("good_value_probability"),
                "prediction": row.get("prediction"),
            }
            for row in rows
            if isinstance(row, Mapping)
        ],
    }


def write_candidate_k_sweep_plot(
    result: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    candidate = result.get("candidate")
    rows = result.get("k_sweep")
    if not isinstance(candidate, Mapping) or not isinstance(rows, list) or not rows:
        raise ValueError("result must contain a non-empty 'k_sweep' list.")

    total_price = resolve_candidate_total_price(candidate)
    plot_rows = [row for row in rows if isinstance(row, Mapping)]
    ks = [int(row.get("k")) for row in plot_rows]
    peer_prices = [_to_optional_float(row.get("peer_price")) for row in plot_rows]
    average_similarities = [
        _to_optional_float(row.get("average_neighbor_similarity")) for row in plot_rows
    ]
    probabilities = [
        _to_optional_float(row.get("good_value_probability")) for row in plot_rows
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Peer Price vs K",
            "Neighbor Price Distribution by K",
            "Average Similarity and Good-Value Probability",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=peer_prices,
            mode="lines+markers",
            name="peer_price",
        ),
        row=1,
        col=1,
    )
    if total_price is not None:
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=[total_price for _ in ks],
                mode="lines",
                name="listing_total_price",
                line={"dash": "dash"},
            ),
            row=1,
            col=1,
        )

    for row in plot_rows:
        neighbor_prices = [
            price for price in row.get("neighbor_prices", []) if price is not None
        ]
        if not neighbor_prices:
            continue
        fig.add_trace(
            go.Box(
                y=neighbor_prices,
                name=f"k={int(row.get('k'))}",
                boxpoints="outliers",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=ks,
            y=average_similarities,
            mode="lines+markers",
            name="avg_similarity",
        ),
        row=3,
        col=1,
    )
    if any(probability is not None for probability in probabilities):
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=probabilities,
                mode="lines+markers",
                name="good_value_probability",
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=f"K Sweep Diagnostics: {candidate.get('title') or candidate.get('source_url')}",
        height=950,
        width=1100,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Neighbor Price", row=2, col=1)
    fig.update_yaxes(title_text="0-1 Scale", row=3, col=1)
    fig.update_xaxes(title_text="K", row=3, col=1)

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_output_path.write_text(
        fig.to_html(full_html=True, include_plotlyjs=True),
        encoding="utf-8",
    )
    return str(resolved_output_path)


def summarize_ebay_candidate_value_result(
    result: Mapping[str, Any],
    *,
    probability_threshold: float = 0.50,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")

    candidate = result.get("candidate")
    bayesian_result = result.get("bayesian_result")
    if not isinstance(candidate, Mapping) or not isinstance(bayesian_result, Mapping):
        raise ValueError("result must contain 'candidate' and 'bayesian_result' mappings.")

    resolved_input = bayesian_result.get("resolved_input")
    resolved_input = resolved_input if isinstance(resolved_input, Mapping) else {}
    derived_metrics = bayesian_result.get("derived_metrics")
    derived_metrics = derived_metrics if isinstance(derived_metrics, Mapping) else {}
    fused_agent_signals = bayesian_result.get("fused_agent_signals")
    fused_agent_signals = (
        fused_agent_signals if isinstance(fused_agent_signals, Mapping) else {}
    )
    probability = _to_optional_float(bayesian_result.get("good_value_probability"))

    return {
        "source_url": candidate.get("source_url"),
        "title": candidate.get("title"),
        "total_price": resolved_input.get("price"),
        "peer_price": resolved_input.get("peer_price"),
        "price_gap_vs_peer": derived_metrics.get("price_gap_vs_peer"),
        "good_value_probability": probability,
        "prediction": _resolve_probability_label(
            probability,
            threshold=probability_threshold,
        ),
        "trust_probability": resolved_input.get("trust_probability"),
        "ewom_score_0_to_100": resolved_input.get("ewom_score_0_to_100"),
        "seller_feedback_review_count": fused_agent_signals.get("review_count"),
    }


def resolve_candidate_total_price(candidate: Candidate | Mapping[str, Any]) -> float | None:
    resolved_candidate = _coerce_candidate(candidate)
    item_price = _extract_money_value(resolved_candidate.price)
    shipping_cost = _extract_min_shipping_cost(resolved_candidate.shipping)

    if item_price is None:
        return None
    if shipping_cost is None:
        return item_price
    return item_price + shipping_cost


def _coerce_candidate(candidate: Candidate | Mapping[str, Any]) -> Candidate:
    if isinstance(candidate, Candidate):
        return candidate
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a Candidate or mapping.")
    return Candidate(
        source_url=str(candidate.get("source_url") or ""),
        page_type=str(candidate.get("page_type") or ""),
        legacy_item_id=_string_or_none(candidate.get("legacy_item_id")),
        rest_item_id=_string_or_none(candidate.get("rest_item_id")),
        product_id=_string_or_none(candidate.get("product_id")),
        title=_string_or_none(candidate.get("title")),
        price=_mapping_or_none(candidate.get("price")),
        shipping=_list_of_mappings(candidate.get("shipping")),
        returns=_mapping_or_none(candidate.get("returns")),
        condition=_string_or_none(candidate.get("condition")),
        seller_id=_string_or_none(candidate.get("seller_id")),
        seller_feedback_score=_int_or_none(candidate.get("seller_feedback_score")),
        seller_feedback_percentage=_to_optional_float(
            candidate.get("seller_feedback_percentage")
        ),
        detailed_seller_ratings=_mapping_or_none(candidate.get("detailed_seller_ratings")),
        product_rating_count=_int_or_none(candidate.get("product_rating_count")),
        product_rating_histogram=_list_of_mappings(candidate.get("product_rating_histogram")),
        product_average_rating=_to_optional_float(candidate.get("product_average_rating")),
        seller_feedback_texts=_list_of_strings(candidate.get("seller_feedback_texts")),
        item_specifics=_mapping_or_none(candidate.get("item_specifics")),
        product_family_key=_string_or_none(candidate.get("product_family_key")),
    )


def _resolve_candidate_identity(candidate: Candidate) -> str:
    for value in (
        candidate.product_family_key,
        candidate.product_id,
        candidate.legacy_item_id,
        candidate.rest_item_id,
        candidate.source_url,
    ):
        text = str(value or "").strip()
        if text:
            return text
    return "ebay_candidate"


def _build_candidate_product_document(candidate: Candidate) -> str:
    parts = [
        str(candidate.title or ""),
        str(candidate.condition or ""),
        _flatten_mapping_text(candidate.item_specifics),
    ]
    return " ".join(part for part in parts if part).strip()


def _flatten_mapping_text(value: Mapping[str, Any] | None) -> str:
    if not isinstance(value, Mapping):
        return ""
    parts: list[str] = []
    for key, raw_value in value.items():
        key_text = str(key).strip()
        if isinstance(raw_value, list):
            value_text = " ".join(str(item).strip() for item in raw_value if str(item).strip())
        else:
            value_text = str(raw_value).strip()
        if key_text or value_text:
            parts.append(f"{key_text} {value_text}".strip())
    return " ".join(parts).strip()


def _extract_money_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        return _to_optional_float(value.get("value"))
    return _to_optional_float(value)


def _extract_min_shipping_cost(shipping_options: list[dict[str, Any]]) -> float | None:
    costs: list[float] = []
    for option in shipping_options:
        if not isinstance(option, Mapping):
            continue
        shipping_cost = _extract_money_value(option.get("shippingCost"))
        if shipping_cost is not None:
            costs.append(max(0.0, shipping_cost))
            continue
        shipping_type = str(option.get("shippingCostType") or "").strip().lower()
        if shipping_type == "free":
            costs.append(0.0)

    if not costs:
        return None
    return min(costs)


def _extract_return_window_days(returns_info: Mapping[str, Any] | None) -> float | None:
    if not isinstance(returns_info, Mapping):
        return None

    direct_candidates = [
        returns_info.get("returnPeriod"),
        returns_info.get("returnWindow"),
        returns_info.get("returnPeriodValue"),
    ]
    for candidate in direct_candidates:
        days = _parse_duration_to_days(candidate)
        if days is not None:
            return days

    for value in returns_info.values():
        days = _parse_duration_to_days(value)
        if days is not None:
            return days
    return None


def _extract_warranty_months(item_specifics: Mapping[str, Any] | None) -> float | None:
    if not isinstance(item_specifics, Mapping):
        return None

    for key, value in item_specifics.items():
        normalized_key = str(key).strip().lower()
        if "warranty" not in normalized_key and "guarantee" not in normalized_key:
            continue
        months = _parse_duration_to_months(value)
        if months is not None:
            return months
    return None


def _parse_duration_to_days(value: Any) -> float | None:
    if isinstance(value, Mapping):
        numeric = _to_optional_float(value.get("value"))
        unit = str(value.get("unit") or value.get("unitType") or "").strip().lower()
        if numeric is not None and unit:
            return _convert_duration(numeric, unit, target="days")

    if isinstance(value, list):
        for item in value:
            parsed = _parse_duration_to_days(item)
            if parsed is not None:
                return parsed
        return None

    return _parse_duration_from_text(str(value or ""), target="days")


def _parse_duration_to_months(value: Any) -> float | None:
    if isinstance(value, Mapping):
        numeric = _to_optional_float(value.get("value"))
        unit = str(value.get("unit") or value.get("unitType") or "").strip().lower()
        if numeric is not None and unit:
            return _convert_duration(numeric, unit, target="months")

    if isinstance(value, list):
        for item in value:
            parsed = _parse_duration_to_months(item)
            if parsed is not None:
                return parsed
        return None

    return _parse_duration_from_text(str(value or ""), target="months")


def _parse_duration_from_text(text: str, *, target: str) -> float | None:
    normalized = text.strip().lower()
    if not normalized:
        return None

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(business\s+days?|days?|weeks?|months?|years?)",
        normalized,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)
    return _convert_duration(value, unit, target=target)


def _convert_duration(value: float, unit: str, *, target: str) -> float | None:
    normalized_unit = unit.strip().lower()
    if normalized_unit.startswith("business day"):
        day_value = value
    elif normalized_unit.startswith("day"):
        day_value = value
    elif normalized_unit.startswith("week"):
        day_value = value * 7.0
    elif normalized_unit.startswith("month"):
        day_value = value * 30.0
    elif normalized_unit.startswith("year"):
        day_value = value * 365.0
    else:
        return None

    if target == "days":
        return day_value
    if target == "months":
        return day_value / 30.0
    return None


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


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return _to_builtin(value.to_dict())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _string_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _list_of_strings(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned or None


def _int_or_none(value: Any) -> int | None:
    numeric = _to_optional_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _normalize_k_values(k_values: list[int] | tuple[int, ...]) -> list[int]:
    normalized: list[int] = []
    for raw_value in k_values:
        value = int(raw_value)
        if value <= 0:
            raise ValueError("k values must be positive integers.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one k value is required.")
    return sorted(normalized)


def _resolve_probability_label(
    probability: float | None,
    *,
    threshold: float,
) -> str | None:
    if probability is None:
        return None
    return "good_value" if probability >= threshold else "not_good_value"
