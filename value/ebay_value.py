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
from .listing_kind import infer_listing_kind_from_parts
from .worth_buying import inspect_worth_buying_catalog_neighbors

SUFFICIENT_RETRIEVAL_STATUSES = {"usable"}
PRICE_GAP_TIE_MARGIN = 0.02
NEUTRAL_PRICE_BUCKET = "fair"

PREFERRED_BRAND_KEYS = (
    "brand",
    "manufacturer",
    "brand name",
    "make",
)

GENERIC_BRAND_VALUES = {
    "does not apply",
    "generic",
    "unbranded",
    "unknown",
}

ACCESSORY_TERMS = {
    "adapter",
    "band",
    "cable",
    "case",
    "charger",
    "cord",
    "cover",
    "dock",
    "earpads",
    "hook",
    "holder",
    "mount",
    "protector",
    "replacement",
    "skin",
    "sleeve",
    "stand",
    "strap",
    "tips",
}

ACCESSORY_CUE_PHRASES = (
    "anti lost",
    "adapter for",
    "case for",
    "cable for",
    "charger for",
    "compatible with",
    "cover for",
    "protective case",
    "protector for",
    "replacement for",
    "skin for",
)

ACCESSORY_CONTEXT_TERMS = {
    "anti",
    "carabiner",
    "cover",
    "cute",
    "dustproof",
    "earhook",
    "fashion",
    "hard",
    "hook",
    "keychain",
    "protective",
    "rugged",
    "shockproof",
    "silicone",
    "soft",
    "tpu",
}

DEVICE_FAMILY_TERMS = {
    "airpods",
    "camera",
    "console",
    "earbuds",
    "galaxy",
    "headphones",
    "iphone",
    "ipad",
    "kindle",
    "laptop",
    "macbook",
    "monitor",
    "phone",
    "pixel",
    "playstation",
    "speaker",
    "switch",
    "tablet",
    "watch",
    "xbox",
}

IMPLAUSIBLY_LOW_PEER_PRICE_RATIO = 0.18
MIN_PEER_NEIGHBOR_COUNT = 3

PRIMARY_PRODUCT_TYPE_TERMS = {
    "camera",
    "console",
    "earbud",
    "earbuds",
    "headphone",
    "headphones",
    "headset",
    "in ear",
    "laptop",
    "monitor",
    "phone",
    "smartphone",
    "speaker",
    "tablet",
    "watch",
}

RETRIEVAL_MODEL_KEYS = (
    "model",
    "model name",
    "product line",
    "series",
)

RETRIEVAL_TYPE_KEYS = (
    "type",
    "product type",
)

TITLE_NOISE_TERMS = (
    "nib",
    "bnib",
    "brand new",
    "new in box",
    "open box",
    "sealed",
)

REMOVABLE_WITH_TERMS = (
    "accessories",
    "bundle",
    "cable",
    "case",
    "charger",
    "charging case",
    "cover",
    "ear tips",
    "keychain",
    "lanyard",
    "magsafe",
    "skin",
)

DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE = 500
DEFAULT_RERANKED_NEIGHBOR_COUNT = 5
DEFAULT_RETRIEVAL_MIN_SIMILARITY = 0.02
MIN_RERANK_MATCH_SCORE = 0.24
RETRIEVAL_STOPWORDS = {
    "apple",
    "bluetooth",
    "earbud",
    "earbuds",
    "generation",
    "in",
    "the",
}
CORE_IDENTITY_STOPWORDS = {
    "1st",
    "2nd",
    "3rd",
    "4th",
    "5th",
    "6th",
    "7th",
    "8th",
    "9th",
    "10th",
    "bluetooth",
    "generation",
    "gen",
    "new",
    "series",
    "true",
    "wireless",
}
IMPORTANT_VARIANT_TOKENS = {"max", "mini", "plus", "pro", "ultra"}
PRODUCT_KIND_ALIASES: dict[str, set[str]] = {
    "camera": {"camera", "camcorder", "dslr", "mirrorless", "webcam"},
    "console": {"console", "playstation", "switch", "xbox"},
    "earbud": {
        "airpod",
        "airpods",
        "earbud",
        "earbuds",
        "earphone",
        "earphones",
        "earset",
        "headset",
        "headsets",
        "iem",
    },
    "headphone": {"headphone", "headphones"},
    "laptop": {"chromebook", "laptop", "macbook", "notebook"},
    "monitor": {"display", "monitor"},
    "pen": {"pencil", "pen", "stylus"},
    "phone": {"galaxy", "iphone", "phone", "pixel", "smartphone"},
    "speaker": {"soundbar", "speaker", "speakers"},
    "tablet": {"ipad", "tablet"},
    "watch": {"applewatch", "smartwatch", "watch"},
    "charger": {"adapter", "charger", "charging", "dock", "powerbank", "usb"},
}
AUDIO_PRODUCT_KINDS = {"earbud", "headphone"}
PRIMARY_PRODUCT_TYPE_TOKENS = {
    token
    for term in PRIMARY_PRODUCT_TYPE_TERMS
    for token in term.split()
}


def build_bayesian_input_from_candidate(
    candidate: Candidate | Mapping[str, Any],
    *,
    peer_price: float | None = None,
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
) -> BayesianValueInput:
    resolved_candidate = _coerce_candidate(candidate)
    return BayesianValueInput(
        average_rating=_to_optional_float(resolved_candidate.product_average_rating),
        rating_count=_to_optional_float(resolved_candidate.product_rating_count),
        price=resolve_candidate_total_price(
            resolved_candidate,
            include_shipping_in_total=include_shipping_in_total,
            prefer_converted_usd=prefer_converted_usd,
        ),
        peer_price=_to_optional_float(peer_price),
        warranty_months=_extract_warranty_months(resolved_candidate.item_specifics),
        return_window_days=_extract_return_window_days(resolved_candidate.returns),
    )


def build_worth_buying_query_row(
    candidate: Candidate | Mapping[str, Any],
    *,
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    average_rating = _to_optional_float(resolved_candidate.product_average_rating)
    rating_count = _to_optional_float(resolved_candidate.product_rating_count)
    review_count = int(rating_count or 0.0)
    title = _build_candidate_retrieval_title(resolved_candidate)
    brand = _extract_candidate_brand(resolved_candidate)
    listing_kind = infer_listing_kind_from_parts(
        title=title,
        main_category="",
        categories=[],
        features=[],
        description=[],
        details_text=_flatten_mapping_text(resolved_candidate.item_specifics),
    )

    return {
        "parent_asin": _resolve_candidate_identity(resolved_candidate),
        "title": title,
        "store": brand or "",
        "main_category": "",
        "listing_kind": listing_kind,
        "product_document": _build_candidate_product_document(resolved_candidate),
        "price": resolve_candidate_total_price(
            resolved_candidate,
            include_shipping_in_total=include_shipping_in_total,
            prefer_converted_usd=prefer_converted_usd,
        ),
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
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
    retrieval_candidate_pool_size: int | None = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    min_peer_price_ratio: float = IMPLAUSIBLY_LOW_PEER_PRICE_RATIO,
    min_peer_neighbor_count: int = MIN_PEER_NEIGHBOR_COUNT,
) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    reranked_top_n = _resolve_reranked_top_n(top_k_neighbors)
    candidate_pool_size = _resolve_candidate_pool_size(
        retrieval_candidate_pool_size,
        reranked_top_n=reranked_top_n,
    )
    query_row = build_worth_buying_query_row(
        resolved_candidate,
        include_shipping_in_total=include_shipping_in_total,
        prefer_converted_usd=prefer_converted_usd,
    )
    diagnostics = inspect_worth_buying_catalog_neighbors(
        pd.DataFrame([query_row]),
        model_path=model_path,
        config_overrides={
            "top_k_neighbors": candidate_pool_size,
            "min_similarity": (
                min_similarity
                if min_similarity is not None
                else DEFAULT_RETRIEVAL_MIN_SIMILARITY
            ),
        },
    )[0]
    return _to_builtin(
        _refine_candidate_market_context(
            resolved_candidate,
            diagnostics,
            reranked_top_n=reranked_top_n,
            candidate_pool_size=candidate_pool_size,
            min_peer_price_ratio=min_peer_price_ratio,
            min_peer_neighbor_count=min_peer_neighbor_count,
        )
    )


def score_ebay_candidate_value(
    candidate: Candidate | Mapping[str, Any],
    *,
    peer_price: float | None = None,
    worth_buying_model_path: str | Path | None = None,
    top_k_neighbors: int | None = None,
    ewom_result: Mapping[str, Any] | None = None,
    ewom_model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
    retrieval_candidate_pool_size: int | None = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    min_peer_price_ratio: float = IMPLAUSIBLY_LOW_PEER_PRICE_RATIO,
    min_peer_neighbor_count: int = MIN_PEER_NEIGHBOR_COUNT,
) -> dict[str, Any]:
    resolved_candidate = _coerce_candidate(candidate)
    peer_price_source = "manual" if peer_price is not None else "none"
    default_relative_price_bucket: str | None = None
    price_considered = peer_price is not None
    price_note: str | None = None
    base_input = build_bayesian_input_from_candidate(
        resolved_candidate,
        peer_price=peer_price,
        include_shipping_in_total=include_shipping_in_total,
        prefer_converted_usd=prefer_converted_usd,
    )

    market_context: dict[str, Any] | None = None
    if worth_buying_model_path is not None:
        market_context = infer_candidate_market_context(
            resolved_candidate,
            model_path=worth_buying_model_path,
            top_k_neighbors=top_k_neighbors,
            include_shipping_in_total=include_shipping_in_total,
            prefer_converted_usd=prefer_converted_usd,
            retrieval_candidate_pool_size=retrieval_candidate_pool_size,
            min_peer_price_ratio=min_peer_price_ratio,
            min_peer_neighbor_count=min_peer_neighbor_count,
        )
        retrieval_status = str(market_context.get("retrieval_status") or "").strip()
        inferred_peer_price = _to_optional_float(market_context.get("peer_price"))
        if (
            inferred_peer_price is not None
            and retrieval_status in SUFFICIENT_RETRIEVAL_STATUSES
        ):
            base_input = build_bayesian_input_from_candidate(
                resolved_candidate,
                peer_price=inferred_peer_price,
                include_shipping_in_total=include_shipping_in_total,
                prefer_converted_usd=prefer_converted_usd,
            )
            peer_price_source = "retrieval"
            price_considered = True
        elif retrieval_status and retrieval_status not in SUFFICIENT_RETRIEVAL_STATUSES:
            default_relative_price_bucket = NEUTRAL_PRICE_BUCKET
            price_considered = False
            price_note = (
                "We could not find enough trustworthy similar-item price data, so the "
                "Bayesian score uses a neutral fair-price bucket and does not consider "
                "price advantage."
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
    else:
        base_input = BayesianValueInput(
            trust_probability=0.0,
            ewom_score_0_to_100=0.0,
            ewom_magnitude_0_to_100=base_input.ewom_magnitude_0_to_100,
            average_rating=base_input.average_rating,
            rating_count=base_input.rating_count,
            verified_purchase_rate=base_input.verified_purchase_rate,
            price=base_input.price,
            peer_price=base_input.peer_price,
            warranty_months=base_input.warranty_months,
            return_window_days=base_input.return_window_days,
        )

    bayesian_result = score_good_value_probability(
        base_input,
        default_relative_price_bucket=default_relative_price_bucket,
    )
    bayesian_result["resolved_input"] = asdict(base_input)
    if fused_agent_signals is not None:
        bayesian_result["fused_agent_signals"] = fused_agent_signals
    decision = _build_ebay_value_decision(
        good_value_probability=_to_optional_float(
            bayesian_result.get("good_value_probability")
        ),
        market_context=market_context,
        peer_price_source=peer_price_source,
        price_considered=price_considered,
        price_note=price_note,
    )

    return {
        "candidate": resolved_candidate.to_output_dict(),
        "market_context": market_context,
        "ewom_result": resolved_ewom_result,
        "bayesian_result": _to_builtin(bayesian_result),
        "decision": decision,
        "pricing": {
            "include_shipping_in_total": bool(include_shipping_in_total),
            "prefer_converted_usd": bool(prefer_converted_usd),
            "total_price_currency": resolve_candidate_total_price_currency(
                resolved_candidate,
                prefer_converted_usd=prefer_converted_usd,
            ),
            "peer_price_source": peer_price_source,
            "price_considered": bool(price_considered),
            "price_note": price_note,
            "default_relative_price_bucket": default_relative_price_bucket,
        },
    }


def sweep_candidate_market_context_k(
    candidate: Candidate | Mapping[str, Any],
    *,
    model_path: str | Path,
    k_values: list[int] | tuple[int, ...],
    ewom_result: Mapping[str, Any] | None = None,
    ewom_model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
    retrieval_candidate_pool_size: int | None = DEFAULT_RETRIEVAL_CANDIDATE_POOL_SIZE,
    min_peer_price_ratio: float = IMPLAUSIBLY_LOW_PEER_PRICE_RATIO,
    min_peer_neighbor_count: int = MIN_PEER_NEIGHBOR_COUNT,
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
    for k in normalized_k_values:
        diagnostics = infer_candidate_market_context(
            resolved_candidate,
            model_path=model_path,
            top_k_neighbors=int(k),
            include_shipping_in_total=include_shipping_in_total,
            prefer_converted_usd=prefer_converted_usd,
            retrieval_candidate_pool_size=retrieval_candidate_pool_size,
            min_peer_price_ratio=min_peer_price_ratio,
            min_peer_neighbor_count=min_peer_neighbor_count,
        )
        peer_price = _to_optional_float(diagnostics.get("peer_price"))
        good_value_probability = None
        prediction = None
        prediction_reason = None
        if peer_price is not None and resolved_ewom_result is not None:
            bayesian_result = score_ebay_candidate_value(
                resolved_candidate,
                peer_price=peer_price,
                ewom_result=resolved_ewom_result,
                include_shipping_in_total=include_shipping_in_total,
                prefer_converted_usd=prefer_converted_usd,
                retrieval_candidate_pool_size=retrieval_candidate_pool_size,
                min_peer_price_ratio=min_peer_price_ratio,
                min_peer_neighbor_count=min_peer_neighbor_count,
            )["bayesian_result"]
            good_value_probability = _to_optional_float(
                bayesian_result.get("good_value_probability")
            )
        prediction, prediction_reason = _resolve_prediction_from_market_context(
            good_value_probability,
            diagnostics.get("retrieval_status"),
            peer_price_source=("retrieval" if peer_price is not None else "none"),
            threshold=0.50,
        )

        sweep_rows.append(
            {
                "k": int(k),
                "peer_price": peer_price,
                "neighbor_count": int(diagnostics.get("neighbor_count", 0) or 0),
                "raw_neighbor_count": int(diagnostics.get("raw_neighbor_count", 0) or 0),
                "average_neighbor_similarity": _to_optional_float(
                    diagnostics.get("average_neighbor_similarity")
                ),
                "retrieval_status": diagnostics.get("retrieval_status"),
                "neighbor_prices": [
                    _to_optional_float(neighbor.get("price"))
                    for neighbor in diagnostics.get("neighbors", [])
                    if _to_optional_float(neighbor.get("price")) is not None
                ],
                "good_value_probability": good_value_probability,
                "prediction": prediction,
                "prediction_reason": prediction_reason,
            }
        )

    return {
        "candidate": resolved_candidate.to_output_dict(),
        "ewom_result": resolved_ewom_result,
        "k_sweep": sweep_rows,
        "pricing": {
            "include_shipping_in_total": bool(include_shipping_in_total),
            "prefer_converted_usd": bool(prefer_converted_usd),
            "total_price_currency": resolve_candidate_total_price_currency(
                resolved_candidate,
                prefer_converted_usd=prefer_converted_usd,
            ),
        },
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

    pricing = result.get("pricing")
    include_shipping_in_total = True
    prefer_converted_usd = False
    if isinstance(pricing, Mapping):
        include_shipping_in_total = bool(pricing.get("include_shipping_in_total", True))
        prefer_converted_usd = bool(pricing.get("prefer_converted_usd", False))

    total_price = resolve_candidate_total_price(
        candidate,
        include_shipping_in_total=include_shipping_in_total,
        prefer_converted_usd=prefer_converted_usd,
    )
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
    decision = result.get("decision")
    decision = decision if isinstance(decision, Mapping) else {}

    return {
        "source_url": candidate.get("source_url"),
        "title": candidate.get("title"),
        "total_price": resolved_input.get("price"),
        "total_price_currency": (
            result.get("pricing", {}).get("total_price_currency")
            if isinstance(result.get("pricing"), Mapping)
            else None
        ),
        "peer_price": resolved_input.get("peer_price"),
        "price_gap_vs_peer": derived_metrics.get("price_gap_vs_peer"),
        "good_value_probability": probability,
        "prediction": (
            decision.get("prediction")
            if decision.get("prediction") is not None
            else _resolve_probability_label(
                probability,
                threshold=probability_threshold,
            )
        ),
        "prediction_reason": decision.get("reason"),
        "trust_probability": resolved_input.get("trust_probability"),
        "ewom_score_0_to_100": resolved_input.get("ewom_score_0_to_100"),
        "seller_feedback_review_count": fused_agent_signals.get("review_count"),
        "retrieval_status": (
            result.get("market_context", {}).get("retrieval_status")
            if isinstance(result.get("market_context"), Mapping)
            else None
        ),
        "retrieved_neighbor_count": (
            result.get("market_context", {}).get("neighbor_count")
            if isinstance(result.get("market_context"), Mapping)
            else None
        ),
        "price_considered": (
            result.get("pricing", {}).get("price_considered")
            if isinstance(result.get("pricing"), Mapping)
            else None
        ),
        "price_note": (
            result.get("pricing", {}).get("price_note")
            if isinstance(result.get("pricing"), Mapping)
            else None
        ),
        "peer_price_source": (
            result.get("pricing", {}).get("peer_price_source")
            if isinstance(result.get("pricing"), Mapping)
            else None
        ),
    }


def _build_comparison_summary(
    result: Mapping[str, Any],
    *,
    probability_threshold: float,
    force_neutral_price: bool,
) -> dict[str, Any]:
    summary = summarize_ebay_candidate_value_result(
        result,
        probability_threshold=probability_threshold,
    )
    if not force_neutral_price:
        return summary

    neutral_probability = _score_neutral_price_probability_from_result(result)
    summary["good_value_probability"] = neutral_probability
    summary["prediction"] = _resolve_probability_label(
        neutral_probability,
        threshold=probability_threshold,
    )
    summary["prediction_reason"] = (
        "bayesian_probability_without_price"
        if neutral_probability is not None
        else None
    )
    summary["peer_price"] = None
    summary["price_gap_vs_peer"] = None
    summary["price_considered"] = False
    summary["price_note"] = (
        "Comparison used a neutral fair-price bucket because at least one listing "
        "lacked trustworthy similar-item price data."
    )
    summary["peer_price_source"] = "none"
    return summary


def compare_ebay_candidate_value_results(
    result_a: Mapping[str, Any],
    result_b: Mapping[str, Any],
    *,
    probability_threshold: float = 0.50,
    tie_margin: float = 0.03,
) -> dict[str, Any]:
    use_neutral_price_compare = not (
        _result_has_trusted_price_context(result_a)
        and _result_has_trusted_price_context(result_b)
    )
    summary_a = _build_comparison_summary(
        result_a,
        probability_threshold=probability_threshold,
        force_neutral_price=use_neutral_price_compare,
    )
    summary_b = _build_comparison_summary(
        result_b,
        probability_threshold=probability_threshold,
        force_neutral_price=use_neutral_price_compare,
    )
    verdict, reasons = _compare_scored_summaries(
        summary_a,
        summary_b,
        tie_margin=tie_margin,
    )
    if use_neutral_price_compare:
        reasons = [
            (
                "At least one listing lacked trustworthy similar-item price data, so both "
                "listings were rescored with a neutral fair-price bucket. This comparison "
                "does not consider price advantage."
            ),
            *reasons,
        ]
    probability_a = _to_optional_float(summary_a.get("good_value_probability"))
    probability_b = _to_optional_float(summary_b.get("good_value_probability"))
    return {
        "listing_a": summary_a,
        "listing_b": summary_b,
        "comparison": {
            "verdict": verdict,
            "price_comparison_mode": (
                "neutral_fallback" if use_neutral_price_compare else "peer_price"
            ),
            "good_value_probability_delta": (
                probability_a - probability_b
                if probability_a is not None and probability_b is not None
                else None
            ),
            "tie_margin_used": float(tie_margin),
            "reasons": reasons,
        },
    }


def _score_neutral_price_probability_from_result(result: Mapping[str, Any]) -> float | None:
    bayesian_result = result.get("bayesian_result")
    if not isinstance(bayesian_result, Mapping):
        return None
    resolved_input = bayesian_result.get("resolved_input")
    if not isinstance(resolved_input, Mapping):
        return None
    neutral_input = dict(resolved_input)
    neutral_input["peer_price"] = None
    rescored = score_good_value_probability(
        BayesianValueInput.from_mapping(neutral_input),
        default_relative_price_bucket=NEUTRAL_PRICE_BUCKET,
    )
    return _to_optional_float(rescored.get("good_value_probability"))


def resolve_candidate_total_price(
    candidate: Candidate | Mapping[str, Any],
    *,
    include_shipping_in_total: bool = True,
    prefer_converted_usd: bool = False,
) -> float | None:
    resolved_candidate = _coerce_candidate(candidate)
    item_price = _extract_money_value(
        resolved_candidate.price,
        prefer_converted_usd=prefer_converted_usd,
    )

    if item_price is None:
        return None
    if not include_shipping_in_total:
        return item_price

    shipping_cost = _extract_min_shipping_cost(
        resolved_candidate.shipping,
        prefer_converted_usd=prefer_converted_usd,
    )
    if shipping_cost is None:
        return item_price
    return item_price + shipping_cost


def resolve_candidate_total_price_currency(
    candidate: Candidate | Mapping[str, Any],
    *,
    prefer_converted_usd: bool = False,
) -> str | None:
    resolved_candidate = _coerce_candidate(candidate)
    item_currency = _extract_money_currency(
        resolved_candidate.price,
        prefer_converted_usd=prefer_converted_usd,
    )
    if item_currency:
        return item_currency
    for option in resolved_candidate.shipping:
        shipping_currency = _extract_money_currency(
            option.get("shippingCost"),
            prefer_converted_usd=prefer_converted_usd,
        )
        if shipping_currency:
            return shipping_currency
    return None


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
    retrieval_title = _build_candidate_retrieval_title(candidate)
    raw_title = _clean_candidate_title_text(str(candidate.title or ""))
    parts = [
        retrieval_title,
        raw_title if raw_title != retrieval_title else "",
        str(candidate.condition or ""),
        _flatten_mapping_text(candidate.item_specifics),
    ]
    return " ".join(part for part in parts if part).strip()


def _build_candidate_retrieval_title(candidate: Candidate) -> str:
    brand = _extract_candidate_brand(candidate)
    model = _extract_candidate_item_specific_text(candidate, RETRIEVAL_MODEL_KEYS)
    type_text = _extract_candidate_item_specific_text(candidate, RETRIEVAL_TYPE_KEYS)
    cleaned_title = _clean_candidate_title_text(str(candidate.title or ""))

    parts: list[str] = []
    if model:
        normalized_model = _clean_candidate_title_text(model)
        if normalized_model:
            if brand and _normalize_matching_text(brand) not in _normalize_matching_text(normalized_model):
                parts.append(brand)
            parts.append(normalized_model)
    elif cleaned_title:
        if brand and _normalize_matching_text(brand) not in _normalize_matching_text(cleaned_title):
            parts.append(brand)
        parts.append(cleaned_title)

    normalized_parts = {_normalize_matching_text(part) for part in parts if part}
    normalized_type = _normalize_matching_text(type_text)
    if type_text and normalized_type and normalized_type not in normalized_parts:
        parts.append(_clean_candidate_title_text(type_text))

    resolved = " ".join(part for part in parts if part).strip()
    return resolved or cleaned_title or str(candidate.title or "").strip()


def _refine_candidate_market_context(
    candidate: Candidate,
    diagnostics: Mapping[str, Any],
    *,
    reranked_top_n: int,
    candidate_pool_size: int,
    min_peer_price_ratio: float,
    min_peer_neighbor_count: int,
) -> dict[str, Any]:
    raw_neighbors = [
        dict(neighbor)
        for neighbor in diagnostics.get("neighbors", [])
        if isinstance(neighbor, Mapping)
    ]
    profile = _build_candidate_retrieval_profile(
        candidate,
        listing_price=_to_optional_float(diagnostics.get("price")),
    )

    reranked_neighbors: list[dict[str, Any]] = []
    rejected_neighbors: list[dict[str, Any]] = []
    rejection_summary: dict[str, int] = {}
    for neighbor in raw_neighbors:
        rejection_reason = _resolve_neighbor_rejection_reason(profile, neighbor)
        match_score = _score_neighbor_match(profile, neighbor)
        enriched_neighbor = {
            **dict(neighbor),
            "match_score": match_score,
        }
        if rejection_reason is None:
            if match_score >= MIN_RERANK_MATCH_SCORE:
                reranked_neighbors.append(enriched_neighbor)
                continue
            rejection_reason = "weak_identity_match"
        rejected_neighbors.append(
            {
                **enriched_neighbor,
                "rejection_reason": rejection_reason,
            }
        )
        rejection_summary[rejection_reason] = rejection_summary.get(rejection_reason, 0) + 1

    reranked_neighbors.sort(
        key=lambda neighbor: (
            float(neighbor.get("match_score", 0.0)),
            float(neighbor.get("similarity", 0.0)),
        ),
        reverse=True,
    )
    filtered_neighbors = reranked_neighbors[: max(1, reranked_top_n)]

    weighted_neighbors = [
        (
            _to_optional_float(neighbor.get("price")),
            _to_optional_float(neighbor.get("similarity")),
        )
        for neighbor in filtered_neighbors
    ]
    peer_price = _weighted_average(
        [price for price, similarity in weighted_neighbors if price is not None and similarity is not None],
        [
            similarity
            for price, similarity in weighted_neighbors
            if price is not None and similarity is not None
        ],
    )
    average_similarity = (
        sum(
            _to_optional_float(neighbor.get("similarity")) or 0.0
            for neighbor in filtered_neighbors
        )
        / len(filtered_neighbors)
        if filtered_neighbors
        else 0.0
    )

    rejected_peer_price_reason = _resolve_peer_price_rejection_reason(
        profile,
        peer_price,
        filtered_neighbor_count=len(filtered_neighbors),
        min_peer_price_ratio=min_peer_price_ratio,
        min_peer_neighbor_count=min_peer_neighbor_count,
    )
    if rejected_peer_price_reason is not None:
        rejection_summary[rejected_peer_price_reason] = (
            rejection_summary.get(rejected_peer_price_reason, 0) + 1
        )
        peer_price = None

    listing_price = _to_optional_float(diagnostics.get("price"))
    raw_neighbor_count = int(diagnostics.get("neighbor_count", len(raw_neighbors)) or 0)
    return {
        "catalog_row_index": diagnostics.get("catalog_row_index"),
        "parent_asin": diagnostics.get("parent_asin"),
        "title": diagnostics.get("title"),
        "price": listing_price,
        "peer_price": peer_price,
        "raw_peer_price": _to_optional_float(diagnostics.get("peer_price")),
        "price_gap_vs_peer": _compute_price_gap_vs_peer(listing_price, peer_price),
        "neighbor_count": len(filtered_neighbors),
        "raw_neighbor_count": raw_neighbor_count,
        "average_neighbor_similarity": average_similarity,
        "top_k_neighbors_used": reranked_top_n,
        "candidate_pool_size_used": candidate_pool_size,
        "min_peer_price_ratio_used": float(min_peer_price_ratio),
        "min_peer_neighbor_count_used": int(min_peer_neighbor_count),
        "min_similarity_used": diagnostics.get("min_similarity_used"),
        "neighbors": filtered_neighbors,
        "rejected_neighbors": rejected_neighbors[:10],
        "rejection_summary": rejection_summary,
        "retrieval_status": _resolve_retrieval_status(
            peer_price=peer_price,
            raw_neighbor_count=raw_neighbor_count,
            filtered_neighbor_count=len(filtered_neighbors),
            rejected_peer_price_reason=rejected_peer_price_reason,
        ),
        "query_brand": profile["brand"],
        "query_accessory_like": profile["is_accessory"],
        "query_product_kind": profile.get("product_kind"),
    }


def _build_candidate_retrieval_profile(
    candidate: Candidate,
    *,
    listing_price: float | None = None,
) -> dict[str, Any]:
    brand = _extract_candidate_brand(candidate)
    retrieval_title = _build_candidate_retrieval_title(candidate)
    combined_text = " ".join(
        part
        for part in (
            str(candidate.title or ""),
            _flatten_mapping_text(candidate.item_specifics),
        )
        if part
    )
    core_identity_tokens = _extract_core_identity_tokens(
        retrieval_title,
        brand_normalized=_normalize_matching_text(brand),
    )
    return {
        "brand": brand,
        "brand_normalized": _normalize_matching_text(brand),
        "identity_tokens": _extract_retrieval_tokens(retrieval_title),
        "core_identity_tokens": core_identity_tokens,
        "required_identity_tokens": _extract_required_identity_tokens(core_identity_tokens),
        "product_kind": _infer_product_kind(retrieval_title) or _infer_product_kind(combined_text),
        "is_accessory": _is_candidate_accessory_listing(candidate, combined_text),
        "listing_price": (
            listing_price
            if listing_price is not None
            else resolve_candidate_total_price(candidate)
        ),
    }


def _extract_candidate_brand(candidate: Candidate) -> str | None:
    if not isinstance(candidate.item_specifics, Mapping):
        return None

    for preferred_key in PREFERRED_BRAND_KEYS:
        for raw_key, raw_value in candidate.item_specifics.items():
            normalized_key = _normalize_matching_text(raw_key)
            if preferred_key not in normalized_key:
                continue
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                cleaned = str(value or "").strip()
                normalized_value = _normalize_matching_text(cleaned)
                if not normalized_value or normalized_value in GENERIC_BRAND_VALUES:
                    continue
                return cleaned
    return None


def _extract_candidate_item_specific_text(
    candidate: Candidate,
    preferred_keys: tuple[str, ...],
) -> str | None:
    if not isinstance(candidate.item_specifics, Mapping):
        return None

    normalized_key_map = {
        _normalize_matching_text(raw_key): raw_value
        for raw_key, raw_value in candidate.item_specifics.items()
    }
    for preferred_key in preferred_keys:
        raw_value = normalized_key_map.get(_normalize_matching_text(preferred_key))
        if raw_value is None:
            continue
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        for value in values:
            cleaned = _clean_candidate_title_text(str(value or ""))
            if cleaned:
                return cleaned
    return None


def _score_neighbor_match(
    profile: Mapping[str, Any],
    neighbor: Mapping[str, Any],
) -> float:
    similarity = _to_optional_float(neighbor.get("similarity")) or 0.0
    neighbor_title = str(neighbor.get("title") or "")
    neighbor_store = str(neighbor.get("store") or "")
    neighbor_category = str(neighbor.get("main_category") or "")
    neighbor_tokens = _extract_retrieval_tokens(
        " ".join(part for part in (neighbor_title, neighbor_store, neighbor_category) if part)
    )

    query_tokens = set(profile.get("identity_tokens", set()) or set())
    core_tokens = set(profile.get("core_identity_tokens", set()) or set())
    required_tokens = set(profile.get("required_identity_tokens", set()) or set())
    token_overlap = (
        len(query_tokens.intersection(neighbor_tokens)) / max(1, len(query_tokens))
        if query_tokens
        else 0.0
    )
    core_overlap = (
        len(core_tokens.intersection(neighbor_tokens)) / max(1, len(core_tokens))
        if core_tokens
        else 0.0
    )
    brand_bonus = 0.0
    brand_normalized = str(profile.get("brand_normalized") or "").strip()
    if brand_normalized and brand_normalized in _normalize_matching_text(
        f"{neighbor_title} {neighbor_store}"
    ):
        brand_bonus = 1.0

    query_kind = str(profile.get("product_kind") or "").strip()
    neighbor_kind = _infer_product_kind(
        " ".join(part for part in (neighbor_title, neighbor_store, neighbor_category) if part)
    )
    kind_bonus = 1.0 if _are_product_kinds_compatible(query_kind, neighbor_kind) else 0.0

    category_penalty = 0.0
    normalized_category = _normalize_matching_text(neighbor_category)
    if (
        not bool(profile.get("is_accessory"))
        and ("case" in normalized_category or "accessories" in normalized_category)
    ):
        category_penalty = 0.15

    family_penalty = 0.0
    if not bool(profile.get("is_accessory")):
        if required_tokens and not required_tokens.intersection(neighbor_tokens):
            family_penalty = 0.30
        elif core_tokens and not core_tokens.intersection(neighbor_tokens):
            family_penalty = 0.18

    kind_penalty = 0.0
    if query_kind and neighbor_kind and not _are_product_kinds_compatible(query_kind, neighbor_kind):
        kind_penalty = 0.25

    return max(
        0.0,
        (0.45 * similarity)
        + (0.17 * token_overlap)
        + (0.28 * core_overlap)
        + (0.08 * kind_bonus)
        + (0.05 * brand_bonus)
        - category_penalty
        - family_penalty
        - kind_penalty,
    )


def _resolve_neighbor_rejection_reason(
    profile: Mapping[str, Any],
    neighbor: Mapping[str, Any],
) -> str | None:
    neighbor_text = " ".join(
        part
        for part in (
            str(neighbor.get("title") or ""),
            str(neighbor.get("store") or ""),
            str(neighbor.get("main_category") or ""),
        )
        if part
    )
    if bool(profile.get("is_accessory")):
        return None
    if _is_accessory_like_text(neighbor_text):
        return "accessory_mismatch"

    query_kind = str(profile.get("product_kind") or "").strip()
    neighbor_kind = _infer_product_kind(neighbor_text)
    if query_kind and neighbor_kind and not _are_product_kinds_compatible(query_kind, neighbor_kind):
        return "product_type_mismatch"
    return None


def _resolve_peer_price_rejection_reason(
    profile: Mapping[str, Any],
    peer_price: float | None,
    *,
    filtered_neighbor_count: int = 0,
    min_peer_price_ratio: float = IMPLAUSIBLY_LOW_PEER_PRICE_RATIO,
    min_peer_neighbor_count: int = MIN_PEER_NEIGHBOR_COUNT,
) -> str | None:
    listing_price = _to_optional_float(profile.get("listing_price"))
    if peer_price is None or listing_price is None or listing_price <= 0.0:
        return None
    if bool(profile.get("is_accessory")):
        return None
    if int(filtered_neighbor_count) < max(1, int(min_peer_neighbor_count)):
        return "insufficient_peer_neighbors"
    if listing_price < 40.0:
        return None
    resolved_ratio = float(min_peer_price_ratio)
    if resolved_ratio <= 0.0:
        return None
    if peer_price / listing_price < resolved_ratio:
        return "implausibly_low_peer_price"
    return None


def _resolve_retrieval_status(
    *,
    peer_price: float | None,
    raw_neighbor_count: int,
    filtered_neighbor_count: int,
    rejected_peer_price_reason: str | None,
) -> str:
    if rejected_peer_price_reason is not None:
        return rejected_peer_price_reason
    if peer_price is not None:
        if filtered_neighbor_count < max(1, min(3, raw_neighbor_count)):
            return "usable_but_thin"
        return "usable"
    if raw_neighbor_count <= 0:
        return "no_neighbors"
    if filtered_neighbor_count <= 0:
        return "all_neighbors_filtered"
    return "insufficient_evidence"


def _is_accessory_like_text(text: str) -> bool:
    normalized = _normalize_matching_text(text)
    if not normalized:
        return False
    if any(phrase in normalized for phrase in ACCESSORY_CUE_PHRASES):
        return True

    tokens = set(normalized.split())
    has_accessory_term = bool(tokens.intersection(ACCESSORY_TERMS))
    if not has_accessory_term and " charging case " not in f" {normalized} ":
        return False

    has_accessory_context = bool(tokens.intersection(ACCESSORY_CONTEXT_TERMS))
    if has_accessory_context:
        return True

    if " for " in f" {normalized} " and tokens.intersection(DEVICE_FAMILY_TERMS):
        return True

    if " charging case " in f" {normalized} " and not has_accessory_context:
        return False

    return has_accessory_term and not tokens.intersection(PRIMARY_PRODUCT_TYPE_TERMS)


def _is_candidate_accessory_listing(candidate: Candidate, combined_text: str) -> bool:
    type_text = ""
    if isinstance(candidate.item_specifics, Mapping):
        raw_type = candidate.item_specifics.get("Type")
        if isinstance(raw_type, list):
            type_text = " ".join(str(value).strip() for value in raw_type if str(value).strip())
        else:
            type_text = str(raw_type or "").strip()

    normalized_type = _normalize_matching_text(type_text)
    if normalized_type:
        if set(normalized_type.split()).intersection(ACCESSORY_TERMS):
            return True
        if any(term in normalized_type for term in PRIMARY_PRODUCT_TYPE_TERMS):
            return False

    return _is_accessory_like_text(combined_text)


def _clean_candidate_title_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""

    for noise_term in TITLE_NOISE_TERMS:
        cleaned = re.sub(rf"\b{re.escape(noise_term)}\b", " ", cleaned, flags=re.IGNORECASE)

    with_match = re.search(r"\bwith\b(.+)$", cleaned, flags=re.IGNORECASE)
    if with_match:
        trailing_text = with_match.group(1)
        normalized_trailing = _normalize_matching_text(trailing_text)
        if any(term in normalized_trailing for term in REMOVABLE_WITH_TERMS):
            cleaned = cleaned[:with_match.start()].strip()

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_,;/")
    return cleaned


def _extract_retrieval_tokens(text: str) -> set[str]:
    normalized = _normalize_matching_text(text)
    if not normalized:
        return set()
    return {
        token
        for token in normalized.split()
        if len(token) >= 2 and token not in RETRIEVAL_STOPWORDS
    }


def _extract_core_identity_tokens(
    text: str,
    *,
    brand_normalized: str = "",
) -> set[str]:
    brand_tokens = set(brand_normalized.split()) if brand_normalized else set()
    tokens = _extract_retrieval_tokens(text)
    core_tokens = {
        token
        for token in tokens
        if token not in brand_tokens
        and token not in CORE_IDENTITY_STOPWORDS
        and token not in PRIMARY_PRODUCT_TYPE_TOKENS
    }
    if core_tokens:
        return core_tokens
    return {token for token in tokens if token not in brand_tokens}


def _extract_required_identity_tokens(tokens: set[str]) -> set[str]:
    return {
        token
        for token in tokens
        if len(token) >= 4
        or any(character.isdigit() for character in token)
        or token in IMPORTANT_VARIANT_TOKENS
    }


def _infer_product_kind(text: str) -> str | None:
    tokens = _extract_retrieval_tokens(text)
    if not tokens:
        return None
    best_kind: str | None = None
    best_score = 0
    for product_kind, aliases in PRODUCT_KIND_ALIASES.items():
        overlap_score = len(tokens.intersection(aliases))
        if overlap_score > best_score:
            best_kind = product_kind
            best_score = overlap_score
    return best_kind


def _are_product_kinds_compatible(
    query_kind: str,
    neighbor_kind: str,
) -> bool:
    resolved_query_kind = str(query_kind or "").strip()
    resolved_neighbor_kind = str(neighbor_kind or "").strip()
    if not resolved_query_kind or not resolved_neighbor_kind:
        return False
    if resolved_query_kind == resolved_neighbor_kind:
        return True
    return (
        resolved_query_kind in AUDIO_PRODUCT_KINDS
        and resolved_neighbor_kind in AUDIO_PRODUCT_KINDS
    )


def _resolve_reranked_top_n(top_k_neighbors: int | None) -> int:
    if top_k_neighbors is None:
        return DEFAULT_RERANKED_NEIGHBOR_COUNT
    value = int(top_k_neighbors)
    if value <= 0:
        raise ValueError("top_k_neighbors must be a positive integer when provided.")
    return value


def _resolve_candidate_pool_size(
    candidate_pool_size: int | None,
    *,
    reranked_top_n: int,
) -> int:
    if candidate_pool_size is None:
        return reranked_top_n
    value = int(candidate_pool_size)
    if value <= 0:
        raise ValueError("retrieval candidate pool size must be a positive integer.")
    return max(value, reranked_top_n)


def _compute_price_gap_vs_peer(
    price: float | None,
    peer_price: float | None,
) -> float | None:
    if price is None or peer_price is None or peer_price <= 0.0:
        return None
    return (peer_price - price) / peer_price


def _weighted_average(values: list[float | None], weights: list[float | None]) -> float | None:
    resolved_pairs = [
        (float(value), float(weight))
        for value, weight in zip(values, weights)
        if value is not None and weight is not None and weight > 0.0
    ]
    if not resolved_pairs:
        return None

    weighted_total = sum(value * weight for value, weight in resolved_pairs)
    total_weight = sum(weight for _, weight in resolved_pairs)
    if total_weight <= 0.0:
        return None
    return weighted_total / total_weight


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


def _normalize_matching_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _extract_money_value(
    value: Any,
    *,
    prefer_converted_usd: bool = False,
) -> float | None:
    if isinstance(value, Mapping):
        if prefer_converted_usd:
            converted_currency = str(value.get("convertedFromCurrency") or "").strip().upper()
            converted_value = _to_optional_float(value.get("convertedFromValue"))
            if converted_currency == "USD" and converted_value is not None:
                return converted_value
        return _to_optional_float(value.get("value"))
    return _to_optional_float(value)


def _extract_money_currency(
    value: Any,
    *,
    prefer_converted_usd: bool = False,
) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if prefer_converted_usd:
        converted_currency = str(value.get("convertedFromCurrency") or "").strip().upper()
        converted_value = _to_optional_float(value.get("convertedFromValue"))
        if converted_currency == "USD" and converted_value is not None:
            return converted_currency
    currency = str(value.get("currency") or "").strip().upper()
    return currency or None


def _extract_min_shipping_cost(
    shipping_options: list[dict[str, Any]],
    *,
    prefer_converted_usd: bool = False,
) -> float | None:
    costs: list[float] = []
    for option in shipping_options:
        if not isinstance(option, Mapping):
            continue
        shipping_cost = _extract_money_value(
            option.get("shippingCost"),
            prefer_converted_usd=prefer_converted_usd,
        )
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


def _build_ebay_value_decision(
    *,
    good_value_probability: float | None,
    market_context: Mapping[str, Any] | None,
    peer_price_source: str,
    price_considered: bool,
    price_note: str | None,
    probability_threshold: float = 0.50,
) -> dict[str, str | bool | None]:
    if not price_considered and price_note:
        prediction = _resolve_probability_label(
            good_value_probability,
            threshold=probability_threshold,
        )
        reason = (
            "bayesian_probability_without_price"
            if good_value_probability is not None
            else None
        )
        return {
            "prediction": prediction,
            "reason": reason,
            "price_considered": False,
            "price_note": price_note,
        }

    prediction, reason = _resolve_prediction_from_market_context(
        good_value_probability,
        market_context.get("retrieval_status") if isinstance(market_context, Mapping) else None,
        peer_price_source=peer_price_source,
        threshold=probability_threshold,
    )
    return {
        "prediction": prediction,
        "reason": reason,
        "price_considered": price_considered,
        "price_note": price_note,
    }


def _compare_scored_summaries(
    summary_a: Mapping[str, Any],
    summary_b: Mapping[str, Any],
    *,
    tie_margin: float,
) -> tuple[str, list[str]]:
    prediction_a = str(summary_a.get("prediction") or "").strip()
    prediction_b = str(summary_b.get("prediction") or "").strip()
    probability_a = _to_optional_float(summary_a.get("good_value_probability"))
    probability_b = _to_optional_float(summary_b.get("good_value_probability"))
    price_gap_a = _to_optional_float(summary_a.get("price_gap_vs_peer"))
    price_gap_b = _to_optional_float(summary_b.get("price_gap_vs_peer"))
    title_a = str(summary_a.get("title") or summary_a.get("source_url") or "listing_a")
    title_b = str(summary_b.get("title") or summary_b.get("source_url") or "listing_b")

    if prediction_a == "insufficient_evidence" or prediction_b == "insufficient_evidence":
        reasons: list[str] = []
        if prediction_a == "insufficient_evidence":
            reasons.append(
                f"Listing A lacks sufficient price evidence: {summary_a.get('prediction_reason') or 'unknown'}."
            )
        if prediction_b == "insufficient_evidence":
            reasons.append(
                f"Listing B lacks sufficient price evidence: {summary_b.get('prediction_reason') or 'unknown'}."
            )
        return "insufficient_evidence", reasons

    if probability_a is None or probability_b is None:
        return "insufficient_evidence", [
            "At least one listing is missing a Bayesian probability."
        ]

    delta = probability_a - probability_b
    if abs(delta) <= max(0.0, float(tie_margin)):
        if price_gap_a is not None and price_gap_b is not None:
            price_gap_delta = price_gap_a - price_gap_b
            if abs(price_gap_delta) > PRICE_GAP_TIE_MARGIN:
                if price_gap_delta > 0.0:
                    return "better_A", [
                        (
                            f"Bayesian probabilities are tied, so the comparison falls back to "
                            f"price gap vs peer. Listing A is cheaper relative to peers "
                            f"({price_gap_a:.4f} vs {price_gap_b:.4f})."
                        )
                    ]
                return "better_B", [
                    (
                        f"Bayesian probabilities are tied, so the comparison falls back to "
                        f"price gap vs peer. Listing B is cheaper relative to peers "
                        f"({price_gap_b:.4f} vs {price_gap_a:.4f})."
                    )
                ]
        return "tie", [
            (
                f"{title_a} and {title_b} are within the tie margin "
                f"({abs(delta):.4f} <= {float(tie_margin):.4f})."
            )
        ]

    if delta > 0.0:
        return "better_A", [
            f"Listing A has the higher good-value probability ({probability_a:.4f} vs {probability_b:.4f})."
        ]
    return "better_B", [
        f"Listing B has the higher good-value probability ({probability_b:.4f} vs {probability_a:.4f})."
    ]


def _resolve_prediction_from_market_context(
    probability: float | None,
    retrieval_status: Any,
    *,
    peer_price_source: str,
    threshold: float,
) -> tuple[str | None, str | None]:
    resolved_status = str(retrieval_status or "").strip()
    resolved_source = str(peer_price_source or "none").strip().lower()

    if resolved_status:
        if resolved_source == "none":
            return "insufficient_evidence", resolved_status
        if (
            resolved_source == "retrieval"
            and resolved_status not in SUFFICIENT_RETRIEVAL_STATUSES
        ):
            return "insufficient_evidence", resolved_status

    return (
        _resolve_probability_label(probability, threshold=threshold),
        "bayesian_probability" if probability is not None else None,
    )


def _result_has_trusted_price_context(result: Mapping[str, Any]) -> bool:
    pricing = result.get("pricing")
    pricing = pricing if isinstance(pricing, Mapping) else {}
    peer_price_source = str(pricing.get("peer_price_source") or "none").strip().lower()
    if peer_price_source == "manual":
        return True
    if peer_price_source != "retrieval":
        return False

    market_context = result.get("market_context")
    market_context = market_context if isinstance(market_context, Mapping) else {}
    retrieval_status = str(market_context.get("retrieval_status") or "").strip()
    return retrieval_status in SUFFICIENT_RETRIEVAL_STATUSES
