from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .listing_kind import (
    LISTING_KIND_VALUES,
    LISTING_KIND_OTHER,
    infer_listing_kind_from_row,
    normalize_listing_kind,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorthBuyingConfig:
    word_max_features: int = 40_000
    char_max_features: int = 30_000
    min_df: int = 2
    top_k_neighbors: int = 20
    neighbor_candidate_multiplier: int = 4
    min_neighbors: int = 5
    min_similarity: float = 0.12
    neighbor_query_chunk_size: int = 2_048
    price_score_scale: float = 0.20
    bayesian_rating_prior_weight: float = 20.0
    review_volume_scale: float = 50.0
    helpful_vote_scale: float = 5.0
    low_rating_penalty_start: float = 3.6
    low_rating_penalty_per_star: float = 0.22
    price_weight: float = 0.45
    review_weight: float = 0.40
    confidence_weight: float = 0.15
    worth_buying_threshold: float = 0.62
    consider_threshold: float = 0.48
    min_confidence_for_verdict: float = 0.35


def load_prepared_catalog(
    split_path: str | Path,
    *,
    max_rows: int | None = None,
) -> pd.DataFrame:
    resolved_path = Path(split_path).expanduser().resolve()
    LOGGER.info("Loading prepared value split from %s", resolved_path)
    rows: list[dict[str, Any]] = []
    with resolved_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_rows is not None and index + 1 >= max_rows:
                break

    if not rows:
        raise ValueError(f"No rows loaded from {resolved_path}")
    frame = pd.DataFrame(rows)
    return _normalize_catalog_frame(frame)


def train_worth_buying_pipeline(
    *,
    train_path: str | Path,
    output_prefix: str | Path,
    config: WorthBuyingConfig | None = None,
    max_rows: int | None = None,
    allowed_listing_kinds: list[str] | tuple[str, ...] | None = None,
    filtered_catalog_output_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config = config or WorthBuyingConfig()
    source_catalog = load_prepared_catalog(train_path, max_rows=max_rows).reset_index(drop=True)
    if source_catalog.empty:
        raise ValueError("Training split is empty after normalization.")
    resolved_allowed_listing_kinds = _resolve_allowed_listing_kinds(allowed_listing_kinds)
    train_catalog = source_catalog.loc[source_catalog["price"].notna()].copy().reset_index(drop=True)
    if resolved_allowed_listing_kinds is not None:
        train_catalog = train_catalog.loc[
            train_catalog["listing_kind"].isin(resolved_allowed_listing_kinds)
        ].reset_index(drop=True)
    if train_catalog.empty:
        raise ValueError("Training split does not contain any priced products to fit neighbors on.")
    resolved_filtered_catalog_output_path: Path | None = None
    if filtered_catalog_output_path is not None:
        resolved_filtered_catalog_output_path = Path(filtered_catalog_output_path).expanduser().resolve()
        _write_jsonl_records(
            resolved_filtered_catalog_output_path,
            train_catalog.to_dict(orient="records"),
        )

    feature_bundle = _fit_product_vectorizers(train_catalog, resolved_config)
    feature_matrix = feature_bundle["feature_matrix"]
    candidate_neighbors = min(
        max(2, resolved_config.top_k_neighbors * resolved_config.neighbor_candidate_multiplier + 1),
        len(train_catalog),
    )
    neighbor_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=candidate_neighbors)
    neighbor_model.fit(feature_matrix)

    global_rating_mean = float(
        train_catalog["average_rating"].dropna().mean()
        if train_catalog["average_rating"].notna().any()
        else 3.8
    )
    kind_models = _build_kind_models(
        train_catalog=train_catalog,
        word_vectorizer=feature_bundle["word_vectorizer"],
        char_vectorizer=feature_bundle["char_vectorizer"],
        config=resolved_config,
    )

    bundle = {
        "config": asdict(resolved_config),
        "word_vectorizer": feature_bundle["word_vectorizer"],
        "char_vectorizer": feature_bundle["char_vectorizer"],
        "neighbor_model": neighbor_model,
        "train_catalog": train_catalog,
        "kind_models": kind_models,
        "global_rating_mean": global_rating_mean,
        "allowed_listing_kinds": resolved_allowed_listing_kinds,
    }

    resolved_output_prefix = Path(output_prefix).expanduser().resolve()
    resolved_output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = resolved_output_prefix.with_suffix(".joblib")
    metadata_path = resolved_output_prefix.with_name(f"{resolved_output_prefix.name}_metadata.json")
    joblib.dump(bundle, model_path)

    metadata = {
        "model_path": str(model_path),
        "train_path": str(Path(train_path).expanduser().resolve()),
        "rows_loaded": int(len(source_catalog)),
        "rows_fitted": int(len(train_catalog)),
        "priced_rows": int(train_catalog["price"].notna().sum()),
        "filtered_catalog_output_path": (
            str(resolved_filtered_catalog_output_path)
            if resolved_filtered_catalog_output_path is not None
            else None
        ),
        "allowed_listing_kinds": (
            list(resolved_allowed_listing_kinds)
            if resolved_allowed_listing_kinds is not None
            else None
        ),
        "rows_with_reviews": int((source_catalog["review_count"] > 0).sum()),
        "listing_kind_counts": {
            key: int(value)
            for key, value in train_catalog["listing_kind"].value_counts(dropna=False).to_dict().items()
        },
        "kind_model_rows": {
            key: int(model_payload["rows"])
            for key, model_payload in kind_models.items()
        },
        "global_rating_mean": global_rating_mean,
        "config": asdict(resolved_config),
    }
    _write_json(metadata_path, metadata)

    LOGGER.info(
        "Saved worth-buying model to %s with fitted_train_rows=%s",
        model_path,
        len(train_catalog),
    )
    return metadata


def score_worth_buying_split(
    *,
    model_path: str | Path,
    split_path: str | Path,
    output_path: str | Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    split_catalog = load_prepared_catalog(split_path, max_rows=max_rows).reset_index(drop=True)
    scored = score_worth_buying_catalog(
        split_catalog,
        model_path=model_path,
    )

    resolved_output_path: Path | None = None
    if output_path is not None:
        resolved_output_path = Path(output_path).expanduser().resolve()
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(resolved_output_path, index=False)

    summary = {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "split_path": str(Path(split_path).expanduser().resolve()),
        "output_path": str(resolved_output_path) if resolved_output_path else None,
        "rows_scored": int(len(scored)),
        "worth_buying_rows": int((scored["verdict"] == "worth_buying").sum()),
        "consider_rows": int((scored["verdict"] == "consider").sum()),
        "skip_rows": int((scored["verdict"] == "skip").sum()),
        "insufficient_evidence_rows": int((scored["verdict"] == "insufficient_evidence").sum()),
        "score_quantiles": {
            "p10": float(scored["worth_buying_score"].quantile(0.10)),
            "p50": float(scored["worth_buying_score"].quantile(0.50)),
            "p90": float(scored["worth_buying_score"].quantile(0.90)),
        },
    }
    return summary


def load_model(model_path: str | Path) -> dict[str, Any]:
    resolved_path = Path(model_path).expanduser().resolve()
    LOGGER.info("Loading worth-buying model from %s", resolved_path)
    return joblib.load(resolved_path)


def score_worth_buying_catalog(
    catalog: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
    model_bundle: dict[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if (model_path is None) == (model_bundle is None):
        raise ValueError("Provide exactly one of model_path or model_bundle.")

    bundle = model_bundle if model_bundle is not None else load_model(model_path)
    config = _resolve_runtime_config(bundle, config_overrides=config_overrides)
    resolved_catalog = _normalize_catalog_frame(catalog).reset_index(drop=True)
    search_results = _collect_neighbor_candidates(
        query_catalog=resolved_catalog,
        bundle=bundle,
        config=config,
    )
    return _score_catalog(
        query_catalog=resolved_catalog,
        search_results=search_results,
        config=config,
        global_rating_mean=float(bundle["global_rating_mean"]),
    )


def inspect_worth_buying_catalog_neighbors(
    catalog: pd.DataFrame,
    *,
    model_path: str | Path | None = None,
    model_bundle: dict[str, Any] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if (model_path is None) == (model_bundle is None):
        raise ValueError("Provide exactly one of model_path or model_bundle.")

    bundle = model_bundle if model_bundle is not None else load_model(model_path)
    config = _resolve_runtime_config(bundle, config_overrides=config_overrides)
    resolved_catalog = _normalize_catalog_frame(catalog).reset_index(drop=True)
    search_results = _collect_neighbor_candidates(
        query_catalog=resolved_catalog,
        bundle=bundle,
        config=config,
    )

    diagnostics: list[dict[str, Any]] = []
    for row_index, (_, product) in enumerate(resolved_catalog.iterrows()):
        search_result = search_results[row_index]
        neighbors = _select_neighbors(
            query_parent_asin=_safe_get_string(product, "parent_asin"),
            train_catalog=search_result["train_catalog"],
            candidate_indices=search_result["indices"],
            candidate_distances=search_result["distances"],
            config=config,
        )
        average_similarity = (
            float(np.mean([neighbor["similarity"] for neighbor in neighbors]))
            if neighbors
            else 0.0
        )
        peer_price = _weighted_average(
            [neighbor["price"] for neighbor in neighbors],
            [neighbor["similarity"] for neighbor in neighbors],
        )
        diagnostics.append(
            {
                "catalog_row_index": int(row_index),
                "parent_asin": _safe_get_string(product, "parent_asin"),
                "title": _safe_get_string(product, "title"),
                "price": _to_optional_float(product.get("price")),
                "peer_price": peer_price,
                "neighbor_count": len(neighbors),
                "average_neighbor_similarity": average_similarity,
                "top_k_neighbors_used": int(config.top_k_neighbors),
                "min_similarity_used": float(config.min_similarity),
                "listing_kind": _safe_get_string(product, "listing_kind"),
                "retrieval_listing_kind": _safe_get_string(search_result, "listing_kind"),
                "neighbors": neighbors,
            }
        )
    return diagnostics


def _normalize_catalog_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in (
        "parent_asin",
        "title",
        "store",
        "main_category",
        "product_document",
        "listing_kind",
    ):
        if column not in normalized.columns:
            normalized[column] = ""
        normalized[column] = normalized[column].map(_safe_text)

    if "listing_kind" not in normalized.columns:
        normalized["listing_kind"] = ""
    normalized["listing_kind"] = normalized["listing_kind"].map(normalize_listing_kind)
    missing_kind_mask = normalized["listing_kind"].eq(LISTING_KIND_OTHER)
    if missing_kind_mask.any():
        inferred_kinds = normalized.loc[missing_kind_mask].apply(
            lambda row: infer_listing_kind_from_row(row.to_dict()),
            axis=1,
        )
        normalized.loc[missing_kind_mask, "listing_kind"] = [
            normalize_listing_kind(value) for value in inferred_kinds.tolist()
        ]

    for column in (
        "price",
        "average_rating",
        "rating_number",
        "verified_purchase_rate",
        "helpful_vote_total",
        "helpful_vote_avg",
        "avg_review_rating",
        "trust_probability",
        "ewom_score_0_to_100",
        "ewom_magnitude_0_to_100",
    ):
        if column not in normalized.columns:
            normalized[column] = np.nan
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "review_count" not in normalized.columns:
        normalized["review_count"] = 0
    normalized["review_count"] = pd.to_numeric(
        normalized["review_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    return normalized


def _build_kind_models(
    *,
    train_catalog: pd.DataFrame,
    word_vectorizer: TfidfVectorizer,
    char_vectorizer: TfidfVectorizer,
    config: WorthBuyingConfig,
) -> dict[str, dict[str, Any]]:
    kind_models: dict[str, dict[str, Any]] = {}
    if "listing_kind" not in train_catalog.columns:
        return kind_models

    for listing_kind in sorted(
        {
            normalize_listing_kind(value)
            for value in train_catalog["listing_kind"].dropna().tolist()
        }
    ):
        if not listing_kind or listing_kind == LISTING_KIND_OTHER:
            continue
        subset = train_catalog.loc[
            train_catalog["listing_kind"] == listing_kind
        ].reset_index(drop=True)
        if len(subset) < 2:
            continue

        feature_matrix = _transform_catalog(
            subset,
            {
                "word_vectorizer": word_vectorizer,
                "char_vectorizer": char_vectorizer,
            },
        )
        candidate_neighbors = min(
            max(2, config.top_k_neighbors * config.neighbor_candidate_multiplier + 1),
            len(subset),
        )
        neighbor_model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=candidate_neighbors,
        )
        neighbor_model.fit(feature_matrix)
        kind_models[listing_kind] = {
            "rows": int(len(subset)),
            "train_catalog": subset,
            "neighbor_model": neighbor_model,
        }

    return kind_models


def _fit_product_vectorizers(
    catalog: pd.DataFrame,
    config: WorthBuyingConfig,
) -> dict[str, Any]:
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=config.min_df,
        max_features=config.word_max_features,
        stop_words="english",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=config.min_df,
        max_features=config.char_max_features,
    )
    documents = catalog["product_document"].fillna("").tolist()
    LOGGER.info(
        "Fitting worth-buying vectorizers on %s products",
        len(documents),
    )
    word_matrix = word_vectorizer.fit_transform(documents)
    char_matrix = char_vectorizer.fit_transform(documents)
    feature_matrix = normalize(hstack([word_matrix, char_matrix]), norm="l2").tocsr()
    return {
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "feature_matrix": feature_matrix,
    }


def _transform_catalog(catalog: pd.DataFrame, bundle: dict[str, Any]) -> csr_matrix:
    documents = catalog["product_document"].fillna("").tolist()
    word_matrix = bundle["word_vectorizer"].transform(documents)
    char_matrix = bundle["char_vectorizer"].transform(documents)
    return normalize(hstack([word_matrix, char_matrix]), norm="l2").tocsr()


def _chunked_kneighbors(
    neighbor_model: NearestNeighbors,
    matrix: csr_matrix,
    n_neighbors: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    distance_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    total_rows = matrix.shape[0]
    for start in range(0, total_rows, max(1, chunk_size)):
        stop = min(total_rows, start + max(1, chunk_size))
        distances, indices = neighbor_model.kneighbors(matrix[start:stop], n_neighbors=n_neighbors)
        distance_chunks.append(distances)
        index_chunks.append(indices)
    return np.vstack(distance_chunks), np.vstack(index_chunks)


def _score_catalog(
    *,
    query_catalog: pd.DataFrame,
    search_results: list[dict[str, Any]],
    config: WorthBuyingConfig,
    global_rating_mean: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_index, (_, product) in enumerate(query_catalog.iterrows()):
        search_result = search_results[row_index]
        neighbors = _select_neighbors(
            query_parent_asin=_safe_text(product["parent_asin"]),
            train_catalog=search_result["train_catalog"],
            candidate_indices=search_result["indices"],
            candidate_distances=search_result["distances"],
            config=config,
        )
        neighbor_count = len(neighbors)
        average_similarity = (
            float(np.mean([neighbor["similarity"] for neighbor in neighbors]))
            if neighbors
            else 0.0
        )
        peer_price = _weighted_average(
            [neighbor["price"] for neighbor in neighbors],
            [neighbor["similarity"] for neighbor in neighbors],
        )
        price_gap_vs_peer = None
        if peer_price is not None and pd.notna(product["price"]) and float(peer_price) > 0.0:
            price_gap_vs_peer = (float(peer_price) - float(product["price"])) / float(peer_price)

        price_score = _price_alignment_score(
            price=float(product["price"]) if pd.notna(product["price"]) else None,
            peer_price=peer_price,
            scale=config.price_score_scale,
        )
        bayesian_rating_score = _bayesian_rating(
            average_rating=float(product["average_rating"]) if pd.notna(product["average_rating"]) else None,
            rating_number=float(product["rating_number"]) if pd.notna(product["rating_number"]) else None,
            prior_mean=global_rating_mean,
            prior_weight=config.bayesian_rating_prior_weight,
        )
        review_quality_score = _review_quality_score(
            average_rating=float(product["avg_review_rating"]) if pd.notna(product["avg_review_rating"]) else None,
            verified_purchase_rate=(
                float(product["verified_purchase_rate"])
                if pd.notna(product["verified_purchase_rate"])
                else None
            ),
            helpful_vote_total=(
                float(product["helpful_vote_total"])
                if pd.notna(product["helpful_vote_total"])
                else 0.0
            ),
            review_count=int(product["review_count"]),
            config=config,
            fallback_rating=float(product["average_rating"]) if pd.notna(product["average_rating"]) else None,
            prior_mean=global_rating_mean,
        )
        confidence_score = _confidence_score(
            neighbor_count=neighbor_count,
            average_similarity=average_similarity,
            review_count=int(product["review_count"]),
            config=config,
        )
        worth_buying_score = _worth_buying_score(
            price_score=price_score,
            bayesian_rating=bayesian_rating_score,
            review_quality=review_quality_score,
            confidence=confidence_score,
            average_rating=float(product["average_rating"]) if pd.notna(product["average_rating"]) else None,
            config=config,
        )
        verdict = _resolve_verdict(
            worth_buying_score=worth_buying_score,
            confidence=confidence_score,
            config=config,
        )
        rows.append(
            {
                "catalog_row_index": int(row_index),
                "parent_asin": product["parent_asin"],
                "title": product["title"],
                "price": float(product["price"]) if pd.notna(product["price"]) else None,
                "average_rating": (
                    float(product["average_rating"]) if pd.notna(product["average_rating"]) else None
                ),
                "rating_number": (
                    float(product["rating_number"]) if pd.notna(product["rating_number"]) else None
                ),
                "review_count": int(product["review_count"]),
                "listing_kind": _safe_text(product.get("listing_kind")),
                "peer_price": peer_price,
                "price_gap_vs_peer": price_gap_vs_peer,
                "neighbor_count": neighbor_count,
                "average_neighbor_similarity": average_similarity,
                "retrieval_listing_kind": _safe_text(search_result.get("listing_kind")),
                "price_alignment_score": price_score,
                "bayesian_rating_score": bayesian_rating_score,
                "review_quality_score": review_quality_score,
                "confidence_score": confidence_score,
                "worth_buying_score": worth_buying_score,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["worth_buying_score", "confidence_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _collect_neighbor_candidates(
    *,
    query_catalog: pd.DataFrame,
    bundle: Mapping[str, Any],
    config: WorthBuyingConfig,
) -> list[dict[str, Any]]:
    kind_models = bundle.get("kind_models")
    kind_models = kind_models if isinstance(kind_models, Mapping) else {}
    allowed_listing_kinds = _resolve_allowed_listing_kinds(bundle.get("allowed_listing_kinds"))
    grouped_indices: dict[str, list[int]] = {}
    for row_index, (_, product) in enumerate(query_catalog.iterrows()):
        listing_kind = normalize_listing_kind(product.get("listing_kind"))
        if allowed_listing_kinds is not None and listing_kind not in allowed_listing_kinds:
            grouped_indices.setdefault("__disallowed__", []).append(row_index)
            continue
        bundle_key = listing_kind if listing_kind in kind_models else "__all__"
        grouped_indices.setdefault(bundle_key, []).append(row_index)

    search_results: list[dict[str, Any] | None] = [None] * len(query_catalog)
    for bundle_key, row_indices in grouped_indices.items():
        if bundle_key == "__disallowed__":
            empty_catalog = pd.DataFrame(columns=query_catalog.columns)
            for row_index in row_indices:
                search_results[row_index] = {
                    "train_catalog": empty_catalog,
                    "distances": np.array([], dtype=float),
                    "indices": np.array([], dtype=int),
                    "listing_kind": "",
                }
            continue
        search_bundle = (
            kind_models[bundle_key]
            if bundle_key != "__all__"
            else {
                "train_catalog": bundle["train_catalog"],
                "neighbor_model": bundle["neighbor_model"],
            }
        )
        query_slice = query_catalog.iloc[row_indices].reset_index(drop=True)
        query_matrix = _transform_catalog(query_slice, bundle)
        train_catalog = search_bundle["train_catalog"]
        candidate_neighbors = min(
            max(2, config.top_k_neighbors * config.neighbor_candidate_multiplier + 1),
            len(train_catalog),
        )
        distances, indices = _chunked_kneighbors(
            search_bundle["neighbor_model"],
            query_matrix,
            candidate_neighbors,
            config.neighbor_query_chunk_size,
        )
        for local_index, row_index in enumerate(row_indices):
            search_results[row_index] = {
                "train_catalog": train_catalog,
                "distances": distances[local_index],
                "indices": indices[local_index],
                "listing_kind": bundle_key if bundle_key != "__all__" else "",
            }

    return [result or {} for result in search_results]


def _select_neighbors(
    *,
    query_parent_asin: str,
    train_catalog: pd.DataFrame,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    config: WorthBuyingConfig,
) -> list[dict[str, float]]:
    neighbors: list[dict[str, float]] = []
    for index, distance in zip(candidate_indices.tolist(), candidate_distances.tolist()):
        candidate = train_catalog.iloc[int(index)]
        candidate_parent_asin = _safe_text(candidate["parent_asin"])
        if candidate_parent_asin == query_parent_asin:
            continue
        if pd.isna(candidate["price"]):
            continue
        similarity = max(0.0, 1.0 - float(distance))
        if similarity < config.min_similarity:
            continue
        neighbors.append(
            {
                "parent_asin": candidate_parent_asin,
                "title": _safe_text(candidate.get("title")),
                "store": _safe_text(candidate.get("store")),
                "main_category": _safe_text(candidate.get("main_category")),
                "listing_kind": _safe_text(candidate.get("listing_kind")),
                "price": float(candidate["price"]),
                "similarity": similarity,
                "average_rating": _to_optional_float(candidate.get("average_rating")),
                "review_count": int(candidate.get("review_count", 0) or 0),
            }
        )
        if len(neighbors) >= config.top_k_neighbors:
            break
    return neighbors


def _price_alignment_score(
    *,
    price: float | None,
    peer_price: float | None,
    scale: float,
) -> float:
    if price is None or peer_price is None or peer_price <= 0.0:
        return 0.50
    relative_gap = (peer_price - price) / peer_price
    return _clamp(0.5 * (1.0 + math.tanh(relative_gap / max(scale, 1e-6))), 0.0, 1.0)


def _bayesian_rating(
    *,
    average_rating: float | None,
    rating_number: float | None,
    prior_mean: float,
    prior_weight: float,
) -> float:
    if average_rating is None:
        return _clamp(prior_mean / 5.0, 0.0, 1.0)
    observed_count = max(0.0, rating_number or 0.0)
    posterior_rating = (
        (prior_weight * prior_mean) + (observed_count * average_rating)
    ) / max(prior_weight + observed_count, 1e-6)
    return _clamp(posterior_rating / 5.0, 0.0, 1.0)


def _review_quality_score(
    *,
    average_rating: float | None,
    verified_purchase_rate: float | None,
    helpful_vote_total: float,
    review_count: int,
    config: WorthBuyingConfig,
    fallback_rating: float | None,
    prior_mean: float,
) -> float:
    resolved_rating = average_rating
    if resolved_rating is None:
        resolved_rating = fallback_rating if fallback_rating is not None else prior_mean
    rating_score = _clamp(resolved_rating / 5.0, 0.0, 1.0)
    verified_score = _clamp(verified_purchase_rate if verified_purchase_rate is not None else 0.5, 0.0, 1.0)
    helpful_score = 1.0 - math.exp(-max(0.0, helpful_vote_total) / max(config.helpful_vote_scale, 1e-6))
    volume_score = 1.0 - math.exp(-max(0, review_count) / max(config.review_volume_scale, 1e-6))
    return _clamp(
        (0.45 * rating_score)
        + (0.25 * verified_score)
        + (0.15 * helpful_score)
        + (0.15 * volume_score),
        0.0,
        1.0,
    )


def _confidence_score(
    *,
    neighbor_count: int,
    average_similarity: float,
    review_count: int,
    config: WorthBuyingConfig,
) -> float:
    neighbor_coverage = _clamp(neighbor_count / max(config.top_k_neighbors, 1), 0.0, 1.0)
    similarity_score = _clamp(average_similarity, 0.0, 1.0)
    volume_score = 1.0 - math.exp(-max(0, review_count) / max(config.review_volume_scale, 1e-6))
    return _clamp(
        (0.45 * neighbor_coverage) + (0.35 * similarity_score) + (0.20 * volume_score),
        0.0,
        1.0,
    )


def _worth_buying_score(
    *,
    price_score: float,
    bayesian_rating: float,
    review_quality: float,
    confidence: float,
    average_rating: float | None,
    config: WorthBuyingConfig,
) -> float:
    review_component = (0.55 * bayesian_rating) + (0.45 * review_quality)
    penalty = 0.0
    if average_rating is not None and average_rating < config.low_rating_penalty_start:
        penalty = (config.low_rating_penalty_start - average_rating) * config.low_rating_penalty_per_star
    raw_score = (
        (config.price_weight * price_score)
        + (config.review_weight * review_component)
        + (config.confidence_weight * confidence)
        - penalty
    )
    return _clamp(raw_score, 0.0, 1.0)


def _resolve_verdict(
    *,
    worth_buying_score: float,
    confidence: float,
    config: WorthBuyingConfig,
) -> str:
    if confidence < config.min_confidence_for_verdict:
        return "insufficient_evidence"
    if worth_buying_score >= config.worth_buying_threshold:
        return "worth_buying"
    if worth_buying_score >= config.consider_threshold:
        return "consider"
    return "skip"


def _weighted_average(values: list[float], weights: list[float]) -> float | None:
    if not values or not weights:
        return None
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if float(weight_array.sum()) <= 0.0:
        return None
    return float(np.average(value_array, weights=weight_array))


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_get_string(row: pd.Series, key: str) -> str:
    value = row.get(key, "")
    return _safe_text(value)


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


def _resolve_allowed_listing_kinds(
    value: Any,
) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_values = [str(item or "").strip() for item in value]
    else:
        raise TypeError(
            "allowed_listing_kinds must be a string, list, tuple, set, or None."
        )

    resolved: list[str] = []
    for raw_value in raw_values:
        if not raw_value:
            continue
        normalized = normalize_listing_kind(raw_value)
        if normalized not in LISTING_KIND_VALUES or normalized == LISTING_KIND_OTHER:
            continue
        if normalized not in resolved:
            resolved.append(normalized)
    return resolved or None


def _resolve_runtime_config(
    bundle: Mapping[str, Any],
    config_overrides: Mapping[str, Any] | None = None,
) -> WorthBuyingConfig:
    config_payload = dict(bundle["config"])
    if config_overrides:
        for key, value in config_overrides.items():
            if value is None:
                continue
            if key not in config_payload:
                raise ValueError(f"Unsupported worth-buying config override: {key}")
            config_payload[key] = value
    return WorthBuyingConfig(**config_payload)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _write_jsonl_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
