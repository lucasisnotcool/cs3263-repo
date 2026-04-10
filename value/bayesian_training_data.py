from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

# Allow `python value/bayesian_training_data.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.bayesian_value import BayesianValueInput, build_value_evidence
from value.electronics_filter import is_actual_electronics_device
from value.listing_kind import LISTING_KIND_DEVICE, normalize_listing_kind
from value.worth_buying import (
    WorthBuyingConfig,
    load_model as load_worth_buying_model,
    load_prepared_catalog,
    score_worth_buying_catalog,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "value" / "artifacts" / "amazon_worth_buying_devices_full.joblib"
DEFAULT_INPUT_PATHS = (
    PROJECT_ROOT / "data" / "value" / "electronics_split" / "electronics_products_train_devices_only.jsonl",
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "value"
    / "bayesian_training"
    / "amazon_electronics_bayesian_train.jsonl"
)


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_bayesian_training_dataset(
    *,
    input_paths: Iterable[str | Path],
    worth_buying_model_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    max_rows_per_input: int | None = None,
    allowed_listing_kinds: Iterable[str] = (LISTING_KIND_DEVICE,),
    strict_actual_device_filter: bool = True,
    min_review_count: int = 1,
    min_confidence: float | None = None,
    include_consider_as_no: bool = False,
    score_chunk_size: int = 2_048,
) -> dict[str, Any]:
    if score_chunk_size <= 0:
        raise ValueError("score_chunk_size must be positive.")

    resolved_input_paths = [Path(path).expanduser().resolve() for path in input_paths]
    if not resolved_input_paths:
        raise ValueError("input_paths must contain at least one path.")

    resolved_output_path = Path(output_path).expanduser().resolve()
    resolved_summary_path = (
        Path(summary_path).expanduser().resolve()
        if summary_path is not None
        else resolved_output_path.with_name(f"{resolved_output_path.stem}_summary.json")
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_summary_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = load_worth_buying_model(worth_buying_model_path)
    model_config = WorthBuyingConfig(**model_bundle["config"])
    confidence_floor = (
        model_config.min_confidence_for_verdict
        if min_confidence is None
        else float(min_confidence)
    )
    if not 0.0 <= confidence_floor <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1.")

    allowed_kinds = _normalize_allowed_listing_kinds(allowed_listing_kinds)
    summary: dict[str, Any] = {
        "worth_buying_model_path": str(Path(worth_buying_model_path).expanduser().resolve()),
        "output_path": str(resolved_output_path),
        "summary_path": str(resolved_summary_path),
        "allowed_listing_kinds": allowed_kinds,
        "strict_actual_device_filter": bool(strict_actual_device_filter),
        "min_review_count": int(min_review_count),
        "min_confidence": confidence_floor,
        "include_consider_as_no": bool(include_consider_as_no),
        "max_rows_per_input": max_rows_per_input,
        "score_chunk_size": int(score_chunk_size),
        "inputs": [],
        "rows_seen": 0,
        "rows_after_filter": 0,
        "rows_scored": 0,
        "rows_written": 0,
        "label_counts": {"yes": 0, "no": 0},
        "skipped_after_scoring": {},
    }

    with resolved_output_path.open("w", encoding="utf-8") as output_handle:
        for input_path in resolved_input_paths:
            input_summary = _process_input_path(
                input_path=input_path,
                output_handle=output_handle,
                model_bundle=model_bundle,
                allowed_listing_kinds=allowed_kinds,
                strict_actual_device_filter=strict_actual_device_filter,
                min_review_count=min_review_count,
                min_confidence=confidence_floor,
                include_consider_as_no=include_consider_as_no,
                max_rows=max_rows_per_input,
                score_chunk_size=score_chunk_size,
            )
            summary["inputs"].append(input_summary)
            summary["rows_seen"] += input_summary["rows_loaded"]
            summary["rows_after_filter"] += input_summary["rows_after_filter"]
            summary["rows_scored"] += input_summary["rows_scored"]
            summary["rows_written"] += input_summary["rows_written"]
            for label, count in input_summary["label_counts"].items():
                summary["label_counts"][label] += count
            for reason, count in input_summary["skipped_after_scoring"].items():
                summary["skipped_after_scoring"][reason] = (
                    summary["skipped_after_scoring"].get(reason, 0) + count
                )

    _write_json(resolved_summary_path, summary)
    return summary


def _process_input_path(
    *,
    input_path: Path,
    output_handle: Any,
    model_bundle: Mapping[str, Any],
    allowed_listing_kinds: list[str],
    strict_actual_device_filter: bool,
    min_review_count: int,
    min_confidence: float,
    include_consider_as_no: bool,
    max_rows: int | None,
    score_chunk_size: int,
) -> dict[str, Any]:
    LOGGER.info("Building Bayesian labels from %s", input_path)
    catalog = load_prepared_catalog(input_path, max_rows=max_rows).reset_index(drop=True)
    filtered_catalog = _filter_catalog(
        catalog,
        allowed_listing_kinds=allowed_listing_kinds,
        strict_actual_device_filter=strict_actual_device_filter,
        min_review_count=min_review_count,
    )
    summary: dict[str, Any] = {
        "input_path": str(input_path),
        "rows_loaded": int(len(catalog)),
        "rows_after_filter": int(len(filtered_catalog)),
        "rows_scored": 0,
        "rows_written": 0,
        "label_counts": {"yes": 0, "no": 0},
        "skipped_after_scoring": {},
    }
    if filtered_catalog.empty:
        return summary

    for start in range(0, len(filtered_catalog), score_chunk_size):
        stop = min(start + score_chunk_size, len(filtered_catalog))
        LOGGER.info(
            "Scoring Bayesian label chunk %s:%s of %s rows from %s",
            start,
            stop,
            len(filtered_catalog),
            input_path,
        )
        scored = score_worth_buying_catalog(
            filtered_catalog.iloc[start:stop].copy().reset_index(drop=True),
            model_bundle=model_bundle,
        )
        summary["rows_scored"] += int(len(scored))
        _write_scored_rows(
            scored=scored,
            source_catalog=filtered_catalog,
            source_offset=start,
            output_handle=output_handle,
            input_path=input_path,
            min_confidence=min_confidence,
            include_consider_as_no=include_consider_as_no,
            summary=summary,
        )
    return summary


def _write_scored_rows(
    *,
    scored: pd.DataFrame,
    source_catalog: pd.DataFrame,
    source_offset: int,
    output_handle: Any,
    input_path: Path,
    min_confidence: float,
    include_consider_as_no: bool,
    summary: dict[str, Any],
) -> None:
    for _, scored_row in scored.iterrows():
        source_row = source_catalog.iloc[source_offset + int(scored_row["catalog_row_index"])]
        label, skip_reason = _resolve_label(
            scored_row,
            min_confidence=min_confidence,
            include_consider_as_no=include_consider_as_no,
        )
        if label is None:
            _increment(summary["skipped_after_scoring"], skip_reason or "unlabeled")
            continue

        record = _build_training_record(
            source_row=source_row,
            scored_row=scored_row,
            input_path=input_path,
            label=label,
        )
        if record is None:
            _increment(summary["skipped_after_scoring"], "missing_bayesian_evidence")
            continue

        json.dump(record, output_handle, ensure_ascii=False)
        output_handle.write("\n")
        summary["rows_written"] += 1
        summary["label_counts"][label] += 1


def _filter_catalog(
    catalog: pd.DataFrame,
    *,
    allowed_listing_kinds: list[str],
    strict_actual_device_filter: bool,
    min_review_count: int,
) -> pd.DataFrame:
    mask = catalog["price"].notna()
    mask &= catalog["listing_kind"].map(normalize_listing_kind).isin(allowed_listing_kinds)
    if min_review_count > 0:
        mask &= catalog["review_count"] >= int(min_review_count)
    filtered = catalog.loc[mask].copy().reset_index(drop=True)
    if strict_actual_device_filter:
        device_mask = filtered.apply(
            lambda row: is_actual_electronics_device(row.to_dict()),
            axis=1,
        )
        filtered = filtered.loc[device_mask].copy().reset_index(drop=True)
    return filtered


def _resolve_label(
    scored_row: Mapping[str, Any],
    *,
    min_confidence: float,
    include_consider_as_no: bool,
) -> tuple[str | None, str | None]:
    confidence = _to_optional_float(scored_row.get("confidence_score"))
    if confidence is None or confidence < min_confidence:
        return None, "low_confidence"

    verdict = str(scored_row.get("verdict") or "").strip()
    if verdict == "worth_buying":
        return "yes", None
    if verdict == "skip":
        return "no", None
    if verdict == "consider" and include_consider_as_no:
        return "no", None
    return None, f"ambiguous_verdict:{verdict or 'missing'}"


def _build_training_record(
    *,
    source_row: pd.Series,
    scored_row: pd.Series,
    input_path: Path,
    label: str,
) -> dict[str, Any] | None:
    rating_count = _resolve_rating_count(source_row)
    bayesian_input = BayesianValueInput(
        trust_probability=_to_optional_float(source_row.get("trust_probability")),
        ewom_score_0_to_100=_to_optional_float(source_row.get("ewom_score_0_to_100")),
        ewom_magnitude_0_to_100=_to_optional_float(source_row.get("ewom_magnitude_0_to_100")),
        average_rating=_to_optional_float(source_row.get("average_rating")),
        rating_count=rating_count,
        verified_purchase_rate=_to_optional_float(source_row.get("verified_purchase_rate")),
        price=_to_optional_float(source_row.get("price")),
        peer_price=_to_optional_float(scored_row.get("peer_price")),
        warranty_months=_to_optional_float(source_row.get("warranty_months")),
        return_window_days=_to_optional_float(source_row.get("return_window_days")),
    )
    evidence, derived = build_value_evidence(bayesian_input)
    if "RelativePriceBucket" not in evidence:
        return None

    record = {
        "parent_asin": _safe_text(source_row.get("parent_asin")),
        "title": _safe_text(source_row.get("title")),
        "store": _safe_text(source_row.get("store")),
        "main_category": _safe_text(source_row.get("main_category")),
        "categories": _to_jsonable(source_row.get("categories")),
        "listing_kind": _safe_text(source_row.get("listing_kind")),
        "source_path": str(input_path),
        "price": bayesian_input.price,
        "peer_price": bayesian_input.peer_price,
        "price_gap_vs_peer": derived.get("price_gap_vs_peer"),
        "average_rating": bayesian_input.average_rating,
        "rating_count": rating_count,
        "rating_number": _to_optional_float(source_row.get("rating_number")),
        "review_count": int(source_row.get("review_count", 0) or 0),
        "verified_purchase_rate": bayesian_input.verified_purchase_rate,
        "trust_probability": bayesian_input.trust_probability,
        "ewom_score_0_to_100": bayesian_input.ewom_score_0_to_100,
        "ewom_magnitude_0_to_100": bayesian_input.ewom_magnitude_0_to_100,
        "warranty_months": bayesian_input.warranty_months,
        "return_window_days": bayesian_input.return_window_days,
        "bayesian_evidence": evidence,
        "good_value_label": label,
        "label_source": "worth_buying_retrieval_verdict",
        "retrieval_verdict": _safe_text(scored_row.get("verdict")),
        "retrieval_worth_buying_score": _to_optional_float(scored_row.get("worth_buying_score")),
        "retrieval_confidence_score": _to_optional_float(scored_row.get("confidence_score")),
        "retrieval_price_alignment_score": _to_optional_float(scored_row.get("price_alignment_score")),
        "retrieval_review_quality_score": _to_optional_float(scored_row.get("review_quality_score")),
        "retrieval_neighbor_count": int(scored_row.get("neighbor_count", 0) or 0),
        "retrieval_average_neighbor_similarity": _to_optional_float(
            scored_row.get("average_neighbor_similarity")
        ),
    }
    return {key: _to_jsonable(value) for key, value in record.items()}


def _resolve_rating_count(source_row: pd.Series) -> float | None:
    rating_number = _to_optional_float(source_row.get("rating_number"))
    if rating_number is not None:
        return rating_number
    return _to_optional_float(source_row.get("review_count"))


def _normalize_allowed_listing_kinds(values: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        normalized = normalize_listing_kind(value)
        if normalized and normalized not in resolved:
            resolved.append(normalized)
    if not resolved:
        raise ValueError("allowed_listing_kinds must resolve to at least one listing kind.")
    return resolved


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


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _to_jsonable(value.item())
    if not isinstance(value, (str, bytes)) and value is not None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a device-only Amazon Electronics dataset labeled for Bayesian "
            "good-value training using the retrieval worth-buying model."
        )
    )
    parser.add_argument("--input-path", type=Path, action="append", default=None)
    parser.add_argument("--worth-buying-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--max-rows-per-input", type=int, default=None)
    parser.add_argument("--allowed-listing-kinds", type=str, default=LISTING_KIND_DEVICE)
    parser.add_argument("--min-review-count", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--include-consider-as-no", action="store_true")
    parser.add_argument("--no-strict-actual-device-filter", action="store_true")
    parser.add_argument("--score-chunk-size", type=int, default=2_048)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    input_paths = args.input_path if args.input_path is not None else DEFAULT_INPUT_PATHS
    summary = build_bayesian_training_dataset(
        input_paths=input_paths,
        worth_buying_model_path=args.worth_buying_model_path,
        output_path=args.output_path,
        summary_path=args.summary_path,
        max_rows_per_input=args.max_rows_per_input,
        allowed_listing_kinds=[
            part.strip()
            for part in str(args.allowed_listing_kinds).split(",")
            if part.strip()
        ],
        strict_actual_device_filter=not args.no_strict_actual_device_filter,
        min_review_count=args.min_review_count,
        min_confidence=args.min_confidence,
        include_consider_as_no=args.include_consider_as_no,
        score_chunk_size=args.score_chunk_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
