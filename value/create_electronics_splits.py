from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow `python value/create_electronics_splits.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.listing_kind import infer_listing_kind_from_parts


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon-product-data"
    / "dataset"
    / "meta_Electronics.jsonl"
)
DEFAULT_REVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amazon-product-data"
    / "dataset"
    / "Electronics.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "value" / "electronics_split"
DEFAULT_VALIDATION_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.10
DEFAULT_RANDOM_STATE = 42
DEFAULT_LOG_EVERY = 100_000


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create product-level train/validation/test splits for the Amazon Electronics "
            "catalog by joining meta_Electronics.jsonl with aggregated review signals from "
            "Electronics.jsonl. Splitting is deterministic by parent_asin."
        )
    )
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META_PATH)
    parser.add_argument("--review-path", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-meta-rows", type=int, default=None)
    parser.add_argument("--max-review-rows", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def create_electronics_splits(
    *,
    meta_path: str | Path = DEFAULT_META_PATH,
    review_path: str | Path = DEFAULT_REVIEW_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_meta_rows: int | None = None,
    max_review_rows: int | None = None,
    log_every: int = DEFAULT_LOG_EVERY,
) -> dict[str, Any]:
    if validation_ratio <= 0.0 or test_ratio <= 0.0:
        raise ValueError("validation_ratio and test_ratio must both be positive.")
    if validation_ratio + test_ratio >= 1.0:
        raise ValueError("validation_ratio + test_ratio must be less than 1.0.")
    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    resolved_meta_path = Path(meta_path).expanduser().resolve()
    resolved_review_path = Path(review_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    review_stats = aggregate_review_signals(
        review_path=resolved_review_path,
        max_rows=max_review_rows,
        log_every=log_every,
    )
    split_summary = write_product_splits(
        meta_path=resolved_meta_path,
        output_dir=resolved_output_dir,
        review_stats=review_stats,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        max_rows=max_meta_rows,
        log_every=log_every,
    )

    summary = {
        "meta_path": str(resolved_meta_path),
        "review_path": str(resolved_review_path),
        "output_dir": str(resolved_output_dir),
        "validation_ratio": float(validation_ratio),
        "test_ratio": float(test_ratio),
        "train_ratio": float(1.0 - validation_ratio - test_ratio),
        "random_state": int(random_state),
        "max_meta_rows": max_meta_rows,
        "max_review_rows": max_review_rows,
        "review_aggregation": {
            "products_with_reviews": int(len(review_stats)),
        },
        "splits": split_summary,
    }
    summary_path = resolved_output_dir / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def choose_split(
    parent_asin: str,
    *,
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
) -> str:
    normalized_parent_asin = str(parent_asin or "").strip()
    digest = hashlib.sha256(
        f"{random_state}:{normalized_parent_asin}".encode("utf-8")
    ).digest()
    sample = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    if sample < test_ratio:
        return "test"
    if sample < test_ratio + validation_ratio:
        return "val"
    return "train"


def aggregate_review_signals(
    *,
    review_path: Path,
    max_rows: int | None,
    log_every: int,
) -> dict[str, list[float]]:
    LOGGER.info("Aggregating review signals from %s", review_path)
    stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    rows_seen = 0

    with review_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            parent_asin = _safe_text(payload.get("parent_asin") or payload.get("asin"))
            if not parent_asin:
                continue

            bucket = stats[parent_asin]
            bucket[0] += 1.0
            bucket[1] += 1.0 if _coerce_bool(payload.get("verified_purchase")) else 0.0
            bucket[2] += max(0.0, _to_float(payload.get("helpful_vote")) or 0.0)
            rating = _to_float(payload.get("rating"))
            if rating is not None:
                bucket[3] += rating

            rows_seen += 1
            if rows_seen % log_every == 0:
                LOGGER.info(
                    "Review aggregation progress: rows_seen=%s products=%s",
                    rows_seen,
                    len(stats),
                )
            if max_rows is not None and rows_seen >= max_rows:
                break

    LOGGER.info(
        "Finished aggregating reviews: rows_seen=%s products=%s",
        rows_seen,
        len(stats),
    )
    return dict(stats)


def write_product_splits(
    *,
    meta_path: Path,
    output_dir: Path,
    review_stats: dict[str, list[float]],
    validation_ratio: float,
    test_ratio: float,
    random_state: int,
    max_rows: int | None,
    log_every: int,
) -> dict[str, Any]:
    split_paths = {
        "train": output_dir / "electronics_products_train.jsonl",
        "val": output_dir / "electronics_products_val.jsonl",
        "test": output_dir / "electronics_products_test.jsonl",
    }
    for path in split_paths.values():
        if path.exists():
            path.unlink()

    summary = {
        "rows_seen": 0,
        "rows_written": 0,
        "priced_rows": 0,
        "rows_with_reviews": 0,
        "by_split": {
            "train": {"rows": 0, "priced_rows": 0, "rows_with_reviews": 0},
            "val": {"rows": 0, "priced_rows": 0, "rows_with_reviews": 0},
            "test": {"rows": 0, "priced_rows": 0, "rows_with_reviews": 0},
        },
        "paths": {name: str(path) for name, path in split_paths.items()},
    }

    with (
        meta_path.open("r", encoding="utf-8") as source,
        split_paths["train"].open("w", encoding="utf-8") as train_out,
        split_paths["val"].open("w", encoding="utf-8") as val_out,
        split_paths["test"].open("w", encoding="utf-8") as test_out,
    ):
        writers = {
            "train": train_out,
            "val": val_out,
            "test": test_out,
        }

        for line in source:
            if not line.strip():
                continue
            payload = json.loads(line)
            record = _build_product_record(payload, review_stats)
            if record is None:
                continue

            split_name = choose_split(
                record["parent_asin"],
                validation_ratio=validation_ratio,
                test_ratio=test_ratio,
                random_state=random_state,
            )
            json.dump(record, writers[split_name], ensure_ascii=False)
            writers[split_name].write("\n")

            summary["rows_seen"] += 1
            summary["rows_written"] += 1
            summary["by_split"][split_name]["rows"] += 1

            if record["price"] is not None:
                summary["priced_rows"] += 1
                summary["by_split"][split_name]["priced_rows"] += 1
            if record["review_count"] > 0:
                summary["rows_with_reviews"] += 1
                summary["by_split"][split_name]["rows_with_reviews"] += 1

            if summary["rows_seen"] % log_every == 0:
                LOGGER.info(
                    "Meta split progress: rows_seen=%s train_rows=%s val_rows=%s test_rows=%s",
                    summary["rows_seen"],
                    summary["by_split"]["train"]["rows"],
                    summary["by_split"]["val"]["rows"],
                    summary["by_split"]["test"]["rows"],
                )
            if max_rows is not None and summary["rows_seen"] >= max_rows:
                break

    LOGGER.info(
        "Finished building product splits: rows=%s train=%s val=%s test=%s",
        summary["rows_written"],
        summary["by_split"]["train"]["rows"],
        summary["by_split"]["val"]["rows"],
        summary["by_split"]["test"]["rows"],
    )
    return summary


def _build_product_record(
    meta_payload: dict[str, Any],
    review_stats: dict[str, list[float]],
) -> dict[str, Any] | None:
    parent_asin = _safe_text(meta_payload.get("parent_asin"))
    if not parent_asin:
        return None

    title = _safe_text(meta_payload.get("title"))
    store = _safe_text(meta_payload.get("store"))
    main_category = _safe_text(meta_payload.get("main_category"))
    categories = _string_list(meta_payload.get("categories"))
    features = _string_list(meta_payload.get("features"))
    description = _string_list(meta_payload.get("description"))
    details_text = _flatten_mapping_text(meta_payload.get("details"))

    price = _to_float(meta_payload.get("price"))
    average_rating = _to_float(meta_payload.get("average_rating"))
    rating_number = _to_float(meta_payload.get("rating_number"))

    review_count, verified_count, helpful_vote_total, rating_sum = review_stats.get(
        parent_asin,
        [0.0, 0.0, 0.0, 0.0],
    )
    review_count_int = int(review_count)
    verified_purchase_rate = (
        _clamp(verified_count / review_count, 0.0, 1.0)
        if review_count > 0.0
        else None
    )
    avg_review_rating = rating_sum / review_count if review_count > 0.0 else None
    helpful_vote_avg = helpful_vote_total / review_count if review_count > 0.0 else None

    review_volume_signal = 1.0 - math.exp(-review_count / 50.0) if review_count > 0.0 else 0.0
    trust_probability = None
    if verified_purchase_rate is not None:
        trust_probability = _clamp(
            (0.70 * verified_purchase_rate) + (0.30 * review_volume_signal),
            0.0,
            1.0,
        )

    ewom_score_0_to_100 = None
    if avg_review_rating is not None:
        ewom_score_0_to_100 = _clamp(((avg_review_rating - 1.0) / 4.0) * 100.0, 0.0, 100.0)

    ewom_magnitude_0_to_100 = None
    if review_count > 0.0:
        helpful_signal = 1.0 - math.exp(-(helpful_vote_avg or 0.0) / 2.0)
        ewom_magnitude_0_to_100 = _clamp(
            100.0 * ((0.70 * review_volume_signal) + (0.30 * helpful_signal)),
            0.0,
            100.0,
        )

    product_document = _build_product_document(
        title=title,
        store=store,
        main_category=main_category,
        categories=categories,
        features=features,
        description=description,
        details_text=details_text,
    )
    listing_kind = infer_listing_kind_from_parts(
        title=title,
        main_category=main_category,
        categories=categories,
        features=features,
        description=description,
        details_text=details_text,
    )

    return {
        "parent_asin": parent_asin,
        "title": title,
        "store": store,
        "main_category": main_category,
        "categories": categories,
        "listing_kind": listing_kind,
        "price": price,
        "average_rating": average_rating,
        "rating_number": rating_number,
        "review_count": review_count_int,
        "verified_purchase_rate": verified_purchase_rate,
        "helpful_vote_total": int(helpful_vote_total),
        "helpful_vote_avg": helpful_vote_avg,
        "avg_review_rating": avg_review_rating,
        "trust_probability": trust_probability,
        "ewom_score_0_to_100": ewom_score_0_to_100,
        "ewom_magnitude_0_to_100": ewom_magnitude_0_to_100,
        "product_document": product_document,
    }


def _build_product_document(
    *,
    title: str,
    store: str,
    main_category: str,
    categories: list[str],
    features: list[str],
    description: list[str],
    details_text: str,
) -> str:
    parts = [
        title,
        store,
        main_category,
        " ".join(categories),
        " ".join(features),
        " ".join(description),
        details_text,
    ]
    return " ".join(part for part in parts if part).strip()


def _flatten_mapping_text(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    parts: list[str] = []
    for key, inner in value.items():
        key_text = _safe_text(key)
        inner_text = _safe_text(inner)
        if key_text or inner_text:
            parts.append(f"{key_text} {inner_text}".strip())
    return " ".join(parts).strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _safe_text(item)
        if text:
            result.append(text)
    return result


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    result = create_electronics_splits(
        meta_path=args.meta_path,
        review_path=args.review_path,
        output_dir=args.output_dir,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        max_meta_rows=args.max_meta_rows,
        max_review_rows=args.max_review_rows,
        log_every=args.log_every,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
