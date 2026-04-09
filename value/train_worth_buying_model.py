from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python value/train_worth_buying_model.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.worth_buying import WorthBuyingConfig, train_worth_buying_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "value" / "electronics_split" / "electronics_products_train.jsonl"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "value" / "artifacts" / "amazon_worth_buying_electronics_tfidf"


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Electronics worth-buying scorer by fitting TF-IDF product retrieval "
            "artifacts on the prepared train split."
        )
    )
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--top-k-neighbors", type=int, default=20)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-similarity", type=float, default=0.12)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--word-max-features", type=int, default=40_000)
    parser.add_argument("--char-max-features", type=int, default=30_000)
    parser.add_argument(
        "--allowed-listing-kinds",
        type=str,
        default=None,
        help=(
            "Optional comma-separated listing_kind allowlist for training, for example "
            "\"device\" or \"device,other\"."
        ),
    )
    parser.add_argument(
        "--filtered-catalog-output",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output path for the filtered priced training catalog "
            "that will be used to fit the model."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    config = WorthBuyingConfig(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
        min_df=args.min_df,
        top_k_neighbors=args.top_k_neighbors,
        min_neighbors=args.min_neighbors,
        min_similarity=args.min_similarity,
    )
    result = train_worth_buying_pipeline(
        train_path=args.train_path,
        output_prefix=args.output_prefix,
        config=config,
        max_rows=args.max_rows,
        allowed_listing_kinds=args.allowed_listing_kinds,
        filtered_catalog_output_path=args.filtered_catalog_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
