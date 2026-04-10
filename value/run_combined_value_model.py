from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow `python value/run_combined_value_model.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.combined_value import score_combined_value_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "value" / "artifacts" / "amazon_worth_buying_electronics_tfidf.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "value" / "electronics_split" / "electronics_products_val.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "value" / "artifacts" / "amazon_combined_validation_scores.csv"


def configure_logging(level: str = "INFO") -> None:
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the combined value pipeline: estimate peer price and retrieval confidence "
            "with the worth-buying model, then score final good-value probability with the "
            "Bayesian model."
        )
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.50)
    parser.add_argument("--min-confidence-for-prediction", type=float, default=None)
    parser.add_argument(
        "--bayesian-network-path",
        type=Path,
        default=None,
        help="Optional trained Bayesian network JSON artifact to use instead of the default CPTs.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)
    result = score_combined_value_split(
        model_path=args.model_path,
        split_path=args.input_path,
        output_path=args.output_path,
        max_rows=args.max_rows,
        probability_threshold=args.probability_threshold,
        min_confidence_for_prediction=args.min_confidence_for_prediction,
        bayesian_network_path=args.bayesian_network_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
