from __future__ import annotations

import argparse
import json
from pathlib import Path

from eWOM.api import EWOMModelPaths, score_review, score_review_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-review or mock review-set scoring through the eWOM pipeline.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional review title.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Main review text.",
    )
    parser.add_argument(
        "--rating",
        type=float,
        default=0.0,
        help="Optional star rating.",
    )
    parser.add_argument(
        "--verified-purchase",
        action="store_true",
        help="Mark the review as a verified purchase.",
    )
    parser.add_argument(
        "--mock-json",
        default="mock/mock_ewom.json",
        help="Path to a mock JSON file containing seller_feedback_texts cases.",
    )
    parser.add_argument(
        "--mock-case-id",
        default=None,
        help="Case ID inside the mock JSON. When provided, the demo scores the whole review set.",
    )
    parser.add_argument(
        "--helpfulness-model-path",
        default=None,
        help="Optional override for the helpfulness model artifact path.",
    )
    parser.add_argument(
        "--helpfulness-feature-builder-path",
        default=None,
        help="Optional override for the helpfulness feature-builder artifact path.",
    )
    parser.add_argument(
        "--sentiment-model-path",
        default=None,
        help="Optional override for the sentiment model artifact path.",
    )
    parser.add_argument(
        "--sentiment-feature-builder-path",
        default=None,
        help="Optional override for the sentiment feature-builder artifact path.",
    )
    return parser


def resolve_model_paths(args: argparse.Namespace) -> EWOMModelPaths:
    defaults = EWOMModelPaths.defaults()
    return EWOMModelPaths(
        helpfulness_model_path=(
            args.helpfulness_model_path or defaults.helpfulness_model_path
        ),
        helpfulness_feature_builder_path=(
            args.helpfulness_feature_builder_path or defaults.helpfulness_feature_builder_path
        ),
        sentiment_model_path=(
            args.sentiment_model_path or defaults.sentiment_model_path
        ),
        sentiment_feature_builder_path=(
            args.sentiment_feature_builder_path
            or defaults.sentiment_feature_builder_path
        ),
    )


def load_mock_case_reviews(mock_json_path: str, case_id: str) -> list[str]:
    payload = json.loads(Path(mock_json_path).read_text(encoding="utf-8"))
    for case in payload.get("cases", []):
        if case.get("case_id") != case_id:
            continue

        review_texts = case.get("seller_feedback_texts")
        if not isinstance(review_texts, list) or not review_texts:
            raise ValueError(
                f"Mock case '{case_id}' does not contain a non-empty seller_feedback_texts array."
            )
        return [str(review_text) for review_text in review_texts]

    raise ValueError(f"Mock case '{case_id}' was not found in {mock_json_path}.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mock_case_id:
        result = score_review_set(
            load_mock_case_reviews(args.mock_json, args.mock_case_id),
            model_paths=resolve_model_paths(args),
        )
    else:
        if not args.text:
            parser.error("--text is required unless --mock-case-id is provided.")
        result = score_review(
            {
                "title": args.title,
                "text": args.text,
                "rating": args.rating,
                "verified_purchase": args.verified_purchase,
            },
            model_paths=resolve_model_paths(args),
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
