from __future__ import annotations

import argparse
import json

from eWOM.api import EWOMModelPaths, score_review


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sample review through the helpfulness + sentiment fusion pipeline.",
    )
    parser.add_argument(
        "--title",
        default="",
        help="Optional review title.",
    )
    parser.add_argument(
        "--text",
        required=True,
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

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
