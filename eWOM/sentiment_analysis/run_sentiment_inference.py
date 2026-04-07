from __future__ import annotations

import argparse
import json
from pathlib import Path

from eWOM.sentiment_analysis.predictor import SentimentPredictor


DEFAULT_MODEL_PREFIX = Path("models/sentiment/amazon_polarity")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sentiment inference directly from a saved model checkpoint.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Main review text.",
    )
    parser.add_argument(
        "--model-prefix",
        default=str(DEFAULT_MODEL_PREFIX),
        help="Artifact prefix used to resolve <prefix>.joblib and <prefix>_feature_builder.joblib.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional explicit override for the model checkpoint path.",
    )
    parser.add_argument(
        "--feature-builder-path",
        default=None,
        help="Optional explicit override for the feature-builder checkpoint path.",
    )
    return parser


def resolve_checkpoint_paths(args: argparse.Namespace) -> tuple[str, str]:
    model_prefix = Path(args.model_prefix).expanduser()
    model_path = Path(args.model_path).expanduser() if args.model_path else model_prefix.with_suffix(".joblib")
    feature_builder_path = (
        Path(args.feature_builder_path).expanduser()
        if args.feature_builder_path
        else model_prefix.with_name(f"{model_prefix.name}_feature_builder.joblib")
    )
    return str(model_path), str(feature_builder_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_path, feature_builder_path = resolve_checkpoint_paths(args)
    predictor = SentimentPredictor(
        model_path=model_path,
        feature_builder_path=feature_builder_path,
    )
    result = predictor.predict_one(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
