from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python value/run_bayesian_value_model.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.bayesian_value import (
    BayesianValueInput,
    fuse_ewom_result_into_bayesian_input,
    score_good_value_probability,
)
from value.train_bayesian_value_model import load_bayesian_value_network


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bayesian value model using raw product signals and optionally "
            "fuse in live eWOM/trust-agent outputs."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--mock", action="store_true", help="Run with a built-in example payload.")
    source_group.add_argument("--input", type=str, help="Path to a JSON file containing Bayesian value inputs.")
    parser.add_argument(
        "--ewom-output",
        type=str,
        help=(
            "Optional path to a JSON file containing a precomputed eWOM score_review or "
            "score_review_set result to fuse into the Bayesian input."
        ),
    )
    parser.add_argument(
        "--ewom-mock-json",
        type=str,
        help="Optional mock JSON file passed through the eWOM review-set demo flow.",
    )
    parser.add_argument(
        "--ewom-mock-case-id",
        type=str,
        help="Case ID inside --ewom-mock-json to score before Bayesian inference.",
    )
    parser.add_argument(
        "--network-path",
        type=str,
        help="Optional trained Bayesian network JSON artifact to use instead of the default CPTs.",
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
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _validate_ewom_args(parser, args)

    payload = _mock_payload() if args.mock else _load_json_file(args.input)
    bayesian_input = BayesianValueInput.from_mapping(payload)
    ewom_result = _resolve_ewom_result(args)
    fused_agent_signals: dict[str, Any] | None = None
    if ewom_result is not None:
        bayesian_input, fused_agent_signals = fuse_ewom_result_into_bayesian_input(
            bayesian_input,
            ewom_result,
        )

    network = _load_bayesian_network(args.network_path)
    result = score_good_value_probability(bayesian_input, network=network)
    result["resolved_input"] = asdict(bayesian_input)
    if args.network_path:
        result["bayesian_network_path"] = str(Path(args.network_path).expanduser().resolve())
    if fused_agent_signals is not None:
        result["fused_agent_signals"] = fused_agent_signals

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result))


def _validate_ewom_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    has_live_mock_case = bool(args.ewom_mock_json or args.ewom_mock_case_id)
    if args.ewom_output and has_live_mock_case:
        parser.error(
            "--ewom-output cannot be combined with --ewom-mock-json/--ewom-mock-case-id."
        )
    if bool(args.ewom_mock_json) != bool(args.ewom_mock_case_id):
        parser.error(
            "--ewom-mock-json and --ewom-mock-case-id must be provided together."
        )


def _load_json_file(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        raise ValueError("--input path is required when --mock is not used.")
    payload_path = Path(path_str).expanduser().resolve()
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Input JSON must be an object.")
    return payload


def _resolve_ewom_result(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.ewom_output:
        return _load_json_file(args.ewom_output)
    if args.ewom_mock_json and args.ewom_mock_case_id:
        return _score_ewom_mock_case(args)
    return None


def _load_bayesian_network(network_path: str | None):
    if not network_path:
        return None
    return load_bayesian_value_network(network_path)


def _score_ewom_mock_case(args: argparse.Namespace) -> dict[str, Any]:
    from eWOM.api import score_review_set
    from eWOM.run_fusion_demo import load_mock_case_reviews, resolve_model_paths

    review_texts = load_mock_case_reviews(args.ewom_mock_json, args.ewom_mock_case_id)
    return score_review_set(
        review_texts,
        model_paths=resolve_model_paths(args),
    )


def _mock_payload() -> dict[str, Any]:
    return {
        "trust_probability": 0.82,
        "ewom_score_0_to_100": 71.0,
        "ewom_magnitude_0_to_100": 64.0,
        "average_rating": 4.4,
        "rating_count": 356,
        "verified_purchase_rate": 0.87,
        "price": 79.0,
        "peer_price": 96.0,
        "warranty_months": 12,
        "return_window_days": 30,
    }


if __name__ == "__main__":
    main()
