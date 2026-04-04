from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python value/run_bayesian_value_model.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from value.bayesian_value import BayesianValueInput, score_good_value_probability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Bayesian value model using raw product signals plus assumed trust/review-agent outputs."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--mock", action="store_true", help="Run with a built-in example payload.")
    source_group.add_argument("--input", type=str, help="Path to a JSON file containing Bayesian value inputs.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payload = _mock_payload() if args.mock else _load_json_file(args.input)
    result = score_good_value_probability(BayesianValueInput.from_mapping(payload))

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result))


def _load_json_file(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        raise ValueError("--input path is required when --mock is not used.")
    payload_path = Path(path_str).expanduser().resolve()
    with payload_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Input JSON must be an object.")
    return payload


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
