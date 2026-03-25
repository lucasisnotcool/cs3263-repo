from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from value.agent import compare_listings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Value Agent with mock or JSON input payload.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--mock",
        action="store_true",
        help="Run with built-in mock listing payload.",
    )
    source_group.add_argument(
        "--input",
        type=str,
        help="Path to JSON payload file for compare_listings.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    payload = _mock_payload() if args.mock else _load_json_file(args.input)
    result = compare_listings(payload)

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result))


def _load_json_file(path_str: str | None) -> dict[str, Any]:
    if not path_str:
        raise ValueError("--input path is required when --mock is not used.")
    payload_path = Path(path_str).expanduser().resolve()
    with payload_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise TypeError("Input JSON must be an object.")
    return payload


def _mock_payload() -> dict[str, Any]:
    return {
        "listing_a": {
            "url": "https://shopee.sg/example-airpods",
            "platform": "Shopee",
            "title": "Apple AirPods Pro 2 USB-C",
            "currency": "SGD",
            "base_price": 329.0,
            "shipping_fee": 2.0,
            "voucher_discount": 10.0,
            "delivery_days": 2,
            "warranty_months": 12,
            "return_window_days": 15,
            "specs": {
                "model": "AirPods Pro 2",
                "connector": "USB-C",
                "battery_life_hours": 6,
            },
        },
        "listing_b": {
            "url": "https://lazada.sg/example-airpods",
            "platform": "Lazada",
            "title": "Apple AirPods Pro (2nd Generation) USB C",
            "currency": "SGD",
            "total_price": 335.0,
            "delivery_days": 3,
            "warranty_months": 6,
            "return_window_days": 7,
            "specs": {
                "model": "AirPods Pro 2",
                "connector": "USB-C",
                "battery_life_hours": 6,
            },
        },
        "required_spec_keys": ["model", "connector", "battery_life_hours"],
    }


if __name__ == "__main__":
    main()
