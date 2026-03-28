from __future__ import annotations

import copy
import unittest

from value import compare_listings


def _base_payload() -> dict:
    return {
        "listing_a": {
            "url": "https://shopee.sg/item-a",
            "platform": "Shopee",
            "title": "Apple AirPods Pro 2 USB-C",
            "currency": "SGD",
            "total_price": 330.0,
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
            "url": "https://lazada.sg/item-b",
            "platform": "Lazada",
            "title": "Apple AirPods Pro 2 with USB C",
            "currency": "SGD",
            "total_price": 340.0,
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


class ValueAgentTests(unittest.TestCase):
    def test_better_a(self) -> None:
        payload = _base_payload()
        result = compare_listings(payload)
        self.assertEqual(result["verdict"], "better_A")

    def test_better_b(self) -> None:
        payload = _base_payload()
        payload["listing_b"]["total_price"] = 300.0
        payload["listing_b"]["delivery_days"] = 1
        payload["listing_b"]["warranty_months"] = 18
        payload["listing_b"]["return_window_days"] = 30

        result = compare_listings(payload)
        self.assertEqual(result["verdict"], "better_B")

    def test_tie(self) -> None:
        payload = _base_payload()
        payload["listing_b"]["total_price"] = payload["listing_a"]["total_price"]
        payload["listing_b"]["delivery_days"] = payload["listing_a"]["delivery_days"]
        payload["listing_b"]["warranty_months"] = payload["listing_a"]["warranty_months"]
        payload["listing_b"]["return_window_days"] = payload["listing_a"]["return_window_days"]
        payload["listing_b"]["specs"] = dict(payload["listing_a"]["specs"])

        result = compare_listings(payload)
        self.assertEqual(result["verdict"], "tie")

    def test_insufficient_evidence(self) -> None:
        payload = _base_payload()
        payload["listing_a"]["total_price"] = None
        payload["listing_b"]["total_price"] = None
        payload["listing_a"]["specs"] = {}
        payload["listing_b"]["specs"] = {}
        payload["listing_a"]["title"] = "Wireless Earbuds"
        payload["listing_b"]["title"] = "Phone Case"
        payload["required_spec_keys"] = ["model", "battery_life_hours", "chip"]
        payload["listing_a"]["delivery_days"] = None
        payload["listing_b"]["delivery_days"] = None
        payload["listing_a"]["warranty_months"] = None
        payload["listing_b"]["warranty_months"] = None
        payload["listing_a"]["return_window_days"] = None
        payload["listing_b"]["return_window_days"] = None

        result = compare_listings(payload)
        self.assertEqual(result["verdict"], "insufficient_evidence")
        self.assertLess(result["confidence"], 0.45)

    def test_deterministic_output(self) -> None:
        payload = _base_payload()
        first = compare_listings(copy.deepcopy(payload))
        second = compare_listings(copy.deepcopy(payload))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
