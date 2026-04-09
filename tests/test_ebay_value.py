from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from core.entities.candidate import Candidate
from value.ebay_value import (
    build_bayesian_input_from_candidate,
    infer_candidate_market_context,
    score_ebay_candidate_value,
    summarize_candidate_market_context_k_sweep,
    summarize_ebay_candidate_value_result,
    sweep_candidate_market_context_k,
)


class EbayValueBridgeTests(unittest.TestCase):
    def test_build_bayesian_input_from_candidate_parses_price_shipping_and_policy_fields(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Noise Cancelling Headphones",
            price={"value": "199.99", "currency": "SGD"},
            shipping=[
                {"shippingCostType": "FIXED", "shippingCost": {"value": "12.50"}},
                {"shippingCostType": "FREE"},
            ],
            returns={"returnPeriod": {"value": 30, "unit": "DAY"}},
            product_rating_count=210,
            product_average_rating=4.6,
            item_specifics={"Manufacturer Warranty": ["2 years"]},
        )

        bayesian_input = build_bayesian_input_from_candidate(candidate, peer_price=249.0)

        self.assertEqual(bayesian_input.price, 199.99)
        self.assertEqual(bayesian_input.peer_price, 249.0)
        self.assertEqual(bayesian_input.average_rating, 4.6)
        self.assertEqual(bayesian_input.rating_count, 210.0)
        self.assertEqual(bayesian_input.return_window_days, 30.0)
        self.assertEqual(bayesian_input.warranty_months, 24.333333333333332)

    def test_score_ebay_candidate_value_fuses_ewom_result_into_bayesian_score(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={"value": "79.0", "currency": "SGD"},
            product_rating_count=320,
            product_average_rating=4.5,
            seller_feedback_texts=[
                "Fast shipping and great communication.",
                "Arrived exactly as described.",
            ],
        )
        ewom_result = {
            "review_count": 2,
            "aggregate": {
                "mean_deception_probability": 0.10,
                "final_ewom_score_0_to_100": 72.0,
                "final_ewom_magnitude_0_to_100": 61.0,
            },
        }

        result = score_ebay_candidate_value(
            candidate,
            peer_price=99.0,
            ewom_result=ewom_result,
        )

        bayesian_result = result["bayesian_result"]
        self.assertIsNone(result["market_context"])
        self.assertEqual(result["ewom_result"]["aggregate"]["final_ewom_score_0_to_100"], 72.0)
        self.assertAlmostEqual(
            bayesian_result["resolved_input"]["trust_probability"],
            0.9,
        )
        self.assertEqual(bayesian_result["resolved_input"]["ewom_score_0_to_100"], 72.0)
        self.assertEqual(
            bayesian_result["resolved_input"]["ewom_magnitude_0_to_100"],
            61.0,
        )
        self.assertEqual(bayesian_result["resolved_input"]["peer_price"], 99.0)
        self.assertGreaterEqual(bayesian_result["good_value_probability"], 0.0)
        self.assertLessEqual(bayesian_result["good_value_probability"], 1.0)

    @patch("value.ebay_value.score_worth_buying_catalog")
    def test_infer_candidate_market_context_builds_single_query_row(
        self,
        score_worth_buying_catalog_mock,
    ) -> None:
        score_worth_buying_catalog_mock.return_value = pd.DataFrame(
            [
                {
                    "catalog_row_index": 0,
                    "parent_asin": "epid:456",
                    "title": "USB-C Charger",
                    "price": 24.99,
                    "average_rating": 4.4,
                    "rating_number": 140.0,
                    "review_count": 140,
                    "peer_price": 29.99,
                    "price_gap_vs_peer": 0.1667,
                    "neighbor_count": 5,
                    "average_neighbor_similarity": 0.71,
                    "price_alignment_score": 0.64,
                    "bayesian_rating_score": 0.88,
                    "review_quality_score": 0.77,
                    "confidence_score": 0.61,
                    "worth_buying_score": 0.73,
                    "verdict": "worth_buying",
                }
            ]
        )
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            product_id="456",
            title="USB-C Charger",
            condition="New",
            price={"value": "24.99"},
            product_rating_count=140,
            product_average_rating=4.4,
            item_specifics={"Brand": ["Anker"], "Power": ["30W"]},
            product_family_key="epid:456",
        )

        result = infer_candidate_market_context(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            top_k_neighbors=3,
        )

        self.assertEqual(result["peer_price"], 29.99)
        self.assertEqual(result["verdict"], "worth_buying")

        args, kwargs = score_worth_buying_catalog_mock.call_args
        frame = args[0]
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["parent_asin"], "epid:456")
        self.assertEqual(frame.iloc[0]["price"], 24.99)
        self.assertEqual(frame.iloc[0]["average_rating"], 4.4)
        self.assertIn("Brand Anker", frame.iloc[0]["product_document"])
        self.assertEqual(kwargs["model_path"], "value/artifacts/amazon_worth_buying_quick.joblib")
        self.assertEqual(kwargs["config_overrides"]["top_k_neighbors"], 3)

    def test_summarize_ebay_candidate_value_result_returns_compact_payload(self) -> None:
        result = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/123",
                "title": "USB-C Charger",
            },
            "bayesian_result": {
                "good_value_probability": 0.73,
                "derived_metrics": {
                    "price_gap_vs_peer": 0.12,
                },
                "resolved_input": {
                    "price": 24.99,
                    "peer_price": 28.50,
                    "trust_probability": 0.81,
                    "ewom_score_0_to_100": 66.0,
                },
                "fused_agent_signals": {
                    "review_count": 8,
                },
            },
        }

        summary = summarize_ebay_candidate_value_result(result)

        self.assertEqual(summary["source_url"], "https://www.ebay.com.sg/itm/123")
        self.assertEqual(summary["title"], "USB-C Charger")
        self.assertEqual(summary["total_price"], 24.99)
        self.assertEqual(summary["peer_price"], 28.50)
        self.assertEqual(summary["price_gap_vs_peer"], 0.12)
        self.assertEqual(summary["good_value_probability"], 0.73)
        self.assertEqual(summary["prediction"], "good_value")
        self.assertEqual(summary["trust_probability"], 0.81)
        self.assertEqual(summary["ewom_score_0_to_100"], 66.0)
        self.assertEqual(summary["seller_feedback_review_count"], 8)

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_sweep_candidate_market_context_k_builds_sorted_k_rows(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.side_effect = [
            [
                {
                    "peer_price": 31.0,
                    "neighbor_count": 1,
                    "average_neighbor_similarity": 0.91,
                    "neighbors": [{"price": 31.0}],
                }
            ],
            [
                {
                    "peer_price": 29.0,
                    "neighbor_count": 3,
                    "average_neighbor_similarity": 0.74,
                    "neighbors": [{"price": 28.0}, {"price": 29.0}, {"price": 30.0}],
                }
            ],
        ]
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="USB-C Charger",
            price={"value": "24.99"},
        )

        sweep = sweep_candidate_market_context_k(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            k_values=[3, 1],
        )

        self.assertEqual([row["k"] for row in sweep["k_sweep"]], [1, 3])
        self.assertEqual(sweep["k_sweep"][0]["peer_price"], 31.0)
        self.assertEqual(sweep["k_sweep"][1]["peer_price"], 29.0)
        self.assertIsNone(sweep["k_sweep"][0]["good_value_probability"])

    def test_summarize_candidate_market_context_k_sweep_returns_compact_rows(self) -> None:
        result = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/123",
                "title": "USB-C Charger",
            },
            "k_sweep": [
                {
                    "k": 1,
                    "peer_price": 31.0,
                    "neighbor_count": 1,
                    "average_neighbor_similarity": 0.91,
                    "good_value_probability": 0.63,
                    "prediction": "good_value",
                    "neighbor_prices": [31.0],
                },
                {
                    "k": 3,
                    "peer_price": 29.0,
                    "neighbor_count": 3,
                    "average_neighbor_similarity": 0.74,
                    "good_value_probability": 0.58,
                    "prediction": "good_value",
                    "neighbor_prices": [28.0, 29.0, 30.0],
                },
            ],
        }

        summary = summarize_candidate_market_context_k_sweep(result)

        self.assertEqual(summary["source_url"], "https://www.ebay.com.sg/itm/123")
        self.assertEqual(summary["title"], "USB-C Charger")
        self.assertEqual(summary["k_sweep"][0]["k"], 1)
        self.assertNotIn("neighbor_prices", summary["k_sweep"][0])


if __name__ == "__main__":
    unittest.main()
