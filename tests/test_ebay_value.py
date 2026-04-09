from __future__ import annotations

import unittest
from unittest.mock import patch

from core.entities.candidate import Candidate
from value.ebay_value import (
    _build_candidate_retrieval_title,
    _is_accessory_like_text,
    build_bayesian_input_from_candidate,
    build_worth_buying_query_row,
    compare_ebay_candidate_value_results,
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

    def test_score_ebay_candidate_value_can_exclude_shipping_from_price(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={"value": "79.0", "currency": "SGD"},
            shipping=[
                {"shippingCostType": "FIXED", "shippingCost": {"value": "12.50"}}
            ],
            product_rating_count=320,
            product_average_rating=4.5,
        )

        result = score_ebay_candidate_value(
            candidate,
            peer_price=99.0,
            ewom_result={
                "review_count": 1,
                "aggregate": {
                    "mean_deception_probability": 0.10,
                    "final_ewom_score_0_to_100": 72.0,
                    "final_ewom_magnitude_0_to_100": 61.0,
                },
            },
            include_shipping_in_total=False,
        )

        self.assertEqual(result["bayesian_result"]["resolved_input"]["price"], 79.0)
        self.assertFalse(result["pricing"]["include_shipping_in_total"])

    def test_score_ebay_candidate_value_defaults_missing_agent_outputs_to_zero(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={"value": "79.0", "currency": "SGD"},
            product_rating_count=320,
            product_average_rating=4.5,
            seller_feedback_texts=[],
        )

        result = score_ebay_candidate_value(
            candidate,
            peer_price=99.0,
        )

        self.assertEqual(result["bayesian_result"]["resolved_input"]["trust_probability"], 0.0)
        self.assertEqual(result["bayesian_result"]["resolved_input"]["ewom_score_0_to_100"], 0.0)

    def test_score_ebay_candidate_value_can_prefer_converted_usd_prices(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={
                "value": "138.89",
                "currency": "SGD",
                "convertedFromValue": "109.00",
                "convertedFromCurrency": "USD",
            },
            shipping=[
                {
                    "shippingCostType": "FIXED",
                    "shippingCost": {
                        "value": "91.64",
                        "currency": "SGD",
                        "convertedFromValue": "71.92",
                        "convertedFromCurrency": "USD",
                    },
                }
            ],
            product_rating_count=320,
            product_average_rating=4.5,
        )

        result = score_ebay_candidate_value(
            candidate,
            peer_price=120.0,
            ewom_result={
                "review_count": 1,
                "aggregate": {
                    "mean_deception_probability": 0.10,
                    "final_ewom_score_0_to_100": 72.0,
                    "final_ewom_magnitude_0_to_100": 61.0,
                },
            },
            prefer_converted_usd=True,
        )

        self.assertAlmostEqual(
            result["bayesian_result"]["resolved_input"]["price"],
            180.92,
        )
        self.assertTrue(result["pricing"]["prefer_converted_usd"])
        self.assertEqual(result["pricing"]["total_price_currency"], "USD")

    @patch("value.ebay_value.infer_candidate_market_context")
    def test_score_ebay_candidate_value_uses_neutral_price_when_retrieval_is_missing(
        self,
        infer_market_context_mock,
    ) -> None:
        infer_market_context_mock.return_value = {
            "peer_price": None,
            "retrieval_status": "all_neighbors_filtered",
            "neighbor_count": 0,
        }
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

        result = score_ebay_candidate_value(
            candidate,
            worth_buying_model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            ewom_result={
                "review_count": 2,
                "aggregate": {
                    "mean_deception_probability": 0.10,
                    "final_ewom_score_0_to_100": 72.0,
                    "final_ewom_magnitude_0_to_100": 61.0,
                },
            },
        )

        self.assertEqual(result["decision"]["prediction"], "good_value")
        self.assertEqual(
            result["decision"]["reason"],
            "bayesian_probability_without_price",
        )
        self.assertFalse(result["decision"]["price_considered"])
        self.assertIn("does not consider price advantage", result["decision"]["price_note"])
        self.assertEqual(result["pricing"]["peer_price_source"], "none")
        self.assertFalse(result["pricing"]["price_considered"])
        self.assertEqual(result["pricing"]["default_relative_price_bucket"], "fair")
        self.assertIsNone(result["bayesian_result"]["resolved_input"]["peer_price"])
        self.assertEqual(
            result["bayesian_result"]["derived_metrics"]["relative_price_bucket"],
            "fair",
        )
        self.assertEqual(
            result["bayesian_result"]["derived_metrics"]["relative_price_bucket_source"],
            "default",
        )

    def test_retrieval_title_prefers_structured_model_over_noisy_listing_title(self) -> None:
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/airpods",
            page_type="listing",
            title="Airpods Pro 2nd Generation with MagSafe Charging Case NIB",
            item_specifics={
                "Brand": "Apple",
                "Model": "Apple AirPods Pro (2nd generation)",
                "Type": "Earbud (In Ear)",
            },
        )

        retrieval_title = _build_candidate_retrieval_title(candidate)
        query_row = build_worth_buying_query_row(candidate)

        self.assertEqual(
            retrieval_title,
            "Apple AirPods Pro (2nd generation) Earbud (In Ear)",
        )
        self.assertEqual(query_row["title"], retrieval_title)
        self.assertEqual(query_row["listing_kind"], "device")
        self.assertNotIn("MagSafe Charging Case", query_row["product_document"])
        self.assertNotIn("NIB", query_row["product_document"])

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_infer_candidate_market_context_builds_single_query_row(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.return_value = [
            {
                "catalog_row_index": 0,
                "parent_asin": "epid:456",
                "title": "USB-C Charger",
                "price": 24.99,
                "peer_price": 29.99,
                "neighbor_count": 3,
                "average_neighbor_similarity": 0.71,
                "top_k_neighbors_used": 3,
                "min_similarity_used": 0.12,
                "neighbors": [
                    {
                        "title": "Anker USB-C Charger 30W",
                        "store": "Anker",
                        "main_category": "Chargers",
                        "price": 29.99,
                        "similarity": 0.81,
                    },
                    {
                        "title": "USB-C Charger 33W Fast Charging",
                        "store": "UGREEN",
                        "main_category": "Chargers",
                        "price": 31.99,
                        "similarity": 0.69,
                    },
                    {
                        "title": "Compact USB-C Charger",
                        "store": "Baseus",
                        "main_category": "Chargers",
                        "price": 27.99,
                        "similarity": 0.63,
                    },
                ],
            }
        ]
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

        self.assertAlmostEqual(result["peer_price"], 30.046338028169016)
        self.assertEqual(result["retrieval_status"], "usable")
        self.assertEqual(result["top_k_neighbors_used"], 3)
        self.assertEqual(result["candidate_pool_size_used"], 500)

        args, kwargs = inspect_neighbors_mock.call_args
        frame = args[0]
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["parent_asin"], "epid:456")
        self.assertEqual(frame.iloc[0]["price"], 24.99)
        self.assertEqual(frame.iloc[0]["store"], "Anker")
        self.assertEqual(frame.iloc[0]["average_rating"], 4.4)
        self.assertIn("Brand Anker", frame.iloc[0]["product_document"])
        self.assertEqual(kwargs["model_path"], "value/artifacts/amazon_worth_buying_quick.joblib")
        self.assertEqual(kwargs["config_overrides"]["top_k_neighbors"], 500)
        self.assertEqual(kwargs["config_overrides"]["min_similarity"], 0.02)

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_infer_candidate_market_context_filters_accessory_neighbors(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.return_value = [
            {
                "catalog_row_index": 0,
                "parent_asin": "epid:airpods",
                "title": "AirPods Pro 2nd Generation",
                "price": 232.05,
                "peer_price": 118.0,
                "neighbor_count": 2,
                "average_neighbor_similarity": 0.77,
                "top_k_neighbors_used": 2,
                "min_similarity_used": 0.12,
                "neighbors": [
                    {
                        "title": "Protective Case for AirPods Pro 2nd Generation",
                        "store": "Spigen",
                        "main_category": "Cases",
                        "price": 14.99,
                        "similarity": 0.89,
                    },
                    {
                        "title": "Apple AirPods Pro (2nd Generation)",
                        "store": "Apple",
                        "main_category": "Earbuds",
                        "price": 189.99,
                        "similarity": 0.74,
                    },
                ],
            }
        ]
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/airpods",
            page_type="listing",
            title="AirPods Pro 2nd Generation with MagSafe Charging Case NIB",
            price={"value": "139.80"},
            shipping=[{"shippingCostType": "FIXED", "shippingCost": {"value": "92.25"}}],
            item_specifics={"Brand": ["Apple"]},
        )

        result = infer_candidate_market_context(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            top_k_neighbors=2,
            min_peer_neighbor_count=1,
        )

        self.assertEqual(result["peer_price"], 189.99)
        self.assertEqual(result["neighbor_count"], 1)
        self.assertEqual(result["raw_neighbor_count"], 2)
        self.assertEqual(result["rejection_summary"]["accessory_mismatch"], 1)
        self.assertEqual(result["retrieval_status"], "usable_but_thin")
        self.assertEqual(result["candidate_pool_size_used"], 500)

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_infer_candidate_market_context_rejects_cross_family_apple_matches(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.return_value = [
            {
                "catalog_row_index": 0,
                "parent_asin": "epid:airpods",
                "title": "AirPods Pro 2nd Generation",
                "price": 138.89,
                "peer_price": 650.0,
                "neighbor_count": 4,
                "average_neighbor_similarity": 0.41,
                "top_k_neighbors_used": 4,
                "min_similarity_used": 0.02,
                "neighbors": [
                    {
                        "title": 'KEEPRO Pencil 2nd Generation for iPad Pro 11"',
                        "store": "KEEPRO",
                        "main_category": "All Electronics",
                        "price": 22.97,
                        "similarity": 0.42,
                    },
                    {
                        "title": 'Apple 2021 MacBook Pro (14-inch, Apple M1 Pro chip with 8-core CPU and 14-core GPU, 16GB RAM, 512GB SSD) - Space Gray',
                        "store": "Apple",
                        "main_category": "Computers",
                        "price": 1999.0,
                        "similarity": 0.40,
                    },
                    {
                        "title": "Back Bay Tempo 30 Wireless Earbuds for Small Ears",
                        "store": "Back Bay Audio",
                        "main_category": "All Electronics",
                        "price": 49.99,
                        "similarity": 0.33,
                    },
                    {
                        "title": "Apple AirPods Pro (2nd Generation)",
                        "store": "Apple",
                        "main_category": "Earbuds",
                        "price": 189.99,
                        "similarity": 0.32,
                    },
                ],
            }
        ]
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/airpods",
            page_type="listing",
            title="AirPods Pro 2nd Generation with MagSafe Charging Case NIB",
            price={"value": "138.89"},
            item_specifics={
                "Brand": ["Apple"],
                "Model": ["Apple AirPods Pro (2nd generation)"],
                "Type": ["Earbud (In Ear)"],
            },
        )

        result = infer_candidate_market_context(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            top_k_neighbors=5,
            min_peer_neighbor_count=1,
        )

        self.assertEqual(result["peer_price"], 189.99)
        self.assertEqual(result["neighbor_count"], 1)
        self.assertEqual(result["query_product_kind"], "earbud")
        self.assertEqual(result["rejection_summary"]["product_type_mismatch"], 2)
        self.assertEqual(result["rejection_summary"]["weak_identity_match"], 1)
        self.assertEqual(result["neighbors"][0]["title"], "Apple AirPods Pro (2nd Generation)")

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_infer_candidate_market_context_can_raise_low_peer_price_cutoff(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.return_value = [
            {
                "catalog_row_index": 0,
                "parent_asin": "epid:earbuds",
                "title": "Wireless Earbuds",
                "price": 200.0,
                "peer_price": 80.0,
                "neighbor_count": 1,
                "average_neighbor_similarity": 0.65,
                "top_k_neighbors_used": 1,
                "min_similarity_used": 0.02,
                "neighbors": [
                    {
                        "title": "Wireless Earbuds Noise Cancelling",
                        "store": "Acme",
                        "main_category": "All Electronics",
                        "price": 80.0,
                        "similarity": 0.65,
                    }
                ],
            }
        ]
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={"value": "200.0"},
            item_specifics={"Type": ["Earbud (In Ear)"]},
        )

        result = infer_candidate_market_context(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            top_k_neighbors=1,
            min_peer_price_ratio=0.50,
            min_peer_neighbor_count=1,
        )

        self.assertIsNone(result["peer_price"])
        self.assertEqual(result["retrieval_status"], "implausibly_low_peer_price")
        self.assertEqual(result["min_peer_price_ratio_used"], 0.50)
        self.assertEqual(
            result["rejection_summary"]["implausibly_low_peer_price"],
            1,
        )

    @patch("value.ebay_value.inspect_worth_buying_catalog_neighbors")
    def test_infer_candidate_market_context_rejects_thin_peer_price_support(
        self,
        inspect_neighbors_mock,
    ) -> None:
        inspect_neighbors_mock.return_value = [
            {
                "catalog_row_index": 0,
                "parent_asin": "epid:earbuds",
                "title": "Wireless Earbuds",
                "price": 150.0,
                "peer_price": 120.0,
                "neighbor_count": 1,
                "average_neighbor_similarity": 0.65,
                "top_k_neighbors_used": 1,
                "min_similarity_used": 0.02,
                "neighbors": [
                    {
                        "title": "Wireless Earbuds Noise Cancelling",
                        "store": "Acme",
                        "main_category": "All Electronics",
                        "price": 120.0,
                        "similarity": 0.65,
                    }
                ],
            }
        ]
        candidate = Candidate(
            source_url="https://www.ebay.com.sg/itm/123",
            page_type="listing",
            title="Wireless Earbuds",
            price={"value": "150.0"},
            item_specifics={"Type": ["Earbud (In Ear)"]},
        )

        result = infer_candidate_market_context(
            candidate,
            model_path="value/artifacts/amazon_worth_buying_quick.joblib",
            top_k_neighbors=1,
            min_peer_price_ratio=0.50,
            min_peer_neighbor_count=3,
        )

        self.assertIsNone(result["peer_price"])
        self.assertEqual(result["neighbor_count"], 1)
        self.assertEqual(result["retrieval_status"], "insufficient_peer_neighbors")
        self.assertEqual(result["min_peer_neighbor_count_used"], 3)
        self.assertEqual(
            result["rejection_summary"]["insufficient_peer_neighbors"],
            1,
        )

    def test_accessory_detector_distinguishes_device_listing_from_case_accessory(self) -> None:
        self.assertFalse(
            _is_accessory_like_text(
                "Airpods Pro 2nd Generation with MagSafe Charging Case NIB"
            )
        )
        self.assertTrue(
            _is_accessory_like_text(
                "AirPods Pro Case with Silicone Lanyard Protective Cover with Keychain"
            )
        )

    def test_summarize_ebay_candidate_value_result_returns_compact_payload(self) -> None:
        result = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/123",
                "title": "USB-C Charger",
            },
            "pricing": {
                "total_price_currency": "USD",
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
        self.assertEqual(summary["total_price_currency"], "USD")
        self.assertEqual(summary["peer_price"], 28.50)
        self.assertEqual(summary["price_gap_vs_peer"], 0.12)
        self.assertEqual(summary["good_value_probability"], 0.73)
        self.assertEqual(summary["prediction"], "good_value")
        self.assertEqual(summary["prediction_reason"], None)
        self.assertEqual(summary["trust_probability"], 0.81)
        self.assertEqual(summary["ewom_score_0_to_100"], 66.0)
        self.assertEqual(summary["seller_feedback_review_count"], 8)
        self.assertIsNone(summary["retrieval_status"])
        self.assertIsNone(summary["retrieved_neighbor_count"])
        self.assertIsNone(summary["price_considered"])
        self.assertIsNone(summary["price_note"])
        self.assertIsNone(summary["peer_price_source"])

    def test_summarize_ebay_candidate_value_result_marks_missing_retrieval_price_as_price_neutral(self) -> None:
        result = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/123",
                "title": "USB-C Charger",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "none",
                "price_considered": False,
                "price_note": "Bayesian score uses a neutral fair-price bucket.",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability_without_price",
            },
            "market_context": {
                "retrieval_status": "all_neighbors_filtered",
                "neighbor_count": 0,
            },
            "bayesian_result": {
                "good_value_probability": 0.73,
                "derived_metrics": {
                    "price_gap_vs_peer": None,
                },
                "resolved_input": {
                    "price": 24.99,
                    "peer_price": None,
                    "trust_probability": 0.81,
                    "ewom_score_0_to_100": 66.0,
                },
                "fused_agent_signals": {
                    "review_count": 8,
                },
            },
        }

        summary = summarize_ebay_candidate_value_result(result)

        self.assertEqual(summary["prediction"], "good_value")
        self.assertEqual(summary["prediction_reason"], "bayesian_probability_without_price")
        self.assertEqual(summary["retrieval_status"], "all_neighbors_filtered")
        self.assertEqual(summary["retrieved_neighbor_count"], 0)
        self.assertFalse(summary["price_considered"])
        self.assertEqual(summary["price_note"], "Bayesian score uses a neutral fair-price bucket.")
        self.assertEqual(summary["peer_price_source"], "none")

    def test_compare_ebay_candidate_value_results_prefers_higher_supported_probability(self) -> None:
        result_a = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/a",
                "title": "Listing A",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "retrieval",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability",
            },
            "market_context": {
                "retrieval_status": "usable",
                "neighbor_count": 5,
            },
            "bayesian_result": {
                "good_value_probability": 0.61,
                "derived_metrics": {
                    "price_gap_vs_peer": 0.05,
                },
                "resolved_input": {
                    "price": 100.0,
                    "peer_price": 105.0,
                    "trust_probability": 0.80,
                    "ewom_score_0_to_100": 66.0,
                },
            },
        }
        result_b = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/b",
                "title": "Listing B",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "retrieval",
            },
            "decision": {
                "prediction": "not_good_value",
                "reason": "bayesian_probability",
            },
            "market_context": {
                "retrieval_status": "usable",
                "neighbor_count": 5,
            },
            "bayesian_result": {
                "good_value_probability": 0.42,
                "derived_metrics": {
                    "price_gap_vs_peer": -0.10,
                },
                "resolved_input": {
                    "price": 110.0,
                    "peer_price": 100.0,
                    "trust_probability": 0.75,
                    "ewom_score_0_to_100": 62.0,
                },
            },
        }

        comparison = compare_ebay_candidate_value_results(result_a, result_b)

        self.assertEqual(comparison["comparison"]["verdict"], "better_A")
        self.assertAlmostEqual(
            comparison["comparison"]["good_value_probability_delta"],
            0.19,
        )

    def test_compare_ebay_candidate_value_results_uses_neutral_price_when_one_side_lacks_neighbors(self) -> None:
        result_a = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/a",
                "title": "Listing A",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "retrieval",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability",
            },
            "market_context": {
                "retrieval_status": "usable",
                "neighbor_count": 5,
            },
            "bayesian_result": {
                "good_value_probability": 0.61,
                "derived_metrics": {
                    "price_gap_vs_peer": 0.05,
                },
                "resolved_input": {
                    "price": 100.0,
                    "peer_price": 105.0,
                    "trust_probability": 0.80,
                    "ewom_score_0_to_100": 66.0,
                    "ewom_magnitude_0_to_100": 52.0,
                },
            },
        }
        result_b = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/b",
                "title": "Listing B",
            },
            "pricing": {
                "total_price_currency": "SGD",
                "peer_price_source": "none",
                "price_considered": False,
                "price_note": "Neutral fair-price fallback used.",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability_without_price",
            },
            "market_context": {
                "retrieval_status": "insufficient_peer_neighbors",
                "neighbor_count": 1,
            },
            "bayesian_result": {
                "good_value_probability": 0.66,
                "derived_metrics": {
                    "price_gap_vs_peer": None,
                },
                "resolved_input": {
                    "price": 153.58,
                    "peer_price": None,
                    "trust_probability": 0.31,
                    "ewom_score_0_to_100": 42.0,
                    "ewom_magnitude_0_to_100": 18.0,
                },
            },
        }

        comparison = compare_ebay_candidate_value_results(result_a, result_b)

        self.assertEqual(comparison["comparison"]["verdict"], "better_A")
        self.assertEqual(comparison["comparison"]["price_comparison_mode"], "neutral_fallback")
        self.assertEqual(comparison["listing_a"]["peer_price"], None)
        self.assertEqual(comparison["listing_b"]["peer_price"], None)
        self.assertIn("does not consider price advantage", comparison["comparison"]["reasons"][0])

    def test_compare_ebay_candidate_value_results_breaks_probability_ties_with_price_gap(self) -> None:
        result_a = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/a",
                "title": "Listing A",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "retrieval",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability",
            },
            "market_context": {
                "retrieval_status": "usable",
                "neighbor_count": 5,
            },
            "bayesian_result": {
                "good_value_probability": 0.8892,
                "derived_metrics": {
                    "price_gap_vs_peer": 0.4737,
                },
                "resolved_input": {
                    "price": 109.0,
                    "peer_price": 207.1,
                    "trust_probability": 0.77,
                    "ewom_score_0_to_100": 60.9,
                },
            },
        }
        result_b = {
            "candidate": {
                "source_url": "https://www.ebay.com.sg/itm/b",
                "title": "Listing B",
            },
            "pricing": {
                "total_price_currency": "USD",
                "peer_price_source": "retrieval",
            },
            "decision": {
                "prediction": "good_value",
                "reason": "bayesian_probability",
            },
            "market_context": {
                "retrieval_status": "usable",
                "neighbor_count": 5,
            },
            "bayesian_result": {
                "good_value_probability": 0.8892,
                "derived_metrics": {
                    "price_gap_vs_peer": 0.2506,
                },
                "resolved_input": {
                    "price": 153.58,
                    "peer_price": 204.93,
                    "trust_probability": 0.84,
                    "ewom_score_0_to_100": 60.96,
                },
            },
        }

        comparison = compare_ebay_candidate_value_results(result_a, result_b)

        self.assertEqual(comparison["comparison"]["verdict"], "better_A")
        self.assertAlmostEqual(
            comparison["comparison"]["good_value_probability_delta"],
            0.0,
        )
        self.assertIn("falls back to price gap vs peer", comparison["comparison"]["reasons"][0])

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
                    "neighbors": [{"price": 31.0, "similarity": 0.91}],
                }
            ],
            [
                {
                    "peer_price": 29.0,
                    "neighbor_count": 3,
                    "average_neighbor_similarity": 0.74,
                    "neighbors": [
                        {"price": 28.0, "similarity": 0.78},
                        {"price": 29.0, "similarity": 0.74},
                        {"price": 30.0, "similarity": 0.70},
                    ],
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
            min_peer_neighbor_count=1,
        )

        self.assertEqual([row["k"] for row in sweep["k_sweep"]], [1, 3])
        self.assertEqual(sweep["k_sweep"][0]["peer_price"], 31.0)
        self.assertEqual(sweep["k_sweep"][1]["peer_price"], 28.963963963963966)
        self.assertIsNone(sweep["k_sweep"][0]["prediction"])
        self.assertIsNone(sweep["k_sweep"][0]["prediction_reason"])
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
