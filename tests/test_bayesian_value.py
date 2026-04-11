from __future__ import annotations

import unittest

from value import (
    BayesianValueInput,
    build_value_evidence,
    extract_ewom_bayesian_signals,
    fuse_ewom_result_into_bayesian_input,
    score_good_value_probability,
)


class BayesianValueModelTests(unittest.TestCase):
    def test_favorable_product_scores_higher_than_unfavorable_product(self) -> None:
        favorable = BayesianValueInput(
            trust_probability=0.88,
            ewom_score_0_to_100=74.0,
            ewom_magnitude_0_to_100=68.0,
            average_rating=4.5,
            rating_count=420,
            verified_purchase_rate=0.91,
            price=79.0,
            peer_price=99.0,
            warranty_months=12.0,
            return_window_days=30.0,
        )
        unfavorable = BayesianValueInput(
            trust_probability=0.22,
            ewom_score_0_to_100=28.0,
            ewom_magnitude_0_to_100=70.0,
            average_rating=2.9,
            rating_count=35,
            verified_purchase_rate=0.41,
            price=129.0,
            peer_price=99.0,
            warranty_months=0.0,
            return_window_days=7.0,
        )

        favorable_result = score_good_value_probability(favorable)
        unfavorable_result = score_good_value_probability(unfavorable)

        self.assertGreater(
            favorable_result["good_value_probability"],
            unfavorable_result["good_value_probability"],
        )
        self.assertGreater(favorable_result["good_value_probability"], 0.70)
        self.assertLess(unfavorable_result["good_value_probability"], 0.35)

    def test_build_value_evidence_derives_relative_price_bucket(self) -> None:
        evidence, derived = build_value_evidence(
            BayesianValueInput(price=80.0, peer_price=100.0)
        )

        self.assertEqual(evidence["RelativePriceBucket"], "cheaper")
        self.assertAlmostEqual(derived["price_gap_vs_peer"], 0.20)

    def test_build_value_evidence_uses_finer_price_bins_for_large_gaps(self) -> None:
        evidence_a, _ = build_value_evidence(
            BayesianValueInput(price=109.0, peer_price=207.10119974788003)
        )
        evidence_b, _ = build_value_evidence(
            BayesianValueInput(price=153.58, peer_price=204.93384007512537)
        )

        self.assertEqual(evidence_a["RelativePriceBucket"], "extreme_cheaper")
        self.assertEqual(evidence_b["RelativePriceBucket"], "much_cheaper")

    def test_build_value_evidence_uses_finer_trust_and_polarity_bins(self) -> None:
        evidence, _ = build_value_evidence(
            BayesianValueInput(
                trust_probability=0.86,
                ewom_score_0_to_100=74.0,
            )
        )

        self.assertEqual(evidence["TrustSignal"], "very_high")
        self.assertEqual(evidence["ReviewPolarity"], "positive")

    def test_build_value_evidence_can_apply_neutral_relative_price_fallback(self) -> None:
        evidence, derived = build_value_evidence(
            BayesianValueInput(price=109.0, peer_price=None),
            default_relative_price_bucket="fair",
        )

        self.assertEqual(evidence["RelativePriceBucket"], "fair")
        self.assertIsNone(derived["price_gap_vs_peer"])
        self.assertEqual(derived["relative_price_bucket"], "fair")
        self.assertEqual(derived["relative_price_bucket_source"], "default")

    def test_missing_optional_signals_still_produces_probability(self) -> None:
        result = score_good_value_probability(
            BayesianValueInput(
                average_rating=4.2,
                rating_count=120,
                price=95.0,
                peer_price=100.0,
            )
        )

        self.assertGreaterEqual(result["good_value_probability"], 0.0)
        self.assertLessEqual(result["good_value_probability"], 1.0)
        self.assertIn("ProductQuality", result["component_posteriors"])

    def test_extract_review_set_agent_signals_uses_aggregate_outputs(self) -> None:
        ewom_result = {
            "review_count": 3,
            "reviews": [
                {
                    "deception": {
                        "trust_probability": 0.80,
                        "deception_probability": 0.20,
                    }
                },
                {
                    "deception": {
                        "trust_probability": 0.50,
                        "deception_probability": 0.50,
                    }
                },
                {
                    "deception": {
                        "trust_probability": 0.65,
                        "deception_probability": 0.35,
                    }
                },
            ],
            "aggregate": {
                "mean_deception_probability": 0.35,
                "mean_deception_weight": 0.65,
                "final_ewom_score_0_to_100": 62.0,
                "final_ewom_magnitude_0_to_100": 47.0,
            },
        }

        extracted = extract_ewom_bayesian_signals(ewom_result)

        self.assertEqual(extracted["source_type"], "review_set")
        self.assertEqual(extracted["review_count"], 3)
        self.assertAlmostEqual(extracted["trust_probability"], 0.65)
        self.assertEqual(extracted["ewom_score_0_to_100"], 62.0)
        self.assertEqual(extracted["ewom_magnitude_0_to_100"], 47.0)

    def test_extract_review_set_agent_signals_does_not_assume_perfect_trust_on_fallback(self) -> None:
        ewom_result = {
            "review_count": 2,
            "reviews": [
                {
                    "deception": {
                        "trust_probability": None,
                        "deception_probability": None,
                    },
                    "fusion": {
                        "deception_weight": 1.0,
                    },
                },
                {
                    "deception": {
                        "trust_probability": None,
                        "deception_probability": None,
                    },
                    "fusion": {
                        "deception_weight": 1.0,
                    },
                },
            ],
            "aggregate": {
                "mean_deception_probability": None,
                "mean_deception_weight": 1.0,
                "final_ewom_score_0_to_100": 53.0,
                "final_ewom_magnitude_0_to_100": 18.0,
            },
        }

        extracted = extract_ewom_bayesian_signals(ewom_result)

        self.assertIsNone(extracted["trust_probability"])

    def test_fuse_ewom_result_into_bayesian_input_preserves_non_agent_fields(self) -> None:
        base_input = BayesianValueInput(
            average_rating=4.4,
            rating_count=200,
            verified_purchase_rate=0.83,
            price=79.0,
            peer_price=99.0,
            warranty_months=12.0,
            return_window_days=30.0,
        )
        ewom_result = {
            "review_count": 2,
            "aggregate": {
                "mean_deception_probability": 0.20,
                "final_ewom_score_0_to_100": 68.0,
                "final_ewom_magnitude_0_to_100": 41.0,
            },
        }

        fused_input, extracted = fuse_ewom_result_into_bayesian_input(
            base_input,
            ewom_result,
        )

        self.assertAlmostEqual(fused_input.trust_probability, 0.80)
        self.assertEqual(fused_input.ewom_score_0_to_100, 68.0)
        self.assertEqual(fused_input.ewom_magnitude_0_to_100, 41.0)
        self.assertEqual(fused_input.average_rating, 4.4)
        self.assertEqual(fused_input.peer_price, 99.0)
        self.assertEqual(extracted["source_type"], "review_set")

    def test_fuse_ewom_result_can_preserve_existing_listing_trust(self) -> None:
        base_input = BayesianValueInput(
            trust_probability=0.91,
            price=79.0,
            peer_price=99.0,
        )
        ewom_result = {
            "review_count": 2,
            "aggregate": {
                "mean_deception_probability": 0.35,
                "final_ewom_score_0_to_100": 68.0,
                "final_ewom_magnitude_0_to_100": 41.0,
            },
        }

        fused_input, extracted = fuse_ewom_result_into_bayesian_input(
            base_input,
            ewom_result,
            include_trust_probability=False,
        )

        self.assertAlmostEqual(fused_input.trust_probability, 0.91)
        self.assertEqual(fused_input.ewom_score_0_to_100, 68.0)
        self.assertEqual(fused_input.ewom_magnitude_0_to_100, 41.0)
        self.assertAlmostEqual(extracted["trust_probability"], 0.65)


if __name__ == "__main__":
    unittest.main()
