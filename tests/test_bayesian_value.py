from __future__ import annotations

import unittest

from value import BayesianValueInput, build_value_evidence, score_good_value_probability


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


if __name__ == "__main__":
    unittest.main()
