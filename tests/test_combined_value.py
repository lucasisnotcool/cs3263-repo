from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from value import WorthBuyingConfig, score_combined_value_split, train_worth_buying_pipeline
from value.train_bayesian_value_model import train_bayesian_value_network


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


class CombinedValuePipelineTests(unittest.TestCase):
    def test_combined_value_pipeline_scores_rows(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"combined_value_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = tmpdir / "train.jsonl"
            val_path = tmpdir / "val.jsonl"
            output_prefix = tmpdir / "artifacts" / "worth_buying"
            score_output_path = tmpdir / "combined_scores.csv"

            train_rows = [
                {
                    "parent_asin": "A1",
                    "title": "wireless earbuds pro",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth noise canceling usb c",
                    "price": 79.0,
                    "average_rating": 4.6,
                    "rating_number": 420,
                    "review_count": 380,
                    "verified_purchase_rate": 0.91,
                    "helpful_vote_total": 120,
                    "helpful_vote_avg": 0.32,
                    "avg_review_rating": 4.5,
                    "trust_probability": 0.92,
                    "ewom_score_0_to_100": 82.0,
                    "ewom_magnitude_0_to_100": 67.0,
                },
                {
                    "parent_asin": "A2",
                    "title": "wireless earbuds basic",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth basic usb c",
                    "price": 99.0,
                    "average_rating": 4.0,
                    "rating_number": 210,
                    "review_count": 200,
                    "verified_purchase_rate": 0.85,
                    "helpful_vote_total": 40,
                    "helpful_vote_avg": 0.20,
                    "avg_review_rating": 4.0,
                    "trust_probability": 0.84,
                    "ewom_score_0_to_100": 70.0,
                    "ewom_magnitude_0_to_100": 53.0,
                },
                {
                    "parent_asin": "A3",
                    "title": "gaming headset wired",
                    "store": "Beta",
                    "main_category": "Electronics",
                    "product_document": "gaming headset wired microphone rgb pc",
                    "price": 59.0,
                    "average_rating": 4.4,
                    "rating_number": 300,
                    "review_count": 260,
                    "verified_purchase_rate": 0.89,
                    "helpful_vote_total": 60,
                    "helpful_vote_avg": 0.23,
                    "avg_review_rating": 4.3,
                    "trust_probability": 0.87,
                    "ewom_score_0_to_100": 78.0,
                    "ewom_magnitude_0_to_100": 59.0,
                },
            ]
            val_rows = [
                {
                    "parent_asin": "B1",
                    "title": "wireless earbuds lite",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth compact usb c",
                    "price": 69.0,
                    "average_rating": 4.5,
                    "rating_number": 130,
                    "review_count": 120,
                    "verified_purchase_rate": 0.90,
                    "helpful_vote_total": 22,
                    "helpful_vote_avg": 0.18,
                    "avg_review_rating": 4.4,
                    "trust_probability": 0.88,
                    "ewom_score_0_to_100": 80.0,
                    "ewom_magnitude_0_to_100": 61.0,
                },
                {
                    "parent_asin": "B2",
                    "title": "wired headset economy",
                    "store": "Beta",
                    "main_category": "Electronics",
                    "product_document": "gaming headset wired economy microphone",
                    "price": 79.0,
                    "average_rating": 3.4,
                    "rating_number": 45,
                    "review_count": 35,
                    "verified_purchase_rate": 0.62,
                    "helpful_vote_total": 3,
                    "helpful_vote_avg": 0.08,
                    "avg_review_rating": 3.5,
                    "trust_probability": 0.55,
                    "ewom_score_0_to_100": 46.0,
                    "ewom_magnitude_0_to_100": 28.0,
                },
            ]

            _write_jsonl(train_path, train_rows)
            _write_jsonl(val_path, val_rows)

            metadata = train_worth_buying_pipeline(
                train_path=train_path,
                output_prefix=output_prefix,
                config=WorthBuyingConfig(min_df=1, top_k_neighbors=2, min_neighbors=1),
            )
            summary = score_combined_value_split(
                model_path=metadata["model_path"],
                split_path=val_path,
                output_path=score_output_path,
                probability_threshold=0.5,
            )

            self.assertEqual(summary["rows_scored"], 2)
            self.assertTrue(score_output_path.exists())

            scored = pd.read_csv(score_output_path)
            self.assertIn("combined_good_value_probability", scored.columns)
            self.assertIn("combined_prediction", scored.columns)
            self.assertIn("peer_price", scored.columns)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_combined_value_pipeline_can_use_trained_bayesian_network_path(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"combined_value_bayes_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = tmpdir / "train.jsonl"
            val_path = tmpdir / "val.jsonl"
            output_prefix = tmpdir / "artifacts" / "worth_buying"
            score_output_path = tmpdir / "combined_scores.csv"
            bayesian_dataset_path = tmpdir / "bayesian_train.jsonl"
            bayesian_network_path = tmpdir / "artifacts" / "trained_bayes.json"

            train_rows = [
                {
                    "parent_asin": "A1",
                    "title": "wireless earbuds pro",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth noise canceling usb c",
                    "price": 79.0,
                    "average_rating": 4.6,
                    "rating_number": 420,
                    "review_count": 380,
                    "verified_purchase_rate": 0.91,
                    "helpful_vote_total": 120,
                    "helpful_vote_avg": 0.32,
                    "avg_review_rating": 4.5,
                    "trust_probability": 0.92,
                    "ewom_score_0_to_100": 82.0,
                    "ewom_magnitude_0_to_100": 67.0,
                },
                {
                    "parent_asin": "A2",
                    "title": "wireless earbuds basic",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth basic usb c",
                    "price": 99.0,
                    "average_rating": 4.0,
                    "rating_number": 210,
                    "review_count": 200,
                    "verified_purchase_rate": 0.85,
                    "helpful_vote_total": 40,
                    "helpful_vote_avg": 0.20,
                    "avg_review_rating": 4.0,
                    "trust_probability": 0.84,
                    "ewom_score_0_to_100": 70.0,
                    "ewom_magnitude_0_to_100": 53.0,
                },
            ]
            val_rows = [
                {
                    "parent_asin": "B1",
                    "title": "wireless earbuds lite",
                    "store": "Acme",
                    "main_category": "Electronics",
                    "product_document": "wireless earbuds bluetooth compact usb c",
                    "price": 69.0,
                    "average_rating": 4.5,
                    "rating_number": 130,
                    "review_count": 120,
                    "verified_purchase_rate": 0.90,
                    "helpful_vote_total": 22,
                    "helpful_vote_avg": 0.18,
                    "avg_review_rating": 4.4,
                    "trust_probability": 0.88,
                    "ewom_score_0_to_100": 80.0,
                    "ewom_magnitude_0_to_100": 61.0,
                }
            ]

            _write_jsonl(train_path, train_rows)
            _write_jsonl(val_path, val_rows)
            _write_jsonl(
                bayesian_dataset_path,
                [
                    {
                        "good_value_label": "yes",
                        "bayesian_evidence": {
                            "TrustSignal": "very_high",
                            "ReviewPolarity": "very_positive",
                            "ReviewStrength": "strong",
                            "RatingSignal": "high",
                            "ReviewVolume": "high",
                            "VerifiedSignal": "high",
                            "RelativePriceBucket": "much_cheaper",
                        },
                    },
                    {
                        "good_value_label": "no",
                        "bayesian_evidence": {
                            "TrustSignal": "low",
                            "ReviewPolarity": "negative",
                            "ReviewStrength": "medium",
                            "RatingSignal": "low",
                            "ReviewVolume": "medium",
                            "VerifiedSignal": "low",
                            "RelativePriceBucket": "much_pricier",
                        },
                    },
                ],
            )
            train_bayesian_value_network(
                dataset_path=bayesian_dataset_path,
                output_path=bayesian_network_path,
                smoothing=0.25,
            )

            metadata = train_worth_buying_pipeline(
                train_path=train_path,
                output_prefix=output_prefix,
                config=WorthBuyingConfig(min_df=1, top_k_neighbors=2, min_neighbors=1),
            )
            summary = score_combined_value_split(
                model_path=metadata["model_path"],
                split_path=val_path,
                output_path=score_output_path,
                probability_threshold=0.5,
                bayesian_network_path=bayesian_network_path,
            )

            self.assertEqual(
                summary["bayesian_network_path"],
                str(bayesian_network_path.resolve()),
            )
            self.assertTrue(score_output_path.exists())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
