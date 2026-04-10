from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from value.bayesian_training_data import build_bayesian_training_dataset
from value.electronics_filter import is_actual_electronics_device
from value.train_bayesian_value_model import (
    load_bayesian_value_network,
    train_bayesian_value_network,
)
from value.worth_buying import WorthBuyingConfig, train_worth_buying_pipeline
from value.bayesian_value import BayesianValueInput, score_good_value_probability


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


class BayesianTrainingTests(unittest.TestCase):
    def test_actual_electronics_filter_excludes_obvious_accessory(self) -> None:
        self.assertTrue(
            is_actual_electronics_device(
                {
                    "title": "GIGABYTE AERO 16 Laptop",
                    "categories": ["Electronics", "Computers & Accessories", "Laptops"],
                    "listing_kind": "device",
                }
            )
        )
        self.assertFalse(
            is_actual_electronics_device(
                {
                    "title": "GGS Swivi HD DSLR LCD Universal Foldable Viewfinder",
                    "categories": ["Electronics", "Camera & Photo", "Accessories", "Viewfinders"],
                    "listing_kind": "device",
                }
            )
        )

    def test_build_bayesian_training_dataset_labels_rows(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"bayesian_training_data_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = tmpdir / "worth_buying_train.jsonl"
            source_path = tmpdir / "source.jsonl"
            output_prefix = tmpdir / "artifacts" / "worth_buying"
            dataset_path = tmpdir / "bayesian" / "dataset.jsonl"

            train_rows = [
                _product("D1", "Wireless Earbuds Pro", 100.0, 4.6, 500),
                _product("D2", "Wireless Earbuds Lite", 80.0, 4.4, 420),
                _product("D3", "Wireless Earbuds Basic", 75.0, 4.1, 250),
                _product("D4", "Wireless Earbuds Overpriced", 210.0, 2.7, 45),
            ]
            source_rows = [
                _product("Q1", "Wireless Earbuds Deal", 55.0, 4.7, 180),
                _product("Q2", "Wireless Earbuds Poor Value", 220.0, 2.8, 60),
                {
                    **_product("A1", "Wireless Earbuds Carrying Case", 12.0, 4.8, 300),
                    "listing_kind": "case",
                    "categories": ["Electronics", "Headphone Cases"],
                },
            ]
            _write_jsonl(train_path, train_rows)
            _write_jsonl(source_path, source_rows)

            metadata = train_worth_buying_pipeline(
                train_path=train_path,
                output_prefix=output_prefix,
                config=WorthBuyingConfig(
                    min_df=1,
                    top_k_neighbors=2,
                    min_neighbors=1,
                    min_similarity=0.01,
                ),
                allowed_listing_kinds=["device"],
            )

            summary = build_bayesian_training_dataset(
                input_paths=[source_path],
                worth_buying_model_path=metadata["model_path"],
                output_path=dataset_path,
                min_confidence=0.0,
                include_consider_as_no=True,
            )

            self.assertEqual(summary["rows_seen"], 3)
            self.assertEqual(summary["rows_after_filter"], 2)
            self.assertGreaterEqual(summary["rows_written"], 1)
            written_rows = [
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(all(row["listing_kind"] == "device" for row in written_rows))
            self.assertTrue(all(row["good_value_label"] in {"yes", "no"} for row in written_rows))
            self.assertTrue(all("RelativePriceBucket" in row["bayesian_evidence"] for row in written_rows))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_train_and_load_bayesian_value_network(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"bayesian_train_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            dataset_path = tmpdir / "dataset.jsonl"
            output_path = tmpdir / "network.json"
            _write_jsonl(
                dataset_path,
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

            summary = train_bayesian_value_network(
                dataset_path=dataset_path,
                output_path=output_path,
                smoothing=0.25,
            )
            trained_network = load_bayesian_value_network(output_path)
            favorable = score_good_value_probability(
                BayesianValueInput(
                    trust_probability=0.90,
                    ewom_score_0_to_100=90.0,
                    ewom_magnitude_0_to_100=75.0,
                    average_rating=4.8,
                    rating_count=500,
                    verified_purchase_rate=0.95,
                    price=80.0,
                    peer_price=120.0,
                ),
                network=trained_network,
            )
            unfavorable = score_good_value_probability(
                BayesianValueInput(
                    trust_probability=0.25,
                    ewom_score_0_to_100=30.0,
                    ewom_magnitude_0_to_100=40.0,
                    average_rating=3.0,
                    rating_count=80,
                    verified_purchase_rate=0.40,
                    price=140.0,
                    peer_price=100.0,
                ),
                network=trained_network,
            )

            self.assertEqual(summary["rows_used"], 2)
            self.assertTrue(output_path.exists())
            self.assertGreater(
                favorable["good_value_probability"],
                unfavorable["good_value_probability"],
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _product(
    parent_asin: str,
    title: str,
    price: float,
    average_rating: float,
    review_count: int,
) -> dict:
    return {
        "parent_asin": parent_asin,
        "title": title,
        "store": "Acme",
        "main_category": "Electronics",
        "categories": ["Electronics", "Headphones & Earbuds", "Earbud Headphones"],
        "listing_kind": "device",
        "product_document": f"{title} bluetooth wireless earbuds noise canceling",
        "price": price,
        "average_rating": average_rating,
        "rating_number": review_count,
        "review_count": review_count,
        "verified_purchase_rate": 0.90 if average_rating >= 4.0 else 0.55,
        "helpful_vote_total": review_count // 3,
        "helpful_vote_avg": 0.33,
        "avg_review_rating": average_rating,
        "trust_probability": 0.85 if average_rating >= 4.0 else 0.35,
        "ewom_score_0_to_100": max(0.0, min(100.0, ((average_rating - 1.0) / 4.0) * 100.0)),
        "ewom_magnitude_0_to_100": 70.0,
    }


if __name__ == "__main__":
    unittest.main()
