from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from value import WorthBuyingConfig, score_worth_buying_split, train_worth_buying_pipeline
from value.worth_buying import inspect_worth_buying_catalog_neighbors, load_model


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


class WorthBuyingPipelineTests(unittest.TestCase):
    def test_train_and_score_split(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"worth_buying_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = tmpdir / "train.jsonl"
            val_path = tmpdir / "val.jsonl"
            output_prefix = tmpdir / "artifacts" / "worth_buying"
            score_output_path = tmpdir / "scores.csv"

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
                },
                {
                    "parent_asin": "A4",
                    "title": "hdmi cable braided",
                    "store": "CableCo",
                    "main_category": "Electronics",
                    "product_document": "hdmi cable braided 4k television cable",
                    "price": 12.0,
                    "average_rating": 4.8,
                    "rating_number": 800,
                    "review_count": 760,
                    "verified_purchase_rate": 0.95,
                    "helpful_vote_total": 80,
                    "helpful_vote_avg": 0.11,
                    "avg_review_rating": 4.7,
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
                },
            ]

            _write_jsonl(train_path, train_rows)
            _write_jsonl(val_path, val_rows)

            metadata = train_worth_buying_pipeline(
                train_path=train_path,
                output_prefix=output_prefix,
                config=WorthBuyingConfig(min_df=1, top_k_neighbors=2, min_neighbors=1),
            )
            self.assertTrue(Path(metadata["model_path"]).exists())

            summary = score_worth_buying_split(
                model_path=metadata["model_path"],
                split_path=val_path,
                output_path=score_output_path,
            )
            self.assertEqual(summary["rows_scored"], 2)
            self.assertTrue(score_output_path.exists())

            scored = pd.read_csv(score_output_path)
            self.assertIn("worth_buying_score", scored.columns)
            self.assertIn("verdict", scored.columns)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_kind_aware_retrieval_prefers_same_listing_kind(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"worth_buying_kind_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            train_path = tmpdir / "train.jsonl"
            output_prefix = tmpdir / "artifacts" / "worth_buying"

            train_rows = [
                {
                    "parent_asin": "D1",
                    "title": "Apple AirPods Pro Wireless Earbuds",
                    "store": "Apple",
                    "main_category": "All Electronics",
                    "listing_kind": "device",
                    "product_document": "apple airpods pro wireless earbuds bluetooth noise canceling",
                    "price": 189.0,
                    "average_rating": 4.7,
                    "rating_number": 1800,
                    "review_count": 1600,
                    "verified_purchase_rate": 0.96,
                    "helpful_vote_total": 400,
                    "helpful_vote_avg": 0.25,
                    "avg_review_rating": 4.7,
                },
                {
                    "parent_asin": "D2",
                    "title": "Wireless Earbuds In Ear Noise Cancelling",
                    "store": "Acme",
                    "main_category": "All Electronics",
                    "listing_kind": "device",
                    "product_document": "wireless earbuds in ear noise canceling bluetooth",
                    "price": 79.0,
                    "average_rating": 4.3,
                    "rating_number": 500,
                    "review_count": 450,
                    "verified_purchase_rate": 0.91,
                    "helpful_vote_total": 80,
                    "helpful_vote_avg": 0.18,
                    "avg_review_rating": 4.2,
                },
                {
                    "parent_asin": "C1",
                    "title": "AirPods Pro Charging Cable",
                    "store": "CableCo",
                    "main_category": "Cell Phones & Accessories",
                    "listing_kind": "cable",
                    "product_document": "airpods pro charging cable lightning usb c cable",
                    "price": 12.0,
                    "average_rating": 4.8,
                    "rating_number": 800,
                    "review_count": 760,
                    "verified_purchase_rate": 0.95,
                    "helpful_vote_total": 80,
                    "helpful_vote_avg": 0.11,
                    "avg_review_rating": 4.7,
                },
            ]
            _write_jsonl(train_path, train_rows)

            metadata = train_worth_buying_pipeline(
                train_path=train_path,
                output_prefix=output_prefix,
                config=WorthBuyingConfig(min_df=1, top_k_neighbors=2, min_neighbors=1),
            )
            bundle = load_model(metadata["model_path"])

            diagnostics = inspect_worth_buying_catalog_neighbors(
                pd.DataFrame(
                    [
                        {
                            "parent_asin": "Q1",
                            "title": "Apple AirPods Pro (2nd generation) Earbud",
                            "store": "Apple",
                            "main_category": "",
                            "listing_kind": "device",
                            "product_document": "apple airpods pro 2nd generation earbud in ear bluetooth",
                            "price": 139.0,
                            "average_rating": None,
                            "rating_number": None,
                            "review_count": 0,
                            "verified_purchase_rate": None,
                            "helpful_vote_total": 0.0,
                            "helpful_vote_avg": 0.0,
                            "avg_review_rating": None,
                            "trust_probability": None,
                            "ewom_score_0_to_100": None,
                            "ewom_magnitude_0_to_100": None,
                        }
                    ]
                ),
                model_bundle=bundle,
                config_overrides={"top_k_neighbors": 2, "min_similarity": 0.02},
            )

            self.assertEqual(diagnostics[0]["retrieval_listing_kind"], "device")
            neighbor_titles = [neighbor["title"] for neighbor in diagnostics[0]["neighbors"]]
            self.assertIn("Apple AirPods Pro Wireless Earbuds", neighbor_titles)
            self.assertNotIn("AirPods Pro Charging Cable", neighbor_titles)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
