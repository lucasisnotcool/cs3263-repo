from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from value.create_electronics_splits import choose_split, create_electronics_splits


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle)
            handle.write("\n")


class ElectronicsSplitBuilderTests(unittest.TestCase):
    def test_choose_split_is_deterministic(self) -> None:
        first = choose_split("PARENT123", validation_ratio=0.1, test_ratio=0.1, random_state=42)
        second = choose_split("PARENT123", validation_ratio=0.1, test_ratio=0.1, random_state=42)
        self.assertEqual(first, second)

    def test_create_electronics_splits_writes_expected_rows(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"electronics_split_{uuid.uuid4().hex}"
        tmpdir.mkdir(parents=True, exist_ok=True)
        try:
            meta_path = tmpdir / "meta.jsonl"
            review_path = tmpdir / "reviews.jsonl"
            output_dir = tmpdir / "splits"

            _write_jsonl(
                meta_path,
                [
                    {
                        "parent_asin": "A1",
                        "title": "Wireless Earbuds",
                        "main_category": "Electronics",
                        "store": "Acme",
                        "categories": ["Electronics", "Audio"],
                        "features": ["Bluetooth", "USB-C"],
                        "description": ["Compact earbuds"],
                        "details": {"Brand": "Acme"},
                        "price": 79.0,
                        "average_rating": 4.5,
                        "rating_number": 100,
                    },
                    {
                        "parent_asin": "A2",
                        "title": "HDMI Cable",
                        "main_category": "Electronics",
                        "store": "CableCo",
                        "categories": ["Electronics", "Cables"],
                        "features": ["4K"],
                        "description": ["Braided cable"],
                        "details": {"Brand": "CableCo"},
                        "price": 15.0,
                        "average_rating": 4.7,
                        "rating_number": 240,
                    },
                ],
            )
            _write_jsonl(
                review_path,
                [
                    {
                        "parent_asin": "A1",
                        "rating": 5.0,
                        "verified_purchase": True,
                        "helpful_vote": 2,
                    },
                    {
                        "parent_asin": "A1",
                        "rating": 4.0,
                        "verified_purchase": False,
                        "helpful_vote": 0,
                    },
                    {
                        "parent_asin": "A2",
                        "rating": 5.0,
                        "verified_purchase": True,
                        "helpful_vote": 1,
                    },
                ],
            )

            summary = create_electronics_splits(
                meta_path=meta_path,
                review_path=review_path,
                output_dir=output_dir,
                validation_ratio=0.2,
                test_ratio=0.2,
                random_state=7,
                log_every=1,
            )

            self.assertEqual(summary["splits"]["rows_written"], 2)
            self.assertTrue((output_dir / "split_summary.json").exists())
            self.assertTrue((output_dir / "electronics_products_train.jsonl").exists())
            self.assertTrue((output_dir / "electronics_products_val.jsonl").exists())
            self.assertTrue((output_dir / "electronics_products_test.jsonl").exists())

            written_rows: list[dict] = []
            for split_name in ("train", "val", "test"):
                split_path = output_dir / f"electronics_products_{split_name}.jsonl"
                with split_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            written_rows.append(json.loads(line))

            rows_by_asin = {row["parent_asin"]: row for row in written_rows}
            self.assertEqual(rows_by_asin["A1"]["listing_kind"], "device")
            self.assertEqual(rows_by_asin["A2"]["listing_kind"], "cable")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
