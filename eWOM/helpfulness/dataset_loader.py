from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


LOGGER = logging.getLogger(__name__)
LABEL_TEXT_BY_ID = {
    0: "not_helpful",
    1: "helpful",
}


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


class AmazonReviewsLoader:
    """
    Loads Amazon Reviews'23 review JSONL or JSONL.GZ files.
    """

    def __init__(self, review_path: str | Path, max_rows: Optional[int] = None):
        self.review_path = Path(review_path)
        self.max_rows = max_rows

    def iter_rows(self) -> Iterator[dict]:
        loaded_rows = 0
        for obj in _iter_jsonl(self.review_path):
            yield {
                "rating": obj.get("rating"),
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
                "verified_purchase": obj.get("verified_purchase", False),
                # dataset page shows helpful_votes in examples, and helpful_vote in fields table
                # so handle both safely
                "helpful_votes": obj.get("helpful_votes", obj.get("helpful_vote", 0)),
                "asin": obj.get("asin"),
                "parent_asin": obj.get("parent_asin"),
                "user_id": obj.get("user_id"),
                "timestamp": obj.get("timestamp", obj.get("sort_timestamp")),
            }
            loaded_rows += 1
            if self.max_rows is not None and loaded_rows >= self.max_rows:
                break

    def load(self) -> pd.DataFrame:
        LOGGER.info(
            "Loading raw Amazon reviews from %s with max_rows=%s",
            self.review_path,
            self.max_rows,
        )
        rows = list(self.iter_rows())

        if not rows:
            raise ValueError(f"No rows loaded from {self.review_path}")

        df = pd.DataFrame(rows)
        LOGGER.info("Loaded %s raw review rows from %s", len(df), self.review_path)
        return df


class PreparedHelpfulnessSplitLoader:
    """
    Loads already-split helpfulness JSONL files and keeps only the columns
    needed by the TF-IDF + logistic regression pipeline.
    """

    def __init__(self, split_path: str | Path, max_rows: Optional[int] = None):
        self.split_path = Path(split_path)
        self.max_rows = max_rows

    def iter_rows(self) -> Iterator[dict]:
        loaded_rows = 0
        for obj in _iter_jsonl(self.split_path):
            yield {
                "rating": obj.get("rating"),
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
                "verified_purchase": obj.get("verified_purchase", False),
                "label": obj.get("label"),
            }
            loaded_rows += 1
            if self.max_rows is not None and loaded_rows >= self.max_rows:
                break

    def load(self) -> pd.DataFrame:
        LOGGER.info(
            "Loading prepared helpfulness split from %s with max_rows=%s",
            self.split_path,
            self.max_rows,
        )
        rows = list(self.iter_rows())

        if not rows:
            raise ValueError(f"No rows loaded from {self.split_path}")

        df = pd.DataFrame(rows)
        if "label" not in df.columns or df["label"].isna().any():
            raise ValueError(
                f"Prepared split at {self.split_path} is missing non-null 'label' values."
            )

        df["label"] = df["label"].astype(int)
        LOGGER.info(
            "Loaded %s prepared helpfulness rows from %s",
            len(df),
            self.split_path,
        )
        return df

    def iter_batches(self, batch_size: int) -> Iterator[pd.DataFrame]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        batch: list[dict] = []
        for row in self.iter_rows():
            batch.append(row)
            if len(batch) >= batch_size:
                yield pd.DataFrame(batch)
                batch = []

        if batch:
            yield pd.DataFrame(batch)
