from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eWOM.helpfulness.dataset_loader import PreparedHelpfulnessSplitLoader
from eWOM.helpfulness.train_test_splitter import (
    DEFAULT_POSITIVE_THRESHOLD,
    prepare_record,
)


class HelpfulnessSplitBuilderTests(unittest.TestCase):
    def test_default_positive_threshold_labels_any_helpful_vote_as_helpful(self) -> None:
        not_helpful = prepare_record(
            {"text": "short review", "helpful_votes": 0},
            positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
            drop_middle=False,
            min_review_words=0,
        )
        helpful = prepare_record(
            {"text": "short review", "helpful_votes": 1},
            positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
            drop_middle=False,
            min_review_words=0,
        )

        self.assertEqual(DEFAULT_POSITIVE_THRESHOLD, 1)
        self.assertEqual(not_helpful["label"], 0)
        self.assertEqual(helpful["label"], 1)

    def test_prepared_loader_relabels_from_helpful_votes_when_threshold_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.jsonl"
            rows = [
                {"text": "no votes", "helpful_votes": 0, "label": 1},
                {"text": "one vote", "helpful_votes": 1, "label": 0},
                {"text": "singular vote key", "helpful_vote": 2, "label": 0},
            ]
            with split_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            loaded = PreparedHelpfulnessSplitLoader(
                split_path,
                positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
            ).load()

        self.assertEqual(loaded["helpful_votes"].tolist(), [0, 1, 2])
        self.assertEqual(loaded["label"].tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
