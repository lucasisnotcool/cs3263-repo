from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eWOM.helpfulness.dataset_loader import PreparedHelpfulnessSplitLoader
from eWOM.helpfulness.train_test_splitter import (
    DEFAULT_POSITIVE_THRESHOLD,
    assign_and_write,
    build_output_paths,
    build_split_plan,
    prepare_record,
    scan_dataset,
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

    def test_balance_labels_undersamples_majority_class_for_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            review_path = tmpdir_path / "reviews.jsonl"
            output_dir = tmpdir_path / "splits"
            output_paths = build_output_paths(output_dir)
            rows = [
                {"text": f"negative {index}", "helpful_vote": 0}
                for index in range(6)
            ] + [
                {"text": f"positive {index}", "helpful_vote": 1}
                for index in range(4)
            ]
            with review_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            stats = scan_dataset(
                review_path,
                max_rows=None,
                positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
                drop_middle=False,
                min_review_words=0,
                log_every_rows=0,
            )
            plan = build_split_plan(
                stats,
                val_size=0.25,
                test_size=0.25,
                balance_labels=True,
            )
            assignment_stats = assign_and_write(
                review_path,
                stats,
                plan,
                output_paths=output_paths,
                max_rows=None,
                positive_threshold=DEFAULT_POSITIVE_THRESHOLD,
                drop_middle=False,
                min_review_words=0,
                random_state=42,
                shuffle_buffer_size=0,
                log_every_rows=0,
            )

        self.assertEqual(stats.label_counts, {0: 6, 1: 4})
        self.assertEqual(plan.train_targets, {0: 2, 1: 2})
        self.assertEqual(plan.val_targets, {0: 1, 1: 1})
        self.assertEqual(plan.test_targets, {0: 1, 1: 1})
        self.assertEqual(assignment_stats.train_label_counts, {0: 2, 1: 2})
        self.assertEqual(assignment_stats.val_label_counts, {0: 1, 1: 1})
        self.assertEqual(assignment_stats.test_label_counts, {0: 1, 1: 1})


if __name__ == "__main__":
    unittest.main()
