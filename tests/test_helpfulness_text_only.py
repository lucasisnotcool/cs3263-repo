from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eWOM.helpfulness.dataset_loader import PreparedHelpfulnessSplitLoader
from eWOM.helpfulness.feature_builder import (
    HelpfulnessFeatureBuilder,
    HelpfulnessFeatureConfig,
)
from eWOM.helpfulness.pipeline import format_pipeline_report, run_pipeline
from eWOM.helpfulness.predictor import HelpfulnessPredictor
from eWOM.helpfulness.preprocess import HelpfulnessPreprocessor


class DummyBinaryModel:
    classes_ = np.array([0, 1], dtype=int)

    def predict_proba(self, x):
        if x.shape[0] != 2:
            raise AssertionError("This dummy model expects exactly two rows.")
        return np.asarray(
            [
                [0.20, 0.80],
                [0.70, 0.30],
            ],
            dtype=float,
        )


class HelpfulnessTextOnlyTests(unittest.TestCase):
    def test_prepared_split_loader_keeps_helpful_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.jsonl"
            split_path.write_text(
                json.dumps(
                    {
                        "title": "Great",
                        "text": "Detailed review text",
                        "rating": 5,
                        "verified_purchase": True,
                        "helpful_votes": 7,
                        "label": 1,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            loaded = PreparedHelpfulnessSplitLoader(split_path).load()

        self.assertEqual(loaded["helpful_votes"].tolist(), [7])

    def test_default_feature_builder_uses_only_text_derived_lengths(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "title": "Battery",
                    "text": "Battery life is excellent",
                    "rating": 1.0,
                    "verified_purchase": False,
                    "label": 0,
                },
                {
                    "title": "Battery",
                    "text": "Battery life is excellent",
                    "rating": 5.0,
                    "verified_purchase": True,
                    "label": 1,
                },
            ]
        )
        transformed = HelpfulnessPreprocessor().transform(frame)
        builder = HelpfulnessFeatureBuilder(
            HelpfulnessFeatureConfig(
                max_features=100,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1),
            )
        )

        matrix = builder.fit_transform(transformed).toarray()

        self.assertEqual(
            builder.active_numeric_feature_names,
            ("review_len_words", "title_len_chars", "text_len_chars"),
        )
        self.assertEqual(matrix.shape[1], len(builder.vectorizer.vocabulary_) + 3)
        self.assertTrue(np.allclose(matrix[0], matrix[1]))

    def test_external_metadata_features_remain_opt_in(self) -> None:
        builder = HelpfulnessFeatureBuilder(
            HelpfulnessFeatureConfig(
                max_features=100,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 1),
                use_rating=True,
                use_verified_purchase=True,
            )
        )

        self.assertEqual(
            builder.active_numeric_feature_names,
            (
                "rating",
                "verified_purchase",
                "review_len_words",
                "title_len_chars",
                "text_len_chars",
            ),
        )

    def test_predictor_supports_text_length_bundle_without_external_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            feature_builder = HelpfulnessFeatureBuilder(
                HelpfulnessFeatureConfig(
                    max_features=50,
                    min_df=1,
                    max_df=1.0,
                    ngram_range=(1, 1),
                )
            )
            frame = HelpfulnessPreprocessor().transform(
                pd.DataFrame(
                    [
                        {"title": "", "text": "first review", "label": 0},
                        {"title": "", "text": "second review", "label": 1},
                    ]
                )
            )
            feature_builder.fit_transform(frame)

            model_path = tmpdir_path / "model.joblib"
            feature_builder_path = tmpdir_path / "feature_builder.joblib"
            joblib.dump(
                {
                    "model": DummyBinaryModel(),
                    "model_name": "dummy_binary_model",
                    "threshold": 0.6,
                },
                model_path,
            )
            joblib.dump(feature_builder, feature_builder_path)

            predictor = HelpfulnessPredictor(
                model_path=str(model_path),
                feature_builder_path=str(feature_builder_path),
            )
            predictions = predictor.predict_many(
                titles=["", ""],
                texts=["first review", "second review"],
            )

        self.assertEqual(predictions[0]["usefulness_probability"], 0.8)
        self.assertIs(predictions[0]["is_useful"], True)
        self.assertEqual(predictions[1]["usefulness_probability"], 0.3)
        self.assertIs(predictions[1]["is_useful"], False)

    def test_pipeline_defaults_to_text_derived_lengths_only_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.jsonl"
            val_path = tmpdir_path / "val.jsonl"
            test_path = tmpdir_path / "test.jsonl"
            model_output = tmpdir_path / "helpfulness_model"

            train_rows = [
                {"title": "Generic", "text": "short note", "helpful_votes": 0},
                {"title": "Detailed", "text": "battery lasted all day during travel", "helpful_votes": 3},
                {"title": "Plain", "text": "ok", "helpful_votes": 0},
                {"title": "Detailed", "text": "setup steps were clear and reproducible", "helpful_votes": 4},
            ]
            val_rows = [
                {"title": "Generic", "text": "fine", "helpful_votes": 0},
                {"title": "Detailed", "text": "long term usage notes with exact timings", "helpful_votes": 2},
            ]
            test_rows = [
                {"title": "Generic", "text": "nice", "helpful_votes": 0},
                {"title": "Detailed", "text": "packaging notes and installation details", "helpful_votes": 5},
            ]

            for path, rows in [
                (train_path, train_rows),
                (val_path, val_rows),
                (test_path, test_rows),
            ]:
                with path.open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row) + "\n")

            result = run_pipeline(
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                model_output=model_output,
                max_features=100,
                min_df=1,
                max_df=1.0,
                ngram_max=1,
                candidate_model_names=["multinomial_nb"],
                positive_threshold=1,
            )
            metadata_path = model_output.with_name(f"{model_output.name}_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_artifacts = metadata["candidate_artifacts"]["multinomial_nb"]
            self.assertTrue(Path(candidate_artifacts["model_path"]).exists())
            self.assertTrue(Path(candidate_artifacts["feature_builder_path"]).exists())
            self.assertEqual(
                result["artifacts"]["candidate_artifacts"]["multinomial_nb"],
                candidate_artifacts,
            )

        self.assertTrue(result["config"]["text_derived_lengths_only"])
        self.assertEqual(
            result["workflow"]["architecture"],
            "TF-IDF + text-derived length features + classifier comparison",
        )
        self.assertFalse(metadata["feature_config"]["use_rating"])
        self.assertFalse(metadata["feature_config"]["use_verified_purchase"])
        self.assertTrue(metadata["feature_config"]["use_text_length_features"])
        self.assertEqual(
            metadata["numeric_feature_names"],
            ["review_len_words", "title_len_chars", "text_len_chars"],
        )
        report = format_pipeline_report(result)
        self.assertIn("Validation candidate comparison:", report)
        self.assertIn("macro_f1", report)
        self.assertIn("multinomial_nb", report)


if __name__ == "__main__":
    unittest.main()
