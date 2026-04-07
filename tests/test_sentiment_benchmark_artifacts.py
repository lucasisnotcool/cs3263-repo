from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from eWOM.sentiment_analysis.run_sentiment_benchmark import (
    format_benchmark_report,
    run_benchmark,
)


class FakeAmazonPolarityLoader:
    def __init__(self, data_dir, random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state

    def load_train_test(self, max_train_rows=None, max_test_rows=None):
        train_df = pd.DataFrame(
            [
                {"text": "bad broken poor", "label": 0, "label_text": "negative"},
                {"text": "terrible waste return", "label": 0, "label_text": "negative"},
                {"text": "awful damaged noisy", "label": 0, "label_text": "negative"},
                {"text": "poor quality failed", "label": 0, "label_text": "negative"},
                {"text": "great useful reliable", "label": 1, "label_text": "positive"},
                {"text": "excellent value sturdy", "label": 1, "label_text": "positive"},
                {"text": "perfect setup easy", "label": 1, "label_text": "positive"},
                {"text": "good battery durable", "label": 1, "label_text": "positive"},
            ]
        )
        test_df = pd.DataFrame(
            [
                {"text": "broken poor", "label": 0, "label_text": "negative"},
                {"text": "terrible return", "label": 0, "label_text": "negative"},
                {"text": "great durable", "label": 1, "label_text": "positive"},
                {"text": "excellent easy", "label": 1, "label_text": "positive"},
            ]
        )
        return train_df, test_df


class SentimentBenchmarkArtifactsTests(unittest.TestCase):
    def test_benchmark_saves_each_candidate_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            model_output = tmpdir_path / "sentiment_model"

            with patch(
                "eWOM.sentiment_analysis.run_sentiment_benchmark.AmazonPolarityLoader",
                FakeAmazonPolarityLoader,
            ):
                result = run_benchmark(
                    data_dir=tmpdir_path / "dataset",
                    model_output=model_output,
                    val_ratio=0.25,
                    max_features=100,
                    min_df=1,
                    max_df=1.0,
                    ngram_max=1,
                    random_state=42,
                    candidate_model_names=["logistic_regression", "multinomial_nb"],
                    log_level="CRITICAL",
                )

            metadata_path = model_output.with_name(f"{model_output.name}_metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertTrue(Path(result["artifacts"]["model_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["feature_builder_path"]).exists())
            self.assertEqual(
                set(result["artifacts"]["candidate_artifacts"]),
                {"logistic_regression", "multinomial_nb"},
            )
            for model_name, candidate_artifacts in result["artifacts"][
                "candidate_artifacts"
            ].items():
                self.assertTrue(Path(candidate_artifacts["model_path"]).exists())
                self.assertTrue(
                    Path(candidate_artifacts["feature_builder_path"]).exists()
                )
                self.assertEqual(
                    metadata["candidate_artifacts"][model_name],
                    candidate_artifacts,
                )
                self.assertEqual(
                    result["model_selection"]["candidate_models"][model_name]["artifacts"],
                    candidate_artifacts,
                )

            report = format_benchmark_report(result)
            self.assertIn("Validation candidate comparison:", report)
            self.assertIn("macro_f1", report)
            self.assertIn("logistic_regression", report)


if __name__ == "__main__":
    unittest.main()
