from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from experiment_trust_fake_review_diffusion.diffusion_review_pipeline import (
    DiffusionReviewConfig,
    run_diffusion_review_experiment,
    score_review_texts,
)


class DiffusionReviewExperimentTests(unittest.TestCase):
    def test_train_and_score_smoke(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"diffusion_exp_{uuid.uuid4().hex}"
        dataset_path = tmpdir / "fake_reviews.csv"
        artifacts_dir = tmpdir / "artifacts"
        tmpdir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(12):
            rows.append({"label": "OR", "text_": f"authentic review {i} build quality reliable shipping fast"})
        for i in range(12):
            rows.append({"label": "CG", "text_": f"best ever buy now limited stock {i} unbelievable miracle"})
        pd.DataFrame(rows).to_csv(dataset_path, index=False)

        try:
            result = run_diffusion_review_experiment(
                DiffusionReviewConfig(
                    dataset_path=dataset_path,
                    artifacts_dir=artifacts_dir,
                    phase_a_target_rows=24,
                    test_size=0.25,
                    random_state=42,
                    max_features=2000,
                    latent_dim=16,
                    denoiser_samples_per_row=2,
                    classifier_samples_per_row=1,
                    inference_samples=4,
                )
            )

            self.assertIn("metrics", result)
            self.assertEqual(len(result["metrics"]), 2)
            self.assertTrue((artifacts_dir / "run_info.json").exists())
            self.assertTrue((artifacts_dir / "phase_a_metrics.csv").exists())
            self.assertTrue((artifacts_dir / "phase_a_test_predictions.csv").exists())
            self.assertTrue((artifacts_dir / "diffusion_model_bundle.joblib").exists())

            scored = score_review_texts(
                [
                    "excellent quality and accurate description",
                    "buy now unbelievable limited stock",
                ],
                artifacts_dir=artifacts_dir,
                inference_samples=4,
                random_state=7,
            )
            self.assertEqual(len(scored), 2)
            for row in scored:
                self.assertGreaterEqual(row["p_real"], 0.0)
                self.assertLessEqual(row["p_real"], 1.0)
                self.assertGreaterEqual(row["p_fake"], 0.0)
                self.assertLessEqual(row["p_fake"], 1.0)
                self.assertGreaterEqual(row["prediction_std"], 0.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
