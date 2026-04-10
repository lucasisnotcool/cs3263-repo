from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

import pandas as pd

from experiment_trust_fake_reviews_plus_detection.deploy_pipeline import (
    DeployConfig,
    run_deployment_pipeline,
)
from experiment_trust_fake_reviews_plus_detection.diffusion_detection_pipeline import (
    DiffusionReviewConfig,
    run_diffusion_review_experiment,
)


class DiffusionPlusDetectionExperimentTests(unittest.TestCase):
    def test_train_and_deploy_smoke(self) -> None:
        tmpdir = Path("data") / "test_artifacts" / f"diffusion_plus_{uuid.uuid4().hex}"
        dataset_path = tmpdir / "fake_reviews.csv"
        artifacts_dir = tmpdir / "artifacts"
        tmpdir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(16):
            rows.append(
                {
                    "label": "OR",
                    "text_": f"authentic review {i} arrived on time quality is good and description matches",
                }
            )
        for i in range(16):
            rows.append(
                {
                    "label": "CG",
                    "text_": f"best ever buy now miracle product limited stock {i} guaranteed result",
                }
            )
        pd.DataFrame(rows).to_csv(dataset_path, index=False)

        try:
            result = run_diffusion_review_experiment(
                DiffusionReviewConfig(
                    dataset_path=dataset_path,
                    artifacts_dir=artifacts_dir,
                    phase_a_target_rows=32,
                    test_size=0.25,
                    random_state=42,
                    max_features=3000,
                    latent_dim=24,
                    denoiser_samples_per_row=2,
                    classifier_samples_per_row=1,
                    inference_samples=4,
                    enforce_text_disjoint_split=True,
                )
            )

            self.assertIn("metrics", result)
            self.assertEqual(len(result["metrics"]), 2)
            self.assertTrue((artifacts_dir / "run_info.json").exists())
            self.assertTrue((artifacts_dir / "phase_a_metrics.csv").exists())
            self.assertTrue((artifacts_dir / "phase_a_test_predictions.csv").exists())
            self.assertTrue((artifacts_dir / "diffusion_model_bundle.joblib").exists())

            deploy = run_deployment_pipeline(
                [
                    {"record_id": "r1", "text": "arrived quickly and works exactly as expected"},
                    {"record_id": "r2", "text": "buy now limited stock miracle deal"},
                ],
                config=DeployConfig(
                    artifacts_dir=artifacts_dir,
                    inference_samples=4,
                    random_state=7,
                ),
                raise_on_environment_error=True,
            )
            self.assertTrue(deploy["environment"]["ok"])
            self.assertEqual(len(deploy["results"]), 2)

            for row in deploy["results"]:
                self.assertEqual(row["status"], "ok")
                self.assertGreaterEqual(row["scores"]["p_real"], 0.0)
                self.assertLessEqual(row["scores"]["p_real"], 1.0)
                self.assertGreaterEqual(row["scores"]["p_fake"], 0.0)
                self.assertLessEqual(row["scores"]["p_fake"], 1.0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
