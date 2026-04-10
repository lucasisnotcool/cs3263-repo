from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiment_trust_fake_reviews_plus_detection.bn_diffusion_fork import (
    DiffusionForkConfig,
    run_bn_diffusion_fork_evaluation,
)
from experiment_trust_fake_reviews_plus_detection.llm_trust_graph_pipeline import PhaseConfig


class BNDiffusionForkTests(unittest.TestCase):
    def test_bn_diffusion_fork_smoke(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        for i in range(30):
            rows.append(
                {
                    "record_id": f"r_real_{i}",
                    "text": f"arrived on time quality good reliable build {i}",
                    "label_truth": 1,
                    "claim_trust_score": float(rng.uniform(0.6, 1.0)),
                    "signal_trust_score": float(rng.uniform(0.55, 1.0)),
                    "heuristic_pressure_score": float(rng.uniform(0.0, 0.4)),
                    "competence_score": float(rng.uniform(0.55, 1.0)),
                    "benevolence_score": float(rng.uniform(0.55, 1.0)),
                    "integrity_score": float(rng.uniform(0.55, 1.0)),
                    "predictability_score": float(rng.uniform(0.55, 1.0)),
                }
            )
        for i in range(30):
            rows.append(
                {
                    "record_id": f"r_fake_{i}",
                    "text": f"buy now miracle best ever limited stock urgent {i}",
                    "label_truth": 0,
                    "claim_trust_score": float(rng.uniform(0.0, 0.45)),
                    "signal_trust_score": float(rng.uniform(0.0, 0.45)),
                    "heuristic_pressure_score": float(rng.uniform(0.55, 1.0)),
                    "competence_score": float(rng.uniform(0.0, 0.5)),
                    "benevolence_score": float(rng.uniform(0.0, 0.5)),
                    "integrity_score": float(rng.uniform(0.0, 0.5)),
                    "predictability_score": float(rng.uniform(0.0, 0.5)),
                }
            )

        labeled = pd.DataFrame(rows)
        out = run_bn_diffusion_fork_evaluation(
            labeled,
            phase_config=PhaseConfig(target_rows=60, test_size=0.25, random_state=42),
            text_col="text",
            config=DiffusionForkConfig(
                random_state=42,
                diffusion_steps=20,
                latent_dim=16,
                max_features=3000,
                denoiser_samples_per_row=2,
                classifier_samples_per_row=1,
                inference_samples=4,
            ),
        )

        metrics = out["metrics"]
        self.assertFalse(metrics.empty)
        self.assertTrue({"bn_baseline", "bn_with_diffusion_factor"}.issubset(set(metrics["model"])))
        self.assertIn("delta_auc_bn", out["comparison_summary"])
        self.assertGreater(out["train_rows"], 0)
        self.assertGreater(out["test_rows"], 0)


if __name__ == "__main__":
    unittest.main()
