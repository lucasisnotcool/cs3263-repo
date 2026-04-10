"""Diffusion-based fake-review detection experiment and deploy package."""

__all__ = [
    "DeployConfig",
    "EnvironmentValidationError",
    "TrustFakeReviewsPlusDetectionDeployPipeline",
    "run_deployment_pipeline",
    "DiffusionReviewConfig",
    "run_diffusion_review_experiment",
    "score_review_texts",
    "DiffusionForkConfig",
    "run_bn_diffusion_fork_evaluation",
]


def __getattr__(name: str):
    if name in {
        "DeployConfig",
        "EnvironmentValidationError",
        "TrustFakeReviewsPlusDetectionDeployPipeline",
        "run_deployment_pipeline",
    }:
        from .deploy_pipeline import (
            DeployConfig,
            EnvironmentValidationError,
            TrustFakeReviewsPlusDetectionDeployPipeline,
            run_deployment_pipeline,
        )

        exports = {
            "DeployConfig": DeployConfig,
            "EnvironmentValidationError": EnvironmentValidationError,
            "TrustFakeReviewsPlusDetectionDeployPipeline": TrustFakeReviewsPlusDetectionDeployPipeline,
            "run_deployment_pipeline": run_deployment_pipeline,
        }
        return exports[name]

    if name in {"DiffusionReviewConfig", "run_diffusion_review_experiment", "score_review_texts"}:
        from .diffusion_detection_pipeline import (
            DiffusionReviewConfig,
            run_diffusion_review_experiment,
            score_review_texts,
        )

        exports = {
            "DiffusionReviewConfig": DiffusionReviewConfig,
            "run_diffusion_review_experiment": run_diffusion_review_experiment,
            "score_review_texts": score_review_texts,
        }
        return exports[name]

    if name in {"DiffusionForkConfig", "run_bn_diffusion_fork_evaluation"}:
        from .bn_diffusion_fork import (
            DiffusionForkConfig,
            run_bn_diffusion_fork_evaluation,
        )

        exports = {
            "DiffusionForkConfig": DiffusionForkConfig,
            "run_bn_diffusion_fork_evaluation": run_bn_diffusion_fork_evaluation,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
