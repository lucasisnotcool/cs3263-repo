"""Deployable trust pipeline for fake-review-calibrated product scoring."""

__all__ = [
    "DeployConfig",
    "EnvironmentValidationError",
    "TrustFakeReviewsDeployPipeline",
    "run_deployment_pipeline",
]


def __getattr__(name: str):
    if name in __all__:
        from .deploy_pipeline import (
            DeployConfig,
            EnvironmentValidationError,
            TrustFakeReviewsDeployPipeline,
            run_deployment_pipeline,
        )

        exports = {
            "DeployConfig": DeployConfig,
            "EnvironmentValidationError": EnvironmentValidationError,
            "TrustFakeReviewsDeployPipeline": TrustFakeReviewsDeployPipeline,
            "run_deployment_pipeline": run_deployment_pipeline,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
