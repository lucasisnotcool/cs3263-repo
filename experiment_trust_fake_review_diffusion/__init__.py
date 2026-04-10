"""Standalone diffusion fake-vs-real review experiment package."""

__all__ = [
    "DiffusionReviewConfig",
    "run_diffusion_review_experiment",
    "score_review_texts",
]


def __getattr__(name: str):
    if name in __all__:
        from .diffusion_review_pipeline import (
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
