from .feature_builder import HelpfulnessFeatureBuilder, HelpfulnessFeatureConfig
from .predictor import HelpfulnessPredictor
from .preprocess import HelpfulnessPreprocessor
from .trainer import HelpfulnessArtifacts, HelpfulnessTrainer

__all__ = [
    "HelpfulnessArtifacts",
    "HelpfulnessFeatureBuilder",
    "HelpfulnessFeatureConfig",
    "HelpfulnessPredictor",
    "HelpfulnessPreprocessor",
    "HelpfulnessTrainer",
    "configure_logging",
    "run_pipeline",
]


def __getattr__(name: str):
    if name == "configure_logging":
        from .pipeline import configure_logging

        return configure_logging
    if name == "run_pipeline":
        from .pipeline import run_pipeline

        return run_pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
