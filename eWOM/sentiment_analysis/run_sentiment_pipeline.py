from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Allow `python eWOM/sentiment_analysis/run_sentiment_pipeline.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from eWOM.sentiment_analysis.dataset_loader import AmazonPolarityLoader
from eWOM.sentiment_analysis.feature_builder import (
    SentimentFeatureBuilder,
    SentimentFeatureConfig,
)
from eWOM.sentiment_analysis.preprocess import SentimentPreprocessor
from eWOM.sentiment_analysis.trainer import SentimentTrainer


LOGGER = logging.getLogger(__name__)

DATA_DIR = "data/raw/amazon_polarity"
MODEL_OUTPUT = "models/sentiment/amazon_polarity_baseline"
MAX_TRAIN_ROWS = None
MAX_TEST_ROWS = None
MAX_FEATURES = 50000
MIN_DF = 5
MAX_DF = 0.95
NGRAM_MAX = 2
RANDOM_STATE = 42
LOG_LEVEL = "INFO"


def configure_logging(level: str = LOG_LEVEL) -> None:
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)


def validate_config(
    *,
    max_features: int,
    min_df: int,
    max_df: float,
    ngram_max: int,
) -> None:
    if max_features <= 0:
        raise ValueError("MAX_FEATURES must be positive.")
    if min_df <= 0:
        raise ValueError("MIN_DF must be positive.")
    if not 0.0 < max_df <= 1.0:
        raise ValueError("MAX_DF must be within (0, 1].")
    if ngram_max < 1:
        raise ValueError("NGRAM_MAX must be at least 1.")


def summarize_labels(df: pd.DataFrame) -> dict[str, int]:
    return {
        str(label_text): int(count)
        for label_text, count in df["label_text"].value_counts().sort_index().items()
    }


def run_pipeline(
    *,
    data_dir: str = DATA_DIR,
    model_output: str = MODEL_OUTPUT,
    max_train_rows: int | None = MAX_TRAIN_ROWS,
    max_test_rows: int | None = MAX_TEST_ROWS,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_max: int = NGRAM_MAX,
    random_state: int = RANDOM_STATE,
) -> dict:
    configure_logging()
    validate_config(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_max=ngram_max,
    )

    LOGGER.info(
        "Starting sentiment pipeline with data_dir=%s, model_output=%s, max_train_rows=%s, max_test_rows=%s",
        data_dir,
        model_output,
        max_train_rows,
        max_test_rows,
    )
    loader = AmazonPolarityLoader(data_dir, random_state=random_state)
    LOGGER.info("Loading train and test splits")
    train_df, test_df = loader.load_train_test(
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )
    LOGGER.info("Loaded %s train rows and %s test rows", len(train_df), len(test_df))

    preprocessor = SentimentPreprocessor()
    LOGGER.info("Preprocessing text fields")
    train_df = preprocessor.transform(train_df)
    test_df = preprocessor.transform(test_df)

    LOGGER.info(
        "Building TF-IDF features with max_features=%s, min_df=%s, max_df=%s, ngram_range=%s",
        max_features,
        min_df,
        max_df,
        (1, ngram_max),
    )
    feature_builder = SentimentFeatureBuilder(
        SentimentFeatureConfig(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, ngram_max),
        )
    )
    trainer = SentimentTrainer(feature_builder, random_state=random_state)
    LOGGER.info("Training logistic regression baseline")
    trainer.fit(train_df["text"].tolist(), train_df["label"].tolist())

    LOGGER.info("Evaluating model on train and test splits")
    train_metrics = trainer.evaluate(train_df["text"].tolist(), train_df["label"].tolist())
    test_metrics = trainer.evaluate(test_df["text"].tolist(), test_df["label"].tolist())
    artifacts = trainer.save(model_output)
    LOGGER.info(
        "Saved model artifacts to %s and %s",
        artifacts.model_path,
        artifacts.feature_builder_path,
    )

    return {
        "data": {
            "data_dir": data_dir,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_label_distribution": summarize_labels(train_df),
            "test_label_distribution": summarize_labels(test_df),
        },
        "config": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
            "random_state": random_state,
            "max_train_rows": max_train_rows,
            "max_test_rows": max_test_rows,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "artifacts": artifacts.__dict__,
    }


def main() -> None:
    result = run_pipeline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
