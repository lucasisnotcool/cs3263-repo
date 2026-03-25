from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow `python eWOM/helpfulness/pipeline.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from eWOM.helpfulness.dataset_loader import LABEL_TEXT_BY_ID, PreparedHelpfulnessSplitLoader
from eWOM.helpfulness.feature_builder import (
    HelpfulnessFeatureBuilder,
    HelpfulnessFeatureConfig,
)
from eWOM.helpfulness.preprocess import HelpfulnessPreprocessor
from eWOM.helpfulness.trainer import HelpfulnessTrainer


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "helpfulness" / "train.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "helpfulness" / "test.jsonl"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "helpfulness" / "amazon_helpfulness_electronics_tfidf_lr"
MAX_TRAIN_ROWS = 300000
MAX_TEST_ROWS = 100000
MAX_FEATURES = 50000
MIN_DF = 5
MAX_DF = 0.95
NGRAM_MAX = 2
DEV_RATIO = 0.1
RANDOM_STATE = 42
LOG_LEVEL = "INFO"


def configure_logging(level: str = LOG_LEVEL) -> None:
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
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
    dev_ratio: float,
) -> None:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if min_df <= 0:
        raise ValueError("min_df must be positive.")
    if not 0.0 < max_df <= 1.0:
        raise ValueError("max_df must be within (0, 1].")
    if ngram_max < 1:
        raise ValueError("ngram_max must be at least 1.")
    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be between 0 and 1.")


def summarize_labels(df) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {
        LABEL_TEXT_BY_ID.get(int(label), str(label)): int(count)
        for label, count in counts.items()
    }


def summarize_loaded_split(source_path: Path, df) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "rows": int(len(df)),
        "label_counts": summarize_labels(df),
        "positive_rate": float(df["label"].mean()),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_pipeline(
    *,
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
    model_output: str | Path = MODEL_OUTPUT,
    max_train_rows: int | None = MAX_TRAIN_ROWS,
    max_test_rows: int | None = MAX_TEST_ROWS,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_max: int = NGRAM_MAX,
    dev_ratio: float = DEV_RATIO,
    random_state: int = RANDOM_STATE,
    log_level: str = LOG_LEVEL,
) -> dict[str, Any]:
    configure_logging(log_level)
    validate_config(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_max=ngram_max,
        dev_ratio=dev_ratio,
    )

    resolved_train_path = Path(train_path).expanduser().resolve()
    resolved_test_path = Path(test_path).expanduser().resolve()
    resolved_model_output = Path(model_output).expanduser().resolve()
    summary_path = resolved_model_output.with_name(f"{resolved_model_output.name}_summary.json")
    metadata_path = resolved_model_output.with_name(f"{resolved_model_output.name}_metadata.json")

    LOGGER.info(
        "Starting helpfulness pipeline with train_path=%s test_path=%s model_output=%s max_train_rows=%s max_test_rows=%s",
        resolved_train_path,
        resolved_test_path,
        resolved_model_output,
        max_train_rows,
        max_test_rows,
    )

    LOGGER.info("Loading prepared helpfulness train/test splits")
    train_loader = PreparedHelpfulnessSplitLoader(resolved_train_path, max_rows=max_train_rows)
    test_loader = PreparedHelpfulnessSplitLoader(resolved_test_path, max_rows=max_test_rows)
    train_raw_df = train_loader.load()
    test_raw_df = test_loader.load()
    LOGGER.info(
        "Loaded prepared splits with train_rows=%s test_rows=%s train_label_distribution=%s test_label_distribution=%s",
        len(train_raw_df),
        len(test_raw_df),
        summarize_labels(train_raw_df),
        summarize_labels(test_raw_df),
    )

    LOGGER.info("Preprocessing helpfulness train/test splits")
    preprocessor = HelpfulnessPreprocessor()
    train_df = preprocessor.transform(train_raw_df)
    test_df = preprocessor.transform(test_raw_df)
    LOGGER.info(
        "Finished preprocessing with train_columns=%s test_columns=%s",
        list(train_df.columns),
        list(test_df.columns),
    )

    feature_config = HelpfulnessFeatureConfig(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, ngram_max),
    )
    LOGGER.info("Using helpfulness feature configuration: %s", feature_config)
    feature_builder = HelpfulnessFeatureBuilder(feature_config)
    trainer = HelpfulnessTrainer(
        feature_builder,
        random_state=random_state,
        log_level=log_level,
    )

    train_subset_df, dev_subset_df = trainer.make_train_dev_split(
        train_df,
        dev_ratio=dev_ratio,
    )
    LOGGER.info("Starting helpfulness model training")
    trainer.fit(train_subset_df, dev_subset_df)

    LOGGER.info("Evaluating helpfulness model on train, dev, and test splits")
    train_metrics = trainer.evaluate(train_subset_df)
    dev_default_metrics = trainer.evaluate(
        dev_subset_df,
        threshold=trainer.DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    dev_selected_metrics = trainer.evaluate(dev_subset_df)
    test_default_metrics = trainer.evaluate(
        test_df,
        threshold=trainer.DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    test_selected_metrics = trainer.evaluate(test_df)

    LOGGER.info(
        "Saving helpfulness artifacts to model_output=%s summary_path=%s metadata_path=%s",
        resolved_model_output,
        summary_path,
        metadata_path,
    )
    artifacts = trainer.save(str(resolved_model_output))

    metadata = {
        "model_path": artifacts.model_path,
        "feature_builder_path": artifacts.feature_builder_path,
        "threshold": float(trainer.threshold),
        "label_text_by_id": LABEL_TEXT_BY_ID,
        "feature_config": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
        },
        "numeric_feature_names": list(feature_builder.NUMERIC_FEATURE_NAMES),
    }
    _write_json(metadata_path, metadata)

    result = {
        "workflow": {
            "architecture": "TF-IDF + metadata + LogisticRegression",
            "threshold_selection": "Threshold chosen on dev by maximizing macro_f1.",
            "primary_ranking_metric": "average_precision",
            "prediction_output": ["usefulness_probability", "is_useful"],
        },
        "config": {
            "train_path": str(resolved_train_path),
            "test_path": str(resolved_test_path),
            "model_output": str(resolved_model_output),
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
            "max_train_rows": max_train_rows,
            "max_test_rows": max_test_rows,
            "dev_ratio": dev_ratio,
            "random_state": random_state,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
            "log_level": str(log_level).upper(),
        },
        "data_summary": {
            "train_load": summarize_loaded_split(resolved_train_path, train_raw_df),
            "train_dev_split": {
                "train_rows": int(len(train_subset_df)),
                "dev_rows": int(len(dev_subset_df)),
                "train_label_counts": summarize_labels(train_subset_df),
                "dev_label_counts": summarize_labels(dev_subset_df),
                "train_positive_rate": float(train_subset_df["label"].mean()),
                "dev_positive_rate": float(dev_subset_df["label"].mean()),
            },
            "test_load": summarize_loaded_split(resolved_test_path, test_raw_df),
        },
        "threshold_selection": trainer.threshold_selection_summary,
        "train_metrics": train_metrics,
        "dev_metrics": {
            "default_threshold_metrics": dev_default_metrics,
            "selected_threshold_metrics": dev_selected_metrics,
        },
        "test_metrics": {
            "default_threshold_metrics": test_default_metrics,
            "selected_threshold_metrics": test_selected_metrics,
        },
        "artifacts": {
            "model_path": artifacts.model_path,
            "feature_builder_path": artifacts.feature_builder_path,
            "metadata_path": str(metadata_path),
            "summary_path": str(summary_path),
        },
    }
    _write_json(summary_path, result)
    LOGGER.info(
        "Helpfulness pipeline complete with threshold=%.6f test_macro_f1=%.6f",
        trainer.threshold,
        test_selected_metrics["macro_f1"],
    )
    return result


def main() -> None:
    result = run_pipeline()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
