from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

# Allow `python eWOM/sentiment_analysis/run_sentiment_benchmark.py` to resolve package imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from eWOM.sentiment_analysis.dataset_loader import AmazonPolarityLoader, LABEL_TEXT_BY_ID
from eWOM.sentiment_analysis.feature_builder import (
    SentimentFeatureBuilder,
    SentimentFeatureConfig,
)
from eWOM.sentiment_analysis.preprocess import SentimentPreprocessor


LOGGER = logging.getLogger(__name__)

DATA_DIR = "data/raw/amazon_polarity"
MODEL_OUTPUT = "models/sentiment/amazon_polarity"
MAX_TRAIN_ROWS = None
MAX_TEST_ROWS = None
MAX_FEATURES = 50000
MIN_DF = 5
MAX_DF = 0.95
NGRAM_MAX = 2
VAL_RATIO = 0.1
RANDOM_STATE = 42
LOG_LEVEL = "INFO"
CANDIDATE_MODEL_NAMES = (
    "logistic_regression",
    "multinomial_nb",
    "complement_nb",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark sentiment TF-IDF classifiers on Amazon Polarity using a train/val split and official test split.",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory containing the saved Amazon Polarity dataset.",
    )
    parser.add_argument(
        "--model-output",
        default=MODEL_OUTPUT,
        help=(
            "Output prefix for the selected model artifacts. Candidate artifacts are "
            "written with suffixed prefixes when multiple candidates are trained."
        ),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=MAX_TRAIN_ROWS,
        help="Optional cap on loaded train rows before the train/val split.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=MAX_TEST_ROWS,
        help="Optional cap on loaded test rows.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Validation split fraction taken from the training split.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=MIN_DF,
        help="Minimum document frequency for TF-IDF vocabulary terms.",
    )
    parser.add_argument(
        "--max-df",
        type=float,
        default=MAX_DF,
        help="Maximum document frequency for TF-IDF vocabulary terms.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=NGRAM_MAX,
        help="Maximum n-gram size for the TF-IDF vectorizer.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed used for the train/val split and model training.",
    )
    parser.add_argument(
        "--candidate-models",
        nargs="+",
        choices=CANDIDATE_MODEL_NAMES,
        default=list(CANDIDATE_MODEL_NAMES),
        help="Candidate model names to benchmark on the validation split.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        help="Logging level.",
    )
    parser.add_argument(
        "--reuse-existing-artifacts",
        action="store_true",
        help="Skip training and return the existing summary if the model artifacts already exist.",
    )
    parser.add_argument(
        "--output-format",
        choices=("text", "json"),
        default="text",
        help=(
            "Terminal output format. Summary JSON is always written to the "
            "<model-output>_summary.json artifact."
        ),
    )
    return parser


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
    val_ratio: float,
) -> None:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if min_df <= 0:
        raise ValueError("min_df must be positive.")
    if not 0.0 < max_df <= 1.0:
        raise ValueError("max_df must be within (0, 1].")
    if ngram_max < 1:
        raise ValueError("ngram_max must be at least 1.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_labels(df) -> dict[str, int]:
    return {
        str(label_text): int(count)
        for label_text, count in df["label_text"].value_counts().sort_index().items()
    }


def build_default_model_candidates(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=500,
            random_state=random_state,
        ),
        "multinomial_nb": MultinomialNB(),
        "complement_nb": ComplementNB(),
    }


def resolve_model_candidates(
    candidate_model_names: list[str] | tuple[str, ...] | None,
    *,
    random_state: int,
) -> dict[str, Any]:
    available = build_default_model_candidates(random_state=random_state)
    if not candidate_model_names:
        return available
    return {model_name: available[model_name] for model_name in candidate_model_names}


def build_feature_builder(
    *,
    max_features: int,
    min_df: int,
    max_df: float,
    ngram_max: int,
) -> SentimentFeatureBuilder:
    return SentimentFeatureBuilder(
        SentimentFeatureConfig(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, ngram_max),
        )
    )


def evaluate_model(model, x, y_true: np.ndarray) -> dict[str, Any]:
    predicted_labels = model.predict(x)
    predicted_probabilities = model.predict_proba(x)
    class_to_index = {
        int(label): index for index, label in enumerate(model.classes_)
    }
    positive_probs = predicted_probabilities[:, class_to_index[1]]

    roc_auc = None
    average_precision = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, positive_probs))
        average_precision = float(average_precision_score(y_true, positive_probs))

    metrics = {
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "macro_f1": float(f1_score(y_true, predicted_labels, average="macro")),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "classification_report": classification_report(
            y_true,
            predicted_labels,
            labels=[0, 1],
            target_names=[LABEL_TEXT_BY_ID[0], LABEL_TEXT_BY_ID[1]],
            output_dict=True,
            zero_division=0,
        ),
    }
    return _to_builtin(metrics)


def candidate_ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    def metric_or_neg_inf(value: float | None) -> float:
        if value is None:
            return float("-inf")
        return float(value)

    return (
        float(metrics["macro_f1"]),
        metric_or_neg_inf(metrics.get("average_precision")),
        metric_or_neg_inf(metrics.get("roc_auc")),
        float(metrics["accuracy"]),
    )


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(row[index])) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    header_line = "  ".join(
        str(value).ljust(widths[index])
        for index, value in enumerate(headers)
    )
    divider_line = "  ".join("-" * width for width in widths)
    row_lines = [
        "  ".join(
            str(value).ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rows
    ]
    return "\n".join([header_line, divider_line, *row_lines])


def _positive_class_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return metrics.get("classification_report", {}).get("positive", {})


def format_benchmark_report(result: dict[str, Any]) -> str:
    selected_model = result["model_selection"]["selected_model"]
    candidate_rows = []
    for model_name, summary in result["model_selection"]["candidate_models"].items():
        metrics = summary["val_metrics"]
        positive_metrics = _positive_class_metrics(metrics)
        candidate_rows.append(
            [
                "*" if model_name == selected_model else "",
                model_name,
                _format_metric(metrics.get("accuracy")),
                _format_metric(metrics.get("macro_f1")),
                _format_metric(positive_metrics.get("precision")),
                _format_metric(positive_metrics.get("recall")),
                _format_metric(positive_metrics.get("f1-score")),
                _format_metric(metrics.get("roc_auc")),
                _format_metric(metrics.get("average_precision")),
            ]
        )

    split_rows = []
    for split_name, metrics_key in [
        ("train", "train_metrics"),
        ("val", "val_metrics"),
        ("test", "test_metrics"),
    ]:
        metrics = result[metrics_key]
        positive_metrics = _positive_class_metrics(metrics)
        split_rows.append(
            [
                split_name,
                _format_metric(metrics.get("accuracy")),
                _format_metric(metrics.get("macro_f1")),
                _format_metric(positive_metrics.get("precision")),
                _format_metric(positive_metrics.get("recall")),
                _format_metric(positive_metrics.get("f1-score")),
                _format_metric(metrics.get("roc_auc")),
                _format_metric(metrics.get("average_precision")),
            ]
        )

    lines = [
        "Sentiment benchmark summary",
        f"Selected model: {selected_model}",
        f"Summary JSON: {result['artifacts']['summary_path']}",
        f"Selected model artifact: {result['artifacts']['model_path']}",
        "",
        "Validation candidate comparison:",
        _format_table(
            [
                "sel",
                "model",
                "accuracy",
                "macro_f1",
                "pos_precision",
                "pos_recall",
                "pos_f1",
                "roc_auc",
                "avg_precision",
            ],
            candidate_rows,
        ),
        "",
        "Selected model metrics:",
        _format_table(
            [
                "split",
                "accuracy",
                "macro_f1",
                "pos_precision",
                "pos_recall",
                "pos_f1",
                "roc_auc",
                "avg_precision",
            ],
            split_rows,
        ),
    ]
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _artifact_paths(model_output: Path) -> dict[str, Path]:
    return {
        "model_path": model_output.with_suffix(".joblib"),
        "feature_builder_path": model_output.with_name(f"{model_output.name}_feature_builder.joblib"),
        "metadata_path": model_output.with_name(f"{model_output.name}_metadata.json"),
        "summary_path": model_output.with_name(f"{model_output.name}_summary.json"),
    }


def _candidate_artifact_paths(model_output: Path, model_name: str) -> dict[str, Path]:
    candidate_output = model_output.with_name(f"{model_output.name}_{model_name}")
    return {
        "model_path": Path(f"{candidate_output}.joblib"),
        "feature_builder_path": Path(f"{candidate_output}_feature_builder.joblib"),
    }


def _load_existing_summary(summary_path: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _should_reuse_existing_artifacts(
    *,
    model_output: Path,
    candidate_model_names: list[str] | tuple[str, ...],
    reuse_existing_artifacts: bool,
) -> tuple[bool, dict[str, Path]]:
    artifact_paths = _artifact_paths(model_output)
    if not reuse_existing_artifacts:
        return False, artifact_paths

    expected_paths = list(artifact_paths.values())
    for model_name in candidate_model_names:
        expected_paths.extend(
            _candidate_artifact_paths(model_output, model_name).values()
        )

    if all(path.exists() for path in expected_paths):
        return True, artifact_paths

    missing_paths = [str(path) for path in expected_paths if not path.exists()]
    LOGGER.info(
        "Requested checkpoint reuse, but some artifacts are missing. Training will proceed. missing=%s",
        missing_paths,
    )
    return False, artifact_paths


def run_benchmark(
    *,
    data_dir: str | Path = DATA_DIR,
    model_output: str | Path = MODEL_OUTPUT,
    max_train_rows: int | None = MAX_TRAIN_ROWS,
    max_test_rows: int | None = MAX_TEST_ROWS,
    val_ratio: float = VAL_RATIO,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_max: int = NGRAM_MAX,
    random_state: int = RANDOM_STATE,
    candidate_model_names: list[str] | tuple[str, ...] | None = None,
    log_level: str = LOG_LEVEL,
    reuse_existing_artifacts: bool = False,
) -> dict[str, Any]:
    configure_logging(log_level)
    validate_config(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_max=ngram_max,
        val_ratio=val_ratio,
    )

    resolved_data_dir = Path(data_dir).expanduser().resolve()
    resolved_model_output = Path(model_output).expanduser().resolve()
    requested_candidate_model_names = tuple(
        candidate_model_names or CANDIDATE_MODEL_NAMES
    )
    should_reuse_artifacts, artifact_paths = _should_reuse_existing_artifacts(
        model_output=resolved_model_output,
        candidate_model_names=requested_candidate_model_names,
        reuse_existing_artifacts=reuse_existing_artifacts,
    )
    summary_path = artifact_paths["summary_path"]
    metadata_path = artifact_paths["metadata_path"]

    if should_reuse_artifacts:
        LOGGER.info(
            "Reusing existing sentiment checkpoint at model_path=%s feature_builder_path=%s",
            artifact_paths["model_path"],
            artifact_paths["feature_builder_path"],
        )
        return _load_existing_summary(summary_path)

    LOGGER.info(
        "Starting sentiment benchmark with data_dir=%s model_output=%s max_train_rows=%s max_test_rows=%s val_ratio=%s candidate_models=%s",
        resolved_data_dir,
        resolved_model_output,
        max_train_rows,
        max_test_rows,
        val_ratio,
        list(requested_candidate_model_names),
    )

    loader = AmazonPolarityLoader(resolved_data_dir, random_state=random_state)
    LOGGER.info("Loading train and test splits")
    train_full_df, test_df = loader.load_train_test(
        max_train_rows=max_train_rows,
        max_test_rows=max_test_rows,
    )
    LOGGER.info("Loaded %s train rows and %s test rows", len(train_full_df), len(test_df))

    preprocessor = SentimentPreprocessor()
    LOGGER.info("Preprocessing text fields")
    train_full_df = preprocessor.transform(train_full_df)
    test_df = preprocessor.transform(test_df)

    train_df, val_df = train_test_split(
        train_full_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_full_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    LOGGER.info(
        "Created train/val split with train_rows=%s val_rows=%s",
        len(train_df),
        len(val_df),
    )

    feature_builder = build_feature_builder(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_max=ngram_max,
    )
    LOGGER.info("Building TF-IDF features with config=%s", feature_builder.config)
    x_train = feature_builder.fit_transform(train_df["text"].tolist())
    x_val = feature_builder.transform(val_df["text"].tolist())
    x_test = feature_builder.transform(test_df["text"].tolist())
    y_train = np.asarray(train_df["label"].tolist(), dtype=int)
    y_val = np.asarray(val_df["label"].tolist(), dtype=int)
    y_test = np.asarray(test_df["label"].tolist(), dtype=int)

    candidate_models = resolve_model_candidates(
        requested_candidate_model_names,
        random_state=random_state,
    )
    candidate_summaries: dict[str, Any] = {}
    trained_candidate_models: dict[str, Any] = {}
    best_model = None
    best_model_name = None
    best_metrics = None
    best_ranking = None

    LOGGER.info("Training and benchmarking %s candidate models", len(candidate_models))
    for model_name, candidate in candidate_models.items():
        LOGGER.info("Training candidate model '%s' (%s)", model_name, candidate.__class__.__name__)
        model = clone(candidate)
        model.fit(x_train, y_train)

        val_metrics = evaluate_model(model, x_val, y_val)
        test_metrics = evaluate_model(model, x_test, y_test)
        ranking = candidate_ranking_key(val_metrics)
        trained_candidate_models[model_name] = model
        candidate_summaries[model_name] = {
            "model_class": model.__class__.__name__,
            "model_params": _to_builtin(model.get_params()),
            "feature_builder_class": feature_builder.__class__.__name__,
            "feature_group": "tfidf",
            "description": "TF-IDF",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

        if best_ranking is None or ranking > best_ranking:
            best_model = model
            best_model_name = model_name
            best_metrics = val_metrics
            best_ranking = ranking

    if best_model is None or best_model_name is None or best_metrics is None:
        raise RuntimeError("Failed to train any sentiment candidate models.")

    model_path = f"{resolved_model_output}.joblib"
    feature_builder_path = f"{resolved_model_output}_feature_builder.joblib"
    resolved_model_output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "label_text_by_id": LABEL_TEXT_BY_ID,
    }
    joblib.dump(bundle, model_path)
    joblib.dump(feature_builder, feature_builder_path)

    candidate_artifacts: dict[str, dict[str, str]] = {}
    for model_name, model in trained_candidate_models.items():
        candidate_artifact_paths = _candidate_artifact_paths(
            resolved_model_output,
            model_name,
        )
        candidate_bundle = {
            "model": model,
            "model_name": model_name,
            "label_text_by_id": LABEL_TEXT_BY_ID,
        }
        joblib.dump(candidate_bundle, candidate_artifact_paths["model_path"])
        joblib.dump(feature_builder, candidate_artifact_paths["feature_builder_path"])
        candidate_artifacts[model_name] = {
            "model_path": str(candidate_artifact_paths["model_path"]),
            "feature_builder_path": str(candidate_artifact_paths["feature_builder_path"]),
        }
        candidate_summaries[model_name]["artifacts"] = candidate_artifacts[model_name]

    metadata = {
        "model_path": model_path,
        "feature_builder_path": feature_builder_path,
        "candidate_artifacts": candidate_artifacts,
        "model_name": best_model_name,
        "model_class": best_model.__class__.__name__,
        "feature_builder_class": feature_builder.__class__.__name__,
        "feature_config": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
        },
    }
    _write_json(metadata_path, metadata)

    train_metrics = evaluate_model(best_model, x_train, y_train)
    val_metrics = evaluate_model(best_model, x_val, y_val)
    test_metrics = evaluate_model(best_model, x_test, y_test)

    result = {
        "workflow": {
            "architecture": "text classifier comparison",
            "selected_model": best_model_name,
            "selected_model_class": best_model.__class__.__name__,
            "selected_feature_builder_class": feature_builder.__class__.__name__,
            "model_selection": "Candidate classifiers compared on the val split using macro_f1.",
        },
        "data": {
            "data_dir": str(resolved_data_dir),
            "train_rows": int(len(train_full_df)),
            "train_subset_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_label_distribution": summarize_labels(train_full_df),
            "train_subset_label_distribution": summarize_labels(train_df),
            "val_label_distribution": summarize_labels(val_df),
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
            "val_ratio": val_ratio,
            "candidate_models": list(candidate_models.keys()),
            "model_output": str(resolved_model_output),
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
        },
        "model_selection": {
            "selection_split": "val",
            "selection_metric": "macro_f1",
            "selected_model": best_model_name,
            "selected_model_class": best_model.__class__.__name__,
            "selected_feature_builder_class": feature_builder.__class__.__name__,
            "selected_feature_group": "tfidf",
            "selected_val_metrics": best_metrics,
            "candidate_models": candidate_summaries,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model_path": model_path,
            "feature_builder_path": feature_builder_path,
            "candidate_artifacts": candidate_artifacts,
            "metadata_path": str(metadata_path),
            "summary_path": str(summary_path),
        },
    }
    _write_json(summary_path, result)
    LOGGER.info(
        "Sentiment benchmark complete with selected_model=%s val_macro_f1=%.6f test_macro_f1=%.6f",
        best_model_name,
        float(val_metrics["macro_f1"]),
        float(test_metrics["macro_f1"]),
    )
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = run_benchmark(
        data_dir=args.data_dir,
        model_output=args.model_output,
        max_train_rows=args.max_train_rows,
        max_test_rows=args.max_test_rows,
        val_ratio=args.val_ratio,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        random_state=args.random_state,
        candidate_model_names=args.candidate_models,
        log_level=args.log_level,
        reuse_existing_artifacts=args.reuse_existing_artifacts,
    )
    if args.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(format_benchmark_report(result))


if __name__ == "__main__":
    main()
