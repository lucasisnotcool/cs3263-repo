from __future__ import annotations

import argparse
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
from eWOM.helpfulness.trainer import HelpfulnessTrainer, build_default_model_candidates
from eWOM.helpfulness.train_test_splitter import DEFAULT_POSITIVE_THRESHOLD


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "helpfulness" / "train.jsonl"
VAL_PATH = PROJECT_ROOT / "data" / "helpfulness" / "val.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "helpfulness" / "test.jsonl"
MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "helpfulness"
    / "amazon_helpfulness"
)
MAX_TRAIN_ROWS = None
MAX_VAL_ROWS = None
MAX_TEST_ROWS = None
MAX_FEATURES = 50000
MIN_DF = 5
MAX_DF = 0.95
NGRAM_MAX = 2
RANDOM_STATE = 42
LOG_LEVEL = "INFO"
DEFAULT_TEXT_DERIVED_LENGTHS_ONLY = True
CANDIDATE_MODEL_NAMES = (
    "logistic_regression",
    "multinomial_nb",
    "complement_nb",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the helpfulness model from explicit train/val/test JSONL splits.",
    )
    parser.add_argument(
        "--train-path",
        default=str(TRAIN_PATH),
        help="Path to the prepared helpfulness training JSONL split.",
    )
    parser.add_argument(
        "--val-path",
        default=str(VAL_PATH),
        help="Path to the prepared helpfulness validation JSONL split.",
    )
    parser.add_argument(
        "--test-path",
        default=str(TEST_PATH),
        help="Path to the prepared helpfulness test JSONL split.",
    )
    parser.add_argument(
        "--model-output",
        default=str(MODEL_OUTPUT),
        help=(
            "Output prefix for the selected model artifacts. Candidate artifacts are "
            "written with suffixed prefixes when multiple candidates are trained."
        ),
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=MAX_TRAIN_ROWS,
        help="Optional cap on loaded training rows.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=MAX_VAL_ROWS,
        help="Optional cap on loaded validation rows.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=MAX_TEST_ROWS,
        help="Optional cap on loaded test rows.",
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
        help="Random seed used for model training.",
    )
    parser.add_argument(
        "--log-level",
        default=LOG_LEVEL,
        help="Logging level.",
    )
    parser.add_argument(
        "--candidate-models",
        nargs="+",
        choices=CANDIDATE_MODEL_NAMES,
        default=list(CANDIDATE_MODEL_NAMES),
        help="Candidate model names to train and compare on the validation split.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=int,
        default=DEFAULT_POSITIVE_THRESHOLD,
        help=(
            "When prepared rows include helpful_votes/helpful_vote, minimum vote count "
            "used to relabel a review as helpful."
        ),
    )
    parser.add_argument(
        "--text-derived-lengths-only",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_TEXT_DERIVED_LENGTHS_ONLY,
        help=(
            "Use TF-IDF text features plus review/title/text length features only, "
            "excluding rating and verified_purchase. Enabled by default; use "
            "--no-text-derived-lengths-only only for legacy metadata-inclusive experiments."
        ),
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
) -> None:
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if min_df <= 0:
        raise ValueError("min_df must be positive.")
    if not 0.0 < max_df <= 1.0:
        raise ValueError("max_df must be within (0, 1].")
    if ngram_max < 1:
        raise ValueError("ngram_max must be at least 1.")


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


def format_pipeline_report(result: dict[str, Any]) -> str:
    selected_model = result["model_selection"]["selected_model"]
    candidate_rows = []
    for model_name, summary in result["model_selection"]["candidate_models"].items():
        selection_summary = summary["selection_summary"]
        metrics = selection_summary["selected_threshold_metrics"]
        candidate_rows.append(
            [
                "*" if model_name == selected_model else "",
                model_name,
                _format_metric(selection_summary.get("best_threshold")),
                _format_metric(metrics.get("accuracy")),
                _format_metric(metrics.get("balanced_accuracy")),
                _format_metric(metrics.get("macro_f1")),
                _format_metric(metrics.get("precision_positive")),
                _format_metric(metrics.get("recall_positive")),
                _format_metric(metrics.get("f1_positive")),
                _format_metric(metrics.get("average_precision")),
            ]
        )

    split_rows = []
    for split_name, metrics in [
        ("train", result["train_metrics"]),
        ("val", result["val_metrics"]["selected_threshold_metrics"]),
        ("test", result["test_metrics"]["selected_threshold_metrics"]),
    ]:
        split_rows.append(
            [
                split_name,
                _format_metric(metrics.get("threshold")),
                _format_metric(metrics.get("accuracy")),
                _format_metric(metrics.get("balanced_accuracy")),
                _format_metric(metrics.get("macro_f1")),
                _format_metric(metrics.get("precision_positive")),
                _format_metric(metrics.get("recall_positive")),
                _format_metric(metrics.get("f1_positive")),
                _format_metric(metrics.get("roc_auc")),
                _format_metric(metrics.get("average_precision")),
            ]
        )

    lines = [
        "Helpfulness training summary",
        f"Selected model: {selected_model}",
        f"Selected threshold: {_format_metric(result['threshold_selection']['best_threshold'])}",
        f"Summary JSON: {result['artifacts']['summary_path']}",
        f"Selected model artifact: {result['artifacts']['model_path']}",
        "",
        "Validation candidate comparison:",
        _format_table(
            [
                "sel",
                "model",
                "threshold",
                "accuracy",
                "bal_acc",
                "macro_f1",
                "helpful_precision",
                "helpful_recall",
                "helpful_f1",
                "avg_precision",
            ],
            candidate_rows,
        ),
        "",
        "Selected model metrics:",
        _format_table(
            [
                "split",
                "threshold",
                "accuracy",
                "bal_acc",
                "macro_f1",
                "helpful_precision",
                "helpful_recall",
                "helpful_f1",
                "roc_auc",
                "avg_precision",
            ],
            split_rows,
        ),
    ]
    return "\n".join(lines)


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


def resolve_model_candidates(
    candidate_model_names: list[str] | tuple[str, ...] | None,
    *,
    random_state: int,
) -> dict[str, Any]:
    available = build_default_model_candidates(random_state=random_state)
    if not candidate_model_names:
        return available

    selected = {
        model_name: available[model_name]
        for model_name in candidate_model_names
    }
    if not selected:
        raise ValueError("At least one candidate model name is required.")
    return selected


def run_pipeline(
    *,
    train_path: str | Path = TRAIN_PATH,
    val_path: str | Path = VAL_PATH,
    test_path: str | Path = TEST_PATH,
    model_output: str | Path = MODEL_OUTPUT,
    max_train_rows: int | None = MAX_TRAIN_ROWS,
    max_val_rows: int | None = MAX_VAL_ROWS,
    max_test_rows: int | None = MAX_TEST_ROWS,
    max_features: int = MAX_FEATURES,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    ngram_max: int = NGRAM_MAX,
    random_state: int = RANDOM_STATE,
    log_level: str = LOG_LEVEL,
    candidate_model_names: list[str] | tuple[str, ...] | None = None,
    positive_threshold: int = DEFAULT_POSITIVE_THRESHOLD,
    text_derived_lengths_only: bool = DEFAULT_TEXT_DERIVED_LENGTHS_ONLY,
    reuse_existing_artifacts: bool = False,
) -> dict[str, Any]:
    configure_logging(log_level)
    validate_config(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_max=ngram_max,
    )

    resolved_train_path = Path(train_path).expanduser().resolve()
    resolved_val_path = Path(val_path).expanduser().resolve()
    resolved_test_path = Path(test_path).expanduser().resolve()
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
            "Reusing existing helpfulness checkpoint at model_path=%s feature_builder_path=%s",
            artifact_paths["model_path"],
            artifact_paths["feature_builder_path"],
        )
        return _load_existing_summary(summary_path)

    LOGGER.info(
        "Starting helpfulness pipeline with train_path=%s val_path=%s test_path=%s model_output=%s max_train_rows=%s max_val_rows=%s max_test_rows=%s positive_threshold=%s text_derived_lengths_only=%s",
        resolved_train_path,
        resolved_val_path,
        resolved_test_path,
        resolved_model_output,
        max_train_rows,
        max_val_rows,
        max_test_rows,
        positive_threshold,
        text_derived_lengths_only,
    )

    LOGGER.info("Loading prepared helpfulness train/val/test splits")
    train_loader = PreparedHelpfulnessSplitLoader(
        resolved_train_path,
        max_rows=max_train_rows,
        positive_threshold=positive_threshold,
    )
    val_loader = PreparedHelpfulnessSplitLoader(
        resolved_val_path,
        max_rows=max_val_rows,
        positive_threshold=positive_threshold,
    )
    test_loader = PreparedHelpfulnessSplitLoader(
        resolved_test_path,
        max_rows=max_test_rows,
        positive_threshold=positive_threshold,
    )
    train_raw_df = train_loader.load()
    val_raw_df = val_loader.load()
    test_raw_df = test_loader.load()
    LOGGER.info(
        "Loaded prepared splits with train_rows=%s val_rows=%s test_rows=%s train_label_distribution=%s val_label_distribution=%s test_label_distribution=%s",
        len(train_raw_df),
        len(val_raw_df),
        len(test_raw_df),
        summarize_labels(train_raw_df),
        summarize_labels(val_raw_df),
        summarize_labels(test_raw_df),
    )

    LOGGER.info("Preprocessing helpfulness train/val/test splits")
    preprocessor = HelpfulnessPreprocessor()
    train_df = preprocessor.transform(train_raw_df)
    val_df = preprocessor.transform(val_raw_df)
    test_df = preprocessor.transform(test_raw_df)
    LOGGER.info(
        "Finished preprocessing with train_columns=%s val_columns=%s test_columns=%s",
        list(train_df.columns),
        list(val_df.columns),
        list(test_df.columns),
    )

    feature_config = HelpfulnessFeatureConfig(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, ngram_max),
        use_rating=not text_derived_lengths_only,
        use_verified_purchase=not text_derived_lengths_only,
        use_text_length_features=True,
    )
    LOGGER.info("Using helpfulness feature configuration: %s", feature_config)
    feature_builder = HelpfulnessFeatureBuilder(feature_config)
    trainer = HelpfulnessTrainer(
        feature_builder,
        random_state=random_state,
        log_level=log_level,
        model_candidates=resolve_model_candidates(
            requested_candidate_model_names,
            random_state=random_state,
        ),
    )

    LOGGER.info("Starting helpfulness model training")
    trainer.fit(train_df, val_df)
    LOGGER.info("Selected helpfulness model '%s'", trainer.model_name)

    LOGGER.info("Evaluating helpfulness model on train, val, and test splits")
    train_metrics = trainer.evaluate(train_df)
    val_default_metrics = trainer.evaluate(
        val_df,
        threshold=trainer.DEFAULT_CLASSIFICATION_THRESHOLD,
    )
    val_selected_metrics = trainer.evaluate(val_df)
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
    if trainer.model_selection_summary is not None:
        for model_name, candidate_artifact in artifacts.candidate_artifacts.items():
            candidate_summary = trainer.model_selection_summary["candidate_models"].get(
                model_name
            )
            if candidate_summary is not None:
                candidate_summary["artifacts"] = candidate_artifact

    metadata = {
        "model_path": artifacts.model_path,
        "feature_builder_path": artifacts.feature_builder_path,
        "candidate_artifacts": artifacts.candidate_artifacts,
        "model_name": trainer.model_name,
        "model_class": trainer.model.__class__.__name__ if trainer.model is not None else None,
        "threshold": float(trainer.threshold),
        "label_text_by_id": LABEL_TEXT_BY_ID,
        "positive_threshold": positive_threshold,
        "feature_config": {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
            "use_rating": feature_config.use_rating,
            "use_verified_purchase": feature_config.use_verified_purchase,
            "use_text_length_features": feature_config.use_text_length_features,
        },
        "numeric_feature_names": list(feature_builder.active_numeric_feature_names),
    }
    _write_json(metadata_path, metadata)

    workflow_architecture = (
        "TF-IDF + text-derived length features + classifier comparison"
        if text_derived_lengths_only
        else "TF-IDF + metadata + classifier comparison"
    )
    result = {
        "workflow": {
            "architecture": workflow_architecture,
            "selected_model": trainer.model_name,
            "selected_model_class": (
                trainer.model.__class__.__name__ if trainer.model is not None else None
            ),
            "threshold_selection": "Threshold chosen on val by maximizing macro_f1.",
            "model_selection": "Candidate classifiers compared on the val split using selected-threshold macro_f1.",
            "primary_ranking_metric": "macro_f1",
            "ranking_tie_breakers": [
                "average_precision",
                "roc_auc",
                "balanced_accuracy",
            ],
            "prediction_output": ["usefulness_probability", "is_useful"],
        },
        "config": {
            "train_path": str(resolved_train_path),
            "val_path": str(resolved_val_path),
            "test_path": str(resolved_test_path),
            "model_output": str(resolved_model_output),
            "summary_path": str(summary_path),
            "metadata_path": str(metadata_path),
            "max_train_rows": max_train_rows,
            "max_val_rows": max_val_rows,
            "max_test_rows": max_test_rows,
            "random_state": random_state,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": [1, ngram_max],
            "log_level": str(log_level).upper(),
            "candidate_models": list(trainer.model_candidates.keys()),
            "positive_threshold": positive_threshold,
            "text_derived_lengths_only": text_derived_lengths_only,
        },
        "data_summary": {
            "train_load": summarize_loaded_split(resolved_train_path, train_raw_df),
            "val_load": summarize_loaded_split(resolved_val_path, val_raw_df),
            "test_load": summarize_loaded_split(resolved_test_path, test_raw_df),
        },
        "model_selection": trainer.model_selection_summary,
        "threshold_selection": trainer.threshold_selection_summary,
        "train_metrics": train_metrics,
        "val_metrics": {
            "default_threshold_metrics": val_default_metrics,
            "selected_threshold_metrics": val_selected_metrics,
        },
        "test_metrics": {
            "default_threshold_metrics": test_default_metrics,
            "selected_threshold_metrics": test_selected_metrics,
        },
        "artifacts": {
            "model_path": artifacts.model_path,
            "feature_builder_path": artifacts.feature_builder_path,
            "candidate_artifacts": artifacts.candidate_artifacts,
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
    parser = build_parser()
    args = parser.parse_args()

    result = run_pipeline(
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        model_output=args.model_output,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        random_state=args.random_state,
        log_level=args.log_level,
        candidate_model_names=args.candidate_models,
        positive_threshold=args.positive_threshold,
        text_derived_lengths_only=args.text_derived_lengths_only,
        reuse_existing_artifacts=args.reuse_existing_artifacts,
    )
    if args.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(format_pipeline_report(result))


if __name__ == "__main__":
    main()
