from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from .dataset_loader import LABEL_TEXT_BY_ID


LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_SELECTION_METRIC = "macro_f1"


@dataclass(frozen=True)
class HelpfulnessArtifacts:
    model_path: str
    feature_builder_path: str


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


def _macro_f1_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    positive_denominator = (2 * tp) + fp + fn
    negative_denominator = (2 * tn) + fp + fn

    positive_f1 = 0.0 if positive_denominator == 0 else (2 * tp) / positive_denominator
    negative_f1 = 0.0 if negative_denominator == 0 else (2 * tn) / negative_denominator
    return (positive_f1 + negative_f1) / 2.0


def build_default_model_candidates(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            solver="saga",
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
        ),
        "multinomial_nb": MultinomialNB(),
        "complement_nb": ComplementNB(),
    }


def _metric_or_neg_inf(value: float | None) -> float:
    if value is None:
        return float("-inf")
    return float(value)


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _candidate_ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics[DEFAULT_MODEL_SELECTION_METRIC]),
        _metric_or_neg_inf(metrics.get("average_precision")),
        _metric_or_neg_inf(metrics.get("roc_auc")),
        float(metrics["balanced_accuracy"]),
    )


class HelpfulnessTrainer:
    DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

    def __init__(
        self,
        feature_builder,
        random_state: int = 42,
        log_level: str = "INFO",
        model_candidates: dict[str, Any] | None = None,
    ):
        self.feature_builder = feature_builder
        self.random_state = random_state
        self.log_level = log_level
        self.model_candidates = model_candidates or build_default_model_candidates(
            random_state=random_state
        )
        self.model = None
        self.model_name: str | None = None
        self.threshold = self.DEFAULT_CLASSIFICATION_THRESHOLD
        self.threshold_selection_summary: dict[str, Any] | None = None
        self.model_selection_summary: dict[str, Any] | None = None

    def make_train_val_split(
        self,
        df: pd.DataFrame,
        *,
        val_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("val_ratio must be between 0 and 1.")

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=df["label"],
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        LOGGER.info(
            "Created stratified train/val split with train_rows=%s, val_rows=%s, train_positive_rate=%.6f, val_positive_rate=%.6f",
            len(train_df),
            len(val_df),
            train_df["label"].mean(),
            val_df["label"].mean(),
        )
        return train_df, val_df

    def make_train_dev_split(
        self,
        df: pd.DataFrame,
        *,
        dev_ratio: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.make_train_val_split(df, val_ratio=dev_ratio)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        if not self.model_candidates:
            raise ValueError("At least one model candidate is required.")

        LOGGER.debug(
            "Trainer hyperparameters: model_candidates=%s feature_builder=%s numeric_features=%s",
            {
                name: _to_builtin(candidate.get_params())
                for name, candidate in self.model_candidates.items()
            },
            asdict(self.feature_builder.config),
            list(self.feature_builder.NUMERIC_FEATURE_NAMES),
        )
        LOGGER.info("Fitting helpfulness features on %s training rows", len(train_df))
        x_train = self.feature_builder.fit_transform(train_df)
        LOGGER.info("Built training feature matrix with shape=%s", x_train.shape)

        LOGGER.info("Building validation feature matrix with %s rows", len(val_df))
        x_val = self.feature_builder.transform(val_df)
        LOGGER.info("Built validation feature matrix with shape=%s", x_val.shape)

        y_train = train_df["label"].tolist()
        y_val = np.asarray(val_df["label"].tolist(), dtype=int)
        best_model = None
        best_model_name = None
        best_selection_summary = None
        best_ranking_key = None
        candidate_models: dict[str, Any] = {}

        LOGGER.info(
            "Training and comparing %s candidate models on the validation split",
            len(self.model_candidates),
        )
        for model_name, candidate in self.model_candidates.items():
            LOGGER.info(
                "Training candidate model '%s' (%s)",
                model_name,
                candidate.__class__.__name__,
            )
            model = clone(candidate)
            model.fit(x_train, y_train)

            val_positive_probs = self._predict_positive_probabilities(model, x_val)
            selection_summary = self._select_threshold(y_val, val_positive_probs)
            selected_metrics = selection_summary["selected_threshold_metrics"]
            ranking_key = _candidate_ranking_key(selected_metrics)
            candidate_models[model_name] = {
                "model_class": model.__class__.__name__,
                "model_params": _to_builtin(model.get_params()),
                "selection_summary": selection_summary,
            }
            LOGGER.info(
                "Candidate '%s' val macro_f1=%.6f average_precision=%s threshold=%.6f",
                model_name,
                selected_metrics["macro_f1"],
                _format_optional_metric(selected_metrics["average_precision"]),
                selection_summary["best_threshold"],
            )
            if best_ranking_key is None or ranking_key > best_ranking_key:
                best_model = model
                best_model_name = model_name
                best_selection_summary = selection_summary
                best_ranking_key = ranking_key

        if best_model is None or best_model_name is None or best_selection_summary is None:
            raise RuntimeError("Failed to train any helpfulness model candidate.")

        self.model = best_model
        self.model_name = best_model_name
        self.threshold = best_selection_summary["best_threshold"]
        self.threshold_selection_summary = best_selection_summary
        self.model_selection_summary = {
            "selection_split": "val",
            "selection_metric": DEFAULT_MODEL_SELECTION_METRIC,
            "selected_model": best_model_name,
            "selected_model_class": best_model.__class__.__name__,
            "candidate_models": candidate_models,
        }
        LOGGER.info(
            "Selected model '%s' with threshold=%.6f and val macro_f1=%.6f across %s candidate thresholds",
            self.model_name,
            self.threshold,
            best_selection_summary["best_metric_value"],
            best_selection_summary["candidate_thresholds"],
        )
        LOGGER.debug(
            "Threshold metric snapshots: default=%s selected=%s",
            json.dumps(best_selection_summary["default_threshold_metrics"], sort_keys=True),
            json.dumps(best_selection_summary["selected_threshold_metrics"], sort_keys=True),
        )

    def evaluate(self, df: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
        resolved_threshold = self.threshold if threshold is None else float(threshold)
        LOGGER.info(
            "Evaluating helpfulness model on %s rows at threshold=%.6f",
            len(df),
            resolved_threshold,
        )
        x = self.feature_builder.transform(df)
        positive_probs = self._predict_positive_probabilities_from_matrix(x)
        return self._build_metrics(
            np.asarray(df["label"].tolist(), dtype=int),
            positive_probs,
            resolved_threshold,
        )

    def save(self, output_prefix: str) -> HelpfulnessArtifacts:
        if self.model is None or self.model_name is None:
            raise ValueError("Model has not been trained yet.")

        output_root = Path(output_prefix)
        output_root.parent.mkdir(parents=True, exist_ok=True)

        model_path = f"{output_prefix}.joblib"
        feature_builder_path = f"{output_prefix}_feature_builder.joblib"
        bundle = {
            "model": self.model,
            "model_name": self.model_name,
            "threshold": float(self.threshold),
            "label_text_by_id": LABEL_TEXT_BY_ID,
        }
        joblib.dump(bundle, model_path)
        joblib.dump(self.feature_builder, feature_builder_path)
        LOGGER.info(
            "Serialized helpfulness artifacts to model_path=%s feature_builder_path=%s",
            model_path,
            feature_builder_path,
        )
        return HelpfulnessArtifacts(
            model_path=model_path,
            feature_builder_path=feature_builder_path,
        )

    def _predict_positive_probabilities_from_matrix(self, x) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self._predict_positive_probabilities(self.model, x)

    def _predict_positive_probabilities(self, model, x) -> np.ndarray:
        probabilities = model.predict_proba(x)
        class_to_index = {
            int(label): index for index, label in enumerate(model.classes_)
        }
        positive_index = class_to_index[1]
        return probabilities[:, positive_index]

    def _build_metrics(
        self,
        y_true: np.ndarray,
        positive_probs: np.ndarray,
        threshold: float,
    ) -> dict[str, Any]:
        clipped_probs = np.clip(positive_probs, 1e-15, 1.0 - 1e-15)
        predicted_labels = (positive_probs >= threshold).astype(int)

        roc_auc = None
        average_precision = None
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, positive_probs))
            average_precision = float(average_precision_score(y_true, positive_probs))

        metrics = {
            "threshold": float(threshold),
            "label_positive_rate": float(np.mean(y_true == 1)),
            "predicted_positive_rate": float(np.mean(predicted_labels == 1)),
            "accuracy": float(accuracy_score(y_true, predicted_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted_labels)),
            "macro_f1": float(f1_score(y_true, predicted_labels, average="macro")),
            "f1_positive": float(
                f1_score(y_true, predicted_labels, pos_label=1, zero_division=0)
            ),
            "precision_positive": float(
                precision_score(y_true, predicted_labels, pos_label=1, zero_division=0)
            ),
            "recall_positive": float(
                recall_score(y_true, predicted_labels, pos_label=1, zero_division=0)
            ),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "log_loss": float(log_loss(y_true, clipped_probs, labels=[0, 1])),
            "positive_support": int(np.sum(y_true == 1)),
            "confusion_matrix": confusion_matrix(
                y_true,
                predicted_labels,
                labels=[0, 1],
            ).tolist(),
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

    def _select_threshold(
        self,
        y_true: np.ndarray,
        positive_probs: np.ndarray,
    ) -> dict[str, Any]:
        default_metrics = self._build_metrics(
            y_true,
            positive_probs,
            self.DEFAULT_CLASSIFICATION_THRESHOLD,
        )
        if len(y_true) == 0:
            return {
                "best_threshold": self.DEFAULT_CLASSIFICATION_THRESHOLD,
                "best_metric_value": default_metrics["macro_f1"],
                "candidate_thresholds": 0,
                "default_threshold_metrics": default_metrics,
                "selected_threshold_metrics": default_metrics,
            }

        sort_order = np.argsort(-positive_probs, kind="mergesort")
        sorted_probs = positive_probs[sort_order]
        sorted_labels = y_true[sort_order]

        total_positive = int(np.sum(sorted_labels == 1))
        total_negative = int(np.sum(sorted_labels == 0))
        tp = 0
        fp = 0
        fn = total_positive
        tn = total_negative

        best_threshold = self.DEFAULT_CLASSIFICATION_THRESHOLD
        best_metric_value = -1.0
        candidate_thresholds = 0
        index = 0

        while index < len(sorted_probs):
            threshold = float(sorted_probs[index])
            group_end = index
            group_positive = 0
            group_negative = 0

            while group_end < len(sorted_probs) and sorted_probs[group_end] == threshold:
                if sorted_labels[group_end] == 1:
                    group_positive += 1
                else:
                    group_negative += 1
                group_end += 1

            tp += group_positive
            fp += group_negative
            fn -= group_positive
            tn -= group_negative

            metric_value = _macro_f1_from_confusion(tp, fp, fn, tn)
            candidate_thresholds += 1
            if (
                metric_value > best_metric_value + 1e-12
                or (
                    abs(metric_value - best_metric_value) <= 1e-12
                    and abs(threshold - self.DEFAULT_CLASSIFICATION_THRESHOLD)
                    < abs(best_threshold - self.DEFAULT_CLASSIFICATION_THRESHOLD)
                )
            ):
                best_metric_value = metric_value
                best_threshold = threshold

            index = group_end

        selected_metrics = self._build_metrics(y_true, positive_probs, best_threshold)
        return {
            "best_threshold": float(best_threshold),
            "best_metric_value": float(best_metric_value),
            "candidate_thresholds": int(candidate_thresholds),
            "default_threshold_metrics": default_metrics,
            "selected_threshold_metrics": selected_metrics,
        }
