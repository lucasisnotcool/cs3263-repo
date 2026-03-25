from __future__ import annotations

import joblib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score

from .dataset_loader import LABEL_TEXT_BY_ID


@dataclass
class SentimentArtifacts:
    model_path: str
    feature_builder_path: str


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


class SentimentTrainer:
    def __init__(self, feature_builder, random_state: int = 42):
        self.feature_builder = feature_builder
        self.model = LogisticRegression(
            solver="saga",
            max_iter=200,
            random_state=random_state,
        )

    def fit(self, train_texts, train_labels) -> None:
        x_train = self.feature_builder.fit_transform(train_texts)
        self.model.fit(x_train, train_labels)

    def evaluate(self, texts, labels) -> dict:
        x = self.feature_builder.transform(texts)
        y_true = np.asarray(list(labels), dtype=int)
        pred_labels = self.model.predict(x)
        pred_proba = self.model.predict_proba(x)
        class_to_index = {int(label): idx for idx, label in enumerate(self.model.classes_)}
        positive_probs = pred_proba[:, class_to_index[1]]

        roc_auc = None
        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, positive_probs))

        metrics = {
            "accuracy": float(accuracy_score(y_true, pred_labels)),
            "macro_f1": float(f1_score(y_true, pred_labels, average="macro")),
            "roc_auc": roc_auc,
            "classification_report": classification_report(
                y_true,
                pred_labels,
                labels=[0, 1],
                target_names=[LABEL_TEXT_BY_ID[0], LABEL_TEXT_BY_ID[1]],
                output_dict=True,
                zero_division=0,
            ),
        }
        return _to_builtin(metrics)

    def save(self, output_prefix: str) -> SentimentArtifacts:
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

        model_path = f"{output_prefix}.joblib"
        feature_builder_path = f"{output_prefix}_feature_builder.joblib"

        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_builder, feature_builder_path)

        return SentimentArtifacts(
            model_path=model_path,
            feature_builder_path=feature_builder_path,
        )
