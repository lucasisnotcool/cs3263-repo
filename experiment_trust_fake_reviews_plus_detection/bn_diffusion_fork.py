from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .diffusion_detection_pipeline import (
    DiffusionReviewConfig,
    _build_classifier_train_data,
    _build_denoiser_train_data,
    _build_schedule,
    _fit_text_pipeline,
    _predict_diffusion_probabilities,
    _transform_texts,
)
from .llm_trust_graph_pipeline import (
    BUCKET_ORDER,
    GRAPH_BUCKET_COLUMNS,
    TRUST_SCORE_COLUMNS,
    PhaseConfig,
    bucketize_score,
    discretize_label_columns,
)


@dataclass(frozen=True)
class DiffusionForkConfig:
    random_state: int = 42
    diffusion_steps: int = 50
    latent_dim: int = 192
    max_features: int = 20000
    min_df: int = 2
    ngram_max: int = 2
    denoiser_samples_per_row: int = 8
    classifier_samples_per_row: int = 4
    inference_samples: int = 32
    denoiser_alpha: float = 1.0
    logistic_c: float = 1.0


def _binary_metrics(y_true: np.ndarray, p_real: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p_real, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    y_hat = (p >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "accuracy": float(accuracy_score(y, y_hat)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
    }


def _fit_graph_with_bucket_columns(
    train_df: pd.DataFrame,
    *,
    bucket_columns: list[str],
    alpha: float = 1.0,
) -> dict[str, Any]:
    class_counts = train_df["label_truth"].value_counts().to_dict()
    total = sum(class_counts.values())
    class_probs = {
        cls: (class_counts.get(cls, 0) + alpha) / (total + alpha * 2)
        for cls in [0, 1]
    }

    cpds: dict[str, dict[int, dict[str, float]]] = {}
    for col in bucket_columns:
        cpds[col] = {}
        for cls in [0, 1]:
            subset = train_df[train_df["label_truth"] == cls]
            counts = subset[col].value_counts().to_dict()
            denom = len(subset) + alpha * len(BUCKET_ORDER)
            cpds[col][cls] = {
                bucket: (counts.get(bucket, 0) + alpha) / denom
                for bucket in BUCKET_ORDER
            }

    return {
        "class_probs": class_probs,
        "cpds": cpds,
        "bucket_columns": bucket_columns,
    }


def _predict_graph(model: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    class_probs = model["class_probs"]
    cpds = model["cpds"]
    bucket_columns = model["bucket_columns"]

    probs = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        logp = {}
        for cls in [0, 1]:
            value = math.log(class_probs[cls])
            for bucket_col in bucket_columns:
                value += math.log(cpds[bucket_col][cls][row_dict[bucket_col]])
            logp[cls] = value

        max_logp = max(logp.values())
        exp0 = math.exp(logp[0] - max_logp)
        exp1 = math.exp(logp[1] - max_logp)
        probs.append(exp1 / (exp0 + exp1))

    return np.asarray(probs, dtype=float)


def _fit_diffusion_text_model(
    train_texts: Iterable[str],
    y_train: np.ndarray,
    *,
    config: DiffusionForkConfig,
) -> dict[str, Any]:
    cfg = DiffusionReviewConfig(
        random_state=int(config.random_state),
        diffusion_steps=int(config.diffusion_steps),
        latent_dim=int(config.latent_dim),
        max_features=int(config.max_features),
        min_df=int(config.min_df),
        ngram_max=int(config.ngram_max),
        denoiser_samples_per_row=int(config.denoiser_samples_per_row),
        classifier_samples_per_row=int(config.classifier_samples_per_row),
        inference_samples=int(config.inference_samples),
        denoiser_alpha=float(config.denoiser_alpha),
        logistic_c=float(config.logistic_c),
    )

    vectorizer, svd, scaler, x_train = _fit_text_pipeline(list(train_texts), cfg)
    schedule = _build_schedule(cfg)
    rng = np.random.default_rng(cfg.random_state)

    denoiser_x, denoiser_y = _build_denoiser_train_data(
        x_train,
        schedule,
        samples_per_row=max(1, cfg.denoiser_samples_per_row),
        rng=rng,
    )
    denoiser = Ridge(alpha=cfg.denoiser_alpha)
    denoiser.fit(denoiser_x, denoiser_y)

    cls_x, cls_y = _build_classifier_train_data(
        x_train,
        y_train,
        denoiser=denoiser,
        schedule=schedule,
        samples_per_row=max(1, cfg.classifier_samples_per_row),
        rng=rng,
    )

    classifier = LogisticRegression(max_iter=1000, C=cfg.logistic_c, class_weight="balanced")
    classifier.fit(cls_x, cls_y)

    return {
        "config": cfg,
        "vectorizer": vectorizer,
        "svd": svd,
        "scaler": scaler,
        "schedule": schedule,
        "denoiser": denoiser,
        "classifier": classifier,
    }


def _score_texts_with_diffusion(
    texts: Iterable[str],
    *,
    model: dict[str, Any],
    inference_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = _transform_texts(
        list(texts),
        vectorizer=model["vectorizer"],
        svd=model["svd"],
        scaler=model["scaler"],
    )

    p_real, p_std = _predict_diffusion_probabilities(
        x,
        denoiser=model["denoiser"],
        classifier=model["classifier"],
        schedule=model["schedule"],
        inference_samples=max(1, int(inference_samples)),
        rng=np.random.default_rng(int(random_state)),
    )
    return p_real, p_std


def run_bn_diffusion_fork_evaluation(
    labeled_df: pd.DataFrame,
    *,
    phase_config: PhaseConfig,
    text_col: str = "text",
    config: DiffusionForkConfig | None = None,
) -> dict[str, Any]:
    cfg = config or DiffusionForkConfig(random_state=int(phase_config.random_state))

    if labeled_df.empty:
        raise ValueError("labeled_df is empty")
    if text_col not in labeled_df.columns:
        raise ValueError(f"text column not found: {text_col}")

    prepared = discretize_label_columns(labeled_df)
    needed = TRUST_SCORE_COLUMNS + GRAPH_BUCKET_COLUMNS + ["label_truth", text_col]
    missing = [c for c in needed if c not in prepared.columns]
    if missing:
        raise ValueError(f"Missing required columns for fork evaluation: {missing}")

    split_df = prepared[needed].copy()
    train_df, test_df = train_test_split(
        split_df,
        test_size=phase_config.test_size,
        random_state=phase_config.random_state,
        stratify=split_df["label_truth"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_train = train_df["label_truth"].to_numpy(dtype=int)
    y_test = test_df["label_truth"].to_numpy(dtype=int)

    baseline_graph_model = _fit_graph_with_bucket_columns(
        train_df,
        bucket_columns=list(GRAPH_BUCKET_COLUMNS),
    )
    p_graph_baseline = _predict_graph(baseline_graph_model, test_df)

    baseline_logistic = LogisticRegression(max_iter=1000)
    baseline_logistic.fit(train_df[TRUST_SCORE_COLUMNS], y_train)
    p_logistic_baseline = baseline_logistic.predict_proba(test_df[TRUST_SCORE_COLUMNS])[:, 1]

    diffusion_model = _fit_diffusion_text_model(
        train_df[text_col].astype(str).tolist(),
        y_train,
        config=cfg,
    )

    p_real_train, p_std_train = _score_texts_with_diffusion(
        train_df[text_col].astype(str).tolist(),
        model=diffusion_model,
        inference_samples=cfg.inference_samples,
        random_state=cfg.random_state + 17,
    )
    p_real_test, p_std_test = _score_texts_with_diffusion(
        test_df[text_col].astype(str).tolist(),
        model=diffusion_model,
        inference_samples=cfg.inference_samples,
        random_state=cfg.random_state + 23,
    )

    train_aug = train_df.copy()
    test_aug = test_df.copy()
    train_aug["diffusion_real_score"] = p_real_train
    test_aug["diffusion_real_score"] = p_real_test
    train_aug["diffusion_bucket"] = train_aug["diffusion_real_score"].apply(bucketize_score)
    test_aug["diffusion_bucket"] = test_aug["diffusion_real_score"].apply(bucketize_score)

    aug_bucket_cols = list(GRAPH_BUCKET_COLUMNS) + ["diffusion_bucket"]
    aug_score_cols = list(TRUST_SCORE_COLUMNS) + ["diffusion_real_score"]

    augmented_graph_model = _fit_graph_with_bucket_columns(
        train_aug,
        bucket_columns=aug_bucket_cols,
    )
    p_graph_aug = _predict_graph(augmented_graph_model, test_aug)

    augmented_logistic = LogisticRegression(max_iter=1000)
    augmented_logistic.fit(train_aug[aug_score_cols], y_train)
    p_logistic_aug = augmented_logistic.predict_proba(test_aug[aug_score_cols])[:, 1]

    metrics_rows = [
        {"model": "bn_baseline", **_binary_metrics(y_test, p_graph_baseline)},
        {"model": "bn_with_diffusion_factor", **_binary_metrics(y_test, p_graph_aug)},
        {"model": "logistic_baseline", **_binary_metrics(y_test, p_logistic_baseline)},
        {"model": "logistic_with_diffusion_factor", **_binary_metrics(y_test, p_logistic_aug)},
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    comparison_summary = {
        "delta_auc_bn": float(metrics_df.loc[metrics_df["model"] == "bn_with_diffusion_factor", "auc"].iloc[0] - metrics_df.loc[metrics_df["model"] == "bn_baseline", "auc"].iloc[0]),
        "delta_brier_bn": float(metrics_df.loc[metrics_df["model"] == "bn_with_diffusion_factor", "brier"].iloc[0] - metrics_df.loc[metrics_df["model"] == "bn_baseline", "brier"].iloc[0]),
        "delta_log_loss_bn": float(metrics_df.loc[metrics_df["model"] == "bn_with_diffusion_factor", "log_loss"].iloc[0] - metrics_df.loc[metrics_df["model"] == "bn_baseline", "log_loss"].iloc[0]),
    }

    test_predictions = pd.DataFrame(
        {
            "label_truth": y_test,
            "p_bn_baseline": p_graph_baseline,
            "p_bn_with_diffusion": p_graph_aug,
            "p_logistic_baseline": p_logistic_baseline,
            "p_logistic_with_diffusion": p_logistic_aug,
            "diffusion_real_score": p_real_test,
            "diffusion_prediction_std": p_std_test,
            text_col: test_df[text_col].astype(str).tolist(),
        }
    )

    return {
        "metrics": metrics_df,
        "comparison_summary": comparison_summary,
        "test_predictions": test_predictions,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "models": {
            "bn_baseline": baseline_graph_model,
            "bn_with_diffusion": augmented_graph_model,
            "logistic_baseline": baseline_logistic,
            "logistic_with_diffusion": augmented_logistic,
            "diffusion_model": diffusion_model,
        },
    }
