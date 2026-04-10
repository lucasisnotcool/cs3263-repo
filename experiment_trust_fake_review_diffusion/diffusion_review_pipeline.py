from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "diffusion_fake_review_experiment/v1"
DATASET_RELATIVE_PATH = Path("data/raw/fake-reviews-dataset/fake reviews dataset.csv")
ARTIFACTS_SUBDIR = Path("artifacts") / "diffusion_fake_review"


@dataclass(frozen=True)
class DiffusionReviewConfig:
    dataset_path: Path | None = None
    artifacts_dir: Path | None = None

    # Mirrors experiment_trust_fake_reviews notebook expectations.
    phase_a_target_rows: int = 240
    test_size: float = 0.25
    random_state: int = 42

    # Text featurization
    max_features: int = 20000
    min_df: int = 2
    ngram_max: int = 2
    latent_dim: int = 192

    # Diffusion schedule
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Training footprint
    denoiser_samples_per_row: int = 8
    classifier_samples_per_row: int = 4

    # Model hyperparameters
    denoiser_alpha: float = 1.0
    logistic_c: float = 1.0

    # Robust inference via repeated stochastic forward corruption.
    inference_samples: int = 32


@dataclass(frozen=True)
class DiffusionSchedule:
    steps: int
    betas: np.ndarray
    alphas: np.ndarray
    alpha_bar: np.ndarray


def _project_root() -> Path:
    cwd = Path.cwd().resolve()
    for root in [cwd, *cwd.parents]:
        if (root / "experiment_trust_fake_review_diffusion").exists():
            return root
    raise FileNotFoundError(
        "Could not locate project root containing experiment_trust_fake_review_diffusion/."
    )


def _resolve_dataset_path(config: DiffusionReviewConfig) -> Path:
    if config.dataset_path is not None:
        return Path(config.dataset_path).expanduser().resolve()
    return (_project_root() / DATASET_RELATIVE_PATH).resolve()


def _resolve_artifacts_dir(config: DiffusionReviewConfig) -> Path:
    if config.artifacts_dir is not None:
        return Path(config.artifacts_dir).expanduser().resolve()
    module_dir = Path(__file__).resolve().parent
    return (module_dir / ARTIFACTS_SUBDIR).resolve()


def _config_to_dict(config: DiffusionReviewConfig) -> dict[str, Any]:
    payload = asdict(config)
    if payload["dataset_path"] is not None:
        payload["dataset_path"] = str(payload["dataset_path"])
    if payload["artifacts_dir"] is not None:
        payload["artifacts_dir"] = str(payload["artifacts_dir"])
    return payload


def _load_fake_review_dataset(dataset_path: Path, target_rows: int, random_state: int) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Fake reviews dataset not found: {dataset_path}")

    raw_df = pd.read_csv(dataset_path)
    required_cols = {"text_", "label"}
    missing = sorted(required_cols - set(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required fake review columns: {missing}")

    out = pd.DataFrame()
    out["text"] = raw_df["text_"].astype(str).fillna("").str.strip()
    out["label_raw"] = raw_df["label"].astype(str).str.strip().str.upper()
    if "category" in raw_df.columns:
        out["category"] = raw_df["category"]
    if "rating" in raw_df.columns:
        out["rating"] = raw_df["rating"]

    label_map = {
        "OR": 1,
        "CG": 0,
    }
    out["label_truth"] = out["label_raw"].map(label_map)
    out = out[out["label_truth"].isin([0, 1])].copy()
    out = out[out["text"].str.len() > 0].copy()

    if target_rows > 0 and len(out) > target_rows:
        per_class = max(1, target_rows // 2)
        sampled_parts: list[pd.DataFrame] = []
        for cls in [0, 1]:
            cls_df = out[out["label_truth"] == cls]
            n = min(per_class, len(cls_df))
            sampled_parts.append(cls_df.sample(n=n, random_state=random_state))
        out = pd.concat(sampled_parts, ignore_index=True)

    out = out.reset_index(drop=True)
    out["record_id"] = "fake_reviews_" + out.index.astype(str)
    return out


def _build_schedule(config: DiffusionReviewConfig) -> DiffusionSchedule:
    betas = np.linspace(config.beta_start, config.beta_end, config.diffusion_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    return DiffusionSchedule(
        steps=config.diffusion_steps,
        betas=betas,
        alphas=alphas,
        alpha_bar=alpha_bar,
    )


def _sample_forward_process(
    x0: np.ndarray,
    schedule: DiffusionSchedule,
    *,
    rng: np.random.Generator,
    t_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x0.shape[0]
    if t_index is None:
        t_index = rng.integers(0, schedule.steps, size=n, endpoint=False)

    alpha_bar_t = schedule.alpha_bar[t_index].reshape(-1, 1)
    eps = rng.normal(0.0, 1.0, size=x0.shape)
    x_t = np.sqrt(alpha_bar_t) * x0 + np.sqrt(1.0 - alpha_bar_t) * eps
    return x_t, eps, t_index


def _timestep_feature(t_index: np.ndarray, total_steps: int) -> np.ndarray:
    return ((t_index.astype(np.float64) + 1.0) / float(total_steps)).reshape(-1, 1)


def _fit_text_pipeline(
    train_texts: Iterable[str],
    config: DiffusionReviewConfig,
) -> tuple[TfidfVectorizer, TruncatedSVD, StandardScaler, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, config.ngram_max),
        max_features=config.max_features,
        min_df=config.min_df,
        sublinear_tf=True,
    )
    x_train_sparse = vectorizer.fit_transform(train_texts)

    if x_train_sparse.shape[1] < 4:
        raise ValueError(
            "Not enough vectorized features to train diffusion model. "
            f"Feature count={x_train_sparse.shape[1]}"
        )

    n_components = min(config.latent_dim, x_train_sparse.shape[1] - 1)
    if n_components < 2:
        raise ValueError(f"Invalid latent dimension after adjustment: {n_components}")

    svd = TruncatedSVD(n_components=n_components, random_state=config.random_state)
    x_train_latent = svd.fit_transform(x_train_sparse)

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train = scaler.fit_transform(x_train_latent)
    return vectorizer, svd, scaler, x_train.astype(np.float64)


def _transform_texts(
    texts: Iterable[str],
    *,
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    scaler: StandardScaler,
) -> np.ndarray:
    x_sparse = vectorizer.transform(texts)
    x_latent = svd.transform(x_sparse)
    x = scaler.transform(x_latent)
    return x.astype(np.float64)


def _build_denoiser_train_data(
    x_train: np.ndarray,
    schedule: DiffusionSchedule,
    *,
    samples_per_row: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.repeat(np.arange(x_train.shape[0]), samples_per_row)
    x0 = x_train[idx]
    x_t, eps, t_index = _sample_forward_process(x0, schedule, rng=rng)
    x_in = np.hstack([x_t, _timestep_feature(t_index, schedule.steps)])
    return x_in, eps


def _denoise_x0_hat(
    denoiser: Ridge,
    x_t: np.ndarray,
    t_index: np.ndarray,
    schedule: DiffusionSchedule,
) -> np.ndarray:
    x_in = np.hstack([x_t, _timestep_feature(t_index, schedule.steps)])
    eps_hat = denoiser.predict(x_in)
    alpha_bar_t = schedule.alpha_bar[t_index].reshape(-1, 1)
    x0_hat = (x_t - np.sqrt(1.0 - alpha_bar_t) * eps_hat) / np.sqrt(alpha_bar_t)
    return x0_hat


def _build_classifier_train_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    denoiser: Ridge,
    schedule: DiffusionSchedule,
    samples_per_row: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.repeat(np.arange(x_train.shape[0]), samples_per_row)
    x0 = x_train[idx]
    y = y_train[idx]
    x_t, _, t_index = _sample_forward_process(x0, schedule, rng=rng)
    x0_hat = _denoise_x0_hat(denoiser, x_t, t_index, schedule)

    x_cls = np.vstack([x_train, x0_hat])
    y_cls = np.concatenate([y_train, y])
    return x_cls, y_cls


def _predict_diffusion_probabilities(
    x: np.ndarray,
    *,
    denoiser: Ridge,
    classifier: LogisticRegression,
    schedule: DiffusionSchedule,
    inference_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    all_probs = []
    for _ in range(inference_samples):
        x_t, _, t_index = _sample_forward_process(x, schedule, rng=rng)
        x0_hat = _denoise_x0_hat(denoiser, x_t, t_index, schedule)
        p_real = classifier.predict_proba(x0_hat)[:, 1]
        all_probs.append(p_real)

    stacked = np.vstack(all_probs)
    return stacked.mean(axis=0), stacked.std(axis=0)


def _safe_log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    p_clipped = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    return float(log_loss(y_true, p_clipped, labels=[0, 1]))


def _binary_metrics(y_true: np.ndarray, p_real: np.ndarray) -> dict[str, float | None]:
    y_true = y_true.astype(int)
    p_real = np.clip(p_real.astype(float), 1e-6, 1.0 - 1e-6)
    y_pred = (p_real >= 0.5).astype(int)

    if np.unique(y_true).size == 2:
        auc_value: float | None = float(roc_auc_score(y_true, p_real))
    else:
        auc_value = None

    return {
        "auc": auc_value,
        "brier": float(brier_score_loss(y_true, p_real)),
        "log_loss": _safe_log_loss(y_true, p_real),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def run_diffusion_review_experiment(config: DiffusionReviewConfig | None = None) -> dict[str, Any]:
    cfg = config or DiffusionReviewConfig()

    dataset_path = _resolve_dataset_path(cfg)
    artifacts_dir = _resolve_artifacts_dir(cfg)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    phase_a_df = _load_fake_review_dataset(
        dataset_path=dataset_path,
        target_rows=cfg.phase_a_target_rows,
        random_state=cfg.random_state,
    )

    train_df, test_df = train_test_split(
        phase_a_df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=phase_a_df["label_truth"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    vectorizer, svd, scaler, x_train = _fit_text_pipeline(train_df["text"].tolist(), cfg)
    x_test = _transform_texts(test_df["text"].tolist(), vectorizer=vectorizer, svd=svd, scaler=scaler)
    y_train = train_df["label_truth"].to_numpy(dtype=int)
    y_test = test_df["label_truth"].to_numpy(dtype=int)

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

    diffusion_classifier = LogisticRegression(
        max_iter=1000,
        C=cfg.logistic_c,
        class_weight="balanced",
    )
    diffusion_classifier.fit(cls_x, cls_y)

    baseline_classifier = LogisticRegression(
        max_iter=1000,
        C=cfg.logistic_c,
        class_weight="balanced",
    )
    baseline_classifier.fit(x_train, y_train)

    p_baseline = baseline_classifier.predict_proba(x_test)[:, 1]
    p_diffusion, p_diffusion_std = _predict_diffusion_probabilities(
        x_test,
        denoiser=denoiser,
        classifier=diffusion_classifier,
        schedule=schedule,
        inference_samples=max(1, cfg.inference_samples),
        rng=np.random.default_rng(cfg.random_state + 17),
    )

    x_noisy, eps_true, t_idx = _sample_forward_process(
        x_test,
        schedule,
        rng=np.random.default_rng(cfg.random_state + 29),
    )
    eps_hat = denoiser.predict(np.hstack([x_noisy, _timestep_feature(t_idx, schedule.steps)]))
    denoiser_mse_test = float(np.mean((eps_true - eps_hat) ** 2))

    baseline_metrics = _binary_metrics(y_test, p_baseline)
    diffusion_metrics = _binary_metrics(y_test, p_diffusion)

    metrics_rows = [
        {"model": "baseline_logistic_clean", **baseline_metrics},
        {
            "model": "diffusion_denoised_logistic",
            **diffusion_metrics,
            "mean_prediction_std": float(np.mean(p_diffusion_std)),
            "denoiser_mse_test": denoiser_mse_test,
        },
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    predictions_df = pd.DataFrame(
        {
            "record_id": test_df["record_id"],
            "label_truth": y_test,
            "p_real_baseline": p_baseline,
            "p_real_diffusion": p_diffusion,
            "p_fake_diffusion": 1.0 - p_diffusion,
            "prediction_std_diffusion": p_diffusion_std,
            "text": test_df["text"],
        }
    )

    model_bundle = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": _config_to_dict(cfg),
        "schedule": {
            "steps": int(schedule.steps),
            "betas": schedule.betas,
            "alphas": schedule.alphas,
            "alpha_bar": schedule.alpha_bar,
        },
        "vectorizer": vectorizer,
        "svd": svd,
        "scaler": scaler,
        "denoiser": denoiser,
        "diffusion_classifier": diffusion_classifier,
        "baseline_classifier": baseline_classifier,
    }

    metrics_path = artifacts_dir / "phase_a_metrics.csv"
    predictions_path = artifacts_dir / "phase_a_test_predictions.csv"
    model_path = artifacts_dir / "diffusion_model_bundle.joblib"
    run_info_path = artifacts_dir / "run_info.json"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    joblib.dump(model_bundle, model_path)

    run_info = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "phase_a_rows_loaded": int(len(phase_a_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "config": _config_to_dict(cfg),
        "metrics": metrics_rows,
        "artifacts": {
            "phase_a_metrics": str(metrics_path),
            "phase_a_test_predictions": str(predictions_path),
            "model_bundle": str(model_path),
        },
    }
    save_json(run_info_path, run_info)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifacts_dir": str(artifacts_dir),
        "run_info_path": str(run_info_path),
        "metrics": metrics_rows,
    }


def _load_model_bundle(artifacts_dir: str | Path) -> dict[str, Any]:
    path = Path(artifacts_dir).expanduser().resolve() / "diffusion_model_bundle.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Unexpected model bundle format at {path}")
    return bundle


def score_review_texts(
    texts: Iterable[str],
    *,
    artifacts_dir: str | Path,
    inference_samples: int | None = None,
    random_state: int = 42,
) -> list[dict[str, float | str]]:
    bundle = _load_model_bundle(artifacts_dir)

    schedule_dict = bundle["schedule"]
    schedule = DiffusionSchedule(
        steps=int(schedule_dict["steps"]),
        betas=np.asarray(schedule_dict["betas"], dtype=np.float64),
        alphas=np.asarray(schedule_dict["alphas"], dtype=np.float64),
        alpha_bar=np.asarray(schedule_dict["alpha_bar"], dtype=np.float64),
    )

    vectorizer: TfidfVectorizer = bundle["vectorizer"]
    svd: TruncatedSVD = bundle["svd"]
    scaler: StandardScaler = bundle["scaler"]
    denoiser: Ridge = bundle["denoiser"]
    classifier: LogisticRegression = bundle["diffusion_classifier"]

    text_list = [str(t) for t in texts]
    x = _transform_texts(text_list, vectorizer=vectorizer, svd=svd, scaler=scaler)

    sample_count = inference_samples if inference_samples is not None else int(bundle["config"]["inference_samples"])
    p_real, p_std = _predict_diffusion_probabilities(
        x,
        denoiser=denoiser,
        classifier=classifier,
        schedule=schedule,
        inference_samples=max(1, sample_count),
        rng=np.random.default_rng(random_state),
    )

    rows: list[dict[str, float | str]] = []
    for i, text in enumerate(text_list):
        rows.append(
            {
                "index": i,
                "text": text,
                "p_real": float(p_real[i]),
                "p_fake": float(1.0 - p_real[i]),
                "prediction_std": float(p_std[i]),
            }
        )
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a standalone diffusion fake-vs-real review model (no LLM, no BN)."
    )
    parser.add_argument("--dataset-path", type=str, help="Optional dataset path override.")
    parser.add_argument("--artifacts-dir", type=str, help="Optional artifacts directory override.")
    parser.add_argument("--phase-a-target-rows", type=int, default=240)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--denoiser-samples-per-row", type=int, default=8)
    parser.add_argument("--classifier-samples-per-row", type=int, default=4)
    parser.add_argument("--inference-samples", type=int, default=32)
    parser.add_argument(
        "--score-text",
        action="append",
        default=[],
        help="Optional review text to score after training. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=str, help="Optional output JSON path.")
    return parser.parse_args(argv)


def _write_output(payload: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = DiffusionReviewConfig(
        dataset_path=None if args.dataset_path is None else Path(args.dataset_path),
        artifacts_dir=None if args.artifacts_dir is None else Path(args.artifacts_dir),
        phase_a_target_rows=int(args.phase_a_target_rows),
        test_size=float(args.test_size),
        random_state=int(args.random_state),
        diffusion_steps=int(args.diffusion_steps),
        latent_dim=int(args.latent_dim),
        max_features=int(args.max_features),
        min_df=int(args.min_df),
        denoiser_samples_per_row=int(args.denoiser_samples_per_row),
        classifier_samples_per_row=int(args.classifier_samples_per_row),
        inference_samples=int(args.inference_samples),
    )

    result = run_diffusion_review_experiment(cfg)

    if args.score_text:
        artifacts_dir = Path(result["artifacts_dir"]) if args.artifacts_dir is None else Path(args.artifacts_dir)
        scored = score_review_texts(
            args.score_text,
            artifacts_dir=artifacts_dir,
            inference_samples=int(args.inference_samples),
            random_state=int(args.random_state) + 123,
        )
        result["scored_texts"] = scored

    _write_output(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
