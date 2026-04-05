from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from eWOM.fusion import EWOMFusionConfig, EWOMFusionPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EWOM_SCORE_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {
        "title": {
            "type": "string",
            "default": "",
            "description": "Optional review title.",
        },
        "text": {
            "type": "string",
            "minLength": 1,
            "description": "Main review text used by helpfulness and sentiment scoring.",
        },
        "rating": {
            "type": "number",
            "default": 0.0,
            "description": "Original star rating if available.",
        },
        "verified_purchase": {
            "type": "boolean",
            "default": False,
            "description": "Whether the review is marked as a verified purchase.",
        },
    },
}

EWOM_REVIEW_SET_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "string",
        "minLength": 1,
    },
    "description": "Array of review texts used to compute a product-level eWOM score.",
}

EWOM_MODEL_PATHS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "helpfulness_model_path",
        "helpfulness_feature_builder_path",
        "sentiment_model_path",
        "sentiment_feature_builder_path",
    ],
    "properties": {
        "helpfulness_model_path": {"type": "string"},
        "helpfulness_feature_builder_path": {"type": "string"},
        "sentiment_model_path": {"type": "string"},
        "sentiment_feature_builder_path": {"type": "string"},
    },
}

EWOM_DECEPTION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "status",
        "source",
        "deception_probability",
        "authenticity_probability",
        "trust_probability",
        "is_deceptive",
        "graph_uncertainty_entropy",
        "overall_confidence",
        "error",
    ],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ok", "unavailable", "error"],
        },
        "source": {"type": "string"},
        "deception_probability": {"type": ["number", "null"]},
        "authenticity_probability": {"type": ["number", "null"]},
        "trust_probability": {"type": ["number", "null"]},
        "is_deceptive": {"type": ["boolean", "null"]},
        "graph_uncertainty_entropy": {"type": ["number", "null"]},
        "overall_confidence": {"type": ["number", "null"]},
        "error": {
            "type": ["object", "null"],
            "additionalProperties": False,
            "required": ["type", "message"],
            "properties": {
                "type": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    },
}

EWOM_SCORE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["helpfulness", "sentiment", "deception", "fusion"],
    "properties": {
        "helpfulness": {
            "type": "object",
            "additionalProperties": False,
            "required": ["usefulness_probability", "is_useful"],
            "properties": {
                "usefulness_probability": {"type": "number"},
                "is_useful": {"type": "boolean"},
            },
        },
        "sentiment": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "negative_probability",
                "positive_probability",
                "predicted_label",
                "predicted_label_text",
            ],
            "properties": {
                "negative_probability": {"type": "number"},
                "positive_probability": {"type": "number"},
                "predicted_label": {"type": "integer"},
                "predicted_label_text": {
                    "type": "string",
                    "enum": ["negative", "positive"],
                },
            },
        },
        "deception": EWOM_DECEPTION_RESPONSE_SCHEMA,
        "fusion": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "usefulness_probability",
                "helpfulness_gate",
                "deception_probability",
                "deception_weight",
                "informative_gate",
                "positive_probability",
                "negative_probability",
                "sentiment_polarity",
                "sentiment_strength",
                "signed_ewom_score",
                "magnitude_ewom_score",
                "ewom_score_0_to_100",
                "ewom_magnitude_0_to_100",
            ],
            "properties": {
                "usefulness_probability": {"type": "number"},
                "helpfulness_gate": {"type": "number"},
                "deception_probability": {"type": ["number", "null"]},
                "deception_weight": {"type": "number"},
                "informative_gate": {"type": "number"},
                "positive_probability": {"type": "number"},
                "negative_probability": {"type": "number"},
                "sentiment_polarity": {"type": "number"},
                "sentiment_strength": {"type": "number"},
                "signed_ewom_score": {"type": "number"},
                "magnitude_ewom_score": {"type": "number"},
                "ewom_score_0_to_100": {"type": "number"},
                "ewom_magnitude_0_to_100": {"type": "number"},
            },
        },
    },
}

EWOM_REVIEW_SET_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["review_count", "reviews", "aggregate"],
    "properties": {
        "review_count": {"type": "integer", "minimum": 1},
        "reviews": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "helpfulness", "sentiment", "deception", "fusion"],
                "properties": {
                    "text": {"type": "string"},
                    "helpfulness": EWOM_SCORE_RESPONSE_SCHEMA["properties"]["helpfulness"],
                    "sentiment": EWOM_SCORE_RESPONSE_SCHEMA["properties"]["sentiment"],
                    "deception": EWOM_SCORE_RESPONSE_SCHEMA["properties"]["deception"],
                    "fusion": EWOM_SCORE_RESPONSE_SCHEMA["properties"]["fusion"],
                },
            },
        },
        "aggregate": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "review_count",
                "informative_review_weight",
                "mean_usefulness_probability",
                "mean_helpfulness_gate",
                "mean_deception_probability",
                "mean_deception_weight",
                "mean_informative_gate",
                "weighted_positive_probability",
                "weighted_negative_probability",
                "weighted_sentiment_polarity",
                "weighted_sentiment_strength",
                "review_set_gate",
                "final_signed_ewom_score",
                "final_magnitude_ewom_score",
                "final_ewom_score_0_to_100",
                "final_ewom_magnitude_0_to_100",
            ],
            "properties": {
                "review_count": {"type": "integer", "minimum": 1},
                "informative_review_weight": {"type": "number"},
                "mean_usefulness_probability": {"type": "number"},
                "mean_helpfulness_gate": {"type": "number"},
                "mean_deception_probability": {"type": ["number", "null"]},
                "mean_deception_weight": {"type": "number"},
                "mean_informative_gate": {"type": "number"},
                "weighted_positive_probability": {"type": "number"},
                "weighted_negative_probability": {"type": "number"},
                "weighted_sentiment_polarity": {"type": "number"},
                "weighted_sentiment_strength": {"type": "number"},
                "review_set_gate": {"type": "number"},
                "final_signed_ewom_score": {"type": "number"},
                "final_magnitude_ewom_score": {"type": "number"},
                "final_ewom_score_0_to_100": {"type": "number"},
                "final_ewom_magnitude_0_to_100": {"type": "number"},
            },
        },
    },
}


@dataclass(frozen=True)
class EWOMModelPaths:
    helpfulness_model_path: str
    helpfulness_feature_builder_path: str
    sentiment_model_path: str
    sentiment_feature_builder_path: str

    @classmethod
    def defaults(cls, project_root: str | Path | None = None) -> EWOMModelPaths:
        root = Path(project_root) if project_root is not None else PROJECT_ROOT
        return cls(
            helpfulness_model_path=str(
                root / "models" / "helpfulness" / "amazon_helpfulness_benchmark.joblib"
            ),
            helpfulness_feature_builder_path=str(
                root
                / "models"
                / "helpfulness"
                / "amazon_helpfulness_benchmark_feature_builder.joblib"
            ),
            sentiment_model_path=str(
                root / "models" / "sentiment" / "amazon_polarity_full_benchmark.joblib"
            ),
            sentiment_feature_builder_path=str(
                root / "models" / "sentiment" / "amazon_polarity_full_benchmark_feature_builder.joblib"
            ),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> EWOMModelPaths:
        missing_keys = [
            key
            for key in EWOM_MODEL_PATHS_SCHEMA["required"]
            if key not in value or value[key] in {None, ""}
        ]
        if missing_keys:
            raise ValueError(
                "model_paths is missing required values: " + ", ".join(sorted(missing_keys))
            )

        return cls(
            helpfulness_model_path=str(value["helpfulness_model_path"]),
            helpfulness_feature_builder_path=str(value["helpfulness_feature_builder_path"]),
            sentiment_model_path=str(value["sentiment_model_path"]),
            sentiment_feature_builder_path=str(value["sentiment_feature_builder_path"]),
        )

    def normalized(self) -> EWOMModelPaths:
        return EWOMModelPaths(
            helpfulness_model_path=str(
                Path(self.helpfulness_model_path).expanduser().resolve()
            ),
            helpfulness_feature_builder_path=str(
                Path(self.helpfulness_feature_builder_path).expanduser().resolve()
            ),
            sentiment_model_path=str(
                Path(self.sentiment_model_path).expanduser().resolve()
            ),
            sentiment_feature_builder_path=str(
                Path(self.sentiment_feature_builder_path).expanduser().resolve()
            ),
        )


def get_ewom_schemas() -> dict[str, dict[str, Any]]:
    return {
        "request": deepcopy(EWOM_SCORE_REQUEST_SCHEMA),
        "review_set_request": deepcopy(EWOM_REVIEW_SET_REQUEST_SCHEMA),
        "model_paths": deepcopy(EWOM_MODEL_PATHS_SCHEMA),
        "response": deepcopy(EWOM_SCORE_RESPONSE_SCHEMA),
        "review_set_response": deepcopy(EWOM_REVIEW_SET_RESPONSE_SCHEMA),
    }


def score_review(
    review: Mapping[str, Any],
    *,
    model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
    fusion_config: EWOMFusionConfig | None = None,
) -> dict[str, Any]:
    normalized_review = _normalize_review_payload(review)
    resolved_model_paths = _normalize_model_paths(model_paths)
    resolved_fusion_config = fusion_config or EWOMFusionConfig()
    predictor = _get_predictor(
        resolved_model_paths.normalized(),
        resolved_fusion_config,
    )
    return predictor.predict_one(**normalized_review)


def score_review_set(
    review_texts: Sequence[str],
    *,
    model_paths: EWOMModelPaths | Mapping[str, Any] | None = None,
    fusion_config: EWOMFusionConfig | None = None,
) -> dict[str, Any]:
    normalized_review_texts = _normalize_review_texts(review_texts)
    resolved_model_paths = _normalize_model_paths(model_paths)
    resolved_fusion_config = fusion_config or EWOMFusionConfig()
    predictor = _get_predictor(
        resolved_model_paths.normalized(),
        resolved_fusion_config,
    )
    return predictor.predict_many(review_texts=normalized_review_texts)


def _normalize_review_payload(review: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(review, Mapping):
        raise TypeError("review must be a mapping that matches EWOM_SCORE_REQUEST_SCHEMA.")

    text = str(review.get("text", "")).strip()
    if not text:
        raise ValueError("review['text'] is required and must not be empty.")

    title = str(review.get("title", "") or "")
    rating = _coerce_float("review['rating']", review.get("rating", 0.0))
    verified_purchase = _coerce_bool(
        "review['verified_purchase']",
        review.get("verified_purchase", False),
    )

    return {
        "title": title,
        "text": text,
        "rating": rating,
        "verified_purchase": verified_purchase,
    }


def _normalize_review_texts(review_texts: Sequence[str]) -> list[str]:
    if isinstance(review_texts, (str, bytes)):
        raise TypeError("review_texts must be an array of strings, not a single string.")

    normalized_review_texts = [str(review_text).strip() for review_text in review_texts]
    if not normalized_review_texts:
        raise ValueError("review_texts must contain at least one review.")

    if any(not review_text for review_text in normalized_review_texts):
        raise ValueError("review_texts must not contain empty reviews.")

    return normalized_review_texts


def _normalize_model_paths(
    model_paths: EWOMModelPaths | Mapping[str, Any] | None,
) -> EWOMModelPaths:
    if model_paths is None:
        return EWOMModelPaths.defaults()
    if isinstance(model_paths, EWOMModelPaths):
        return model_paths
    if isinstance(model_paths, Mapping):
        return EWOMModelPaths.from_mapping(model_paths)
    raise TypeError("model_paths must be None, EWOMModelPaths, or a mapping.")


def _coerce_float(name: str, value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric.") from exc


def _coerce_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise TypeError(f"{name} must be a boolean.")


def _ensure_model_artifacts_exist(model_paths: EWOMModelPaths) -> None:
    missing_paths = [
        path
        for path in [
            model_paths.helpfulness_model_path,
            model_paths.helpfulness_feature_builder_path,
            model_paths.sentiment_model_path,
            model_paths.sentiment_feature_builder_path,
        ]
        if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing eWOM model artifacts: " + ", ".join(sorted(missing_paths))
        )


@lru_cache(maxsize=8)
def _get_predictor(
    model_paths: EWOMModelPaths,
    fusion_config: EWOMFusionConfig,
) -> EWOMFusionPredictor:
    _ensure_model_artifacts_exist(model_paths)
    return EWOMFusionPredictor(
        helpfulness_model_path=model_paths.helpfulness_model_path,
        helpfulness_feature_builder_path=model_paths.helpfulness_feature_builder_path,
        sentiment_model_path=model_paths.sentiment_model_path,
        sentiment_feature_builder_path=model_paths.sentiment_feature_builder_path,
        fusion_config=fusion_config,
    )
