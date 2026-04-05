from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from experiment_trust_fake_reviews import DeployConfig, TrustFakeReviewsDeployPipeline


DEFAULT_SOURCE = "experiment_trust_fake_reviews"


class DeceptionPredictor:
    def __init__(self, config: DeployConfig | None = None):
        self.config = config or DeployConfig()
        self.pipeline = TrustFakeReviewsDeployPipeline(config=self.config)

    def predict_one(
        self,
        text: str,
        *,
        title: str = "",
    ) -> dict[str, Any]:
        return self.predict_many([text], titles=[title])[0]

    def predict_many(
        self,
        texts: Sequence[str],
        *,
        titles: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        if not texts:
            raise ValueError("texts must contain at least one review.")

        review_count = len(texts)
        if titles is None:
            titles = [""] * review_count
        elif len(titles) != review_count:
            raise ValueError("titles and texts must have the same length.")

        payload = [
            {
                "text": self._build_review_text(title=title, text=text),
            }
            for title, text in zip(titles, texts)
        ]

        result = self.pipeline.run(payload, raise_on_environment_error=False)
        environment = result.get("environment", {})
        if not environment.get("ok", False) and not result.get("results"):
            error = self._build_environment_error(environment)
            return [error.copy() for _ in range(review_count)]

        result_rows = list(result.get("results", []))
        predictions: list[dict[str, Any]] = []
        for index in range(review_count):
            row = result_rows[index] if index < len(result_rows) else None
            if row is None:
                predictions.append(
                    self._build_error_prediction(
                        error_type="MissingPredictionError",
                        message="Deception pipeline did not return a row for this review.",
                    )
                )
                continue
            predictions.append(self._normalize_row(row))

        return predictions

    def _normalize_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("status") != "ok":
            error = row.get("error") or {}
            return self._build_error_prediction(
                error_type=str(error.get("type") or "DeceptionPredictionError"),
                message=str(error.get("message") or "Unknown deception prediction error."),
            )

        scores = row.get("scores") or {}
        labels = row.get("labels") or {}
        trust_probability = self._to_optional_float(
            scores.get("phase_b_truth_likelihood_graph")
        )
        deception_probability = self._to_optional_float(
            scores.get("trust_risk_index_graph")
        )
        if deception_probability is None and trust_probability is not None:
            deception_probability = max(0.0, min(1.0, 1.0 - trust_probability))
        if trust_probability is None and deception_probability is not None:
            trust_probability = max(0.0, min(1.0, 1.0 - deception_probability))

        return {
            "status": "ok",
            "source": DEFAULT_SOURCE,
            "deception_probability": deception_probability,
            "authenticity_probability": trust_probability,
            "trust_probability": trust_probability,
            "is_deceptive": None
            if deception_probability is None
            else deception_probability >= 0.5,
            "graph_uncertainty_entropy": self._to_optional_float(
                scores.get("graph_uncertainty_entropy")
            ),
            "overall_confidence": self._to_optional_float(
                labels.get("overall_confidence")
            ),
            "error": None,
        }

    def _build_environment_error(self, environment: dict[str, Any]) -> dict[str, Any]:
        message = "; ".join(str(error) for error in environment.get("errors", []))
        if not message:
            message = "Deception pipeline environment is unavailable."
        return {
            "status": "unavailable",
            "source": DEFAULT_SOURCE,
            "deception_probability": None,
            "authenticity_probability": None,
            "trust_probability": None,
            "is_deceptive": None,
            "graph_uncertainty_entropy": None,
            "overall_confidence": None,
            "error": {
                "type": "EnvironmentValidationError",
                "message": message,
            },
        }

    def _build_error_prediction(
        self,
        *,
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "source": DEFAULT_SOURCE,
            "deception_probability": None,
            "authenticity_probability": None,
            "trust_probability": None,
            "is_deceptive": None,
            "graph_uncertainty_entropy": None,
            "overall_confidence": None,
            "error": {
                "type": error_type,
                "message": message,
            },
        }

    def _build_review_text(self, *, title: str, text: str) -> str:
        stripped_title = str(title).strip()
        stripped_text = str(text).strip()
        if stripped_title:
            return f"Review Title: {stripped_title}\nReview Text: {stripped_text}"
        return stripped_text

    def _to_optional_float(self, value: Any) -> float | None:
        if value in {None, ""}:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, numeric))
