from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .llm_trust_graph_pipeline import (
    GRAPH_BUCKET_COLUMNS,
    TRUST_SCORE_COLUMNS,
    LabelingConfig,
    _clean_text,
    _is_valid_cached_label_record,
    build_label_prompt,
    build_product_text,
    discretize_label_columns,
    entropy_binary,
    predict_naive_bayes_graph,
    run_ollama_label,
)


SCHEMA_VERSION = "trust_fake_reviews_deploy/v1"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
ARTIFACTS_SUBDIR = Path("artifacts") / "llm_trust_graph"


@dataclass(frozen=True)
class DeployConfig:
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: int = 240
    keepalive: str = "10m"
    max_text_chars: int = 1200
    max_output_tokens: int = 220
    context_tokens: int = 1024
    artifacts_dir: Path | None = None
    cache_path: Path | None = None


class EnvironmentValidationError(RuntimeError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class TrustFakeReviewsDeployPipeline:
    def __init__(self, config: DeployConfig | None = None) -> None:
        self.config = config or DeployConfig()
        self.module_dir = Path(__file__).resolve().parent
        self.artifacts_dir = (
            Path(self.config.artifacts_dir).expanduser().resolve()
            if self.config.artifacts_dir is not None
            else (self.module_dir / ARTIFACTS_SUBDIR).resolve()
        )
        self.graph_model_path = self.artifacts_dir / "graph_model.json"
        self.logistic_model_path = self.artifacts_dir / "logistic_model.json"
        self.cache_path = (
            Path(self.config.cache_path).expanduser().resolve()
            if self.config.cache_path is not None
            else (self.artifacts_dir / "deploy_labels.jsonl").resolve()
        )
        self.label_config = LabelingConfig(
            model=self.config.ollama_model,
            timeout_seconds=self.config.timeout_seconds,
            keepalive=self.config.keepalive,
            max_text_chars=self.config.max_text_chars,
            max_output_tokens=self.config.max_output_tokens,
            context_tokens=self.config.context_tokens,
        )
        self._graph_model: dict[str, Any] | None = None
        self._logistic_artifact: dict[str, Any] | None = None

    def validate_environment(self) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        suggested_commands: list[str] = []

        artifact_status = {
            "graph_model": {
                "path": str(self.graph_model_path),
                "exists": self.graph_model_path.exists(),
            },
            "logistic_model": {
                "path": str(self.logistic_model_path),
                "exists": self.logistic_model_path.exists(),
            },
            "label_cache": {
                "path": str(self.cache_path),
                "exists": self.cache_path.exists(),
            },
        }
        for name, status in artifact_status.items():
            if name == "label_cache":
                continue
            if not status["exists"]:
                errors.append(f"Missing deploy artifact: {status['path']}")

        ollama_binary = shutil.which("ollama")
        tags_error: str | None = None
        cli_error: str | None = None
        api_models: list[str] = []
        cli_models: list[str] = []
        service_reachable = False
        cli_reachable = False

        if ollama_binary is None:
            errors.append(
                "Ollama was not found on PATH. Install it and ensure the `ollama` command is available."
            )
            suggested_commands.extend(
                [
                    "ollama serve",
                    f"ollama pull {self.config.ollama_model}",
                ]
            )
        else:
            try:
                api_models = self._fetch_ollama_models()
                service_reachable = True
            except Exception as exc:  # noqa: BLE001
                tags_error = str(exc)

            try:
                cli_models = self._list_ollama_models_via_cli(ollama_binary)
                cli_reachable = True
            except Exception as exc:  # noqa: BLE001
                cli_error = str(exc)

            if not service_reachable and not cli_reachable:
                errors.append(
                    "Could not reach local Ollama via API or CLI. Start the service with `ollama serve`."
                )
                suggested_commands.append("ollama serve")
            elif not service_reachable:
                warnings.append(
                    "Ollama HTTP API was not reachable. Runtime can still rely on CLI fallback, "
                    "but `ollama serve` is recommended."
                )
                suggested_commands.append("ollama serve")

            available_models = sorted(set(api_models + cli_models))
            model_present = self.config.ollama_model in available_models
            if not model_present:
                errors.append(
                    f"Ollama model `{self.config.ollama_model}` is not installed locally."
                )
                suggested_commands.append(f"ollama pull {self.config.ollama_model}")
        if ollama_binary is None:
            available_models = []
            model_present = False
        else:
            available_models = sorted(set(api_models + cli_models))
            model_present = self.config.ollama_model in available_models

        return {
            "ok": not errors,
            "ollama": {
                "binary_path": ollama_binary,
                "service_reachable": service_reachable,
                "cli_reachable": cli_reachable,
                "api_models": api_models,
                "cli_models": cli_models,
                "available_models": available_models,
                "model_required": self.config.ollama_model,
                "model_present": model_present,
                "api_error": tags_error,
                "cli_error": cli_error,
            },
            "artifacts": artifact_status,
            "errors": errors,
            "warnings": warnings,
            "suggested_commands": list(dict.fromkeys(suggested_commands)),
        }

    def run(
        self,
        products: Any,
        *,
        raise_on_environment_error: bool = True,
    ) -> dict[str, Any]:
        environment = self.validate_environment()
        if not environment["ok"]:
            result = self._build_error_response(environment)
            if raise_on_environment_error:
                raise EnvironmentValidationError(self._format_environment_error(environment), result)
            return result

        try:
            self._load_models()
        except Exception as exc:  # noqa: BLE001
            environment = dict(environment)
            environment["ok"] = False
            environment["errors"] = list(environment.get("errors", [])) + [
                f"Failed to load deploy artifacts: {exc}"
            ]
            result = self._build_error_response(environment)
            if raise_on_environment_error:
                raise EnvironmentValidationError(self._format_environment_error(environment), result)
            return result

        normalized_products, input_errors = self._normalize_products(products)
        cache_by_id = self._load_cache_by_id()

        successful_rows: list[dict[str, Any]] = []
        error_by_record_id = {item["record_id"]: item for item in input_errors}
        cache_writes: list[dict[str, Any]] = []

        for item in normalized_products:
            cached = cache_by_id.get(item["record_id"])
            if cached is not None and _is_valid_cached_label_record(cached):
                label_payload = cached
            else:
                prompt = build_label_prompt(
                    domain="product",
                    text=item["text"][: self.label_config.max_text_chars],
                )
                try:
                    label = run_ollama_label(
                        prompt=prompt,
                        model=self.label_config.model,
                        timeout_seconds=self.label_config.timeout_seconds,
                        keepalive=self.label_config.keepalive,
                        max_output_tokens=self.label_config.max_output_tokens,
                        context_tokens=self.label_config.context_tokens,
                    )
                    label_payload = {
                        "record_id": item["record_id"],
                        "domain": "product",
                        **label,
                    }
                    cache_by_id[item["record_id"]] = label_payload
                    cache_writes.append(label_payload)
                except Exception as exc:  # noqa: BLE001
                    error_by_record_id[item["record_id"]] = self._build_item_error(
                        item=item,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                    continue

            successful_rows.append({**item, **label_payload})

        self._append_cache_rows(cache_writes)

        scored_by_record_id = self._score_successful_rows(successful_rows)
        ordered_results = []
        for item in normalized_products:
            record_id = item["record_id"]
            if record_id in scored_by_record_id:
                ordered_results.append(scored_by_record_id[record_id])
            elif record_id in error_by_record_id:
                ordered_results.append(error_by_record_id[record_id])

        for input_error in input_errors:
            if input_error["record_id"] not in {row["record_id"] for row in ordered_results}:
                ordered_results.append(input_error)

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "pipeline": {
                "mode": "deploy",
                "module": "experiment_trust_fake_reviews.deploy_pipeline",
                "ollama_model": self.config.ollama_model,
                "artifacts_dir": str(self.artifacts_dir),
                "cache_path": str(self.cache_path),
                "artifacts": {
                    "graph_model": str(self.graph_model_path),
                    "logistic_model": str(self.logistic_model_path),
                },
            },
            "environment": environment,
            "results": ordered_results,
            "summary": self._build_summary(ordered_results),
        }

    def _load_models(self) -> None:
        if self._graph_model is None:
            self._graph_model = self._load_graph_model(self.graph_model_path)
        if self._logistic_artifact is None:
            self._logistic_artifact = self._load_logistic_artifact(self.logistic_model_path)

    def _fetch_ollama_models(self) -> list[str]:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        models = []
        for entry in payload.get("models", []):
            if not isinstance(entry, dict):
                continue
            model_name = entry.get("model") or entry.get("name")
            if model_name:
                models.append(str(model_name))
        return sorted(set(models))

    def _list_ollama_models_via_cli(self, ollama_binary: str) -> list[str]:
        result = subprocess.run(
            [ollama_binary, "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "unknown cli error")
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) <= 1:
            return []
        models = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                models.append(parts[0])
        return sorted(set(models))

    def _load_graph_model(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            "class_probs": {int(key): float(value) for key, value in payload["class_probs"].items()},
            "cpds": {
                column: {
                    int(class_key): {
                        bucket: float(probability)
                        for bucket, probability in bucket_map.items()
                    }
                    for class_key, bucket_map in class_map.items()
                }
                for column, class_map in payload["cpds"].items()
            },
            "bucket_columns": list(payload.get("bucket_columns", GRAPH_BUCKET_COLUMNS)),
        }

    def _load_logistic_artifact(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"feature_columns", "coef", "intercept", "classes"}
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Invalid logistic artifact at {path}: missing {missing}")
        return payload

    def _load_cache_by_id(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        cache_rows = []
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                cache_rows.append(json.loads(line))
        return {
            str(row["record_id"]): row
            for row in cache_rows
            if "record_id" in row and _is_valid_cached_label_record(row)
        }

    def _append_cache_rows(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _normalize_products(self, products: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        raw_items = self._coerce_product_items(products)
        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}

        for index, raw_item in enumerate(raw_items):
            try:
                normalized_item = self._normalize_single_product(raw_item)
                base_record_id = normalized_item["record_id"]
                count = seen_ids.get(base_record_id, 0)
                seen_ids[base_record_id] = count + 1
                if count:
                    normalized_item["record_id"] = f"{base_record_id}__{count + 1}"
                normalized.append(normalized_item)
            except Exception as exc:  # noqa: BLE001
                record_id = f"invalid_product_{index}"
                errors.append(
                    {
                        "record_id": record_id,
                        "status": "error",
                        "input": {"index": index},
                        "labels": None,
                        "scores": None,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
        return normalized, errors

    def _coerce_product_items(self, products: Any) -> list[Any]:
        if isinstance(products, pd.DataFrame):
            return products.to_dict(orient="records")
        if isinstance(products, Mapping):
            if "products" in products and isinstance(products["products"], list):
                return list(products["products"])
            return [dict(products)]
        if hasattr(products, "__dataclass_fields__"):
            return [asdict(products)]
        if isinstance(products, Iterable) and not isinstance(products, (str, bytes)):
            items = list(products)
            if items and all(hasattr(item, "__dataclass_fields__") for item in items):
                return [asdict(item) for item in items]
            return items
        raise TypeError("products must be a mapping, DataFrame, or iterable of product-like mappings")

    def _normalize_single_product(self, raw_item: Any) -> dict[str, Any]:
        if hasattr(raw_item, "__dataclass_fields__"):
            item = asdict(raw_item)
        elif isinstance(raw_item, Mapping):
            item = dict(raw_item)
        else:
            raise TypeError("Each product must be a mapping or dataclass instance.")

        title = self._pick_first(item, "title", "TITLE")
        bullet_points = self._pick_first(item, "bullet_points", "BULLET_POINTS")
        description = self._pick_first(item, "description", "DESCRIPTION")
        raw_text = self._pick_first(item, "text", "raw_text", "content")
        product_id = self._pick_first(item, "product_id", "PRODUCT_ID", "id")
        product_type_id = self._pick_first(item, "product_type_id", "PRODUCT_TYPE_ID")
        record_id = self._pick_first(item, "record_id")

        if any([title, bullet_points, description]):
            text = build_product_text(
                pd.Series(
                    {
                        "TITLE": title,
                        "BULLET_POINTS": bullet_points,
                        "DESCRIPTION": description,
                    }
                )
            )
        else:
            text = _clean_text(raw_text)

        if not text:
            raise ValueError(
                "Product input must include either `text` or at least one of "
                "`title`, `bullet_points`, `description`."
            )

        if not record_id:
            record_id = str(product_id) if product_id not in {None, ""} else self._build_record_id(item, text)

        return {
            "record_id": str(record_id),
            "product_id": None if product_id in {None, ""} else str(product_id),
            "product_type_id": product_type_id,
            "title": title,
            "bullet_points": bullet_points,
            "description": description,
            "text": text,
        }

    def _pick_first(self, payload: Mapping[str, Any], *keys: str) -> Any | None:
        for key in keys:
            if key not in payload:
                continue
            value = payload[key]
            if value is None:
                continue
            cleaned = _clean_text(value)
            if cleaned:
                return cleaned
        return None

    def _build_record_id(self, item: Mapping[str, Any], text: str) -> str:
        digest_source = json.dumps(
            {
                "title": self._pick_first(item, "title", "TITLE"),
                "bullet_points": self._pick_first(item, "bullet_points", "BULLET_POINTS"),
                "description": self._pick_first(item, "description", "DESCRIPTION"),
                "text": text,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
        return f"product_{digest}"

    def _score_successful_rows(self, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not rows:
            return {}

        frame = pd.DataFrame(rows)
        prepared = discretize_label_columns(frame)
        graph_scores = predict_naive_bayes_graph(self._graph_model, prepared)
        logistic_scores = self._predict_logistic_probabilities(prepared)

        prepared["phase_b_truth_likelihood_graph"] = graph_scores
        prepared["phase_b_truth_likelihood_logistic"] = logistic_scores
        prepared["trust_risk_index_graph"] = 1.0 - prepared["phase_b_truth_likelihood_graph"]
        prepared["trust_risk_index_logistic"] = 1.0 - prepared["phase_b_truth_likelihood_logistic"]
        prepared["graph_uncertainty_entropy"] = prepared["phase_b_truth_likelihood_graph"].apply(
            entropy_binary
        )

        result_by_record_id = {}
        for row in prepared.to_dict(orient="records"):
            result_by_record_id[str(row["record_id"])] = {
                "record_id": str(row["record_id"]),
                "status": "ok",
                "input": {
                    "product_id": row.get("product_id"),
                    "product_type_id": row.get("product_type_id"),
                    "title": row.get("title"),
                    "bullet_points": row.get("bullet_points"),
                    "description": row.get("description"),
                    "text": row.get("text"),
                },
                "labels": {
                    "claim_trust_score": float(row["claim_trust_score"]),
                    "signal_trust_score": float(row["signal_trust_score"]),
                    "heuristic_pressure_score": float(row["heuristic_pressure_score"]),
                    "competence_score": float(row["competence_score"]),
                    "benevolence_score": float(row["benevolence_score"]),
                    "integrity_score": float(row["integrity_score"]),
                    "predictability_score": float(row["predictability_score"]),
                    "claim_trust_bucket": row["claim_trust_bucket"],
                    "signal_trust_bucket": row["signal_trust_bucket"],
                    "heuristic_pressure_bucket": row["heuristic_pressure_bucket"],
                    "competence_bucket": row["competence_bucket"],
                    "benevolence_bucket": row["benevolence_bucket"],
                    "integrity_bucket": row["integrity_bucket"],
                    "predictability_bucket": row["predictability_bucket"],
                    "rationale_claim": row["rationale_claim"],
                    "rationale_signal": row["rationale_signal"],
                    "rationale_pressure": row["rationale_pressure"],
                    "overall_confidence": float(row["overall_confidence"]),
                },
                "scores": {
                    "phase_b_truth_likelihood_graph": float(row["phase_b_truth_likelihood_graph"]),
                    "phase_b_truth_likelihood_logistic": float(
                        row["phase_b_truth_likelihood_logistic"]
                    ),
                    "trust_risk_index_graph": float(row["trust_risk_index_graph"]),
                    "trust_risk_index_logistic": float(row["trust_risk_index_logistic"]),
                    "graph_uncertainty_entropy": float(row["graph_uncertainty_entropy"]),
                },
                "error": None,
            }
        return result_by_record_id

    def _predict_logistic_probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        feature_columns = self._logistic_artifact["feature_columns"]
        coefficient_matrix = np.asarray(self._logistic_artifact["coef"], dtype=float)
        intercept = np.asarray(self._logistic_artifact["intercept"], dtype=float)
        features = frame[feature_columns].astype(float).to_numpy()
        logits = features @ coefficient_matrix.T + intercept
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return 1.0 / (1.0 + np.exp(-logits))

    def _build_item_error(
        self,
        *,
        item: Mapping[str, Any],
        error_type: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "record_id": str(item["record_id"]),
            "status": "error",
            "input": {
                "product_id": item.get("product_id"),
                "product_type_id": item.get("product_type_id"),
                "title": item.get("title"),
                "bullet_points": item.get("bullet_points"),
                "description": item.get("description"),
                "text": item.get("text"),
            },
            "labels": None,
            "scores": None,
            "error": {
                "type": error_type,
                "message": message,
            },
        }

    def _build_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [row for row in results if row["status"] == "ok"]
        failed = [row for row in results if row["status"] != "ok"]
        if successful:
            average_graph_truth = float(
                sum(row["scores"]["phase_b_truth_likelihood_graph"] for row in successful)
                / len(successful)
            )
            average_graph_risk = float(
                sum(row["scores"]["trust_risk_index_graph"] for row in successful) / len(successful)
            )
            highest_risk = max(successful, key=lambda row: row["scores"]["trust_risk_index_graph"])
            highest_entropy = max(
                successful,
                key=lambda row: row["scores"]["graph_uncertainty_entropy"],
            )
        else:
            average_graph_truth = None
            average_graph_risk = None
            highest_risk = None
            highest_entropy = None

        return {
            "total_products": len(results),
            "successful_products": len(successful),
            "failed_products": len(failed),
            "average_phase_b_truth_likelihood_graph": average_graph_truth,
            "average_trust_risk_index_graph": average_graph_risk,
            "highest_graph_risk_record_id": None if highest_risk is None else highest_risk["record_id"],
            "highest_entropy_record_id": None if highest_entropy is None else highest_entropy["record_id"],
        }

    def _build_error_response(self, environment: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "pipeline": {
                "mode": "deploy",
                "module": "experiment_trust_fake_reviews.deploy_pipeline",
                "ollama_model": self.config.ollama_model,
                "artifacts_dir": str(self.artifacts_dir),
                "cache_path": str(self.cache_path),
            },
            "environment": environment,
            "results": [],
            "summary": {
                "total_products": 0,
                "successful_products": 0,
                "failed_products": 0,
                "average_phase_b_truth_likelihood_graph": None,
                "average_trust_risk_index_graph": None,
                "highest_graph_risk_record_id": None,
                "highest_entropy_record_id": None,
            },
        }

    def _format_environment_error(self, environment: dict[str, Any]) -> str:
        lines = ["Deployment environment is not ready."]
        for error in environment.get("errors", []):
            lines.append(f"- {error}")
        suggested = environment.get("suggested_commands", [])
        if suggested:
            lines.append("Suggested fixes:")
            for command in suggested:
                lines.append(f"- {command}")
        return "\n".join(lines)


def run_deployment_pipeline(
    products: Any,
    *,
    config: DeployConfig | None = None,
    raise_on_environment_error: bool = True,
) -> dict[str, Any]:
    pipeline = TrustFakeReviewsDeployPipeline(config=config)
    return pipeline.run(products, raise_on_environment_error=raise_on_environment_error)


def _load_products_from_json_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "products" in payload:
        return payload["products"]
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fake-review-calibrated trust deployment pipeline on one or more products."
    )
    parser.add_argument("--input", type=str, help="Path to a JSON file containing a product or list of products.")
    parser.add_argument("--stdin", action="store_true", help="Read JSON payload from stdin.")
    parser.add_argument(
        "--product-json",
        action="append",
        default=[],
        help="Inline JSON for a single product. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=str, help="Optional path to write the output JSON.")
    parser.add_argument("--check-env", action="store_true", help="Only validate artifacts and Ollama runtime.")
    parser.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model to use.")
    parser.add_argument("--cache-path", type=str, help="Optional JSONL cache path for successful labels.")
    parser.add_argument("--artifacts-dir", type=str, help="Optional override for deploy artifacts directory.")
    args = parser.parse_args(argv)

    if not args.check_env and not any([args.input, args.stdin, args.product_json]):
        parser.error("Provide one of --input, --stdin, or --product-json, or use --check-env.")
    return args


def _write_output(payload: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if output_path:
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = DeployConfig(
        ollama_model=args.model,
        artifacts_dir=None if args.artifacts_dir is None else Path(args.artifacts_dir),
        cache_path=None if args.cache_path is None else Path(args.cache_path),
    )
    pipeline = TrustFakeReviewsDeployPipeline(config=config)

    if args.check_env:
        environment = pipeline.validate_environment()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "environment": environment,
        }
        _write_output(payload, args.output)
        return 0 if environment["ok"] else 1

    raw_products: list[Any] = []
    if args.input:
        payload = json.loads(Path(args.input).expanduser().resolve().read_text(encoding="utf-8"))
        loaded = _load_products_from_json_payload(payload)
        if isinstance(loaded, list):
            raw_products.extend(loaded)
        else:
            raw_products.append(loaded)

    if args.stdin:
        payload = json.load(sys.stdin)
        loaded = _load_products_from_json_payload(payload)
        if isinstance(loaded, list):
            raw_products.extend(loaded)
        else:
            raw_products.append(loaded)

    for product_json in args.product_json:
        raw_products.append(json.loads(product_json))

    result = pipeline.run(raw_products, raise_on_environment_error=False)
    _write_output(result, args.output)
    return 0 if result["environment"]["ok"] and result["summary"]["failed_products"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
