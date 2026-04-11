from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.entities.candidate import Candidate
from value.listing_trust import (
    build_candidate_listing_payload,
    build_listing_trust_runtime_status,
    score_candidate_listing_trust,
)


class FakeTrustPipeline:
    def validate_environment(self) -> dict:
        return {
            "ok": True,
            "ollama": {
                "binary_path": "/usr/bin/ollama",
                "service_reachable": True,
                "cli_reachable": True,
                "api_models": ["llama3.1:8b"],
                "cli_models": ["llama3.1:8b"],
                "available_models": ["llama3.1:8b"],
                "model_required": "llama3.1:8b",
                "model_present": True,
                "api_error": None,
                "cli_error": None,
            },
            "errors": [],
            "warnings": [],
        }

    def run(self, products, *, raise_on_environment_error=False) -> dict:
        assert raise_on_environment_error is False
        return {
            "environment": self.validate_environment(),
            "results": [
                {
                    "status": "ok",
                    "input": products[0],
                    "labels": {
                        "overall_confidence": 0.78,
                    },
                    "scores": {
                        "phase_b_truth_likelihood_graph": 0.82,
                        "phase_b_truth_likelihood_logistic": 0.64,
                        "trust_risk_index_graph": 0.18,
                        "trust_risk_index_logistic": 0.36,
                        "graph_uncertainty_entropy": 0.21,
                    },
                    "error": None,
                }
            ],
        }


class FakeGraphOnlyTrustPipeline(FakeTrustPipeline):
    def run(self, products, *, raise_on_environment_error=False) -> dict:
        payload = super().run(products, raise_on_environment_error=raise_on_environment_error)
        payload["results"][0]["scores"].pop("phase_b_truth_likelihood_logistic", None)
        payload["results"][0]["scores"].pop("trust_risk_index_logistic", None)
        return payload


def test_build_candidate_listing_payload_uses_structured_listing_content():
    candidate = Candidate(
        source_url="https://www.ebay.com.sg/itm/123",
        page_type="listing",
        product_id="123",
        title="USB-C Charger",
        condition="New",
        listing_bullet_points=["Brand: Anker", "Power: 30W"],
        listing_description="Compact fast charger for travel.",
    )

    payload = build_candidate_listing_payload(candidate)

    assert payload["product_id"] == "123"
    assert payload["title"] == "USB-C Charger"
    assert payload["bullet_points"] == "Brand: Anker; Power: 30W"
    assert payload["description"] == "Compact fast charger for travel."


@patch("value.listing_trust._get_trust_pipeline", return_value=FakeTrustPipeline())
def test_score_candidate_listing_trust_returns_normalized_probabilities(_pipeline_mock):
    candidate = Candidate(
        source_url="https://www.ebay.com.sg/itm/123",
        page_type="listing",
        product_id="123",
        title="USB-C Charger",
        listing_bullet_points=["Brand: Anker", "Power: 30W"],
        listing_description="Compact fast charger for travel.",
    )

    result = score_candidate_listing_trust(candidate)

    assert result["status"] == "ok"
    assert result["trust_probability"] == 0.64
    assert result["listing_trust_probability"] == 0.64
    assert result["trust_probability_graph"] == 0.82
    assert result["trust_probability_logistic"] == 0.64
    assert result["deception_probability"] == 0.36
    assert result["deception_probability_graph"] == 0.18
    assert result["deception_probability_logistic"] == 0.36
    assert result["title_deception_score"] == 0.36
    assert result["score_head"] == "logistic"
    assert result["overall_confidence"] == 0.78


@patch(
    "value.listing_trust._get_trust_pipeline",
    return_value=FakeGraphOnlyTrustPipeline(),
)
def test_score_candidate_listing_trust_falls_back_to_graph_when_logistic_is_missing(
    _pipeline_mock,
):
    candidate = Candidate(
        source_url="https://www.ebay.com.sg/itm/123",
        page_type="listing",
        product_id="123",
        title="USB-C Charger",
        listing_bullet_points=["Brand: Anker", "Power: 30W"],
        listing_description="Compact fast charger for travel.",
    )

    result = score_candidate_listing_trust(candidate)

    assert result["trust_probability"] == 0.82
    assert result["deception_probability"] == 0.18
    assert result["trust_probability_graph"] == 0.82
    assert result["trust_probability_logistic"] is None
    assert result["score_head"] == "graph"


@patch("value.listing_trust._get_trust_pipeline", return_value=FakeTrustPipeline())
def test_build_listing_trust_runtime_status_exposes_pipeline_readiness(_pipeline_mock):
    status = build_listing_trust_runtime_status()

    assert status["ready"] is True
    assert status["provider"] == "ollama"
    assert status["model_required"] == "llama3.1:8b"
    assert status["model_present"] is True
