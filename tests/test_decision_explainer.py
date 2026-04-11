import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import value.decision_explainer as decision_explainer
from value.decision_explainer import (
    OllamaExplanationConfig,
    build_fallback_decision_explanation,
    build_ollama_runtime_status,
    normalize_decision_explanation_payload,
)


def _sample_comparison_payload(*, verdict: str = "better_A", price_mode: str = "peer_price"):
    return {
        "listing_a": {
            "title": "Canon EOS R50 Mirrorless Camera Body",
            "source_url": "https://www.ebay.com.sg/itm/111",
            "prediction": "good_value",
            "good_value_probability": 0.71,
            "total_price": 899.0,
            "total_price_currency": "USD",
            "peer_price": 1049.0,
            "price_gap_vs_peer": 0.1430,
            "trust_probability": 0.86,
            "ewom_score_0_to_100": 82.4,
            "seller_feedback_review_count": 18,
            "retrieval_status": "usable",
            "retrieved_neighbor_count": 6,
            "price_considered": True,
            "price_note": None,
            "peer_price_source": "retrieval",
        },
        "listing_b": {
            "title": "Canon EOS R50 Camera Kit",
            "source_url": "https://www.ebay.com.sg/itm/222",
            "prediction": "not_good_value",
            "good_value_probability": 0.56,
            "total_price": 979.0,
            "total_price_currency": "USD",
            "peer_price": 1002.0,
            "price_gap_vs_peer": 0.0230,
            "trust_probability": 0.75,
            "ewom_score_0_to_100": 71.2,
            "seller_feedback_review_count": 11,
            "retrieval_status": "usable",
            "retrieved_neighbor_count": 5,
            "price_considered": True,
            "price_note": None,
            "peer_price_source": "retrieval",
        },
        "comparison": {
            "verdict": verdict,
            "price_comparison_mode": price_mode,
            "good_value_probability_delta": 0.15,
            "tie_margin_used": 0.03,
            "reasons": [
                "Listing A holds the stronger overall value read.",
                "Listing A is cheaper relative to similar listings.",
            ],
        },
    }


def test_normalize_decision_explanation_payload_accepts_valid_long_form():
    paragraph = (
        "This paragraph is intentionally long enough to behave like real output from a local "
        "language model. It explains the decision in plain buyer-facing language, connects the "
        "headline call back to the available evidence, and avoids collapsing into a thin one-line summary."
    )
    payload = {
        "headline": "Listing A is the stronger buy",
        "summary": "Listing A wins because the overall value case is stronger and better supported.",
        "paragraphs": [paragraph, paragraph, paragraph, paragraph],
        "watchouts": [
            "Verify condition and accessories on the listing page.",
            "Check shipping, returns, and seller handling details.",
            "Confirm the cheaper listing is not missing anything important.",
        ],
    }

    normalized = normalize_decision_explanation_payload(payload)

    assert normalized["title"] == "Listing A is the stronger buy"
    assert normalized["lead"].startswith("Listing A wins")
    assert len(normalized["paragraphs"]) == 4
    assert len(normalized["watchouts"]) == 3


def test_build_fallback_decision_explanation_mentions_neutral_price_mode():
    explanation = build_fallback_decision_explanation(
        _sample_comparison_payload(price_mode="neutral_fallback"),
        model="llama3.1:8b",
    )

    assert explanation["status"] == "fallback"
    assert explanation["provider"] == "ollama"
    assert explanation["model"] == "llama3.1:8b"
    assert len(explanation["paragraphs"]) == 4
    assert "price needs to be read carefully" in explanation["paragraphs"][1].lower()
    assert explanation["watchouts"][2].startswith("Do a manual price sanity check")


def test_build_ollama_runtime_status_marks_ready_when_model_is_available(monkeypatch):
    config = OllamaExplanationConfig(
        model="llama3.1:8b",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=30,
        keepalive="10m",
        max_output_tokens=700,
        context_tokens=4096,
    )

    monkeypatch.setattr(decision_explainer.shutil, "which", lambda _: "/usr/bin/ollama")
    monkeypatch.setattr(
        decision_explainer,
        "_fetch_ollama_models",
        lambda **kwargs: ["llama3.1:8b", "qwen2.5:7b"],
    )
    monkeypatch.setattr(
        decision_explainer,
        "_list_ollama_models_via_cli",
        lambda _binary: ["llama3.1:8b"],
    )

    status = build_ollama_runtime_status(config)

    assert status["ready"] is True
    assert status["service_reachable"] is True
    assert status["cli_reachable"] is True
    assert status["model_present"] is True
