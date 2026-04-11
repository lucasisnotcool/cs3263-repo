from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_KEEPALIVE = "10m"
DEFAULT_MAX_OUTPUT_TOKENS = 700
DEFAULT_CONTEXT_TOKENS = 4096


@dataclass(frozen=True)
class OllamaExplanationConfig:
    model: str
    base_url: str
    timeout_seconds: int
    keepalive: str
    max_output_tokens: int
    context_tokens: int

    @classmethod
    def from_env(cls) -> "OllamaExplanationConfig":
        return cls(
            model=os.environ.get("OLLAMA_EXPLANATION_MODEL", DEFAULT_OLLAMA_MODEL).strip()
            or DEFAULT_OLLAMA_MODEL,
            base_url=(
                os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
                or DEFAULT_OLLAMA_BASE_URL
            ).rstrip("/"),
            timeout_seconds=_read_int_env(
                "OLLAMA_EXPLANATION_TIMEOUT_SECONDS",
                DEFAULT_TIMEOUT_SECONDS,
            ),
            keepalive=os.environ.get("OLLAMA_EXPLANATION_KEEPALIVE", DEFAULT_KEEPALIVE).strip()
            or DEFAULT_KEEPALIVE,
            max_output_tokens=_read_int_env(
                "OLLAMA_EXPLANATION_MAX_OUTPUT_TOKENS",
                DEFAULT_MAX_OUTPUT_TOKENS,
            ),
            context_tokens=_read_int_env(
                "OLLAMA_EXPLANATION_CONTEXT_TOKENS",
                DEFAULT_CONTEXT_TOKENS,
            ),
        )


def build_ollama_runtime_status(
    config: OllamaExplanationConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or OllamaExplanationConfig.from_env()
    ollama_binary = shutil.which("ollama")
    api_models: list[str] = []
    cli_models: list[str] = []
    api_error: str | None = None
    cli_error: str | None = None
    service_reachable = False
    cli_reachable = False

    try:
        api_models = _fetch_ollama_models(
            base_url=resolved_config.base_url,
            timeout_seconds=min(5, resolved_config.timeout_seconds),
        )
        service_reachable = True
    except Exception as exc:  # noqa: BLE001
        api_error = str(exc)

    if ollama_binary:
        try:
            cli_models = _list_ollama_models_via_cli(ollama_binary)
            cli_reachable = True
        except Exception as exc:  # noqa: BLE001
            cli_error = str(exc)
    else:
        cli_error = "Ollama binary not found on PATH."

    available_models = sorted(set(api_models + cli_models))
    model_present = resolved_config.model in available_models if available_models else False
    ready = model_present and (service_reachable or cli_reachable)

    return {
        "ready": ready,
        "provider": "ollama",
        "base_url": resolved_config.base_url,
        "binary_path": ollama_binary,
        "service_reachable": service_reachable,
        "cli_reachable": cli_reachable,
        "api_models": api_models,
        "cli_models": cli_models,
        "available_models": available_models,
        "model_required": resolved_config.model,
        "model_present": model_present,
        "api_error": api_error,
        "cli_error": cli_error,
    }


def resolve_decision_explanation(
    comparison_payload: Mapping[str, Any],
    *,
    config: OllamaExplanationConfig | None = None,
    runtime_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_config = config or OllamaExplanationConfig.from_env()
    resolved_runtime = (
        dict(runtime_status)
        if isinstance(runtime_status, Mapping)
        else build_ollama_runtime_status(resolved_config)
    )

    if resolved_runtime.get("ready"):
        try:
            return generate_decision_explanation(
                comparison_payload,
                config=resolved_config,
            )
        except Exception as exc:  # noqa: BLE001
            return build_fallback_decision_explanation(
                comparison_payload,
                model=resolved_config.model,
                note=(
                    "Ollama could not finish the long explanation, so this version was "
                    "assembled from the comparison signals instead."
                ),
                error=str(exc),
            )

    reason = "Ollama is not ready locally yet."
    if not resolved_runtime.get("model_present"):
        reason = (
            f"The configured Ollama model `{resolved_config.model}` is not available "
            "locally yet."
        )
    elif not (
        resolved_runtime.get("service_reachable") or resolved_runtime.get("cli_reachable")
    ):
        reason = "The local Ollama service is not reachable right now."

    return build_fallback_decision_explanation(
        comparison_payload,
        model=resolved_config.model,
        note=f"{reason} This explanation was assembled from the structured comparison output.",
    )


def generate_decision_explanation(
    comparison_payload: Mapping[str, Any],
    *,
    config: OllamaExplanationConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or OllamaExplanationConfig.from_env()
    evidence = build_decision_explanation_evidence(comparison_payload)
    prompt = build_decision_explanation_prompt(evidence)
    raw_payload = _run_ollama_json(prompt=prompt, config=resolved_config)
    explanation = normalize_decision_explanation_payload(raw_payload)
    explanation["status"] = "generated"
    explanation["provider"] = "ollama"
    explanation["model"] = resolved_config.model
    explanation["note"] = f"Generated locally with Ollama using `{resolved_config.model}`."
    return explanation


def build_decision_explanation_evidence(
    comparison_payload: Mapping[str, Any],
) -> dict[str, Any]:
    listing_a = _mapping(comparison_payload.get("listing_a"))
    listing_b = _mapping(comparison_payload.get("listing_b"))
    comparison = _mapping(comparison_payload.get("comparison"))
    return {
        "listing_a": _compact_listing_summary(listing_a),
        "listing_b": _compact_listing_summary(listing_b),
        "comparison": {
            "verdict": str(comparison.get("verdict") or "insufficient_evidence"),
            "price_comparison_mode": str(
                comparison.get("price_comparison_mode") or "unknown"
            ),
            "good_value_probability_delta": _round_number(
                comparison.get("good_value_probability_delta"),
                digits=4,
            ),
            "tie_margin_used": _round_number(
                comparison.get("tie_margin_used"),
                digits=4,
            ),
            "reasons": [
                str(reason).strip()
                for reason in comparison.get("reasons", [])
                if str(reason).strip()
            ]
            if isinstance(comparison.get("reasons"), list)
            else [],
        },
    }


def build_decision_explanation_prompt(evidence: Mapping[str, Any]) -> str:
    evidence_json = json.dumps(evidence, indent=2, ensure_ascii=False)
    return f"""
You are writing the final buyer-facing explanation for an eBay comparison interface.
Use only the facts in EVIDENCE_JSON and return valid JSON.

Return an object with exactly these keys:
{{
  "title": "short headline",
  "lead": "1-2 sentence opening summary",
  "paragraphs": [
    "paragraph 1",
    "paragraph 2",
    "paragraph 3",
    "paragraph 4"
  ],
  "watchouts": [
    "short actionable check 1",
    "short actionable check 2",
    "short actionable check 3"
  ]
}}

Requirements:
- Write in plain English for a buyer, not for an engineer.
- Make the explanation long and detailed overall, roughly 220 to 380 words.
- `paragraphs` must contain exactly 4 substantial paragraphs.
- Each paragraph should contain 2 to 4 sentences.
- Explain the final decision first, then price versus market, then trust and buyer-signal support, then uncertainty and what could still change the call.
- If the comparison is a tie or lacks evidence, say that directly and avoid fake certainty.
- If `price_comparison_mode` is `neutral_fallback`, explicitly say that price advantage was not reliable enough to drive the decision.
- Do not invent missing item details, shipping promises, warranty terms, or condition claims.
- Do not mention internal pipeline terms such as Bayesian scoring, retrieval candidate pools, or JSON.
- Keep each `watchouts` item under 20 words.

EVIDENCE_JSON:
{evidence_json}
""".strip()


def normalize_decision_explanation_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")

    title = _pick_first_text(payload, "title", "headline", "heading")
    lead = _pick_first_text(payload, "lead", "summary", "intro")
    paragraphs = _normalize_text_list(
        _pick_first(payload, "paragraphs", "body", "narrative", "sections")
    )
    watchouts = _normalize_text_list(
        _pick_first(payload, "watchouts", "cautions", "risks", "checks")
    )

    word_count = sum(len(paragraph.split()) for paragraph in paragraphs)
    if not title:
        raise ValueError("Explanation payload is missing a title.")
    if not lead:
        raise ValueError("Explanation payload is missing a lead.")
    if len(paragraphs) < 4:
        raise ValueError("Explanation payload must contain at least 4 paragraphs.")
    if word_count < 120:
        raise ValueError("Explanation payload is too short to be useful.")
    if len(watchouts) < 3:
        raise ValueError("Explanation payload must contain at least 3 watchouts.")

    return {
        "title": title,
        "lead": lead,
        "paragraphs": paragraphs[:4],
        "watchouts": watchouts[:3],
    }


def build_fallback_decision_explanation(
    comparison_payload: Mapping[str, Any],
    *,
    model: str | None = None,
    note: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    evidence = build_decision_explanation_evidence(comparison_payload)
    listing_a = evidence["listing_a"]
    listing_b = evidence["listing_b"]
    comparison = evidence["comparison"]
    verdict = str(comparison.get("verdict") or "insufficient_evidence")

    title, lead = _build_fallback_heading_and_lead(listing_a, listing_b, verdict)
    paragraphs = [
        _build_fallback_decision_paragraph(listing_a, listing_b, comparison),
        _build_fallback_price_paragraph(listing_a, listing_b, comparison),
        _build_fallback_signal_paragraph(listing_a, listing_b),
        _build_fallback_caution_paragraph(listing_a, listing_b, comparison),
    ]

    response = {
        "status": "fallback",
        "provider": "ollama",
        "model": model or DEFAULT_OLLAMA_MODEL,
        "title": title,
        "lead": lead,
        "paragraphs": paragraphs,
        "watchouts": _build_fallback_watchouts(listing_a, listing_b, comparison),
        "note": note
        or "Ollama was unavailable, so this explanation was assembled from the comparison signals.",
    }
    if error:
        response["error"] = error
    return response


def _build_fallback_heading_and_lead(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    verdict: str,
) -> tuple[str, str]:
    title_a = str(listing_a.get("title") or "Listing A")
    title_b = str(listing_b.get("title") or "Listing B")

    if verdict == "better_A":
        return (
            "Listing A has the stronger overall buying case",
            f"{title_a} comes out ahead in the current comparison. The edge is not just about a single number; it reflects the combined read across value, trust, and the strength of the supporting market evidence.",
        )
    if verdict == "better_B":
        return (
            "Listing B has the stronger overall buying case",
            f"{title_b} comes out ahead in the current comparison. The recommendation is based on the full mix of value, trust, and supporting buyer-signal evidence rather than a narrow price-only read.",
        )
    if verdict == "tie":
        return (
            "The two listings are effectively in a near tie",
            f"{title_a} and {title_b} land close enough together that the comparison does not justify forcing a winner. The safer reading is that both options remain viable until you break the tie with more listing-specific details.",
        )
    return (
        "The comparison needs more evidence before making a clean call",
        "The current result does not have enough support to make a confident final recommendation. There are still useful signals in the comparison, but they should be treated as directional rather than decisive.",
    )


def _build_fallback_decision_paragraph(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    verdict = str(comparison.get("verdict") or "insufficient_evidence")
    delta = _round_number(comparison.get("good_value_probability_delta"), digits=4)
    delta_text = _format_ratio(delta, signed=False)
    title_a = str(listing_a.get("title") or "Listing A")
    title_b = str(listing_b.get("title") or "Listing B")
    probability_a = _format_probability(listing_a.get("good_value_probability"))
    probability_b = _format_probability(listing_b.get("good_value_probability"))

    if verdict == "better_A":
        return (
            f"{title_a} currently carries the higher overall value score at {probability_a}, "
            f"while {title_b} sits at {probability_b}. That leaves Listing A with an edge of "
            f"about {delta_text}, which is meaningful enough to break the tie in its favor without "
            "looking like a random fluctuation."
        )
    if verdict == "better_B":
        return (
            f"{title_b} currently carries the higher overall value score at {probability_b}, "
            f"while {title_a} sits at {probability_a}. That gives Listing B an advantage of about "
            f"{delta_text}, which is enough to justify preferring it over the alternative in this run."
        )
    if verdict == "tie":
        return (
            f"The headline scores are extremely close, with {title_a} at {probability_a} and "
            f"{title_b} at {probability_b}. When the gap is that small, the result should be read "
            "as a practical draw rather than a reliable signal that one listing is materially better."
        )
    return (
        f"The comparison does not produce a dependable winner because the evidence is still incomplete. "
        f"{title_a} reads at {probability_a} and {title_b} reads at {probability_b}, but the system does "
        "not treat that as enough support for a confident final recommendation."
    )


def _build_fallback_price_paragraph(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    price_mode = str(comparison.get("price_comparison_mode") or "unknown")
    if price_mode == "neutral_fallback":
        return (
            "Price needs to be read carefully here because at least one listing did not have a strong enough "
            "similar-item benchmark. In practice, that means the final recommendation does not fully trust any "
            "headline price advantage, so you should treat raw asking price as something to verify manually on "
            "the listing page instead of assuming the cheaper-looking option is automatically the better deal."
        )

    title_a = str(listing_a.get("title") or "Listing A")
    title_b = str(listing_b.get("title") or "Listing B")
    price_a = _format_currency(listing_a.get("total_price"), listing_a.get("total_price_currency"))
    price_b = _format_currency(listing_b.get("total_price"), listing_b.get("total_price_currency"))
    peer_a = _format_currency(listing_a.get("peer_price"), listing_a.get("total_price_currency"))
    peer_b = _format_currency(listing_b.get("peer_price"), listing_b.get("total_price_currency"))
    gap_a = _format_gap_phrase(listing_a.get("price_gap_vs_peer"))
    gap_b = _format_gap_phrase(listing_b.get("price_gap_vs_peer"))

    return (
        f"On the pricing side, {title_a} is currently listed around {price_a} against an estimated market read "
        f"of {peer_a}, which puts it {gap_a}. {title_b} is listed around {price_b} against an estimated market "
        f"read of {peer_b}, which puts it {gap_b}. That comparison matters because a listing that is cheaper than "
        "its peer set without collapsing on trust can justify paying attention even when the two options look similar at first glance."
    )


def _build_fallback_signal_paragraph(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
) -> str:
    title_a = str(listing_a.get("title") or "Listing A")
    title_b = str(listing_b.get("title") or "Listing B")
    trust_a = _format_probability(listing_a.get("trust_probability"))
    trust_b = _format_probability(listing_b.get("trust_probability"))
    ewom_a = _format_score(listing_a.get("ewom_score_0_to_100"))
    ewom_b = _format_score(listing_b.get("ewom_score_0_to_100"))
    comments_a = int(_round_number(listing_a.get("seller_feedback_review_count"), digits=0) or 0)
    comments_b = int(_round_number(listing_b.get("seller_feedback_review_count"), digits=0) or 0)

    return (
        f"Trust and review support also help explain the call. {title_a} shows listing trust around "
        f"{trust_a} with a review-signal score of {ewom_a}, backed by roughly {comments_a} review texts, while "
        f"{title_b} shows listing trust around {trust_b} with a review-signal score of {ewom_b}, backed by roughly "
        f"{comments_b} review texts. Those signals should not overpower everything else, but they do matter because "
        "a value-looking listing can still be a weaker purchase if the listing content reads poorly or the review evidence looks thin."
    )


def _build_fallback_caution_paragraph(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    verdict = str(comparison.get("verdict") or "insufficient_evidence")
    neighbors_a = int(_round_number(listing_a.get("retrieved_neighbor_count"), digits=0) or 0)
    neighbors_b = int(_round_number(listing_b.get("retrieved_neighbor_count"), digits=0) or 0)
    retrieval_a = str(listing_a.get("retrieval_status") or "limited")
    retrieval_b = str(listing_b.get("retrieval_status") or "limited")

    if verdict == "tie":
        return (
            f"The main reason to stay cautious is that both listings are landing in almost the same decision band. "
            f"Listing A is supported by {neighbors_a} market matches with a `{retrieval_a}` market read, and Listing B "
            f"is supported by {neighbors_b} market matches with a `{retrieval_b}` market read, so the remaining tie-breakers "
            "are likely to be condition details, included accessories, return policy, or any mismatch you notice on the listing pages directly."
        )
    if verdict == "insufficient_evidence":
        return (
            f"The biggest caveat is evidence quality. Listing A has {neighbors_a} market matches with a `{retrieval_a}` market "
            f"read, while Listing B has {neighbors_b} matches with a `{retrieval_b}` read, and that is not strong enough to treat "
            "the result as a locked-in final answer. Use the comparison to narrow the field, then verify the listing pages before buying."
        )
    return (
        f"The recommendation still deserves a final manual check because the model can only see structured price and trust signals, not every listing nuance. "
        f"Listing A is supported by {neighbors_a} market matches and Listing B by {neighbors_b}, which is helpful but still leaves room for photos, bundle contents, "
        "warranty wording, or condition notes to change which option is actually better for you."
    )


def _build_fallback_watchouts(
    listing_a: Mapping[str, Any],
    listing_b: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> list[str]:
    watchouts = [
        "Open both listing pages and verify condition, photos, and included accessories.",
        "Check shipping, returns, and handling details before paying.",
    ]
    if str(comparison.get("price_comparison_mode") or "") == "neutral_fallback":
        watchouts.append("Do a manual price sanity check because market matching was limited.")
    else:
        watchouts.append(
            "Confirm the cheaper-versus-market listing is not missing anything important."
        )
    return watchouts


def _run_ollama_json(
    *,
    prompt: str,
    config: OllamaExplanationConfig,
) -> dict[str, Any]:
    api_error: Exception | None = None
    request_payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": config.keepalive,
        "options": {
            "temperature": 0.2,
            "num_predict": int(config.max_output_tokens),
            "num_ctx": int(config.context_tokens),
        },
    }

    try:
        req = urllib.request.Request(
            f"{config.base_url}/api/generate",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        response_text = str(payload.get("response") or "").strip()
        if not response_text:
            raise ValueError("Empty response from Ollama API.")
        return _extract_json_object(response_text)
    except Exception as exc:  # noqa: BLE001
        api_error = exc

    ollama_binary = shutil.which("ollama")
    if ollama_binary is None:
        raise RuntimeError(
            "Ollama API call failed and the `ollama` binary is not available for CLI fallback."
        ) from api_error

    try:
        result = subprocess.run(
            [
                ollama_binary,
                "run",
                config.model,
                "--format",
                "json",
                "--hidethinking",
                "--keepalive",
                config.keepalive,
                prompt,
            ],
            capture_output=True,
            text=True,
            timeout=max(30, int(config.timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Ollama CLI fallback timed out after {max(30, int(config.timeout_seconds))} seconds."
        ) from exc

    if result.returncode != 0:
        api_context = f" API error: {api_error!s}" if api_error is not None else ""
        raise RuntimeError(
            f"Ollama CLI fallback failed with exit code {result.returncode}: "
            f"{result.stderr.strip()[:400]}{api_context}"
        )

    return _extract_json_object(result.stdout)


def _fetch_ollama_models(*, base_url: str, timeout_seconds: int) -> list[str]:
    req = urllib.request.Request(
        f"{base_url}/api/tags",
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=max(1, int(timeout_seconds))) as response:
        payload = json.loads(response.read().decode("utf-8"))

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for entry in models:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("model") or entry.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _list_ollama_models_via_cli(ollama_binary: str) -> list[str]:
    result = subprocess.run(
        [ollama_binary, "list"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama list failed.")

    models: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("name "):
            continue
        model = stripped.split()[0]
        if model:
            models.append(model)
    return models


def _compact_listing_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "title": str(summary.get("title") or "").strip() or None,
        "source_url": str(summary.get("source_url") or "").strip() or None,
        "prediction": str(summary.get("prediction") or "").strip() or None,
        "good_value_probability": _round_number(
            summary.get("good_value_probability"),
            digits=4,
        ),
        "total_price": _round_number(summary.get("total_price"), digits=2),
        "total_price_currency": str(summary.get("total_price_currency") or "").strip() or None,
        "peer_price": _round_number(summary.get("peer_price"), digits=2),
        "price_gap_vs_peer": _round_number(summary.get("price_gap_vs_peer"), digits=4),
        "trust_probability": _round_number(summary.get("trust_probability"), digits=4),
        "ewom_score_0_to_100": _round_number(summary.get("ewom_score_0_to_100"), digits=2),
        "seller_feedback_review_count": _round_number(
            summary.get("seller_feedback_review_count"),
            digits=0,
        ),
        "retrieval_status": str(summary.get("retrieval_status") or "").strip() or None,
        "retrieved_neighbor_count": _round_number(
            summary.get("retrieved_neighbor_count"),
            digits=0,
        ),
        "price_considered": bool(summary.get("price_considered"))
        if summary.get("price_considered") is not None
        else None,
        "price_note": str(summary.get("price_note") or "").strip() or None,
        "peer_price_source": str(summary.get("peer_price_source") or "").strip() or None,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        snippet = candidate[start : end + 1]
        parsed = json.loads(snippet)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from model output.")


def _pick_first(payload: Mapping[str, Any], *keys: str) -> Any | None:
    normalized = {str(key).strip().lower(): key for key in payload.keys()}
    for key in keys:
        if key in payload:
            return payload[key]
        match = normalized.get(key.lower())
        if match is not None:
            return payload[match]
    return None


def _pick_first_text(payload: Mapping[str, Any], *keys: str) -> str:
    value = _pick_first(payload, *keys)
    return _clean_text(value)


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("\n") if part.strip()]
        return [_clean_text(part) for part in parts if _clean_text(part)]
    if not isinstance(value, list):
        return []
    return [_clean_text(item) for item in value if _clean_text(item)]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _round_number(value: Any, *, digits: int) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, digits)


def _format_probability(value: Any) -> str:
    numeric = _round_number(value, digits=4)
    if numeric is None:
        return "unknown"
    return f"{numeric * 100:.0f}%"


def _format_ratio(value: Any, *, signed: bool) -> str:
    numeric = _round_number(value, digits=4)
    if numeric is None:
        return "unknown"
    percent = numeric * 100
    if signed and percent > 0:
        return f"+{percent:.0f}%"
    return f"{abs(percent):.0f}%"


def _format_score(value: Any) -> str:
    numeric = _round_number(value, digits=1)
    if numeric is None:
        return "unknown"
    return f"{numeric:.1f}/100"


def _format_currency(value: Any, currency: Any) -> str:
    numeric = _round_number(value, digits=2)
    if numeric is None:
        return "unknown"
    resolved_currency = str(currency or "").strip()
    if resolved_currency:
        return f"{resolved_currency} {numeric:.2f}"
    return f"{numeric:.2f}"


def _format_gap_phrase(value: Any) -> str:
    numeric = _round_number(value, digits=4)
    if numeric is None:
        return "without a reliable market gap estimate"
    percent = abs(numeric) * 100
    if percent < 0.1:
        return "roughly at market"
    if numeric > 0:
        return f"about {percent:.0f}% below its market read"
    return f"about {percent:.0f}% above its market read"


def _read_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return max(1, int(raw_value))
    except ValueError:
        return default
