import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

# Allow direct execution via `python scripts/run_normalization.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.external_clients.ebay.ebay_auth_client import EbayAuthClient
from infrastructure.external_clients.ebay.ebay_browse_client import EbayBrowseClient
from infrastructure.external_clients.ebay.ebay_feedback_client import EbayFeedbackClient
from infrastructure.external_clients.ebay.ebay_url_parser import EbayUrlParser
from core.services.normalization_service import NormalizationService
from value.ebay_value import (
    compare_ebay_candidate_value_results,
    score_ebay_candidate_value,
    summarize_candidate_market_context_k_sweep,
    summarize_ebay_candidate_value_result,
    sweep_candidate_market_context_k,
    write_candidate_k_sweep_plot,
)


AUTH_SCOPES = [
    "https://api.ebay.com/oauth/api_scope",
    "https://api.ebay.com/oauth/api_scope/commerce.feedback.readonly",
]
SELLER_FEEDBACK_LIMIT = 10

DEFAULT_URLS = [
    "https://www.ebay.com.sg/itm/206158794969?itmmeta=01KMF0940PYQKEP45AJ0BP7K4M&hash=item30000590d9:g:JnYAAeSwiEVpwANm",
    "https://www.ebay.com.sg/p/4062765295?iid=377055098797",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize eBay URLs or inspect raw eBay API responses.")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        help="eBay URL to process. Repeat the flag to process multiple URLs.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the raw eBay API response(s) instead of normalized candidates.",
    )
    parser.add_argument(
        "--auth-raw",
        action="store_true",
        help="Print the raw OAuth token response and exit.",
    )
    parser.add_argument(
        "--score-bayesian",
        action="store_true",
        help=(
            "After normalization, run seller feedback through eWOM and score the listing "
            "with the Bayesian value model."
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print a compact scoring summary instead of the full JSON payload. "
            "Best used with --score-bayesian."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Compare exactly two scored listings. Requires two --url values and "
            "--score-bayesian."
        ),
    )
    parser.add_argument(
        "--peer-price",
        type=float,
        default=None,
        help="Optional manual peer price to supply to the Bayesian value model.",
    )
    parser.add_argument(
        "--exclude-shipping",
        action="store_true",
        help=(
            "Use item price only for retrieval and Bayesian scoring instead of "
            "item price plus shipping."
        ),
    )
    parser.add_argument(
        "--use-converted-usd",
        action="store_true",
        help=(
            "Prefer eBay's convertedFromValue in USD for item and shipping prices "
            "when available."
        ),
    )
    parser.add_argument(
        "--top-k-neighbors",
        type=int,
        default=None,
        help="Optional override for the final reranked neighbor count used for peer-price averaging.",
    )
    parser.add_argument(
        "--retrieval-candidate-pool-size",
        type=int,
        default=500,
        help=(
            "Number of raw retrieval candidates to inspect before reranking. "
            "The eBay bridge will rerank this pool and average the top final matches."
        ),
    )
    parser.add_argument(
        "--min-peer-price-ratio",
        type=float,
        default=0.18,
        help=(
            "Ignore retrieved peer prices below this fraction of the listing price. "
            "For example, 0.50 drops peer prices that are more than 50%% below the listing."
        ),
    )
    parser.add_argument(
        "--min-peer-neighbors",
        type=int,
        default=3,
        help=(
            "Ignore retrieved peer prices unless at least this many reranked neighbors "
            "remain after filtering."
        ),
    )
    parser.add_argument(
        "--k-values",
        default=None,
        help=(
            "Comma-separated k values for retrieval diagnostics, for example "
            "\"1,3,5,10,20\". Requires --worth-buying-model-path."
        ),
    )
    parser.add_argument(
        "--k-sweep-output",
        type=Path,
        default=None,
        help="Optional HTML output path for the k-sweep diagnostic plot.",
    )
    parser.add_argument(
        "--worth-buying-model-path",
        type=Path,
        default=None,
        help=(
            "Optional worth-buying retrieval model path. When provided, infer peer_price "
            "from the Electronics neighbor model before Bayesian scoring."
        ),
    )
    parser.add_argument(
        "--helpfulness-model-path",
        default=None,
        help="Optional override for the helpfulness model artifact path.",
    )
    parser.add_argument(
        "--helpfulness-feature-builder-path",
        default=None,
        help="Optional override for the helpfulness feature-builder artifact path.",
    )
    parser.add_argument(
        "--sentiment-model-path",
        default=None,
        help="Optional override for the sentiment model artifact path.",
    )
    parser.add_argument(
        "--sentiment-feature-builder-path",
        default=None,
        help="Optional override for the sentiment feature-builder artifact path.",
    )
    return parser.parse_args()


def build_response_snapshot(response):
    try:
        body = response.json()
    except ValueError:
        body = response.text

    return {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "body": body,
    }


def print_json(payload):
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    sys.stdout.buffer.write((serialized + "\n").encode("utf-8", errors="replace"))
    sys.stdout.flush()


def get_seller_id_from_item_body(item_body):
    if not isinstance(item_body, dict):
        return None

    seller = item_body.get("seller")
    if not isinstance(seller, dict):
        return None

    return seller.get("username") or seller.get("sellerId") or seller.get("userId")


def append_seller_feedback_call(result, seller_id: str | None, feedback_client: EbayFeedbackClient | None):
    if not seller_id or feedback_client is None:
        return

    response = feedback_client.get_feedback_response(
        user_id=seller_id,
        feedback_type="FEEDBACK_RECEIVED",
        limit=max(25, SELLER_FEEDBACK_LIMIT),
        role="SELLER",
    )
    result["api_calls"].append(
        {
            "operation": "get_feedback",
            "request_params": {
                "user_id": seller_id,
                "feedback_type": "FEEDBACK_RECEIVED",
                "limit": max(25, SELLER_FEEDBACK_LIMIT),
                "filter": "role:SELLER",
            },
            "response": build_response_snapshot(response),
        }
    )


def get_raw_response_for_url(
    url: str,
    browse_client: EbayBrowseClient,
    url_parser: EbayUrlParser,
    feedback_client: EbayFeedbackClient | None = None,
):
    parsed_url = url_parser.parse(url)
    result = {
        "source_url": url,
        "parsed_url": asdict(parsed_url),
        "api_calls": [],
    }

    if parsed_url.legacy_item_id:
        response = browse_client.get_item_by_legacy_id_response(parsed_url.legacy_item_id)
        result["api_calls"].append(
            {
                "operation": "get_item_by_legacy_id",
                "request_params": {
                    "legacy_item_id": parsed_url.legacy_item_id,
                    "fieldgroups": "PRODUCT",
                },
                "response": build_response_snapshot(response),
            }
        )
        append_seller_feedback_call(
            result,
            get_seller_id_from_item_body(result["api_calls"][-1]["response"]["body"]),
            feedback_client,
        )
        return result

    if parsed_url.epid:
        search_response = browse_client.search_by_epid_response(parsed_url.epid, limit=1)
        result["api_calls"].append(
            {
                "operation": "search_by_epid",
                "request_params": {
                    "epid": parsed_url.epid,
                    "limit": 1,
                },
                "response": build_response_snapshot(search_response),
            }
        )

        try:
            search_body = search_response.json()
        except ValueError:
            search_body = None

        if isinstance(search_body, dict):
            items = search_body.get("itemSummaries", [])
            first_item = items[0] if items else None
            legacy_item_id = first_item.get("legacyItemId") if isinstance(first_item, dict) else None

            if legacy_item_id:
                item_response = browse_client.get_item_by_legacy_id_response(legacy_item_id)
                result["api_calls"].append(
                    {
                        "operation": "get_item_by_legacy_id",
                        "request_params": {
                            "legacy_item_id": legacy_item_id,
                            "fieldgroups": "PRODUCT",
                        },
                        "response": build_response_snapshot(item_response),
                    }
                )
                append_seller_feedback_call(
                    result,
                    get_seller_id_from_item_body(result["api_calls"][-1]["response"]["body"]),
                    feedback_client,
                )
            else:
                seller_id = None
                if isinstance(first_item, dict):
                    seller = first_item.get("seller")
                    if isinstance(seller, dict):
                        seller_id = seller.get("username") or seller.get("sellerId") or seller.get("userId")
                append_seller_feedback_call(result, seller_id, feedback_client)

        return result

    raise ValueError(f"Unsupported or invalid eBay URL: {url}")


def print_separator():
    print("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    if args.summary and not (args.score_bayesian or args.k_values):
        raise ValueError("--summary currently requires --score-bayesian or --k-values.")
    if args.k_values and args.worth_buying_model_path is None:
        raise ValueError("--k-values requires --worth-buying-model-path.")
    if args.compare and not args.score_bayesian:
        raise ValueError("--compare requires --score-bayesian.")
    urls = args.urls or DEFAULT_URLS
    if args.compare and len(urls) != 2:
        raise ValueError("--compare requires exactly two --url values.")

    auth_client = EbayAuthClient(scopes=AUTH_SCOPES)

    if args.auth_raw:
        print_json(build_response_snapshot(auth_client.request_access_token()))
        return

    access_token = auth_client.get_access_token()

    browse_client = EbayBrowseClient(access_token=access_token, marketplace_id="EBAY_SG")
    feedback_client = EbayFeedbackClient(access_token=access_token)
    url_parser = EbayUrlParser()

    normalization_service = NormalizationService(
        marketplace_client=browse_client,
        url_parser=url_parser,
        feedback_client=feedback_client,
        seller_feedback_limit=SELLER_FEEDBACK_LIMIT,
    )

    if args.raw:
        for index, url in enumerate(urls):
            if index:
                print_separator()
            print_json(get_raw_response_for_url(url, browse_client, url_parser, feedback_client=feedback_client))
        return

    if args.compare:
        scored_results = []
        for url in urls:
            candidate = normalization_service.normalize(url)
            scored_results.append(
                score_ebay_candidate_value(
                    candidate,
                    peer_price=args.peer_price,
                    worth_buying_model_path=args.worth_buying_model_path,
                    top_k_neighbors=args.top_k_neighbors,
                    ewom_model_paths=_resolve_ewom_model_paths(args),
                    include_shipping_in_total=not args.exclude_shipping,
                    prefer_converted_usd=args.use_converted_usd,
                    retrieval_candidate_pool_size=args.retrieval_candidate_pool_size,
                    min_peer_price_ratio=args.min_peer_price_ratio,
                    min_peer_neighbor_count=args.min_peer_neighbors,
                )
            )
        comparison_payload = compare_ebay_candidate_value_results(
            scored_results[0],
            scored_results[1],
        )
        if not args.summary:
            comparison_payload = {
                "listing_a_result": scored_results[0],
                "listing_b_result": scored_results[1],
                **comparison_payload,
            }
        print_json(comparison_payload)
        return

    for index, url in enumerate(urls):
        if index:
            print_separator()
        candidate = normalization_service.normalize(url)
        if args.k_values:
            k_sweep_result = sweep_candidate_market_context_k(
                candidate,
                model_path=args.worth_buying_model_path,
                k_values=_parse_k_values(args.k_values),
                ewom_model_paths=_resolve_ewom_model_paths(args)
                if args.score_bayesian
                else None,
                include_shipping_in_total=not args.exclude_shipping,
                prefer_converted_usd=args.use_converted_usd,
                retrieval_candidate_pool_size=args.retrieval_candidate_pool_size,
                min_peer_price_ratio=args.min_peer_price_ratio,
                min_peer_neighbor_count=args.min_peer_neighbors,
            )
            plot_path = write_candidate_k_sweep_plot(
                k_sweep_result,
                output_path=(
                    args.k_sweep_output
                    or _default_k_sweep_output_path(candidate)
                ),
            )
            payload = (
                summarize_candidate_market_context_k_sweep(k_sweep_result)
                if args.summary
                else k_sweep_result
            )
            payload["k_sweep_plot_path"] = plot_path
            print_json(payload)
            continue
        if args.score_bayesian:
            scored_result = score_ebay_candidate_value(
                candidate,
                peer_price=args.peer_price,
                worth_buying_model_path=args.worth_buying_model_path,
                top_k_neighbors=args.top_k_neighbors,
                ewom_model_paths=_resolve_ewom_model_paths(args),
                include_shipping_in_total=not args.exclude_shipping,
                prefer_converted_usd=args.use_converted_usd,
                retrieval_candidate_pool_size=args.retrieval_candidate_pool_size,
                min_peer_price_ratio=args.min_peer_price_ratio,
                min_peer_neighbor_count=args.min_peer_neighbors,
            )
            print_json(
                summarize_ebay_candidate_value_result(scored_result)
                if args.summary
                else scored_result
            )
            continue
        print_json(candidate.to_output_dict())


def _resolve_ewom_model_paths(args) -> dict[str, str] | None:
    overrides = {
        "helpfulness_model_path": args.helpfulness_model_path,
        "helpfulness_feature_builder_path": args.helpfulness_feature_builder_path,
        "sentiment_model_path": args.sentiment_model_path,
        "sentiment_feature_builder_path": args.sentiment_feature_builder_path,
    }
    resolved = {key: value for key, value in overrides.items() if value}
    return resolved or None


def _parse_k_values(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in str(raw_value).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("--k-values must contain at least one integer.")
    return values


def _default_k_sweep_output_path(candidate) -> Path:
    item_id = (
        getattr(candidate, "legacy_item_id", None)
        or getattr(candidate, "product_id", None)
        or "ebay_candidate"
    )
    return PROJECT_ROOT / "value" / "artifacts" / f"ebay_k_sweep_{item_id}.html"


if __name__ == "__main__":
    main()
