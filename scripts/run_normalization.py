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
    print(json.dumps(payload, indent=2, ensure_ascii=False))


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
        role="BUYER",
    )
    result["api_calls"].append(
        {
            "operation": "get_feedback",
            "request_params": {
                "user_id": seller_id,
                "feedback_type": "FEEDBACK_RECEIVED",
                "limit": max(25, SELLER_FEEDBACK_LIMIT),
                "filter": "role:BUYER",
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
    urls = args.urls or DEFAULT_URLS

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

    for index, url in enumerate(urls):
        if index:
            print_separator()
        print_json(normalization_service.normalize(url).to_output_dict())


if __name__ == "__main__":
    main()
