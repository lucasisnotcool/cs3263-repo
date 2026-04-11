import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.services.normalization_service import NormalizationService
from core.interfaces.feedback_client import FeedbackClient
from core.interfaces.marketplace_client import MarketplaceClient
from infrastructure.external_clients.ebay.ebay_url_parser import EbayUrlParser


class FakeMarketplaceClient(MarketplaceClient):
    def get_item_by_legacy_id(self, legacy_item_id: str):
        return {
            "itemId": "v1|123|0",
            "title": "Test listing",
            "condition": "New",
            "seller": {
                "username": "seller_123",
                "feedbackScore": 42,
                "feedbackPercentage": 99.8,
            },
            "product": {"epid": "4062765295"},
        }

    def get_items_by_item_group(self, item_group_id: str):
        return {
            "items": [],
        }

    def search_by_epid(self, epid: str, limit: int = 5):
        return {
            "itemSummaries": [],
        }


class FakeFeedbackClient(FeedbackClient):
    def get_feedback(
        self,
        user_id: str,
        feedback_type: str = "FEEDBACK_RECEIVED",
        limit: int = 25,
        offset: int = 0,
        role: str | None = None,
    ):
        assert user_id == "seller_123"
        assert feedback_type == "FEEDBACK_RECEIVED"
        assert role == "SELLER"
        return {
            "feedbackEntries": [
                {
                    "providerUserDetail": {"role": "BUYER"},
                    "feedbackComment": {"commentText": "Great seller!"},
                },
                {
                    "providerUserDetail": {"role": "BUYER"},
                    "feedbackComment": {"commentText": "Fast shipping."},
                },
                {
                    "providerUserDetail": {"role": "SELLER"},
                    "feedbackComment": {"commentText": "Prompt payment, thanks!"},
                },
                {"feedbackComment": {"commentText": "", "commentTextRemovedPerPolicy": False}},
                {"feedbackComment": {"commentText": "Removed", "commentTextRemovedPerPolicy": True}},
            ]
        }


def test_normalize_populates_seller_feedback_texts():
    service = NormalizationService(
        marketplace_client=FakeMarketplaceClient(),
        url_parser=EbayUrlParser(),
        feedback_client=FakeFeedbackClient(),
    )

    candidate = service.normalize("https://www.ebay.com.sg/itm/206158794969")

    assert candidate.seller_id == "seller_123"
    assert candidate.seller_feedback_texts == ["Great seller!", "Fast shipping."]


def test_normalize_without_feedback_client_leaves_seller_feedback_texts_empty():
    service = NormalizationService(
        marketplace_client=FakeMarketplaceClient(),
        url_parser=EbayUrlParser(),
    )

    candidate = service.normalize("https://www.ebay.com.sg/itm/206158794969")

    assert candidate.seller_feedback_texts is None


def test_candidate_output_dict_has_fixed_schema():
    service = NormalizationService(
        marketplace_client=FakeMarketplaceClient(),
        url_parser=EbayUrlParser(),
    )

    candidate = service.normalize("https://www.ebay.com.sg/itm/206158794969")
    output = candidate.to_output_dict()

    assert list(output.keys()) == [
        "source_url",
        "page_type",
        "legacy_item_id",
        "rest_item_id",
        "product_id",
        "title",
        "price",
        "shipping",
        "returns",
        "condition",
        "seller_id",
        "seller_feedback_score",
        "seller_feedback_percentage",
        "detailed_seller_ratings",
        "product_rating_count",
        "product_rating_histogram",
        "product_average_rating",
        "listing_bullet_points",
        "listing_description",
        "seller_feedback_texts",
        "item_specifics",
        "product_family_key",
    ]
    assert output["price"] is None
    assert output["shipping"] == []
    assert output["returns"] is None
    assert output["listing_bullet_points"] is None
    assert output["listing_description"] == "Condition: New"
    assert output["seller_feedback_texts"] is None
    assert output["product_rating_count"] is None
    assert output["product_rating_histogram"] is None
    assert output["product_average_rating"] is None


class FakeVariationMarketplaceClient(MarketplaceClient):
    def get_item_by_legacy_id(self, legacy_item_id: str):
        response = requests.Response()
        response.status_code = 400
        response._content = (
            b'{"errors":[{"errorId":11006,"message":"The legacy ID is invalid.",'
            b'"parameters":[{"name":"itemGroupHref","value":"https://api.ebay.com/buy/browse/v1/item/get_items_by_item_group?item_group_id=99887766"}]}]}'
        )
        raise requests.HTTPError("400 Client Error", response=response)

    def get_items_by_item_group(self, item_group_id: str):
        assert item_group_id == "99887766"
        return {
            "items": [
                {
                    "itemId": "v1|275276813011|100000000000001|0",
                    "legacyItemId": "275276813011",
                    "title": "JFJ Easy Pro Compatible Buffing Pad/s (JFJ EasyPro) - 2 Pads",
                    "condition": "New",
                    "price": {"value": "9.99", "currency": "GBP"},
                    "estimatedAvailabilities": [
                        {"estimatedAvailabilityStatus": "IN_STOCK"}
                    ],
                    "seller": {
                        "username": "seller_123",
                        "feedbackScore": 42,
                        "feedbackPercentage": 99.8,
                    },
                    "localizedAspects": [
                        {"name": "MPN", "value": "2 Pads"},
                        {"name": "Brand", "value": "JFJ"},
                    ],
                    "product": {"epid": "4062765295"},
                },
                {
                    "itemId": "v1|275276813011|100000000000002|0",
                    "legacyItemId": "275276813011",
                    "title": "JFJ Easy Pro Compatible Buffing Pad/s (JFJ EasyPro) - 5 Pads",
                    "condition": "New",
                    "price": {"value": "19.99", "currency": "GBP"},
                    "estimatedAvailabilities": [
                        {"estimatedAvailabilityStatus": "LIMITED_STOCK"}
                    ],
                    "seller": {
                        "username": "seller_123",
                        "feedbackScore": 42,
                        "feedbackPercentage": 99.8,
                    },
                    "localizedAspects": [
                        {"name": "MPN", "value": "5 Pads"},
                        {"name": "Brand", "value": "JFJ"},
                    ],
                    "product": {"epid": "4062765295"},
                },
            ]
        }

    def search_by_epid(self, epid: str, limit: int = 5):
        return {"itemSummaries": []}


def test_normalize_falls_back_to_item_group_for_variation_listing():
    service = NormalizationService(
        marketplace_client=FakeVariationMarketplaceClient(),
        url_parser=EbayUrlParser(),
        feedback_client=FakeFeedbackClient(),
    )

    candidate = service.normalize("https://www.ebay.com.sg/itm/275276813011")

    assert candidate.legacy_item_id == "275276813011"
    assert candidate.rest_item_id == "v1|275276813011|100000000000001|0"
    assert candidate.title == "JFJ Easy Pro Compatible Buffing Pad/s (JFJ EasyPro) - 2 Pads"
    assert candidate.price == {"value": "9.99", "currency": "GBP"}
    assert candidate.item_specifics == {"MPN": "2 Pads", "Brand": "JFJ"}
    assert candidate.listing_bullet_points == ["MPN: 2 Pads", "Brand: JFJ"]
    assert "Key listing details:" in candidate.listing_description
