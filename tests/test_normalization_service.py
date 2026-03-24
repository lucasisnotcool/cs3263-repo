import sys
from pathlib import Path

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
        assert role == "BUYER"
        return {
            "feedbackEntries": [
                {"feedbackComment": {"commentText": "Great seller!"}},
                {"feedbackComment": {"commentText": "Fast shipping."}},
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
        "seller_feedback_texts",
        "item_specifics",
        "product_family_key",
    ]
    assert output["price"] is None
    assert output["shipping"] == []
    assert output["returns"] is None
    assert output["seller_feedback_texts"] is None
    assert output["product_rating_count"] is None
    assert output["product_rating_histogram"] is None
    assert output["product_average_rating"] is None
