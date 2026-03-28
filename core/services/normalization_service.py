import re
from typing import Any, Dict, List, Optional

from core.entities.candidate import Candidate
from core.interfaces.feedback_client import FeedbackClient
from core.interfaces.marketplace_client import MarketplaceClient
from core.interfaces.normalizer import Normalizer
from infrastructure.external_clients.ebay.ebay_url_parser import EbayUrlParser, ParsedEbayUrl


class NormalizationService(Normalizer):
    def __init__(
        self,
        marketplace_client: MarketplaceClient,
        url_parser: EbayUrlParser,
        feedback_client: FeedbackClient | None = None,
        seller_feedback_limit: int = 10,
    ):
        self.marketplace_client = marketplace_client
        self.url_parser = url_parser
        self.feedback_client = feedback_client
        self.seller_feedback_limit = seller_feedback_limit

    def normalize(self, url: str) -> Candidate:
        parsed_url = self.url_parser.parse(url)

        if parsed_url.legacy_item_id:
            item = self.marketplace_client.get_item_by_legacy_id(parsed_url.legacy_item_id)
            return self._normalize_item_response(url, parsed_url, item)

        if parsed_url.epid:
            search_result = self.marketplace_client.search_by_epid(parsed_url.epid, limit=1)
            items = search_result.get("itemSummaries", [])

            if not items:
                raise ValueError(f"No item found for ePID={parsed_url.epid}")

            first = items[0]
            legacy_item_id = first.get("legacyItemId")

            if legacy_item_id:
                parsed_url.legacy_item_id = legacy_item_id
                item = self.marketplace_client.get_item_by_legacy_id(legacy_item_id)
                return self._normalize_item_response(url, parsed_url, item)

            summary_like_item = {
                "itemId": first.get("itemId"),
                "title": first.get("title"),
                "price": first.get("price"),
                "condition": first.get("condition"),
                "seller": {
                    "username": self._safe_get(first, "seller", "username"),
                    "feedbackScore": self._safe_get(first, "seller", "feedbackScore"),
                    "feedbackPercentage": self._safe_get(first, "seller", "feedbackPercentage"),
                },
                "shippingOptions": first.get("shippingOptions", []),
                "product": {"epid": parsed_url.epid},
            }
            return self._normalize_item_response(url, parsed_url, summary_like_item)

        raise ValueError(f"Unsupported or invalid eBay URL: {url}")

    def _normalize_item_response(
        self,
        source_url: str,
        parsed_url: ParsedEbayUrl,
        item: Dict[str, Any],
    ) -> Candidate:
        seller = item.get("seller", {})
        review_count, histogram, avg_rating = self._parse_product_rating_data(item)

        shipping_options = item.get("shippingOptions", [])
        if not isinstance(shipping_options, list):
            shipping_options = []

        returns_info = item.get("returnTerms") or item.get("returns") or item.get("returnPolicy")
        seller_id = seller.get("username") or seller.get("sellerId") or seller.get("userId")

        return Candidate(
            source_url=source_url,
            page_type=parsed_url.page_type,
            legacy_item_id=parsed_url.legacy_item_id,
            rest_item_id=item.get("itemId"),
            product_id=self._extract_product_id(item, parsed_url),
            title=item.get("title"),
            price=item.get("price"),
            shipping=shipping_options,
            returns=returns_info,
            condition=item.get("condition"),
            seller_id=seller_id,
            seller_feedback_score=seller.get("feedbackScore"),
            seller_feedback_percentage=seller.get("feedbackPercentage"),
            detailed_seller_ratings=self._parse_detailed_seller_ratings(item),
            product_rating_count=review_count,
            product_rating_histogram=histogram,
            product_average_rating=avg_rating,
            seller_feedback_texts=self._fetch_seller_feedback_texts(seller_id),
            item_specifics=self._extract_item_specifics(item),
            product_family_key=self._build_product_family_key(item, parsed_url),
        )

    def _safe_get(self, d: Optional[Dict[str, Any]], *keys, default=None):
        current = d
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def _parse_detailed_seller_ratings(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seller = item.get("seller", {})
        candidate_keys = [
            "feedbackPercentage",
            "feedbackScore",
            "sellerAccountType",
            "username",
        ]
        found = {k: seller.get(k) for k in candidate_keys if k in seller}
        return found or None

    def _parse_product_rating_data(
        self,
        item: Dict[str, Any],
    ) -> tuple[Optional[int], Optional[List[Dict[str, Any]]], Optional[float]]:
        avg_rating = item.get("averageRating")
        review_count = item.get("reviewCount")
        histogram = item.get("ratingHistograms")

        if avg_rating is None:
            avg_rating = self._safe_get(item, "product", "averageRating")
        if review_count is None:
            review_count = self._safe_get(item, "product", "reviewCount")
        if histogram is None:
            histogram = self._safe_get(item, "product", "ratingHistograms")

        return review_count, histogram, avg_rating

    def _fetch_seller_feedback_texts(self, seller_id: Optional[str]) -> Optional[List[str]]:
        if not seller_id or self.feedback_client is None:
            return None

        feedback = self.feedback_client.get_feedback(
            user_id=seller_id,
            feedback_type="FEEDBACK_RECEIVED",
            limit=max(25, self.seller_feedback_limit),
            role="BUYER",
        )

        feedback_entries = feedback.get("feedbackEntries", [])
        texts = []
        for entry in feedback_entries:
            comment_text = self._safe_get(entry, "feedbackComment", "commentText")
            text_removed = self._safe_get(entry, "feedbackComment", "commentTextRemovedPerPolicy", default=False)

            if comment_text and not text_removed:
                texts.append(comment_text)

        if not texts:
            return None

        return texts[: self.seller_feedback_limit]

    def _extract_product_id(self, item: Dict[str, Any], parsed_url: ParsedEbayUrl) -> Optional[str]:
        product = item.get("product", {})
        if isinstance(product, dict):
            epid = product.get("epid") or product.get("ePID")
            if epid:
                return str(epid)
        return parsed_url.epid

    def _extract_item_specifics(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        result = {}

        for aspect in item.get("localizedAspects", []):
            name = aspect.get("name")
            values = aspect.get("value", [])
            if name:
                result[name] = values

        aspect_groups = self._safe_get(item, "product", "aspectGroups", default=[])
        for group in aspect_groups or []:
            for aspect in group.get("aspects", []):
                name = aspect.get("name")
                values = aspect.get("values", [])
                if name:
                    result[name] = values

        return result or None

    def _build_product_family_key(self, item: Dict[str, Any], parsed_url: ParsedEbayUrl) -> Optional[str]:
        product_id = self._extract_product_id(item, parsed_url)
        if product_id:
            return f"epid:{product_id}"

        specifics = self._extract_item_specifics(item) or {}

        brand = specifics.get("Brand", [None])[0] if isinstance(specifics.get("Brand"), list) else specifics.get("Brand")
        mpn = specifics.get("MPN", [None])[0] if isinstance(specifics.get("MPN"), list) else specifics.get("MPN")
        upc = specifics.get("UPC", [None])[0] if isinstance(specifics.get("UPC"), list) else specifics.get("UPC")
        ean = specifics.get("EAN", [None])[0] if isinstance(specifics.get("EAN"), list) else specifics.get("EAN")

        if upc:
            return f"upc:{upc}"
        if ean:
            return f"ean:{ean}"
        if brand and mpn:
            return f"brand_mpn:{str(brand).strip().lower()}::{str(mpn).strip().lower()}"

        title = item.get("title")
        if title:
            normalized_title = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
            normalized_title = re.sub(r"\s+", " ", normalized_title)
            return f"title:{normalized_title}"

        return None
