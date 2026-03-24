from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    source_url: str
    page_type: str
    legacy_item_id: Optional[str] = None
    rest_item_id: Optional[str] = None
    product_id: Optional[str] = None
    title: Optional[str] = None
    price: Optional[Dict[str, Any]] = None
    shipping: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[Dict[str, Any]] = None
    condition: Optional[str] = None
    seller_id: Optional[str] = None
    seller_feedback_score: Optional[int] = None
    seller_feedback_percentage: Optional[float] = None
    detailed_seller_ratings: Optional[Dict[str, Any]] = None
    product_rating_count: Optional[int] = None
    product_rating_histogram: Optional[List[Dict[str, Any]]] = None
    product_average_rating: Optional[float] = None
    seller_feedback_texts: Optional[List[str]] = None
    item_specifics: Optional[Dict[str, Any]] = None
    product_family_key: Optional[str] = None

    def to_output_dict(self) -> Dict[str, Any]:
        return {
            "source_url": self.source_url,
            "page_type": self.page_type,
            "legacy_item_id": self.legacy_item_id,
            "rest_item_id": self.rest_item_id,
            "product_id": self.product_id,
            "title": self.title,
            "price": self.price,
            "shipping": self.shipping,
            "returns": self.returns,
            "condition": self.condition,
            "seller_id": self.seller_id,
            "seller_feedback_score": self.seller_feedback_score,
            "seller_feedback_percentage": self.seller_feedback_percentage,
            "detailed_seller_ratings": self.detailed_seller_ratings,
            "product_rating_count": self.product_rating_count,
            "product_rating_histogram": self.product_rating_histogram,
            "product_average_rating": self.product_average_rating,
            "seller_feedback_texts": self.seller_feedback_texts,
            "item_specifics": self.item_specifics,
            "product_family_key": self.product_family_key,
        }
