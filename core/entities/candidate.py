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