from typing import Any, Dict
import requests

from core.interfaces.marketplace_client import MarketplaceClient


class EbayBrowseClient(MarketplaceClient):
    BASE_URL = "https://api.ebay.com/buy/browse/v1"

    def __init__(self, access_token: str, marketplace_id: str = "EBAY_SG"):
        self.access_token = access_token
        self.marketplace_id = marketplace_id

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
            "Content-Type": "application/json",
        }

    def get_item_by_legacy_id(self, legacy_item_id: str) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/item/get_item_by_legacy_id"
        params = {
            "legacy_item_id": legacy_item_id,
            "fieldgroups": "PRODUCT",
        }

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def search_by_epid(self, epid: str, limit: int = 5) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/item_summary/search"
        params = {
            "epid": epid,
            "limit": limit,
        }

        response = requests.get(url, headers=self.headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json()