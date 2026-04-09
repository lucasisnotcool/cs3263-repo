from typing import Any, Dict
import os
import requests

from core.interfaces.marketplace_client import MarketplaceClient


class EbayBrowseClient(MarketplaceClient):
    BASE_URLS = {
        "production": "https://api.ebay.com/buy/browse/v1",
        "sandbox": "https://api.sandbox.ebay.com/buy/browse/v1",
    }

    def __init__(
        self,
        access_token: str,
        marketplace_id: str = "EBAY_SG",
        environment: str | None = None,
    ):
        self.access_token = access_token
        self.marketplace_id = marketplace_id
        self.environment = (environment or os.getenv("EBAY_ENVIRONMENT", "production")).strip().lower()

        if self.environment not in self.BASE_URLS:
            raise ValueError("EBAY_ENVIRONMENT must be 'production' or 'sandbox'")

    @property
    def base_url(self) -> str:
        return self.BASE_URLS[self.environment]

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
            "Content-Type": "application/json",
        }

    def get_item_by_legacy_id_response(self, legacy_item_id: str) -> requests.Response:
        url = f"{self.base_url}/item/get_item_by_legacy_id"
        params = {
            "legacy_item_id": legacy_item_id,
            "fieldgroups": "PRODUCT",
        }

        return requests.get(url, headers=self.headers, params=params, timeout=30)

    def get_item_by_legacy_id(self, legacy_item_id: str) -> Dict[str, Any]:
        response = self.get_item_by_legacy_id_response(legacy_item_id)
        response.raise_for_status()
        return response.json()

    def get_items_by_item_group_response(self, item_group_id: str) -> requests.Response:
        url = f"{self.base_url}/item/get_items_by_item_group"
        params = {
            "item_group_id": item_group_id,
        }

        return requests.get(url, headers=self.headers, params=params, timeout=30)

    def get_items_by_item_group(self, item_group_id: str) -> Dict[str, Any]:
        response = self.get_items_by_item_group_response(item_group_id)
        response.raise_for_status()
        return response.json()

    def search_by_epid_response(self, epid: str, limit: int = 5) -> requests.Response:
        url = f"{self.base_url}/item_summary/search"
        params = {
            "epid": epid,
            "limit": limit,
        }

        return requests.get(url, headers=self.headers, params=params, timeout=30)

    def search_by_epid(self, epid: str, limit: int = 5) -> Dict[str, Any]:
        response = self.search_by_epid_response(epid, limit=limit)
        response.raise_for_status()
        return response.json()
