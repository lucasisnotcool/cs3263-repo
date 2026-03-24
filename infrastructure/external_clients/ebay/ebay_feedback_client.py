from typing import Any, Dict
import os
import requests

from core.interfaces.feedback_client import FeedbackClient


class EbayFeedbackClient(FeedbackClient):
    BASE_URLS = {
        "production": "https://api.ebay.com/commerce/feedback/v1",
        "sandbox": "https://api.sandbox.ebay.com/commerce/feedback/v1",
    }

    def __init__(self, access_token: str, environment: str | None = None):
        self.access_token = access_token
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
            "Content-Type": "application/json",
        }

    def get_feedback_response(
        self,
        user_id: str,
        feedback_type: str = "FEEDBACK_RECEIVED",
        limit: int = 25,
        offset: int = 0,
        role: str | None = None,
    ) -> requests.Response:
        api_limit = max(25, limit)
        params = {
            "user_id": user_id,
            "feedback_type": feedback_type,
            "limit": api_limit,
            "offset": offset,
        }

        filter_parts = []
        if role:
            filter_parts.append(f"role:{role}")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        return requests.get(
            f"{self.base_url}/feedback",
            headers=self.headers,
            params=params,
            timeout=30,
        )

    def get_feedback(
        self,
        user_id: str,
        feedback_type: str = "FEEDBACK_RECEIVED",
        limit: int = 25,
        offset: int = 0,
        role: str | None = None,
    ) -> Dict[str, Any]:
        response = self.get_feedback_response(
            user_id=user_id,
            feedback_type=feedback_type,
            limit=limit,
            offset=offset,
            role=role,
        )
        response.raise_for_status()
        return response.json()
