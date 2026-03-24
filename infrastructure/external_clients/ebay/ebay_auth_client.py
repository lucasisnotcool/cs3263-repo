import os
import requests


class EbayAuthClient:
    TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.getenv("EBAY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("EBAY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            raise ValueError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set")

    def get_access_token(self) -> str:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        }

        response = requests.post(
            self.TOKEN_URL,
            headers=headers,
            data=data,
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["access_token"]