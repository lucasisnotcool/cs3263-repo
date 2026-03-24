import os
import requests


class EbayAuthClient:
    DEFAULT_SCOPES = [
        "https://api.ebay.com/oauth/api_scope",
    ]
    TOKEN_URLS = {
        "production": "https://api.ebay.com/identity/v1/oauth2/token",
        "sandbox": "https://api.sandbox.ebay.com/identity/v1/oauth2/token",
    }

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        environment: str | None = None,
        scopes: list[str] | None = None,
    ):
        self.client_id = (client_id or os.getenv("EBAY_CLIENT_ID", "")).strip()
        self.client_secret = (client_secret or os.getenv("EBAY_CLIENT_SECRET", "")).strip()
        self.environment = (environment or os.getenv("EBAY_ENVIRONMENT", "production")).strip().lower()
        self.scopes = self._resolve_scopes(scopes)

        if not self.client_id or not self.client_secret:
            raise ValueError("EBAY_CLIENT_ID and EBAY_CLIENT_SECRET must be set")
        if self.environment not in self.TOKEN_URLS:
            raise ValueError("EBAY_ENVIRONMENT must be 'production' or 'sandbox'")

    @property
    def token_url(self) -> str:
        return self.TOKEN_URLS[self.environment]

    def request_access_token(self) -> requests.Response:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": " ".join(self.scopes),
        }

        return requests.post(
            self.token_url,
            headers=headers,
            data=data,
            auth=(self.client_id, self.client_secret),
            timeout=30,
        )

    def get_access_token(self) -> str:
        response = self.request_access_token()

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = self._build_error_detail(response)
            raise requests.HTTPError(
                f"{exc}. eBay token request failed for environment='{self.environment}' with scopes={self.scopes}. {detail}"
            ) from exc

        return response.json()["access_token"]

    def _resolve_scopes(self, scopes: list[str] | None) -> list[str]:
        if scopes is not None:
            cleaned_scopes = [scope.strip() for scope in scopes if scope and scope.strip()]
        else:
            env_scopes = os.getenv("EBAY_OAUTH_SCOPES", "").split()
            cleaned_scopes = env_scopes or self.DEFAULT_SCOPES.copy()

        if not cleaned_scopes:
            raise ValueError("At least one OAuth scope must be provided")

        return cleaned_scopes

    def _build_error_detail(self, response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            pieces = []
            for key in ("error", "error_description", "message"):
                value = payload.get(key)
                if value:
                    pieces.append(f"{key}={value!r}")
            if pieces:
                return "Response payload: " + ", ".join(pieces)

        body = response.text.strip()
        if body:
            return f"Response body: {body[:300]}"

        return "No response body returned."
