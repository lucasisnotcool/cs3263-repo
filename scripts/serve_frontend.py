from __future__ import annotations

import argparse
import json
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FRONTEND_ROOT = PROJECT_ROOT / "frontend"
MODEL_PATH = PROJECT_ROOT / "value" / "artifacts" / "amazon_worth_buying_devices_full.joblib"

from scripts.run_normalization import compare_urls, load_project_env


load_project_env()

CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

DEFAULT_COMPARE_OPTIONS = {
    "summary": True,
    "exclude_shipping": True,
    "worth_buying_model_path": MODEL_PATH,
    "use_converted_usd": True,
    "retrieval_candidate_pool_size": 500,
    "top_k_neighbors": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the eBay comparison frontend and call the normalization pipeline directly."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to.")
    return parser.parse_args()


def build_runtime_health() -> dict[str, object]:
    env_path = PROJECT_ROOT / ".env"
    client_id_set = bool(os.getenv("EBAY_CLIENT_ID", "").strip())
    client_secret_set = bool(os.getenv("EBAY_CLIENT_SECRET", "").strip())
    model_exists = MODEL_PATH.exists()

    missing: list[str] = []
    if not env_path.exists():
        missing.append(".env file")
    if not client_id_set:
        missing.append("EBAY_CLIENT_ID")
    if not client_secret_set:
        missing.append("EBAY_CLIENT_SECRET")
    if not model_exists:
        missing.append("worth-buying model artifact")

    return {
        "ready": not missing,
        "missing": missing,
        "pipeline_mode": "direct_function",
        "entrypoint": "scripts.run_normalization.compare_urls",
        "env_file_path": str(env_path),
        "env_file_exists": env_path.exists(),
        "ebay_environment": os.getenv("EBAY_ENVIRONMENT", "production"),
        "client_id_set": client_id_set,
        "client_secret_set": client_secret_set,
        "worth_buying_model_path": str(MODEL_PATH),
        "worth_buying_model_exists": model_exists,
        "compare_defaults": {
            "summary": True,
            "exclude_shipping": True,
            "worth_buying_model_path": str(MODEL_PATH),
            "use_converted_usd": True,
            "retrieval_candidate_pool_size": 500,
            "top_k_neighbors": 5,
        },
    }


class FrontendRequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self) -> None:
        self._handle_request(include_body=False)

    def do_GET(self) -> None:
        self._handle_request(include_body=True)

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path != "/api/compare":
            self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")
            return

        payload = self._read_json_body()
        url_a = str(payload.get("url_a") or "").strip()
        url_b = str(payload.get("url_b") or "").strip()

        if not url_a or not url_b:
            self._send_json(
                {
                    "error": {
                        "type": "validation_error",
                        "message": "Both url_a and url_b are required.",
                    }
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        health = build_runtime_health()
        if not health["ready"]:
            self._send_json(
                {
                    "error": {
                        "type": "runtime_not_ready",
                        "message": "The live comparison pipeline is not ready yet.",
                        "details": "Missing: " + ", ".join(health["missing"]),
                    },
                    "runtime": health,
                },
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            result = compare_urls(
                [url_a, url_b],
                **DEFAULT_COMPARE_OPTIONS,
            )
        except Exception as exc:
            self._send_json(
                {
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                    "runtime": build_runtime_health(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        if isinstance(result, dict):
            result["_meta"] = {
                "pipeline_mode": "direct_function",
                "entrypoint": "scripts.run_normalization.compare_urls",
                "compare_defaults": health["compare_defaults"],
            }
        self._send_json(result)

    def _handle_request(self, *, include_body: bool) -> None:
        request_path = urlparse(self.path).path

        if request_path in {"/", "/index.html"}:
            self._send_file(FRONTEND_ROOT / "index.html", include_body=include_body)
            return

        if request_path == "/styles.css":
            self._send_file(FRONTEND_ROOT / "styles.css", include_body=include_body)
            return

        if request_path == "/app.js":
            self._send_file(FRONTEND_ROOT / "app.js", include_body=include_body)
            return

        if request_path == "/api/health":
            self._send_json(build_runtime_health(), include_body=include_body)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found.")

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            parsed = {}
        return parsed if isinstance(parsed, dict) else {}

    def _send_file(self, path: Path, *, include_body: bool = True) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return

        content_type = CONTENT_TYPES.get(path.suffix, "text/plain; charset=utf-8")
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if include_body:
            self.wfile.write(body)

    def _send_json(
        self,
        payload: dict,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        include_body: bool = True,
    ) -> None:
        response_body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if include_body:
            self.wfile.write(response_body)


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), FrontendRequestHandler)
    print(f"Serving frontend at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
