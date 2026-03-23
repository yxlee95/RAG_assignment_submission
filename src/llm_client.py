from __future__ import annotations

from pathlib import Path

import requests
from requests import Response
from requests.exceptions import RequestException, SSLError


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: int = 45,
        verify_ssl: bool = True,
        ca_bundle: str = "",
    ) -> None:
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY.")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.ca_bundle = ca_bundle.strip()

    @property
    def endpoint(self) -> str:
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )

    @property
    def verify(self) -> bool | str:
        if not self.verify_ssl:
            return False
        if self.ca_bundle:
            ca_path = Path(self.ca_bundle)
            if not ca_path.exists():
                raise ValueError(
                    f"GEMINI_CA_BUNDLE was set but the file was not found: {self.ca_bundle}"
                )
            return str(ca_path)
        return True

    def _post(self, payload: dict) -> Response:
        return requests.post(
            self.endpoint,
            json=payload,
            timeout=self.timeout,
            verify=self.verify,
        )

    def generate(self, prompt: str) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512,
            },
        }

        try:
            response = self._post(payload)
            response.raise_for_status()
        except SSLError as exc:
            raise RuntimeError(
                "Gemini API SSL verification failed. If you are behind a corporate proxy or "
                "missing local CA certificates, set GEMINI_VERIFY_SSL=false in your .env "
                "as a temporary workaround, or set GEMINI_CA_BUNDLE to your CA bundle path."
            ) from exc
        except RequestException as exc:
            details = ""
            if getattr(exc, "response", None) is not None:
                response_text = exc.response.text.strip()
                if response_text:
                    details = f" Response body: {response_text}"
            raise RuntimeError(f"Gemini API request failed: {exc}.{details}") from exc

        data = response.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return "No response generated."

        content_parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in content_parts if "text" in part]
        return "\n".join(texts).strip() or "No response generated."
