from __future__ import annotations

import json
import os
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import settings


RetrievalMode = Literal["dense", "orca", "qr_future"]


class LongPsychAPIError(RuntimeError):
    """Raised when the LongPsychAgent HTTP API cannot complete a request."""


class LongPsychAPIClient:
    """
    Small standard-library HTTP client for the existing FastAPI service.

    The MCP process does not load BGE, FAISS, or QR-Future checkpoints. It
    delegates to the already-running FastAPI process to avoid duplicate model
    and embedding memory on the GPU/host.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or settings.api_base_url
        ).rstrip("/")
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else settings.api_timeout_seconds
        )
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("LONGPSYCH_API_BASE_URL must start with http:// or https://.")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = None if payload is None else json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")
        request = Request(
            url=url,
            data=body,
            method=method.upper(),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
                parsed = json.loads(error_body) if error_body else {}
                detail = parsed.get("detail", error_body or str(exc))
            except Exception:
                detail = str(exc)
            raise LongPsychAPIError(
                f"LongPsychAgent API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LongPsychAPIError(
                f"Cannot reach LongPsychAgent API at {url}: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LongPsychAPIError(
                f"LongPsychAgent API returned invalid JSON from {path}."
            ) from exc

        if not isinstance(parsed, dict):
            raise LongPsychAPIError(
                f"LongPsychAgent API returned an unexpected payload from {path}."
            )
        return parsed

    def health(self) -> dict[str, Any]:
        result = self._request_json("GET", "/health")
        if result.get("status") != "ready":
            raise LongPsychAPIError(
                f"LongPsychAgent is not ready: {result!r}"
            )
        return result

    @staticmethod
    def _validate_search(
        query: str,
        mode: RetrievalMode,
        top_k: int,
        candidate_k: int | None,
    ) -> None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if len(query) > 4000:
            raise ValueError("query must not exceed 4000 characters.")
        if mode not in {"dense", "orca", "qr_future"}:
            raise ValueError("mode must be one of: dense, orca, qr_future.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100.")
        if candidate_k is not None:
            if (
                not isinstance(candidate_k, int)
                or isinstance(candidate_k, bool)
                or not 1 <= candidate_k <= 5000
            ):
                raise ValueError("candidate_k must be an integer between 1 and 5000.")
            if mode != "qr_future":
                raise ValueError("candidate_k is only valid for mode='qr_future'.")
            if candidate_k < top_k:
                raise ValueError("candidate_k must be greater than or equal to top_k.")

    def search(
        self,
        query: str,
        mode: RetrievalMode = "qr_future",
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> dict[str, Any]:
        self._validate_search(query, mode, top_k, candidate_k)
        payload: dict[str, Any] = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
        }
        if candidate_k is not None:
            payload["candidate_k"] = candidate_k

        response = self._request_json("POST", "/v1/search", payload)
        if response.get("status") != "ok" or "data" not in response:
            raise LongPsychAPIError(
                f"LongPsychAgent search returned an unexpected response: {response!r}"
            )
        data = response["data"]
        if not isinstance(data, dict):
            raise LongPsychAPIError(
                "LongPsychAgent single-query search did not return an object."
            )
        return data

    def compare(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 100,
    ) -> dict[str, Any]:
        self._validate_search(query, "qr_future", top_k, candidate_k)
        results: dict[str, Any] = {}
        for mode in ("dense", "orca", "qr_future"):
            results[mode] = self.search(
                query=query,
                mode=mode,  # type: ignore[arg-type]
                top_k=top_k,
                candidate_k=candidate_k if mode == "qr_future" else None,
            )
        return {
            "query": query,
            "comparison_modes": ["dense", "orca", "qr_future"],
            "candidate_protocol": {
                "qr_future_first_stage": "dense",
                "qr_future_candidate_k": candidate_k,
            },
            "results": results,
        }
