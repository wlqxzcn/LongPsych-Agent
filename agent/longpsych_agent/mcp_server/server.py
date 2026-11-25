from __future__ import annotations

import logging
import os
import sys
from typing import Literal

from mcp.server.mcpserver import MCPServer

from .api_client import LongPsychAPIClient, LongPsychAPIError
from ..config import settings


logging.basicConfig(
    level=settings.mcp_log_level,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("longpsych_agent.mcp")

api = LongPsychAPIClient()

mcp = MCPServer(
    "longpsych-retrieval",
    title="LongPsychAgent Retrieval",
    description=(
        "Psychological dialogue retrieval using BGE Dense, ORCA k=2, "
        "and five-fold QR-Future reranking."
    ),
    instructions=(
        "Use longpsych_search_dialogue for retrieval. The default mode is "
        "qr_future, which performs Dense candidate recall followed by the "
        "frozen five-fold QR-Future FULL reranker. Use "
        "longpsych_compare_retrievers only when an explicit comparison is "
        "needed. QR-Future scores are raw ranking scores, not probabilities. "
        "Retrieved responses are candidates, not diagnoses or guaranteed "
        "clinical recommendations."
    ),
    version="1.0.0",
)


def _error_payload(exc: Exception) -> dict:
    LOGGER.warning("Tool request failed: %s", exc)
    return {
        "status": "error",
        "error": type(exc).__name__,
        "message": str(exc),
        "action": (
            "Check that the LongPsychAgent FastAPI service is running and "
            "that LONGPSYCH_API_BASE_URL points to it."
        ),
    }


@mcp.tool()
def longpsych_search_dialogue(
    query: str,
    mode: Literal["dense", "orca", "qr_future"] = "qr_future",
    top_k: int = 5,
    candidate_k: int | None = None,
) -> dict:
    """
    Retrieve psychological dialogue response candidates for one query.

    Default mode qr_future uses Dense BGE candidate recall followed by the
    frozen five-fold QR-Future FULL reranker. candidate_k defaults to the
    FastAPI service setting (normally 100) and is only accepted in qr_future
    mode. qr_future_score is a raw ranking score, not a probability.
    """
    try:
        result = api.search(
            query=query,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return {"status": "ok", "data": result}
    except (LongPsychAPIError, ValueError, TypeError) as exc:
        return _error_payload(exc)


@mcp.tool()
def longpsych_compare_retrievers(
    query: str,
    top_k: int = 5,
    candidate_k: int = 100,
) -> dict:
    """
    Compare Dense, ORCA, and QR-Future results for the same query.

    QR-Future uses Dense candidate recall with candidate_k candidates.
    Use this only when a retrieval-mode comparison is requested.
    """
    try:
        result = api.compare(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )
        return {"status": "ok", "data": result}
    except (LongPsychAPIError, ValueError, TypeError) as exc:
        return _error_payload(exc)


@mcp.tool()
def longpsych_retrieval_health() -> dict:
    """
    Check whether the LongPsychAgent FastAPI retrieval service is ready.
    """
    try:
        return {"status": "ok", "data": api.health()}
    except (LongPsychAPIError, ValueError, TypeError) as exc:
        return _error_payload(exc)


def main() -> None:
    transport = settings.mcp_transport

    if transport == "stdio":
        # For MCP hosts that launch this server as a local subprocess.
        mcp.run(transport="stdio")
        return

    if transport == "streamable-http":
        host = settings.mcp_host
        port = settings.mcp_port
        LOGGER.info(
            "Starting LongPsychAgent MCP server at http://%s:%s/mcp",
            host,
            port,
        )
        mcp.run(
            transport="streamable-http",
            host=host,
            port=port,
            stateless_http=True,
            json_response=True,
        )
        return

    raise ValueError(
        "LONGPSYCH_MCP_TRANSPORT must be 'stdio' or 'streamable-http'."
    )


if __name__ == "__main__":
    main()
