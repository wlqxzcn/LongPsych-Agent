# LongPsychAgent MCP Tool + Agent Skill V1

This package exposes the already-running LongPsychAgent FastAPI service as MCP
tools and provides a portable Agent Skill.

## Architecture

MCP client / Agent
    -> Streamable HTTP MCP server (127.0.0.1:8001)
        -> existing FastAPI service (127.0.0.1:8000)
            -> existing LongPsychRetrievalPipeline
                -> BGE + Dense FAISS
                -> five-fold QR-Future FULL scorer
                -> original response IDs and text

The MCP process does not load model checkpoints, FAISS indexes, or the 61,502 x
1,024 response embedding matrix. It calls the existing FastAPI service to avoid
duplicating model and embedding memory.

## Included tools

- `longpsych_search_dialogue`
- `longpsych_compare_retrievers`
- `longpsych_retrieval_health`

The default search mode is `qr_future`, using Dense candidate recall followed
by the five-fold QR-Future FULL reranker. `candidate_k` defaults to the
FastAPI service setting (normally 100).

## Install MCP in an isolated environment

The MCP SDK v2 is used. A separate virtual environment avoids changing the
model-serving environment used by FastAPI, BGE, FAISS, and PyTorch.

```bash
bash scripts/agent/setup_mcp_env.sh
```

## Start order

Terminal 1: keep the existing FastAPI service running on `127.0.0.1:8000`.

Terminal 2:

```bash
bash scripts/agent/run_qr_future_mcp.sh
```

Default MCP endpoint:

`http://127.0.0.1:8001/mcp`

The MCP server binds to localhost only. Do not expose it directly to the
public internet without authentication and TLS.

## Remote MCP client access

From the local computer, open an SSH tunnel:

```bash
ssh -L 8001:127.0.0.1:8001 root@YOUR_SERVER
```

Configure an MCP-compatible client to connect to:

`http://127.0.0.1:8001/mcp`

The client must support Streamable HTTP MCP transport. Client-specific
configuration syntax varies; use the URL above as the server endpoint.

## Optional stdio transport

For an MCP host that launches a local subprocess on the same machine where
the FastAPI service is reachable:

```bash
source .venv-mcp/bin/activate
export PYTHONPATH=agent:.
export LONGPSYCH_API_BASE_URL=http://127.0.0.1:8000
export LONGPSYCH_MCP_TRANSPORT=stdio
python -m longpsych_agent.mcp_server.server
```

Do not use the stdio command in a terminal for interactive output; stdout is
reserved for MCP protocol messages.

## Unit tests

```bash
PYTHONPATH=agent:. python -m pytest -q tests/agent/test_longpsych_mcp_client.py
```

These tests mock HTTP responses. They do not require model assets or an active
FastAPI service.

## Verify tools

With the MCP server running, use an MCP Inspector or a compatible MCP client
to connect to `http://127.0.0.1:8001/mcp`.

Expected tool names:

- `longpsych_search_dialogue`
- `longpsych_compare_retrievers`
- `longpsych_retrieval_health`

Call `longpsych_retrieval_health` first. Then call
`longpsych_search_dialogue` with a query, `mode="qr_future"`, and `top_k=5`.

## Agent Skill

The portable skill is in:

`skills/longpsych-retrieval/SKILL.md`

It defines tool selection, default retrieval settings, query construction,
score interpretation, failure handling, and scope boundaries. A compatible
agent must have the MCP server configured separately; the Skill does not
itself create the network connection.
