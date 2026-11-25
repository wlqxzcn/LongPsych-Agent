from __future__ import annotations

import json
import operator
from typing import Any, Literal, TypedDict, Annotated

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver


class RetrievalState(TypedDict, total=False):
    query: str
    conversation_context: str

    mode: Literal["dense", "orca", "qr_future"]
    top_k: int
    candidate_k: int

    retrieval_query: str
    retrieval_result: dict[str, Any]
    results: list[dict[str, Any]]

    status: str
    error: str

    # 同一 thread_id 下累积的检索历史
    history: Annotated[list[dict[str, Any]], operator.add]


class LongPsychRetrievalGraph:

    def __init__(
        self,
        mcp_url: str = "http://127.0.0.1:8001/mcp",
    ):
        self.client = MultiServerMCPClient({
            "longpsych": {
                "transport": "http",
                "url": mcp_url,
            }
        })

        self._tools = None
        self._graph = None

    async def _load_tools(self):
        if self._tools is None:
            tools = await self.client.get_tools()
            self._tools = {tool.name: tool for tool in tools}

            required = {
                "longpsych_search_dialogue",
                "longpsych_compare_retrievers",
                "longpsych_retrieval_health",
            }

            missing = required - set(self._tools)

            if missing:
                raise RuntimeError(
                    f"MCP tools missing: {sorted(missing)}"
                )

        return self._tools

    @staticmethod
    def _prepare_query(state: RetrievalState) -> dict:

        query = state.get("query", "")

        if not isinstance(query, str) or not query.strip():
            return {
                "status": "error",
                "error": "Query must be a non-empty string.",
            }

        mode = state.get("mode", "qr_future")
        top_k = state.get("top_k", 5)
        candidate_k = state.get("candidate_k", 100)

        if mode not in {"dense", "orca", "qr_future"}:
            return {
                "status": "error",
                "error": f"Unsupported retrieval mode: {mode}",
            }

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= 100
        ):
            return {
                "status": "error",
                "error": "top_k must be between 1 and 100.",
            }

        if mode == "qr_future":
            if (
                not isinstance(candidate_k, int)
                or isinstance(candidate_k, bool)
                or not top_k <= candidate_k <= 5000
            ):
                return {
                    "status": "error",
                    "error": "candidate_k must be >= top_k and <= 5000.",
                }

        context = state.get("conversation_context", "")

        if not isinstance(context, str):
            return {
                "status": "error",
                "error": "conversation_context must be a string.",
            }

        if context.strip():
            retrieval_query = (
                f"会话背景：{context.strip()}\n"
                f"当前用户表达：{query.strip()}"
            )
        else:
            retrieval_query = query.strip()

        if len(retrieval_query) > 4000:
            return {
                "status": "error",
                "error": "Retrieval query exceeds 4000 characters.",
            }

        return {
            "retrieval_query": retrieval_query,
            "mode": mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "status": "prepared",
        }

    @staticmethod
    def _decode_tool_result(raw: Any) -> dict:

        if isinstance(raw, dict):
            return raw

        if hasattr(raw, "content"):
            raw = raw.content

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    content = item.get("text")
                else:
                    content = getattr(item, "text", None)

                if isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        continue

        if isinstance(raw, str):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed

        raise RuntimeError(
            f"Cannot decode MCP tool result: {type(raw).__name__}"
        )

    async def _retrieve(self, state: RetrievalState) -> dict:

        try:
            tools = await self._load_tools()

            search_tool = tools["longpsych_search_dialogue"]

            mode = state.get("mode", "qr_future")

            arguments = {
                "query": state["retrieval_query"],
                "mode": mode,
                "top_k": state.get("top_k", 5),
            }

            # candidate_k 只用于 QR-Future
            if mode == "qr_future":
                arguments["candidate_k"] = state.get(
                    "candidate_k", 100
                )

            raw = await search_tool.ainvoke(arguments)

            payload = self._decode_tool_result(raw)

            if payload.get("status") != "ok":
                raise RuntimeError(
                    f"MCP retrieval failed: {payload}"
                )

            data = payload.get("data")

            if not isinstance(data, dict):
                raise RuntimeError(
                    "MCP retrieval returned invalid data."
                )

            results = data.get("results", [])

            if not isinstance(results, list):
                raise RuntimeError(
                    "Retrieval results must be a list."
                )

            return {
                "retrieval_result": data,
                "results": results,
                "status": "ok",
                "error": "",
            }

        except Exception as exc:
            return {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "results": [],
            }

    @staticmethod
    def _finish(state: RetrievalState) -> dict:

        results = state.get("results", [])

        history_entry = {
            "query": state.get("query", ""),
            "retrieval_query": state.get("retrieval_query", ""),
            "mode": state.get("mode", "qr_future"),
            "status": state.get("status", "unknown"),
            "result_uids": [
                item.get("response_uid")
                for item in results
                if isinstance(item, dict)
            ],
        }

        return {
            "history": [history_entry],
        }

    @staticmethod
    def _route_after_prepare(state: RetrievalState) -> str:
        if state.get("status") == "prepared":
            return "retrieve"
        return "finish"

    def build(self):

        if self._graph is not None:
            return self._graph

        builder = StateGraph(RetrievalState)

        builder.add_node(
            "prepare_query",
            self._prepare_query,
        )

        builder.add_node(
            "retrieve",
            self._retrieve,
        )

        builder.add_node(
            "finish",
            self._finish,
        )

        builder.add_edge(START, "prepare_query")

        builder.add_conditional_edges(
            "prepare_query",
            self._route_after_prepare,
            {
                "retrieve": "retrieve",
                "finish": "finish",
            },
        )

        builder.add_edge("retrieve", "finish")
        builder.add_edge("finish", END)

        # V1 使用进程内 Checkpointer
        memory = InMemorySaver()

        self._graph = builder.compile(
            checkpointer=memory
        )

        return self._graph

    async def ainvoke(
        self,
        payload: dict,
        thread_id: str = "longpsych-default",
    ) -> dict:

        graph = self.build()

        return await graph.ainvoke(
            payload,
            config={
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )
