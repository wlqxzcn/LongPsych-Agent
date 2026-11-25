from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

from longpsych_agent.config import settings

from longpsych_agent.graph.agent_graph_v3 import LongPsychAgentV3


ROOT = settings.root
_db_raw = Path(os.getenv("LONGPSYCH_V3_TEST_DB", str(ROOT / "data/agent/longpsych_agent_v3_test.sqlite"))).expanduser()
DB_PATH = (_db_raw if _db_raw.is_absolute() else ROOT / _db_raw).resolve()
MCP_URL = os.getenv("LONGPSYCH_MCP_URL", settings.mcp_url)


async def main() -> None:
    thread_id = f"persistence-{uuid.uuid4().hex[:12]}"

    first_query = (
        "最近工作任务不断增加，我担心无法按时完成，"
        "晚上也经常因为工作安排而难以放松。"
    )
    second_query = "回到工作这件事，我还是担心自己做不完。"

    # First process/session lifetime: write one turn, then close SQLite.
    async with LongPsychAgentV3(MCP_URL, DB_PATH) as agent:
        first = await agent.ainvoke(
            {"query": first_query, "mode": "qr_future", "top_k": 3},
            thread_id=thread_id,
        )
        assert first.get("status") == "ok", first.get("error")
        assert len(first.get("history", [])) == 1

    # Reopen a new agent/checkpointer against the same SQLite file.
    async with LongPsychAgentV3(MCP_URL, DB_PATH) as agent:
        second = await agent.ainvoke(
            {"query": second_query, "mode": "qr_future", "top_k": 3},
            thread_id=thread_id,
        )

    plan = second.get("context_plan", {})
    actual_query = second.get("retrieval_arguments", {}).get("query", "")

    assert plan.get("should_search") is True, plan
    assert len(plan.get("selected_history", [])) >= 1, plan
    assert "最近工作任务不断增加" in actual_query, actual_query
    assert actual_query.startswith(second_query), actual_query
    assert len(second.get("history", [])) == 2

    print("SQLITE DATABASE:", DB_PATH.resolve())
    print("THREAD_ID:", thread_id)
    print("PERSISTED HISTORY:", len(second.get("history", [])))
    print("RESTORED CONTEXT:", plan.get("selected_history"))
    print("ACTUAL QUERY:", actual_query)
    print("LONGPSYCH V3 SQLITE PERSISTENCE = PASS")


if __name__ == "__main__":
    asyncio.run(main())
