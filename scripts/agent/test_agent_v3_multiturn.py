from __future__ import annotations

import asyncio
import json
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
    thread_id = f"multiturn-{uuid.uuid4().hex[:12]}"
    checks: dict[str, bool] = {}
    records = []

    scenarios = [
        (
            "A_INITIAL",
            "最近工作任务不断增加，我担心无法按时完成，晚上也经常因为工作安排而难以放松。",
        ),
        (
            "B_FOLLOWUP",
            "今天主管又临时给我加了任务，我压力很大，不知道怎么安排。",
        ),
        (
            "C_UNRELATED",
            "周末想去杭州玩两天，有什么值得推荐的地方？",
        ),
        (
            "D_RETURN",
            "回到工作这件事，我还是担心自己做不完。",
        ),
    ]

    async with LongPsychAgentV3(MCP_URL, DB_PATH) as agent:
        for label, query in scenarios:
            print("\n" + "=" * 68)
            print(label, ":", query)

            result = await agent.ainvoke(
                {"query": query, "mode": "qr_future", "top_k": 5},
                thread_id=thread_id,
            )
            records.append(result)

            plan = result.get("context_plan", {})
            print("CONTEXT PLAN:")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print("STATUS:", result.get("status"))
            print("ACTUAL MCP ARGUMENTS:")
            print(json.dumps(
                result.get("retrieval_arguments", {}),
                ensure_ascii=False,
                indent=2,
            ))
            print("RESULT COUNT:", len(result.get("results", [])))
            print("HISTORY COUNT:", len(result.get("history", [])))

    A, B, C, D = records

    a_plan = A.get("context_plan", {})
    b_plan = B.get("context_plan", {})
    c_plan = C.get("context_plan", {})
    d_plan = D.get("context_plan", {})

    checks["A_INITIAL_SEARCH"] = (
        a_plan.get("should_search") is True
        and len(A.get("results", [])) > 0
        and A.get("retrieval_arguments", {}).get("query")
            == a_plan.get("final_query")
    )

    b_actual_query = B.get("retrieval_arguments", {}).get("query", "")
    checks["B_FOLLOWUP_SEARCH"] = (
        b_plan.get("should_search") is True
        and len(B.get("results", [])) > 0
    )
    checks["B_HISTORY_CONTEXT"] = (
        bool(b_plan.get("selected_history"))
        and "最近工作任务不断增加" in b_actual_query
    )

    checks["C_NO_UNRELATED_RETRIEVAL"] = (
        c_plan.get("should_search") is False
        and not C.get("retrieval_arguments")
        and not C.get("results")
    )

    d_actual_query = D.get("retrieval_arguments", {}).get("query", "")
    checks["D_RETURN_SEARCH"] = (
        d_plan.get("should_search") is True
        and len(D.get("results", [])) > 0
    )
    checks["D_HISTORY_CONTEXT"] = (
        bool(d_plan.get("selected_history"))
        and (
            "最近工作任务不断增加" in d_actual_query
            or "主管又临时给我加了任务" in d_actual_query
        )
    )
    checks["D_NO_TRAVEL_CONTAMINATION"] = (
        "杭州" not in d_actual_query
        and "旅游" not in d_actual_query
    )

    checks["CURRENT_QUERY_PRESERVED"] = all(
        result.get("retrieval_arguments", {}).get("query", "").startswith(
            result.get("query", "")
        )
        for result in [A, B, D]
    )
    checks["HISTORY_LENGTH"] = len(D.get("history", [])) == 4
    checks["HISTORY_USER_ONLY"] = all(
        isinstance(item.get("query"), str)
        and "response_text" not in item
        for item in D.get("history", [])
    )

    print("\n" + "=" * 68)
    print("LONGPSYCH V3 MULTI-TURN ACCEPTANCE")
    for name, passed in checks.items():
        print(name, "=>", "PASS" if passed else "FAIL")

    passed = sum(checks.values())
    print(f"PASSED: {passed}/{len(checks)}")
    print("STATUS:", "PASS" if all(checks.values()) else "REVIEW")

    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
