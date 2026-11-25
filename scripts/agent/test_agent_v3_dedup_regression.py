from __future__ import annotations

import asyncio
import os
import uuid
import unicodedata

from difflib import SequenceMatcher
from pathlib import Path

from longpsych_agent.config import settings

from longpsych_agent.graph.agent_graph_v3 import LongPsychAgentV3


ROOT = settings.root
_db_raw = Path(os.getenv(
    "LONGPSYCH_V3_DEDUP_TEST_DB",
    str(ROOT / "data/agent/longpsych_agent_v3_dedup_test.sqlite"),
)).expanduser()
DB_PATH = (_db_raw if _db_raw.is_absolute() else ROOT / _db_raw).resolve()
MCP_URL = os.getenv("LONGPSYCH_MCP_URL", settings.mcp_url)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char for char in text
        if not char.isspace()
        and not unicodedata.category(char).startswith("P")
    )


def near_duplicate(left: str, right: str) -> bool:
    a = normalize(left)
    b = normalize(right)

    if not a or not b:
        return False

    if a == b:
        return True

    shorter = min(len(a), len(b))
    longer = max(len(a), len(b))

    if shorter < 12 or shorter / longer < 0.85:
        return False

    return SequenceMatcher(
        None, a, b, autojunk=False
    ).ratio() >= 0.92


def get_plan(result):
    return result.get("context_plan") or {}


def get_actual_query(result):
    arguments = result.get("retrieval_arguments") or {}
    return arguments.get("query", "")


def check_no_duplicate_history(current, selected):
    assert isinstance(selected, list)

    for quote in selected:
        assert isinstance(quote, str)
        assert not near_duplicate(current, quote), (
            "CURRENT_QUERY_DUPLICATED_IN_HISTORY: " + quote
        )

    for i, left in enumerate(selected):
        for right in selected[i + 1:]:
            assert not near_duplicate(left, right), (
                "DUPLICATE_HISTORY_ITEMS: " + left + " / " + right
            )


async def main():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    thread_id = "dedup-" + uuid.uuid4().hex[:12]

    A = "最近工作压力大"

    B = (
        "今天主管又临时给我加了任务，"
        "我压力很大，不知道怎么安排。"
    )

    C = B

    D = "周末想去杭州玩两天，有什么值得推荐的地方？"

    E = "回到刚才的工作问题，我还是担心自己做不完。"

    async with LongPsychAgentV3(MCP_URL, DB_PATH) as agent:

        async def turn(query):
            return await agent.ainvoke(
                {
                    "query": query,
                    "mode": "qr_future",
                    "top_k": 5,
                },
                thread_id=thread_id,
            )

        # =====================================
        # A: 首次心理问题
        # =====================================

        print("\n========== A INITIAL ==========")

        a = await turn(A)
        ap = get_plan(a)

        assert a.get("status") == "ok", a.get("error")
        assert ap.get("should_search") is True
        assert ap.get("selected_history") == []
        assert get_actual_query(a) == A

        print("INITIAL SEARCH: PASS")

        # =====================================
        # B: 同主题追问
        # =====================================

        print("\n========== B FOLLOWUP ==========")

        b = await turn(B)
        bp = get_plan(b)
        bq = get_actual_query(b)

        assert b.get("status") == "ok", b.get("error")
        assert bp.get("should_search") is True
        assert bq.startswith(B)

        check_no_duplicate_history(
            B, bp.get("selected_history", [])
        )

        print("FOLLOWUP SEARCH: PASS")
        print("B HISTORY:", bp.get("selected_history"))

        # =====================================
        # C: 完全重复上一轮
        # =====================================

        print("\n========== C EXACT DUPLICATE ==========")

        c = await turn(C)
        cp = get_plan(c)
        cq = get_actual_query(c)

        assert c.get("status") == "ok", c.get("error")
        assert cp.get("should_search") is True
        assert cq.startswith(C)

        check_no_duplicate_history(
            C, cp.get("selected_history", [])
        )

        assert cq.count(C) == 1, (
            "CURRENT_QUERY_APPENDED_MORE_THAN_ONCE: " + cq
        )

        print("EXACT DUPLICATE FILTER: PASS")
        print("C HISTORY:", cp.get("selected_history"))
        print("C ACTUAL QUERY:", cq)

        # =====================================
        # D: 无关旅游话题
        # =====================================

        print("\n========== D UNRELATED TOPIC ==========")

        d = await turn(D)
        dp = get_plan(d)

        assert dp.get("should_search") is False
        assert dp.get("selected_history") == []
        assert not d.get("retrieval_arguments")

        print("UNRELATED TOPIC BLOCK: PASS")
        print("STATUS:", d.get("status"))

        # =====================================
        # E: 返回原来的工作话题
        # =====================================

        print("\n========== E RETURN TO TOPIC ==========")

        e = await turn(E)
        ep = get_plan(e)
        eq = get_actual_query(e)

        assert e.get("status") == "ok", e.get("error")
        assert ep.get("should_search") is True
        assert eq.startswith(E)

        selected = ep.get("selected_history", [])

        check_no_duplicate_history(E, selected)

        assert len(selected) <= 2

        # 重复的 B 可以存在于原始历史中，
        # 但不能重复进入选中的历史背景。
        assert sum(
            near_duplicate(item, B)
            for item in selected
        ) <= 1

        assert eq.count(B) <= 1

        history = e.get("history", [])
        history_queries = [
            item.get("query", "")
            for item in history
            if isinstance(item, dict)
        ]

        assert len(history_queries) == 5, history_queries
        assert history_queries.count(B) == 2

        # 每条选中的历史必须能够追溯到此前用户消息。
        prior_queries = history_queries[:-1]

        for quote in selected:
            assert any(
                quote in old_query
                for old_query in prior_queries
            ), quote

        print("RETURN TO TOPIC: PASS")
        print("SELECTED HISTORY:", selected)
        print("ACTUAL QUERY:", eq)
        print("RAW HISTORY TURNS:", len(history_queries))
        print("REPEATED B IN RAW HISTORY:", history_queries.count(B))

    # =====================================
    # Final acceptance
    # =====================================

    checks = [
        "INITIAL_SEARCH",
        "FOLLOWUP_SEARCH",
        "EXACT_DUPLICATE_FILTER",
        "UNRELATED_TOPIC_BLOCK",
        "RETURN_TO_TOPIC",
        "HISTORY_DEDUP",
        "HISTORY_TRACEABILITY",
        "CURRENT_QUERY_PRESERVED",
    ]

    print("\n" + "=" * 60)
    print("LONGPSYCH V3 DEDUP REGRESSION")
    print("=" * 60)

    for item in checks:
        print(item, "=> PASS")

    print("PASSED:", len(checks), "/", len(checks))
    print("STATUS: PASS")
    print("THREAD_ID:", thread_id)
    print("DATABASE:", DB_PATH.resolve())


if __name__ == "__main__":
    asyncio.run(main())
