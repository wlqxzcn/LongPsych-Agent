from __future__ import annotations

import argparse
import asyncio
import os
import uuid
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = PROJECT_ROOT / "agent"

for import_root in (PROJECT_ROOT, AGENT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from longpsych_agent.graph.agent_graph_v3 import LongPsychAgentV3
from longpsych_agent.config import settings




def print_result(result: dict) -> None:
    print("\n========== 本轮处理结果 ==========")
    print("状态：", result.get("status"))
    if result.get("error"):
        print("错误：", result["error"])

    plan = result.get("context_plan", {})
    if plan:
        print("话题关系：", plan.get("topic_relation"))
        print("是否检索：", plan.get("should_search"))
        print("上下文判断：", plan.get("reason", ""))

    args = result.get("retrieval_arguments", {})
    if args:
        print("实际检索模式：", args.get("mode"))
        print("实际检索 Query：", args.get("query"))

    results = result.get("results", [])
    if results:
        print("\n========== 候选回复 ==========")
        for item in results:
            print(
                f"\n{item.get('rank', '-')}. "
                f"{item.get('response_text', '')}"
            )
            if "retrieval_score" in item:
                print("   Dense 召回分数：", item["retrieval_score"])
            if "qr_future_score" in item:
                print("   QR-Future 排序分数：", item["qr_future_score"])
    elif plan and plan.get("should_search") is False:
        print("本轮属于技能范围外请求，没有调用心理检索工具。")

    print("\n历史轮数：", len(result.get("history", [])))
    print("================================\n")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongPsychAgent V3 interactive runner"
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="稳定的会话 ID；恢复旧会话时必须使用相同 ID",
    )
    parser.add_argument(
        "--db-path",
        default=str(settings.agent_db_path),
    )
    parser.add_argument(
        "--mcp-url",
        default=settings.mcp_url,
    )
    parser.add_argument("--mode", default="qr_future")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=settings.candidate_k)
    args = parser.parse_args()

    thread_id = args.thread_id or f"session-{uuid.uuid4().hex[:12]}"
    db_arg = Path(args.db_path).expanduser()
    db_path = (db_arg if db_arg.is_absolute() else settings.root / db_arg).resolve()

    print("LongPsychAgent V3")
    print("MCP:", args.mcp_url)
    print("SQLite:", db_path)
    print("THREAD_ID:", thread_id)
    print("恢复会话时请再次使用相同 --thread-id。")
    print("输入 /exit 退出。\n")

    async with LongPsychAgentV3(
        mcp_url=args.mcp_url,
        db_path=db_path,
    ) as agent:
        while True:
            query = await asyncio.to_thread(input, "你：")
            if query.strip().lower() in {"/exit", "/quit"}:
                break
            if not query.strip():
                continue

            result = await agent.ainvoke(
                {
                    "query": query,
                    "mode": args.mode,
                    "top_k": args.top_k,
                    "candidate_k": args.candidate_k,
                },
                thread_id=thread_id,
            )
            print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
