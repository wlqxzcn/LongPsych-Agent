import argparse
import asyncio
import json

from longpsych_agent.graph.agent_graph_v2 import (
    LongPsychAgentV2,
)


async def main():

    parser = argparse.ArgumentParser(
        description="LongPsychAgent V2"
    )

    parser.add_argument("--query", required=True)
    parser.add_argument("--context", default="")

    parser.add_argument(
        "--mode",
        choices=["dense", "orca", "qr_future"],
        default="qr_future",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)

    parser.add_argument(
        "--thread-id",
        default="longpsych-v2-demo",
    )

    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8001/mcp",
    )

    args = parser.parse_args()

    agent = LongPsychAgentV2(
        mcp_url=args.mcp_url
    )

    payload = {
        "query": args.query,
        "conversation_context": args.context,
        "mode": args.mode,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
    }

    print("\n========== LONGPSYCH AGENT V2 ==========")

    result = await agent.ainvoke(
        payload,
        thread_id=args.thread_id,
    )

    print("\n========== 1. STATUS ==========")
    print(result.get("status"))

    if result.get("status") != "ok":
        print("ERROR:", result.get("error"))
        raise SystemExit(1)

    print("\n========== 2. SESSION UNDERSTANDING ==========")
    print(json.dumps(
        result.get("session_analysis", {}),
        ensure_ascii=False,
        indent=2,
    ))

    print("\n========== 3. QUERY REWRITING ==========")
    print(json.dumps(
        result.get("rewrite_metadata", {}),
        ensure_ascii=False,
        indent=2,
    ))

    print("FINAL RETRIEVAL QUERY:")
    print(result.get("retrieval_query"))

    print("\n========== 4. RETRIEVAL ==========")
    print("MODE:", result.get("mode"))
    print("CANDIDATE_K:", result.get("candidate_k"))

    for item in result.get("results", []):
        print(json.dumps(
            item,
            ensure_ascii=False,
            indent=2,
        ))

    print("\n========== 5. HISTORY ==========")
    print(json.dumps(
        result.get("history", []),
        ensure_ascii=False,
        indent=2,
    ))

    print("\n========================================")
    print("LONGPSYCH_AGENT_V2 = PASS")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
