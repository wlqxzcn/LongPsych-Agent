import argparse
import asyncio
import json

from longpsych_agent.graph.retrieval_graph import (
    LongPsychRetrievalGraph,
)


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument(
        "--mode",
        choices=["dense", "orca", "qr_future"],
        default="qr_future",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--thread-id", default="longpsych-demo")
    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8001/mcp",
    )

    args = parser.parse_args()

    agent = LongPsychRetrievalGraph(
        mcp_url=args.mcp_url
    )

    payload = {
        "query": args.query,
        "conversation_context": args.context,
        "mode": args.mode,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
    }

    print("========== LONGPSYCH LANGGRAPH ==========")

    result = await agent.ainvoke(
        payload,
        thread_id=args.thread_id,
    )

    print("\nSTATUS:", result.get("status"))
    print("QUERY:", result.get("query"))
    print("RETRIEVAL QUERY:", result.get("retrieval_query"))
    print("MODE:", result.get("mode"))
    print("HISTORY COUNT:", len(result.get("history", [])))

    if result.get("status") != "ok":
        print("ERROR:", result.get("error"))
        raise SystemExit(1)

    print("\n========== TOP RESULTS ==========")

    for item in result.get("results", []):
        print(
            json.dumps(
                item,
                ensure_ascii=False,
                indent=2,
            )
        )

    print("\n========================================")
    print("LONGPSYCH_LANGGRAPH_RETRIEVAL = PASS")
    print("========================================")


if __name__ == "__main__":
    asyncio.run(main())
