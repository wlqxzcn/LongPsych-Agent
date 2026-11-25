import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT = Path(__file__).resolve().parents[2]


def decode_result(raw):

    if isinstance(raw, dict):
        return raw

    if hasattr(raw, "content"):
        raw = raw.content

    if isinstance(raw, str):
        return json.loads(raw)

    if isinstance(raw, list):
        for item in raw:
            text = (
                item.get("text")
                if isinstance(item, dict)
                else getattr(item, "text", None)
            )

            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue

    raise RuntimeError("Cannot decode MCP response.")


def get_uids(results):
    return {
        x["response_uid"]
        for x in results
        if x.get("response_uid")
    }


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--query", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--rewrite", required=True)

    parser.add_argument(
        "--selected-context-item",
        action="append",
        default=[],
    )

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument(
        "--mcp-url",
        default="http://127.0.0.1:8001/mcp",
    )

    args = parser.parse_args()

    selected = args.selected_context_item

    if not selected:
        raise RuntimeError("D group needs selected context items.")

    # 四组查询
    variants = {
        "A_ORIGINAL": args.query,

        "B_GENERIC_REWRITE": args.rewrite,

        "C_CONTEXT_CONCAT": (
            args.query
            + "\n相关历史背景："
            + args.context
        ),

        "D_ANCHOR_PRESERVING": (
            args.query
            + "\n相关历史背景："
            + "".join(selected)
        ),
    }

    # D组必须以原始查询为锚点
    assert variants["D_ANCHOR_PRESERVING"].startswith(
        args.query
    )

    client = MultiServerMCPClient({
        "longpsych": {
            "transport": "http",
            "url": args.mcp_url,
        }
    })

    tools = await client.get_tools()

    tool = next(
        (
            x for x in tools
            if x.name == "longpsych_search_dialogue"
        ),
        None,
    )

    if tool is None:
        raise RuntimeError("MCP retrieval tool missing.")

    output = {
        "protocol": "QR_FUTURE_FULL_5FOLD",
        "first_stage": "BGE_DENSE",
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,
        "variants": {},
        "topk_jaccard": {},
    }

    print("=" * 65)
    print("FOUR-WAY QUERY CONSTRUCTION AUDIT")
    print("=" * 65)

    for name, query in variants.items():

        print(f"\n========== {name} ==========")
        print("QUERY:", query)

        raw = await tool.ainvoke({
            "query": query,
            "mode": "qr_future",
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
        })

        payload = decode_result(raw)

        if payload.get("status") != "ok":
            raise RuntimeError(
                f"{name} failed: {payload}"
            )

        data = payload["data"]
        results = data["results"]

        output["variants"][name] = {
            "query": query,
            "results": results,
        }

        for item in results:
            print(
                item["rank"],
                item["response_uid"],
                item["response_text"],
                "candidate_rank=",
                item.get("candidate_rank"),
                "qr_score=",
                item.get("qr_future_score"),
            )

    names = list(variants)

    print("\n========== TOP-10 OVERLAP ==========")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):

            a, b = names[i], names[j]

            score = jaccard(
                get_uids(output["variants"][a]["results"]),
                get_uids(output["variants"][b]["results"]),
            )

            key = f"{a}_vs_{b}"

            output["topk_jaccard"][key] = score

            print(key, f"{score:.4f}")

    out_dir = ROOT / "results/agent"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (
        "query_rewriting_v2_audit_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    out_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nAUDIT FILE:", out_path)
    print("FOUR_WAY_QUERY_AUDIT = PASS")


if __name__ == "__main__":
    asyncio.run(main())
