import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():

    print("========== 1. CONNECT MCP ==========")

    client = MultiServerMCPClient({
        "longpsych": {
            "transport": "http",
            "url": "http://127.0.0.1:8001/mcp"
        }
    })

    tools = await client.get_tools()

    print("\n========== 2. DISCOVER TOOLS ==========")

    for tool in tools:
        print("TOOL:", tool.name)

    expected = {
        "longpsych_search_dialogue",
        "longpsych_compare_retrievers",
        "longpsych_retrieval_health"
    }

    actual = {tool.name for tool in tools}

    assert expected.issubset(actual), (
        f"Missing tools: {expected - actual}"
    )

    print("LANGCHAIN_MCP_DISCOVERY_PASS")

    health_tool = next(
        t for t in tools
        if t.name == "longpsych_retrieval_health"
    )

    search_tool = next(
        t for t in tools
        if t.name == "longpsych_search_dialogue"
    )

    print("\n========== 3. HEALTH TOOL ==========")

    health = await health_tool.ainvoke({})

    print(health)
    print("LANGCHAIN_HEALTH_PASS")

    print("\n========== 4. QR-FUTURE SEARCH ==========")

    result = await search_tool.ainvoke({
        "query": "最近总是感到焦虑，不知道该怎么办。",
        "mode": "qr_future",
        "top_k": 5,
        "candidate_k": 100
    })

    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            default=str
        ))

    print("\nLANGCHAIN_QR_FUTURE_SEARCH_PASS")

    print("\n====================================")
    print("LANGCHAIN_MCP_ADAPTER = PASS")
    print("====================================")


if __name__ == "__main__":
    asyncio.run(main())
