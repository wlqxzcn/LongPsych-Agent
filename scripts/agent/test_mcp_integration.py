import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:8001/mcp"

async def main():
    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:

            print("\n========== 1. INITIALIZE ==========")
            init = await session.initialize()
            print("INITIALIZE PASS:", getattr(init, "server_info", getattr(init, "serverInfo", None)))

            print("\n========== 2. LIST TOOLS ==========")
            tools = await session.list_tools()
            actual = {tool.name for tool in tools.tools}

            for tool in tools.tools:
                print("TOOL:", tool.name)

            expected = {
                "longpsych_search_dialogue",
                "longpsych_compare_retrievers",
                "longpsych_retrieval_health",
            }
            assert expected.issubset(actual), f"Missing: {expected - actual}"
            print("TOOL REGISTRATION PASS")

            print("\n========== 3. HEALTH ==========")
            health = await session.call_tool(
                "longpsych_retrieval_health",
                arguments={}
            )
            print(health)
            assert not getattr(health, "is_error", getattr(health, "isError", False))
            print("HEALTH PASS")

            print("\n========== 4. QR-FUTURE SEARCH ==========")
            result = await session.call_tool(
                "longpsych_search_dialogue",
                arguments={
                    "query": "最近总是感到焦虑，不知道该怎么办。",
                    "mode": "qr_future",
                    "top_k": 5,
                    "candidate_k": 100
                }
            )
            print(result)
            assert not getattr(result, "is_error", getattr(result, "isError", False))
            print("QR_FUTURE SEARCH PASS")

            print("\n========== 5. THREE-WAY COMPARISON ==========")
            comparison = await session.call_tool(
                "longpsych_compare_retrievers",
                arguments={
                    "query": "最近总是感到焦虑，不知道该怎么办。",
                    "top_k": 3,
                    "candidate_k": 100
                }
            )
            print(comparison)
            assert not getattr(comparison, "is_error", getattr(comparison, "isError", False))
            print("COMPARISON PASS")

            print("\n========================================")
            print("LONGPSYCH_MCP_INTEGRATION = PASS")
            print("========================================")

asyncio.run(main())
