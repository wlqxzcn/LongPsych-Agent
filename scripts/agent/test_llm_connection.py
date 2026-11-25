import asyncio

from langchain_core.messages import HumanMessage, SystemMessage

from longpsych_agent.llm.client import (
    create_chat_model,
    get_llm_config,
)


async def main():

    config = get_llm_config()

    print("========== LLM CONFIG ==========")
    print("MODEL:", config.model)
    print("CUSTOM_ENDPOINT:", bool(config.base_url))
    print("API_KEY: CONFIGURED")

    llm = create_chat_model(temperature=0)

    print("\n========== LLM REQUEST ==========")

    response = await llm.ainvoke([
        SystemMessage(
            content="You are a helpful assistant. "
                    "Reply with exactly: LONGPSYCH_LLM_READY"
        ),
        HumanMessage(
            content="Test the LLM connection."
        ),
    ])

    content = response.content

    if isinstance(content, list):
        content = " ".join(
            str(item) for item in content
        )

    print("RESPONSE:", content)

    if not content or "LONGPSYCH_LLM_READY" not in str(content):
        raise RuntimeError("Unexpected LLM response.")

    print("\nLONGPSYCH_LLM_CONNECTION = PASS")


if __name__ == "__main__":
    asyncio.run(main())
