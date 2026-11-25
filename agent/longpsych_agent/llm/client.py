from __future__ import annotations

from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from ..config import settings


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str | None
    model: str
    temperature: float
    timeout: float


def get_llm_config() -> LLMConfig:
    if not settings.llm_api_key:
        raise RuntimeError(
            "Missing LONGPSYCH_LLM_API_KEY in .env.agent"
        )

    return LLMConfig(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
    )


def create_chat_model(
    temperature: float | None = None,
) -> ChatOpenAI:

    config = get_llm_config()

    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=(
            config.temperature
            if temperature is None
            else temperature
        ),
        timeout=config.timeout,
        max_retries=2,
    )
