from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load local configuration without overriding existing environment variables.
load_dotenv(PROJECT_ROOT / ".env.agent", override=False)


def _path(
    name: str,
    default: Path,
    base: Path,
) -> Path:
    raw = os.getenv(name, "").strip()
    value = Path(raw).expanduser() if raw else default

    if not value.is_absolute():
        value = base / value

    return value.resolve()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    value = raw.strip().lower()

    if value in {"1", "true", "yes", "on"}:
        return True

    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean configuration: {name}")


@dataclass(frozen=True)
class LongPsychSettings:

    project_root: Path
    root: Path
    source_root: Path

    model_path: Path
    index_dir: Path

    device: str
    batch_size: int
    candidate_k: int
    verify_hashes: bool

    api_base_url: str
    api_timeout_seconds: float

    mcp_transport: str
    mcp_host: str
    mcp_port: int
    mcp_log_level: str
    mcp_url: str

    agent_db_path: Path

    llm_api_key: str
    llm_base_url: str | None
    llm_model: str
    llm_temperature: float
    llm_timeout: float

    @classmethod
    def from_env(cls) -> "LongPsychSettings":

        project_root = PROJECT_ROOT

        root = _path(
            "LONGPSYCH_ROOT",
            project_root,
            project_root,
        )

        model_path = _path(
            "LONGPSYCH_MODEL",
            root / "models/BAAI/bge-large-zh-v1___5",
            root,
        )

        index_dir = _path(
            "LONGPSYCH_INDEX_DIR",
            root / "data/indexes",
            root,
        )

        source_root = _path(
            "LONGPSYCH_SOURCE_ROOT",
            root,
            root,
        )

        mcp_host = os.getenv(
            "LONGPSYCH_MCP_HOST", "127.0.0.1"
        )

        mcp_port = int(
            os.getenv("LONGPSYCH_MCP_PORT", "8001")
        )

        mcp_url = os.getenv(
            "LONGPSYCH_MCP_URL",
            f"http://{mcp_host}:{mcp_port}/mcp",
        )

        base_url = (
            os.getenv("LONGPSYCH_LLM_BASE_URL", "").strip()
            or None
        )

        return cls(
            project_root=project_root,
            root=root,
            source_root=source_root,
            model_path=model_path,
            index_dir=index_dir,

            device=os.getenv(
                "LONGPSYCH_DEVICE", "cuda"
            ),

            batch_size=int(
                os.getenv("LONGPSYCH_BATCH_SIZE", "16")
            ),

            candidate_k=int(
                os.getenv("LONGPSYCH_CANDIDATE_K", "100")
            ),

            verify_hashes=_bool(
                "LONGPSYCH_VERIFY_HASHES", True
            ),

            api_base_url=os.getenv(
                "LONGPSYCH_API_BASE_URL",
                "http://127.0.0.1:8000",
            ).rstrip("/"),

            api_timeout_seconds=float(
                os.getenv(
                    "LONGPSYCH_API_TIMEOUT_SECONDS", "60"
                )
            ),

            mcp_transport=os.getenv(
                "LONGPSYCH_MCP_TRANSPORT",
                "streamable-http",
            ).strip().lower(),

            mcp_host=mcp_host,
            mcp_port=mcp_port,

            mcp_log_level=os.getenv(
                "LONGPSYCH_MCP_LOG_LEVEL", "INFO"
            ).upper(),

            mcp_url=mcp_url,

            agent_db_path=_path(
                "LONGPSYCH_AGENT_DB_PATH",
                root / "data/agent/longpsych_agent_v3.sqlite",
                root,
            ),

            llm_api_key=(
                os.getenv("LONGPSYCH_LLM_API_KEY")
                or os.getenv("OPENAI_API_KEY", "")
            ),

            llm_base_url=base_url,

            llm_model=os.getenv(
                "LONGPSYCH_LLM_MODEL", "gpt-4o-mini"
            ),

            llm_temperature=float(
                os.getenv("LONGPSYCH_LLM_TEMPERATURE", "0.2")
            ),

            llm_timeout=float(
                os.getenv("LONGPSYCH_LLM_TIMEOUT", "45")
            ),
        )


settings = LongPsychSettings.from_env()
