from pathlib import Path
from getpass import getpass

from dotenv import dotenv_values, set_key

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env.agent"

old = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}

print("=" * 55)
print("LongPsychAgent LLM Configuration")
print("=" * 55)

api_key = getpass("LLM API Key (hidden): ").strip()
if not api_key:
    api_key = old.get("LONGPSYCH_LLM_API_KEY", "")

if not api_key:
    raise RuntimeError("API Key cannot be empty.")

base_url = input(
    "LLM Base URL (leave blank for default OpenAI): "
).strip()

if not base_url:
    base_url = old.get("LONGPSYCH_LLM_BASE_URL", "")

model = input(
    f"Model [{old.get('LONGPSYCH_LLM_MODEL') or 'gpt-4o-mini'}]: "
).strip()

if not model:
    model = old.get("LONGPSYCH_LLM_MODEL") or "gpt-4o-mini"

set_key(str(ENV_FILE), "LONGPSYCH_LLM_API_KEY", api_key)
set_key(str(ENV_FILE), "LONGPSYCH_LLM_BASE_URL", base_url)
set_key(str(ENV_FILE), "LONGPSYCH_LLM_MODEL", model)
set_key(str(ENV_FILE), "LONGPSYCH_LLM_TEMPERATURE", "0.2")
set_key(str(ENV_FILE), "LONGPSYCH_LLM_TIMEOUT", "45")

ENV_FILE.chmod(0o600)

print("\nLLM CONFIGURATION SAVED")
print("Model:", model)
print("Custom Base URL:", bool(base_url))
print("API Key: configured (hidden)")
print("File permissions: 600")
