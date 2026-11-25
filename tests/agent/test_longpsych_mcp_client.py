from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
from io import BytesIO

import pytest


CLIENT_PATH = (
    Path(__file__).resolve().parents[2]
    / "agent"
    / "longpsych_agent"
    / "mcp_server"
    / "api_client.py"
)
spec = importlib.util.spec_from_file_location("longpsych_api_client_test", CLIENT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

LongPsychAPIClient = module.LongPsychAPIClient
LongPsychAPIError = module.LongPsychAPIError


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_health_request(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:8000/health"
        return FakeResponse({"status": "ready", "corpus_size": 61501})

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    client = LongPsychAPIClient()
    assert client.health()["corpus_size"] == 61501


def test_qr_future_search_payload(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({
            "status": "ok",
            "data": {
                "mode": "qr_future",
                "candidate_k": 100,
                "results": [{"rank": 1}],
            },
        })

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    client = LongPsychAPIClient()
    result = client.search("焦虑", mode="qr_future", top_k=5, candidate_k=100)

    assert seen["url"] == "http://127.0.0.1:8000/v1/search"
    assert seen["payload"] == {
        "query": "焦虑",
        "mode": "qr_future",
        "top_k": 5,
        "candidate_k": 100,
    }
    assert result["candidate_k"] == 100


def test_dense_search_omits_candidate_k(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"status": "ok", "data": {"mode": "dense"}})

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    client = LongPsychAPIClient()
    client.search("query", mode="dense", top_k=5)
    assert "candidate_k" not in seen["payload"]


def test_compare_uses_three_modes(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        seen.append(payload)
        return FakeResponse({
            "status": "ok",
            "data": {"mode": payload["mode"], "results": []},
        })

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    client = LongPsychAPIClient()
    result = client.compare("query", top_k=3, candidate_k=100)

    assert [x["mode"] for x in seen] == ["dense", "orca", "qr_future"]
    assert "candidate_k" not in seen[0]
    assert "candidate_k" not in seen[1]
    assert seen[2]["candidate_k"] == 100
    assert result["comparison_modes"] == ["dense", "orca", "qr_future"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": " "},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 101},
        {"query": "x", "mode": "unknown"},
        {"query": "x", "mode": "dense", "candidate_k": 100},
        {"query": "x", "mode": "qr_future", "top_k": 5, "candidate_k": 4},
    ],
)
def test_invalid_search_inputs(kwargs):
    client = LongPsychAPIClient()
    with pytest.raises(ValueError):
        client.search(**kwargs)


def test_http_error_is_translated(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            503,
            "Service Unavailable",
            hdrs=None,
            fp=BytesIO(b'{"detail":"Pipeline is not ready."}'),
        )

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    client = LongPsychAPIClient()
    with pytest.raises(LongPsychAPIError, match="HTTP 503"):
        client.health()
