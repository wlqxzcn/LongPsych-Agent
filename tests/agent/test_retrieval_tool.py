import pytest

from longpsych_agent.tools import LongPsychTool


class DummyRetriever:

    def __init__(self):
        self.calls = []
        self.payload = {
            "query": "测试查询",
            "mode": "orca",
            "results": [
                {
                    "rank": 1,
                    "response_uid": "resp_test_001",
                    "response_text": "示例回复",
                    "retrieval_score": 0.95,
                    "faiss_row": 0,
                }
            ],
        }

    def search(self, query, top_k=5, mode=None):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "mode": mode,
        })

        if isinstance(query, str):
            return self.payload

        return [self.payload for _ in query]


@pytest.fixture
def tool():
    retriever = DummyRetriever()
    return LongPsychTool(retriever), retriever


def test_single_search(tool):
    obj, retriever = tool

    result = obj.search("测试查询")

    assert result is retriever.payload
    assert retriever.calls[-1]["query"] == "测试查询"
    assert retriever.calls[-1]["top_k"] == 5


def test_dense_search(tool):
    obj, retriever = tool

    obj.search("测试查询", mode="dense")

    assert retriever.calls[-1]["mode"] == "dense"


def test_orca_search(tool):
    obj, retriever = tool

    obj.search("测试查询", mode="orca")

    assert retriever.calls[-1]["mode"] == "orca"


def test_batch_search(tool):
    obj, retriever = tool

    queries = ["第一个查询", "第二个查询"]

    result = obj.batch_search(queries)

    assert len(result) == 2
    assert retriever.calls[-1]["query"] == queries


def test_batch_orca_search(tool):
    obj, retriever = tool

    obj.batch_search(
        ["查询一", "查询二"],
        mode="orca",
        top_k=10,
    )

    call = retriever.calls[-1]

    assert call["mode"] == "orca"
    assert call["top_k"] == 10


def test_callable_interface(tool):
    obj, retriever = tool

    result = obj("测试查询")

    assert result is retriever.payload


def test_empty_query_rejected(tool):
    obj, _ = tool

    with pytest.raises(ValueError):
        obj.search("")


def test_whitespace_query_rejected(tool):
    obj, _ = tool

    with pytest.raises(ValueError):
        obj.search("   ")


def test_invalid_batch_rejected(tool):
    obj, _ = tool

    with pytest.raises(TypeError):
        obj.batch_search("不是列表")


def test_empty_batch_rejected(tool):
    obj, _ = tool

    with pytest.raises(ValueError):
        obj.batch_search([])


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_rejected(tool, top_k):
    obj, _ = tool

    with pytest.raises(ValueError):
        obj.search("测试查询", top_k=top_k)


def test_non_integer_top_k_rejected(tool):
    obj, _ = tool

    with pytest.raises(TypeError):
        obj.search("测试查询", top_k=2.5)


def test_tool_schema(tool):
    obj, _ = tool

    schema = obj.tool_schema

    assert schema["type"] == "function"

    function = schema["function"]

    assert function["name"] == "longpsych_search"

    assert "query" in function["parameters"]["required"]

    assert function["parameters"]["additionalProperties"] is False
