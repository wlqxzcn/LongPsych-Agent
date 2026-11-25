import numpy as np
import pytest

from longpsych_agent.retrieval.unified_retriever import (
    LongPsychRetriever,
)


class DummyEncoder:
    dimension = 3

    def encode(self, texts, normalize=True):
        return np.tile(
            np.array([[3, 4, 0]], dtype=np.float32),
            (len(texts), 1),
        )


class DummyRetriever:
    dimension = 3

    def __init__(self):
        self.documents = ["示例回复"]
        self.document_ids = ["resp_test_001"]
        self.last_queries = None

    def search(self, embeddings, top_k=5):
        self.last_queries = embeddings.copy()

        return [
            [{
                "id": "resp_test_001",
                "row": 0,
                "text": "示例回复",
                "score": float(q[0]),
            }]
            for q in embeddings
        ]


class DummyProjector:
    def __init__(self):
        self.basis = np.array(
            [[1], [0], [0]],
            dtype=np.float32,
        )
        self.calls = 0

    def transform_queries(self, embeddings):
        self.calls += 1
        return embeddings.copy()


@pytest.fixture
def retriever():
    dense = DummyRetriever()
    orca = DummyRetriever()
    projector = DummyProjector()

    obj = LongPsychRetriever(
        encoder=DummyEncoder(),
        dense_retriever=dense,
        orca_retriever=orca,
        orca_projector=projector,
        default_mode="orca",
    )

    return obj, dense, orca, projector


def test_default_orca_mode(retriever):
    obj, dense, orca, projector = retriever

    result = obj.search("测试查询", top_k=5)

    assert result["mode"] == "orca"
    assert len(result["results"]) == 1
    assert projector.calls == 1


def test_dense_mode(retriever):
    obj, dense, orca, projector = retriever

    result = obj.search(
        "测试查询",
        mode="dense",
    )

    assert result["mode"] == "dense"
    assert projector.calls == 0
    assert dense.last_queries is not None


def test_orca_alias(retriever):
    obj, _, _, _ = retriever

    result = obj.search(
        "测试查询",
        mode="orca_k2",
    )

    assert result["mode"] == "orca"


def test_batch_search(retriever):
    obj, _, _, _ = retriever

    results = obj.search([
        "第一个查询",
        "第二个查询",
    ])

    assert len(results) == 2
    assert results[0]["query"] == "第一个查询"
    assert results[1]["query"] == "第二个查询"


def test_result_schema(retriever):
    obj, _, _, _ = retriever

    result = obj.search("测试查询")

    item = result["results"][0]

    assert item["rank"] == 1
    assert item["response_uid"] == "resp_test_001"
    assert item["response_text"] == "示例回复"
    assert "retrieval_score" in item
    assert "faiss_row" in item


def test_empty_query_rejected(retriever):
    obj, _, _, _ = retriever

    with pytest.raises(ValueError):
        obj.search("")


def test_invalid_mode_rejected(retriever):
    obj, _, _, _ = retriever

    with pytest.raises(ValueError):
        obj.search("测试查询", mode="invalid")


def test_dimension_mismatch(retriever):
    obj, _, _, _ = retriever

    with pytest.raises(ValueError):
        obj.search_embeddings(
            np.zeros(768, dtype=np.float32)
        )


def test_zero_embedding_rejected(retriever):
    obj, _, _, _ = retriever

    with pytest.raises(ValueError):
        obj.search_embeddings(
            np.zeros(3, dtype=np.float32)
        )
