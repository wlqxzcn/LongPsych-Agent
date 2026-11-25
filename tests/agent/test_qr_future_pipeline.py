from __future__ import annotations

import numpy as np
import pytest

from longpsych_agent.reranking.qr_future_pipeline import QRFutureRetrievalPipeline


class FakeIndex:
    ntotal = 3


class FakeDenseRetriever:
    def __init__(self):
        self.index = FakeIndex()
        self.document_ids = ["u0", "u1", "u2"]


class FakeEncoder:
    def encode(self, texts, normalize=True):
        assert normalize is True
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)


class FakeRetriever:
    dimension = 3
    corpus_size = 3

    def __init__(self):
        self.encoder = FakeEncoder()
        self.dense_retriever = FakeDenseRetriever()
        self.last_mode = None
        self.last_top_k = None

    def search_embeddings(self, embeddings, top_k=5, mode=None, query_texts=None):
        self.last_mode = mode
        self.last_top_k = top_k
        return {
            "query": query_texts[0],
            "mode": mode,
            "results": [
                {
                    "rank": 1,
                    "response_uid": "u0",
                    "response_text": "text 0",
                    "retrieval_score": 0.9,
                    "faiss_row": 0,
                },
                {
                    "rank": 2,
                    "response_uid": "u1",
                    "response_text": "text 1",
                    "retrieval_score": 0.8,
                    "faiss_row": 1,
                },
                {
                    "rank": 3,
                    "response_uid": "u2",
                    "response_text": "text 2",
                    "retrieval_score": 0.7,
                    "faiss_row": 2,
                },
            ][:top_k],
        }

    def search(self, query, top_k=5, mode=None):
        if isinstance(query, str):
            return {"query": query, "mode": mode, "results": []}
        return [{"query": q, "mode": mode, "results": []} for q in query]


class FakeScorer:
    EXPECTED_DIMENSION = 3
    EXPECTED_CONDITION = "QR_FUTURE_FULL"

    def __init__(self):
        self.response_embeddings = np.zeros((3, 3), dtype=np.float32)
        self.faiss_to_original = np.arange(3, dtype=np.int64)
        self.row_mapping = [
            {"faiss_row": 0, "original_embedding_row": 0, "response_uid": "u0"},
            {"faiss_row": 1, "original_embedding_row": 1, "response_uid": "u1"},
            {"faiss_row": 2, "original_embedding_row": 2, "response_uid": "u2"},
        ]
        self.folds = [object() for _ in range(5)]

    def rerank_results(self, query_embedding, retrieval_results):
        # Deliberately reverse candidate order to test that the pipeline uses
        # QR-Future scores/ranks rather than preserving first-stage order.
        output = []
        for rank, item in enumerate(reversed(retrieval_results), start=1):
            output.append(
                {
                    **item,
                    "retrieval_score": item["score"],
                    "qr_future_score": float(10 - rank),
                    "rerank_rank": rank,
                }
            )
        return output

    def health(self):
        return {"status": "ready", "fold_count": 5}


@pytest.fixture
def pipeline():
    return QRFutureRetrievalPipeline(
        retriever=FakeRetriever(),
        qr_future_scorer=FakeScorer(),
        default_candidate_k=3,
    )


def test_qr_future_uses_dense_first_stage_and_reranks(pipeline):
    result = pipeline.search("test query", top_k=2, mode="qr_future")
    assert result["first_stage_mode"] == "dense"
    assert result["candidate_k"] == 3
    assert pipeline.retriever.last_mode == "dense"
    assert pipeline.retriever.last_top_k == 3
    assert [item["response_uid"] for item in result["results"]] == ["u2", "u1"]
    assert [item["rank"] for item in result["results"]] == [1, 2]


def test_dense_and_orca_modes_are_forwarded(pipeline):
    assert pipeline.search("q", mode="dense")["mode"] == "dense"
    assert pipeline.search("q", mode="orca")["mode"] == "orca"


def test_batch_qr_future(pipeline):
    results = pipeline.search(["q1", "q2"], top_k=1, mode="qr_future")
    assert len(results) == 2
    assert all(item["mode"] == "qr_future" for item in results)


def test_candidate_k_must_cover_top_k(pipeline):
    with pytest.raises(ValueError):
        pipeline.search("q", top_k=3, mode="qr_future", candidate_k=2)


def test_empty_query_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.search("   ", mode="qr_future")


def test_row_mapping_uid_mismatch_rejected():
    scorer = FakeScorer()
    scorer.row_mapping[1]["response_uid"] = "wrong_uid"
    with pytest.raises(ValueError, match="UID mismatch"):
        QRFutureRetrievalPipeline(FakeRetriever(), scorer, default_candidate_k=3)
