import numpy as np
import pytest

from longpsych_agent.retrieval import FAISSRetriever


def test_build_and_search():
    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(100, 32)
    ).astype("float32")

    documents = [
        f"response_{i}" for i in range(100)
    ]

    model = FAISSRetriever()
    model.build(vectors, documents)

    query = rng.normal(
        size=32
    ).astype("float32")

    results = model.search(query, top_k=5)

    assert len(results) == 5
    assert results[0]["score"] >= results[-1]["score"]


def test_exact_bruteforce_parity():
    rng = np.random.default_rng(123)

    vectors = rng.normal(
        size=(200, 32)
    ).astype("float32")

    query = rng.normal(
        size=32
    ).astype("float32")

    docs = [str(i) for i in range(200)]

    model = FAISSRetriever()
    model.build(vectors, docs)

    actual = model.search(query, top_k=10)

    v = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True),
        1e-12,
    )

    q = query / np.maximum(
        np.linalg.norm(query),
        1e-12,
    )

    scores = v @ q

    expected = np.argsort(-scores)[:10]

    actual_rows = [
        item["row"] for item in actual
    ]

    assert actual_rows == expected.tolist()


def test_batch_search():
    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(100, 32)
    ).astype("float32")

    queries = rng.normal(
        size=(5, 32)
    ).astype("float32")

    model = FAISSRetriever()
    model.build(vectors, [str(i) for i in range(100)])

    results = model.search(
        queries,
        top_k=5,
    )

    assert len(results) == 5
    assert all(len(x) == 5 for x in results)


def test_topk_larger_than_corpus():
    vectors = np.random.randn(
        10, 32
    ).astype("float32")

    model = FAISSRetriever()
    model.build(vectors, [str(i) for i in range(10)])

    results = model.search(vectors[0], top_k=100)

    assert len(results) == 10


def test_custom_document_ids():
    vectors = np.random.randn(
        5, 32
    ).astype("float32")

    ids = ["a", "b", "c", "d", "e"]

    model = FAISSRetriever()
    model.build(
        vectors,
        ["A", "B", "C", "D", "E"],
        document_ids=ids,
    )

    results = model.search(vectors[0], top_k=1)

    assert results[0]["id"] == "a"


def test_duplicate_ids_rejected():
    vectors = np.random.randn(
        3, 32
    ).astype("float32")

    model = FAISSRetriever()

    with pytest.raises(ValueError):
        model.build(
            vectors,
            ["A", "B", "C"],
            document_ids=["x", "x", "y"],
        )


def test_dimension_mismatch():
    vectors = np.random.randn(
        10, 32
    ).astype("float32")

    model = FAISSRetriever()
    model.build(vectors, [str(i) for i in range(10)])

    with pytest.raises(ValueError):
        model.search(
            np.random.randn(64),
            top_k=5,
        )


def test_save_load_parity(tmp_path):
    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(100, 32)
    ).astype("float32")

    queries = rng.normal(
        size=(5, 32)
    ).astype("float32")

    docs = [f"text_{i}" for i in range(100)]

    model = FAISSRetriever()
    model.build(vectors, docs)

    expected = model.search(queries, top_k=10)

    model.save(tmp_path)

    loaded = FAISSRetriever.load(tmp_path)

    actual = loaded.search(queries, top_k=10)

    for a, b in zip(expected, actual):
        assert [x["row"] for x in a] == [
            x["row"] for x in b
        ]

        np.testing.assert_allclose(
            [x["score"] for x in a],
            [x["score"] for x in b],
            atol=0,
            rtol=0,
        )


def test_zero_vector():
    vectors = np.zeros(
        (5, 32),
        dtype="float32",
    )

    model = FAISSRetriever()
    model.build(vectors, [str(i) for i in range(5)])

    results = model.search(vectors[0], top_k=3)

    assert len(results) == 3
    assert all(
        np.isfinite(x["score"]) for x in results
    )


def test_unbuilt_index():
    model = FAISSRetriever()

    with pytest.raises(RuntimeError):
        model.search(
            np.random.randn(32),
            top_k=5,
        )
