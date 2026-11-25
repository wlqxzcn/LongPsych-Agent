import numpy as np
import pytest

from longpsych_agent.retrieval import ORCAProjector


def legacy_fit(source, k):
    _, _, vt = np.linalg.svd(
        source.astype("float32"),
        full_matrices=False,
    )
    return vt[:k].T.astype("float32")


def legacy_project(x, basis):
    return (
        x - (x @ basis) @ basis.T
    ).astype("float32")


def test_raw_svd_basis_parity():
    rng = np.random.default_rng(42)
    source = rng.normal(
        size=(100, 32)
    ).astype("float32")

    expected = legacy_fit(source, 2)

    model = ORCAProjector(k=2)
    model.fit(source)

    np.testing.assert_allclose(
        model.basis,
        expected,
        atol=0,
        rtol=0,
    )


def test_projection_parity():
    rng = np.random.default_rng(123)

    source = rng.normal(
        size=(100, 32)
    ).astype("float32")

    queries = rng.normal(
        size=(20, 32)
    ).astype("float32")

    basis = legacy_fit(source, 2)
    expected = legacy_project(queries, basis)

    model = ORCAProjector(k=2)
    model.fit(source)

    actual = model.project(queries)

    np.testing.assert_allclose(
        actual,
        expected,
        atol=0,
        rtol=0,
    )


def test_transform_is_l2_normalized():
    rng = np.random.default_rng(42)

    source = rng.normal(
        size=(100, 32)
    ).astype("float32")

    model = ORCAProjector(k=2)
    model.fit(source)

    result = model.transform(source[:10])

    norms = np.linalg.norm(
        result,
        axis=1,
    )

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_single_vector_input():
    rng = np.random.default_rng(42)

    source = rng.normal(
        size=(100, 32)
    ).astype("float32")

    model = ORCAProjector(k=2)
    model.fit(source)

    result = model.transform(source[0])

    assert result.shape == (32,)
    assert np.isfinite(result).all()


def test_dimension_mismatch():
    source = np.random.randn(
        100, 32
    ).astype("float32")

    model = ORCAProjector(k=2)
    model.fit(source)

    with pytest.raises(ValueError):
        model.transform(
            np.random.randn(10, 64)
        )


def test_reject_nan_source():
    source = np.random.randn(
        100, 32
    ).astype("float32")

    source[0, 0] = np.nan

    model = ORCAProjector(k=2)

    with pytest.raises(ValueError):
        model.fit(source)


def test_save_load_parity(tmp_path):
    rng = np.random.default_rng(42)

    source = rng.normal(
        size=(100, 32)
    ).astype("float32")

    queries = rng.normal(
        size=(10, 32)
    ).astype("float32")

    model = ORCAProjector(k=2)
    model.fit(source)

    expected = model.transform(queries)

    path = tmp_path / "orca_basis.npz"
    model.save(path)

    loaded = ORCAProjector.load(path)

    actual = loaded.transform(queries)

    np.testing.assert_array_equal(
        actual,
        expected,
    )

    assert (
        loaded.fit_metadata["basis_sha256"]
        == model.fit_metadata["basis_sha256"]
    )


def test_unfitted_projector():
    model = ORCAProjector(k=2)

    with pytest.raises(RuntimeError):
        model.transform(
            np.random.randn(10, 32)
        )
