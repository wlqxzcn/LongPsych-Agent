import numpy as np

from longpsych_agent.encoders import (
    normalize_text,
    l2_normalize,
)


def test_normalize_text():
    assert normalize_text(
        "  你好\n\n世界  "
    ) == "你好 世界"


def test_normalize_none():
    assert normalize_text(None) == ""


def test_normalize_multiple_spaces():
    assert normalize_text(
        "hello    world"
    ) == "hello world"


def test_l2_normalize():
    x = np.array(
        [[3.0, 4.0], [0.0, 2.0]],
        dtype=np.float32,
    )

    result = l2_normalize(x)

    assert result.shape == x.shape

    norms = np.linalg.norm(result, axis=1)

    assert np.allclose(norms, [1.0, 1.0])


def test_l2_normalize_zero_vector():
    x = np.zeros((1, 1024), dtype=np.float32)

    result = l2_normalize(x)

    assert np.isfinite(result).all()
    assert np.allclose(result, 0.0)
