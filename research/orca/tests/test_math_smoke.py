#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper_experiment_utils import fit_raw_svd_basis, l2_normalize, project_remove, vector_angle_degrees

rng = np.random.default_rng(42)
x = rng.normal(size=(50, 8)).astype("float32")
basis, values = fit_raw_svd_basis(x, 2)
assert basis.shape == (8, 2)
assert values.shape == (8,)
assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-5)
y = project_remove(x, basis)
assert np.max(np.abs(y @ basis)) < 1e-4
assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0, atol=1e-5)
assert vector_angle_degrees(basis[:, 0], basis[:, 0]) < 2e-6
print("math smoke test passed")
