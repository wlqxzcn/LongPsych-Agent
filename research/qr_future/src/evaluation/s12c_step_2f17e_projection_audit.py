import json
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASIS_DIR = ROOT / "results/phase_s12c/step_2f17c_orca_fold_basis_v1"
PROJ_DIR = ROOT / "results/phase_s12c/step_2f17d_orca_fold_projection_v1"

QUERY_SRC = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
RESPONSE_SRC = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

REPORT = PROJ_DIR / "integrity_audit_v1.json"

def sha_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()

def sha_memmap(x, chunk=4096):
    h = hashlib.sha256()
    for s in range(0, len(x), chunk):
        h.update(np.ascontiguousarray(x[s:s+chunk]).tobytes())
    return h.hexdigest()

def inspect_array(x, expected_shape, expected_hash):
    assert x.shape == expected_shape, x.shape
    assert x.dtype == np.float32, x.dtype

    max_norm_error = 0.0

    for s in range(0, len(x), 4096):
        z = np.asarray(x[s:s+4096])

        assert np.isfinite(z).all()

        norms = np.linalg.norm(z, axis=1)

        max_norm_error = max(
            max_norm_error,
            float(np.max(np.abs(norms - 1.0)))
        )

    actual_hash = sha_memmap(x)

    assert actual_hash == expected_hash, "ARRAY_HASH_MISMATCH"
    assert max_norm_error < 2e-5, "NORMALIZATION_FAILED"

    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "sha256": actual_hash,
        "max_norm_error": max_norm_error
    }

def verify_projection(src, projected, basis, seed):
    rng = np.random.default_rng(seed)
    indices = np.sort(
        rng.choice(len(src), size=32, replace=False)
    )

    x = np.asarray(src[indices], dtype=np.float32)

    z = x - (x @ basis) @ basis.T
    z /= np.linalg.norm(z, axis=1, keepdims=True)

    diff = np.max(
        np.abs(z - np.asarray(projected[indices]))
    )

    assert diff < 2e-6, f"PROJECTION_FORMULA_MISMATCH: {diff}"

    return float(diff)

assert not REPORT.exists(), "OUTPUT_COLLISION"

assert QUERY_SRC.exists()
assert RESPONSE_SRC.exists()
assert BASIS_DIR.exists()
assert PROJ_DIR.exists()

basis_meta = json.loads(
    (BASIS_DIR / "summary.json").read_text()
)

proj_meta = json.loads(
    (PROJ_DIR / "summary.json").read_text()
)

assert basis_meta["status"] == "COMPLETED"
assert proj_meta["status"] == "COMPLETED"

query_src = np.load(QUERY_SRC, mmap_mode="r")
response_src = np.load(RESPONSE_SRC, mmap_mode="r")

assert query_src.shape == (62889, 1024)
assert response_src.shape == (61502, 1024)

print("=" * 72)
print("STEP 2F17E: PROJECTION INTEGRITY AUDIT")
print("=" * 72)

results = {}

for f in range(5):
    print(f"\nFOLD {f}")

    basis_path = BASIS_DIR / f"fold{f}_orca_basis_k2_v1.npy"
    basis = np.load(basis_path).astype(np.float32)

    basis_expected = basis_meta["fold_results"][str(f)]["basis_sha256"]

    assert sha_array(basis) == basis_expected
    assert basis.shape == (1024, 2)

    meta = proj_meta["folds"][str(f)]

    fold_dir = PROJ_DIR / f"fold{f}"

    q = np.load(
        fold_dir / "query_orca_k2_v1.npy",
        mmap_mode="r"
    )

    r = np.load(
        fold_dir / "response_orca_k2_v1.npy",
        mmap_mode="r"
    )

    q_result = inspect_array(
        q, (62889, 1024), meta["query"]["array_sha256"]
    )

    r_result = inspect_array(
        r, (61502, 1024), meta["response"]["array_sha256"]
    )

    q_diff = verify_projection(query_src, q, basis, 20261109 + f)
    r_diff = verify_projection(response_src, r, basis, 20261209 + f)

    assert meta["basis_sha256"] == basis_expected

    results[str(f)] = {
        "basis_hash_pass": True,
        "query": q_result,
        "response": r_result,
        "query_formula_max_abs_diff": q_diff,
        "response_formula_max_abs_diff": r_diff,
        "status": "PASS"
    }

    print("BASIS_HASH=PASS")
    print("QUERY_HASH=PASS")
    print("RESPONSE_HASH=PASS")
    print("QUERY_FORMULA=PASS", q_diff)
    print("RESPONSE_FORMULA=PASS", r_diff)

report = {
    "artifact": "STEP_2F17E_PROJECTION_INTEGRITY_AUDIT",
    "status": "COMPLETED",
    "folds_passed": 5,
    "results": results,
    "training": False,
    "inference": False,
    "test_evaluation": False,
    "original_artifacts_modified": False
}

REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2)
)

print("\n" + "=" * 72)
print("STEP_2F17E_COMPLETE")
print("FOLDS_PASSED = 5/5")
print("PROJECTION_HASH = PASS")
print("PROJECTION_FORMULA = PASS")
print("NORMALIZATION = PASS")
print("TRAINING = NO")
print("TEST_EVALUATION = NO")
print("OUTPUT =", REPORT)
print("=" * 72)
