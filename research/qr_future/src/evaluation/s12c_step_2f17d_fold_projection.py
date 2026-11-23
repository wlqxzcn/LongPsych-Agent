import os
import json
import hashlib
import shutil
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

BASIS_DIR = ROOT / "results/phase_s12c/step_2f17c_orca_fold_basis_v1"

QUERY_PATH = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
RESPONSE_PATH = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

OUT = ROOT / "results/phase_s12c/step_2f17d_orca_fold_projection_v1"

CHUNK = 2048

def sha_array(x):
    return hashlib.sha256(
        np.ascontiguousarray(x).tobytes()
    ).hexdigest()

def project_block(x, basis):
    z = x - (x @ basis) @ basis.T
    norms = np.linalg.norm(z, axis=1, keepdims=True)

    if np.any(norms < 1e-10):
        raise RuntimeError("ZERO_PROJECTED_VECTOR")

    z /= norms
    return z.astype(np.float32)

def project_to_file(source, basis, output):
    tmp = output.with_suffix(output.suffix + ".tmp")

    if output.exists() or tmp.exists():
        raise RuntimeError(f"OUTPUT_COLLISION: {output}")

    n, d = source.shape

    mm = np.lib.format.open_memmap(
        tmp,
        mode="w+",
        dtype=np.float32,
        shape=(n, d)
    )

    digest = hashlib.sha256()
    max_norm_error = 0.0

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)

        x = np.asarray(source[start:end], dtype=np.float32)
        z = project_block(x, basis)

        norms = np.linalg.norm(z, axis=1)
        max_norm_error = max(
            max_norm_error,
            float(np.max(np.abs(norms - 1.0)))
        )

        mm[start:end] = z
        digest.update(z.tobytes())

        if end % 8192 == 0 or end == n:
            print(
                f"PROGRESS {output.name}: {end}/{n}",
                flush=True
            )

    mm.flush()
    del mm

    os.replace(tmp, output)

    return {
        "shape": [n, d],
        "dtype": "float32",
        "array_sha256": digest.hexdigest(),
        "max_norm_error": max_norm_error,
        "file_size": output.stat().st_size
    }

print("=" * 76)
print("STEP 2F17D: FIVE-FOLD ORCA EMBEDDING PROJECTION")
print("=" * 76)

assert BASIS_DIR.exists()
assert QUERY_PATH.exists()
assert RESPONSE_PATH.exists()
assert not OUT.exists()

basis_summary = json.loads(
    (BASIS_DIR / "summary.json").read_text()
)

assert basis_summary["status"] == "COMPLETED"

query = np.load(QUERY_PATH, mmap_mode="r")
response = np.load(RESPONSE_PATH, mmap_mode="r")

assert query.shape == (62889, 1024)
assert response.shape == (61502, 1024)

required_bytes = (
    (query.shape[0] + response.shape[0])
    * query.shape[1] * 4 * 5
)

free_bytes = shutil.disk_usage(ROOT).free

print("QUERY_SHAPE =", query.shape)
print("RESPONSE_SHAPE =", response.shape)
print("EXPECTED_OUTPUT_GB =", required_bytes / 1e9)
print("FREE_DISK_GB =", free_bytes / 1e9)

if free_bytes < required_bytes + 512 * 1024**2:
    raise RuntimeError("INSUFFICIENT_DISK_SPACE")

OUT.mkdir(parents=True)

results = {}

for f in range(5):

    print("\n" + "=" * 70)
    print("FOLD =", f)
    print("=" * 70)

    basis_path = BASIS_DIR / f"fold{f}_orca_basis_k2_v1.npy"

    basis = np.load(basis_path).astype(np.float32)

    assert basis.shape == (1024, 2)

    expected_hash = basis_summary["fold_results"][str(f)]["basis_sha256"]

    assert sha_array(basis) == expected_hash

    assert np.max(
        np.abs(basis.T @ basis - np.eye(2))
    ) < 1e-5

    fold_out = OUT / f"fold{f}"
    fold_out.mkdir()

    q_result = project_to_file(
        query,
        basis,
        fold_out / "query_orca_k2_v1.npy"
    )

    r_result = project_to_file(
        response,
        basis,
        fold_out / "response_orca_k2_v1.npy"
    )

    results[str(f)] = {
        "basis_sha256": expected_hash,
        "query": q_result,
        "response": r_result,
        "projection_method": "x - (xB)B.T",
        "post_projection_normalization": "L2"
    }

    (fold_out / "manifest.json").write_text(
        json.dumps(
            results[str(f)],
            ensure_ascii=False,
            indent=2
        )
    )

    print("QUERY_PROJECTION=PASS")
    print("RESPONSE_PROJECTION=PASS")
    print("QUERY_MAX_NORM_ERROR =", q_result["max_norm_error"])
    print("RESPONSE_MAX_NORM_ERROR =", r_result["max_norm_error"])

summary = {
    "artifact": "STEP_2F17D_FIVE_FOLD_ORCA_PROJECTIONS",
    "status": "COMPLETED",
    "folds": results,
    "training": False,
    "inference": False,
    "test_evaluation": False,
    "original_embeddings_modified": False
}

(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2)
)

print("\n" + "=" * 76)
print("STEP_2F17D_COMPLETE")
print("FOLDS = 5")
print("QUERY_PROJECTION = PASS")
print("RESPONSE_PROJECTION = PASS")
print("TRAINING = NO")
print("TEST_EVALUATION = NO")
print("ORIGINAL_EMBEDDINGS_MODIFIED = NO")
print("OUTPUT =", OUT)
print("=" * 76)
