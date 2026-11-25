import sys
import json
import hashlib
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT / "agent"))

from longpsych_agent.config import settings

SOURCE_ROOT = settings.source_root

from longpsych_agent.retrieval.faiss_retriever import FAISSRetriever
from longpsych_agent.retrieval.orca_projector import ORCAProjector


# ============================================================
# 1. PATHS
# ============================================================

RESPONSE_FILE = (
    ORIGINAL / "data/canonical/fsar_s2/"
    "train_unique_responses_v1.jsonl"
)

RESPONSE_INDEX_FILE = (
    ORIGINAL / "cache/embeddings/"
    "fsar_train_unique_responses_bge_index_v1.jsonl"
)

DENSE_EMBEDDING_FILE = (
    ORIGINAL / "cache/embeddings/"
    "fsar_train_unique_responses_bge_v1.npy"
)

QUERY_EMBEDDING_FILE = (
    ORIGINAL / "cache/embeddings/"
    "fsar_train_queries_bge_v1.npy"
)

EMBEDDING_MANIFEST = (
    ORIGINAL / "cache/embeddings/"
    "fsar_s2b2_full_embedding_manifest_v1.json"
)

ORCA_BASIS_FILE = (
    ORIGINAL / "results/phase_s12c/"
    "orca_baseline_v1/orca_source_basis_k2_v1.npy"
)

OUTPUT_ROOT = PROJECT / "data/indexes"

DENSE_OUTPUT = OUTPUT_ROOT / "dense_bge_v1"
ORCA_OUTPUT = OUTPUT_ROOT / "orca_k2_v1"

EXPECTED_N = 61502
EXPECTED_DIM = 1024

EXPECTED_BASIS_SHA = (
    "aaa0e7d3f04aeea62498d3dd8a7ab917"
    "b02841f6fc3bce90a7524baa87666be3"
)


# ============================================================
# 2. UTILITIES
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def sha256_array(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# 3. INPUT PROVENANCE
# ============================================================

print("\n========== 1. INPUT PROVENANCE ==========")

with open(EMBEDDING_MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

require(
    sha256_file(RESPONSE_FILE)
    == manifest["response_input_sha256"],
    "Response source SHA mismatch"
)

require(
    sha256_file(DENSE_EMBEDDING_FILE)
    == manifest["artifact_sha256"][
        "fsar_train_unique_responses_bge_v1.npy"
    ],
    "Dense embedding SHA mismatch"
)

require(
    sha256_file(RESPONSE_INDEX_FILE)
    == manifest["artifact_sha256"][
        "fsar_train_unique_responses_bge_index_v1.jsonl"
    ],
    "Response index SHA mismatch"
)

print("RESPONSE_SOURCE_SHA=PASS")
print("DENSE_EMBEDDING_SHA=PASS")
print("RESPONSE_INDEX_SHA=PASS")


# ============================================================
# 4. LOAD AND VERIFY ROW ALIGNMENT
# ============================================================

print("\n========== 2. RESPONSE ALIGNMENT ==========")

responses = load_jsonl(RESPONSE_FILE)
index_rows = load_jsonl(RESPONSE_INDEX_FILE)

require(len(responses) == EXPECTED_N, "Response count mismatch")
require(len(index_rows) == EXPECTED_N, "Index count mismatch")

texts = []
uids = []

for i, (response, index_row) in enumerate(
    zip(responses, index_rows)
):
    uid = str(response["response_uid"])

    require(
        int(index_row["embedding_row"]) == i,
        f"Embedding row mismatch at {i}"
    )

    require(
        str(index_row["response_uid"]) == uid,
        f"UID mismatch at row {i}"
    )

    text = str(response["response_text"])

    require(bool(text.strip()), f"Empty response at {i}")

    texts.append(text)
    uids.append(uid)

require(len(set(uids)) == EXPECTED_N, "Duplicate UID")

print("RESPONSE_ROWS:", len(responses))
print("UNIQUE_UID:", len(set(uids)))
print("FULL_ROW_ALIGNMENT=PASS")


# ============================================================
# 5. LOAD DENSE EMBEDDINGS
# ============================================================

print("\n========== 3. DENSE EMBEDDINGS ==========")

dense_vectors = np.load(
    DENSE_EMBEDDING_FILE,
    mmap_mode="r",
    allow_pickle=False
)

require(
    dense_vectors.shape == (EXPECTED_N, EXPECTED_DIM),
    "Dense shape mismatch"
)

require(dense_vectors.dtype == np.float32, "Dense dtype mismatch")

require(
    np.isfinite(dense_vectors).all(),
    "Dense embeddings contain NaN or Inf"
)

dense_norms = np.linalg.norm(
    dense_vectors, axis=1
)

require(
    np.allclose(dense_norms, 1.0, atol=1e-5),
    "Dense vectors are not normalized"
)

print("SHAPE:", dense_vectors.shape)
print("DTYPE:", dense_vectors.dtype)
print("DENSE_VALIDATION=PASS")


# ============================================================
# 6. LOAD FROZEN GLOBAL ORCA BASIS
# ============================================================

print("\n========== 4. ORCA BASIS ==========")

basis = np.load(
    ORCA_BASIS_FILE,
    allow_pickle=False
)

basis = np.asarray(basis, dtype=np.float32)

require(
    basis.shape == (EXPECTED_DIM, 2),
    "ORCA basis shape mismatch"
)

require(np.isfinite(basis).all(), "Invalid ORCA basis")

basis_sha = sha256_array(basis)

require(
    basis_sha == EXPECTED_BASIS_SHA,
    "ORCA basis SHA mismatch"
)

gram = basis.T @ basis

require(
    np.allclose(gram, np.eye(2), atol=2e-6),
    "ORCA basis orthogonality failure"
)

print("BASIS_SHAPE:", basis.shape)
print("BASIS_SHA256:", basis_sha)
print("BASIS_INTEGRITY=PASS")


# ============================================================
# 7. INITIALIZE FROZEN PROJECTOR
# ============================================================

print("\n========== 5. ORCA PROJECTOR ==========")

projector = ORCAProjector(
    k=2,
    normalize_output=True
)

# Load the already verified basis.
# IMPORTANT: Do not refit the basis.

projector.basis = basis
projector.input_dim = EXPECTED_DIM

# Verify implementation against historical formula.

sample = np.asarray(
    dense_vectors[:10],
    dtype=np.float32
)

manual = sample - (sample @ basis) @ basis.T

manual_norm = np.linalg.norm(
    manual, axis=1, keepdims=True
)

manual = (
    manual / np.maximum(manual_norm, 1e-12)
).astype(np.float32)

projected_sample = projector.transform_responses(sample)

max_diff = float(
    np.max(np.abs(manual - projected_sample))
)

require(max_diff < 1e-6, "ORCA projection parity failure")

print("PROJECTION_MAX_DIFF:", max_diff)
print("PROJECTOR_PARITY=PASS")


# ============================================================
# 8. BUILD DENSE INDEX
# ============================================================

print("\n========== 6. BUILD DENSE INDEX ==========")

dense_retriever = FAISSRetriever(
    normalize=False
)

dense_retriever.build(
    embeddings=dense_vectors,
    documents=texts,
    document_ids=uids
)

require(
    dense_retriever.index.ntotal == EXPECTED_N,
    "Dense FAISS size mismatch"
)

print("DENSE_INDEX_BUILD=PASS")


# ============================================================
# 9. BUILD ORCA INDEX
# ============================================================

print("\n========== 7. BUILD ORCA INDEX ==========")

orca_vectors = projector.transform_responses(
    dense_vectors
)

require(
    orca_vectors.shape == (EXPECTED_N, EXPECTED_DIM),
    "ORCA embedding shape mismatch"
)

require(
    np.isfinite(orca_vectors).all(),
    "Invalid ORCA embeddings"
)

orca_norms = np.linalg.norm(
    orca_vectors, axis=1
)

require(
    np.allclose(orca_norms, 1.0, atol=1e-5),
    "ORCA vectors not normalized"
)

orca_retriever = FAISSRetriever(
    normalize=False
)

orca_retriever.build(
    embeddings=orca_vectors,
    documents=texts,
    document_ids=uids
)

require(
    orca_retriever.index.ntotal == EXPECTED_N,
    "ORCA FAISS size mismatch"
)

print("ORCA_INDEX_BUILD=PASS")


# ============================================================
# 10. RETRIEVAL PARITY AUDIT
# ============================================================

print("\n========== 8. RETRIEVAL PARITY ==========")

queries = np.load(
    QUERY_EMBEDDING_FILE,
    mmap_mode="r",
    allow_pickle=False
)

require(
    queries.shape == (62889, EXPECTED_DIM),
    "Query embedding shape mismatch"
)

positions = [0, len(queries) // 2, len(queries) - 1]

def verify_search(retriever, corpus, query, label):
    top_k = 10

    results = retriever.search(
        query,
        top_k=top_k
    )

    brute_scores = np.asarray(
        corpus @ query,
        dtype=np.float32
    )

    expected_rows = np.argsort(
        -brute_scores,
        kind="stable"
    )[:top_k]

    actual_rows = np.asarray(
        [x["row"] for x in results],
        dtype=np.int64
    )

    require(
        set(actual_rows.tolist())
        == set(expected_rows.tolist()),
        f"{label} top-10 retrieval mismatch"
    )

    score_map = {
        x["row"]: x["score"]
        for x in results
    }

    score_diff = max(
        abs(score_map[int(row)] - float(brute_scores[row]))
        for row in expected_rows
    )

    require(
        score_diff < 2e-5,
        f"{label} score mismatch: {score_diff}"
    )

    return score_diff


for pos in positions:
    q = np.asarray(queries[pos], dtype=np.float32)

    dense_diff = verify_search(
        dense_retriever,
        dense_vectors,
        q,
        "DENSE"
    )

    q_orca = projector.transform_queries(q)

    orca_diff = verify_search(
        orca_retriever,
        orca_vectors,
        q_orca,
        "ORCA"
    )

    print(
        f"QUERY={pos} "
        f"DENSE_DIFF={dense_diff:.8e} "
        f"ORCA_DIFF={orca_diff:.8e}"
    )

print("RETRIEVAL_PARITY=PASS")


# ============================================================
# 11. SAVE INDEXES
# ============================================================

print("\n========== 9. SAVE INDEXES ==========")

require(
    not DENSE_OUTPUT.exists(),
    f"Dense output already exists: {DENSE_OUTPUT}"
)

require(
    not ORCA_OUTPUT.exists(),
    f"ORCA output already exists: {ORCA_OUTPUT}"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

dense_retriever.metadata.update({
    "retrieval_mode": "dense_bge",
    "encoder": "BAAI/bge-large-zh-v1.5",
    "corpus": "TRAIN_UNIQUE_RESPONSES",
    "response_uid_aligned": True
})

orca_retriever.metadata.update({
    "retrieval_mode": "orca_k2",
    "encoder": "BAAI/bge-large-zh-v1.5",
    "orca_algorithm": "raw_svd",
    "orca_k": 2,
    "orca_basis_sha256": basis_sha,
    "corpus": "TRAIN_UNIQUE_RESPONSES",
    "response_uid_aligned": True
})

dense_retriever.save(DENSE_OUTPUT)
orca_retriever.save(ORCA_OUTPUT)

build_manifest = {
    "status": "PASS",
    "corpus_size": EXPECTED_N,
    "embedding_dimension": EXPECTED_DIM,
    "encoder": "BAAI/bge-large-zh-v1.5",
    "response_text_field": "response_text",
    "response_uid_alignment": True,
    "dense_index": str(DENSE_OUTPUT),
    "orca_index": str(ORCA_OUTPUT),
    "orca_k": 2,
    "orca_basis_sha256": basis_sha,
    "dense_index_size": int(dense_retriever.index.ntotal),
    "orca_index_size": int(orca_retriever.index.ntotal),
    "retrieval_parity": "PASS"
}

with open(
    OUTPUT_ROOT / "build_manifest_v1.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        build_manifest,
        f,
        ensure_ascii=False,
        indent=2
    )

print("DENSE_SAVE=PASS")
print("ORCA_SAVE=PASS")
print("BUILD_MANIFEST=PASS")

print("\n========== FINAL RESULT ==========")
print("DENSE_CORPUS:", dense_retriever.index.ntotal)
print("ORCA_CORPUS:", orca_retriever.index.ntotal)
print("ORCA_K: 2")
print("DUAL_INDEX_BUILD=PASS")
