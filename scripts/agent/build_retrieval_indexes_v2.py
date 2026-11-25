#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the LongPsychAgent dual retrieval indexes:
  - Dense BGE
  - ORCA k=2 using the frozen global basis

The single known empty response is excluded at index-build time only.
Original source files and embedding caches are never modified.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

import numpy as np


# ============================================================
# 1. PROJECT / IMPORTS
# ============================================================

PROJECT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT / "agent"))

from longpsych_agent.config import settings

SOURCE_ROOT = settings.source_root

from longpsych_agent.retrieval.faiss_retriever import FAISSRetriever
from longpsych_agent.retrieval.orca_projector import ORCAProjector


# ============================================================
# 2. PATHS AND FROZEN EXPECTATIONS
# ============================================================

RESPONSE_FILE = (
    ORIGINAL / "data/canonical/fsar_s2/"
    "train_unique_responses_v1.jsonl"
)

QUERY_GOLD_MAP_FILE = (
    ORIGINAL / "data/canonical/fsar_s2/"
    "train_query_gold_map_v1.jsonl"
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

# Build into staging directories first. Promote only after reload/parity passes.
DENSE_BUILD = OUTPUT_ROOT / ".dense_bge_v1.building"
ORCA_BUILD = OUTPUT_ROOT / ".orca_k2_v1.building"

EXCLUDED_RESPONSES_BUILD = OUTPUT_ROOT / ".excluded_responses_v1.jsonl.building"
EXCLUDED_GOLD_QUERIES_BUILD = OUTPUT_ROOT / ".excluded_gold_queries_v1.jsonl.building"

EXCLUDED_RESPONSES_OUTPUT = (
    OUTPUT_ROOT / "excluded_responses_v1.jsonl"
)

EXCLUDED_GOLD_QUERIES_OUTPUT = (
    OUTPUT_ROOT / "excluded_gold_queries_v1.jsonl"
)

BUILD_MANIFEST_OUTPUT = OUTPUT_ROOT / "build_manifest_v1.json"

EXPECTED_N = 61502
EXPECTED_VALID_N = 61501
EXPECTED_QUERY_N = 62889
EXPECTED_DIM = 1024

EXPECTED_EMPTY_ROW = 54735
EXPECTED_EMPTY_UID = "resp_e3b0c44298fc1c149afbf4c8"
EXPECTED_EMPTY_GOLD_QUERY_ID = "psydial_2113_12"

EXPECTED_BASIS_SHA = (
    "aaa0e7d3f04aeea62498d3dd8a7ab917"
    "b02841f6fc3bce90a7524baa87666be3"
)


# ============================================================
# 3. UTILITIES
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(arr):
    arr = np.ascontiguousarray(arr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_directory(directory):
    directory = Path(directory)
    return {
        p.name: sha256_file(p)
        for p in sorted(directory.iterdir())
        if p.is_file()
    }


def verify_search(retriever, corpus, query, label, top_k=10):
    results = retriever.search(query, top_k=top_k)

    brute_scores = np.asarray(corpus @ query, dtype=np.float32)
    expected_rows = np.argsort(-brute_scores, kind="stable")[:top_k]

    actual_rows = np.asarray(
        [int(x["row"]) for x in results],
        dtype=np.int64
    )

    require(
        set(actual_rows.tolist()) == set(expected_rows.tolist()),
        f"{label} top-{top_k} retrieval mismatch"
    )

    score_map = {
        int(x["row"]): float(x["score"])
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

    return float(score_diff)


# ============================================================
# 4. INPUT / OUTPUT PROTECTION
# ============================================================

print("=" * 72)
print("LONGPSYCHAGENT DUAL INDEX BUILD")
print("=" * 72)

for path in [
    RESPONSE_FILE,
    QUERY_GOLD_MAP_FILE,
    RESPONSE_INDEX_FILE,
    DENSE_EMBEDDING_FILE,
    QUERY_EMBEDDING_FILE,
    EMBEDDING_MANIFEST,
    ORCA_BASIS_FILE,
]:
    require(path.is_file(), f"Required input file missing: {path}")

for path in [
    DENSE_OUTPUT,
    ORCA_OUTPUT,
    DENSE_BUILD,
    ORCA_BUILD,
    EXCLUDED_RESPONSES_OUTPUT,
    EXCLUDED_GOLD_QUERIES_OUTPUT,
    EXCLUDED_RESPONSES_BUILD,
    EXCLUDED_GOLD_QUERIES_BUILD,
    BUILD_MANIFEST_OUTPUT,
]:
    require(
        not path.exists(),
        f"Output already exists; refusing overwrite: {path}"
    )


# ============================================================
# 5. INPUT PROVENANCE
# ============================================================

print("\n========== 1. INPUT PROVENANCE ==========")

with EMBEDDING_MANIFEST.open("r", encoding="utf-8") as f:
    embedding_manifest = json.load(f)

response_source_sha = sha256_file(RESPONSE_FILE)
query_gold_map_sha = sha256_file(QUERY_GOLD_MAP_FILE)
dense_embedding_sha = sha256_file(DENSE_EMBEDDING_FILE)
query_embedding_sha = sha256_file(QUERY_EMBEDDING_FILE)
response_index_sha = sha256_file(RESPONSE_INDEX_FILE)

require(
    response_source_sha == embedding_manifest["response_input_sha256"],
    "Response source SHA mismatch"
)

require(
    query_gold_map_sha == embedding_manifest["query_input_sha256"],
    "Query Gold map SHA mismatch"
)

require(
    dense_embedding_sha
    == embedding_manifest["artifact_sha256"][
        "fsar_train_unique_responses_bge_v1.npy"
    ],
    "Dense embedding SHA mismatch"
)

require(
    query_embedding_sha
    == embedding_manifest["artifact_sha256"][
        "fsar_train_queries_bge_v1.npy"
    ],
    "Query embedding SHA mismatch"
)

require(
    response_index_sha
    == embedding_manifest["artifact_sha256"][
        "fsar_train_unique_responses_bge_index_v1.jsonl"
    ],
    "Response index SHA mismatch"
)

print("RESPONSE_SOURCE_SHA=PASS")
print("QUERY_GOLD_MAP_SHA=PASS")
print("DENSE_EMBEDDING_SHA=PASS")
print("QUERY_EMBEDDING_SHA=PASS")
print("RESPONSE_INDEX_SHA=PASS")


# ============================================================
# 6. LOAD AND VERIFY RESPONSE ALIGNMENT
# ============================================================

print("\n========== 2. RESPONSE ALIGNMENT ==========")

responses = load_jsonl(RESPONSE_FILE)
index_rows = load_jsonl(RESPONSE_INDEX_FILE)

require(len(responses) == EXPECTED_N, "Response count mismatch")
require(len(index_rows) == EXPECTED_N, "Response index count mismatch")

texts = []
uids = []
valid_embedding_rows = []
excluded_responses = []
all_uids = []

for i, (response, index_row) in enumerate(zip(responses, index_rows)):
    uid = str(response["response_uid"])
    all_uids.append(uid)

    require(
        int(index_row["embedding_row"]) == i,
        f"Embedding row mismatch at {i}"
    )

    require(
        str(index_row["response_uid"]) == uid,
        f"UID mismatch at row {i}"
    )

    text = response.get("response_text")

    if not isinstance(text, str) or not text.strip():
        excluded_responses.append({
            "original_embedding_row": i,
            "response_uid": uid,
            "response_text": text,
            "reason": "empty_response"
        })
        continue

    valid_embedding_rows.append(i)
    texts.append(text)
    uids.append(uid)

require(
    len(set(all_uids)) == EXPECTED_N,
    "Duplicate UID in original response corpus"
)

require(
    len(excluded_responses) == 1,
    f"Expected exactly one empty response, got {len(excluded_responses)}"
)

require(
    excluded_responses[0]["original_embedding_row"] == EXPECTED_EMPTY_ROW,
    "Unexpected empty response row"
)

require(
    excluded_responses[0]["response_uid"] == EXPECTED_EMPTY_UID,
    "Unexpected empty response UID"
)

require(
    len(texts) == EXPECTED_VALID_N,
    f"Valid response count mismatch: {len(texts)}"
)

require(
    len(set(uids)) == EXPECTED_VALID_N,
    "Duplicate UID in valid response corpus"
)

print("ORIGINAL_RESPONSE_ROWS:", len(responses))
print("VALID_RESPONSE_ROWS:", len(texts))
print("EXCLUDED_RESPONSE_ROWS:", len(excluded_responses))
print("EXCLUDED_ORIGINAL_ROW:", excluded_responses[0]["original_embedding_row"])
print("EXCLUDED_UID:", excluded_responses[0]["response_uid"])
print("FULL_ROW_ALIGNMENT=PASS")


# ============================================================
# 7. AUDIT GOLD REFERENCES TO THE EXCLUDED RESPONSE
# ============================================================

print("\n========== 3. EXCLUDED GOLD REFERENCE AUDIT ==========")

query_gold_rows = load_jsonl(QUERY_GOLD_MAP_FILE)

require(
    len(query_gold_rows) == EXPECTED_QUERY_N,
    "Query Gold map row count mismatch"
)

query_ids = [str(row["query_id"]) for row in query_gold_rows]
require(
    len(set(query_ids)) == EXPECTED_QUERY_N,
    "Duplicate query_id in query Gold map"
)

original_uid_set = set(all_uids)

missing_gold_uids = [
    row["query_id"]
    for row in query_gold_rows
    if str(row["gold_response_uid"]) not in original_uid_set
]

require(
    len(missing_gold_uids) == 0,
    f"Gold UID missing from original response corpus: {missing_gold_uids[:5]}"
)

excluded_uid_set = {
    row["response_uid"] for row in excluded_responses
}

excluded_gold_queries = [
    {
        "query_id": str(row["query_id"]),
        "dialogue_id": row.get("dialogue_id"),
        "target_pair_index": row.get("target_pair_index"),
        "gold_response_uid": str(row["gold_response_uid"]),
        "reason": "gold_response_is_empty"
    }
    for row in query_gold_rows
    if str(row["gold_response_uid"]) in excluded_uid_set
]

require(
    len(excluded_gold_queries) == 1,
    f"Expected one Gold reference to excluded response, got {len(excluded_gold_queries)}"
)

require(
    excluded_gold_queries[0]["query_id"] == EXPECTED_EMPTY_GOLD_QUERY_ID,
    "Unexpected query references the excluded empty response"
)

print("QUERY_GOLD_ROWS:", len(query_gold_rows))
print("MISSING_GOLD_UIDS:", len(missing_gold_uids))
print("EXCLUDED_GOLD_QUERY_COUNT:", len(excluded_gold_queries))
print("EXCLUDED_GOLD_QUERY_ID:", excluded_gold_queries[0]["query_id"])
print("GOLD_REFERENCE_AUDIT=PASS")


# ============================================================
# 8. LOAD AND VALIDATE DENSE EMBEDDINGS
# ============================================================

print("\n========== 4. DENSE EMBEDDINGS ==========")

dense_vectors = np.load(
    DENSE_EMBEDDING_FILE,
    mmap_mode="r",
    allow_pickle=False
)

require(
    dense_vectors.shape == (EXPECTED_N, EXPECTED_DIM),
    f"Dense shape mismatch: {dense_vectors.shape}"
)

require(
    dense_vectors.dtype == np.float32,
    f"Dense dtype mismatch: {dense_vectors.dtype}"
)

require(
    np.isfinite(dense_vectors).all(),
    "Dense embeddings contain NaN or Inf"
)

dense_norms = np.linalg.norm(dense_vectors, axis=1)

require(
    np.allclose(dense_norms, 1.0, atol=1e-5),
    "Dense vectors are not normalized"
)

# Apply the exact same valid-row selection to embeddings, texts and UIDs.
valid_dense_vectors = np.ascontiguousarray(
    dense_vectors[valid_embedding_rows],
    dtype=np.float32
)

require(
    valid_dense_vectors.shape == (EXPECTED_VALID_N, EXPECTED_DIM),
    f"Filtered Dense shape mismatch: {valid_dense_vectors.shape}"
)

require(
    np.isfinite(valid_dense_vectors).all(),
    "Filtered Dense embeddings contain NaN or Inf"
)

print("ORIGINAL_DENSE_SHAPE:", dense_vectors.shape)
print("VALID_DENSE_SHAPE:", valid_dense_vectors.shape)
print("DENSE_VALIDATION=PASS")
print("DENSE_FILTER_ALIGNMENT=PASS")


# ============================================================
# 9. LOAD AND VERIFY FROZEN GLOBAL ORCA BASIS
# ============================================================

print("\n========== 5. ORCA BASIS ==========")

basis = np.load(
    ORCA_BASIS_FILE,
    allow_pickle=False
)

basis = np.asarray(basis, dtype=np.float32)

require(
    basis.shape == (EXPECTED_DIM, 2),
    f"ORCA basis shape mismatch: {basis.shape}"
)

require(
    np.isfinite(basis).all(),
    "Invalid ORCA basis"
)

# The frozen expected SHA recorded in the earlier integrity audit is the
# SHA256 of the complete .npy file (including its NPY header), not just the
# contiguous ndarray payload. Check the file hash against that frozen value.
basis_file_sha = sha256_file(ORCA_BASIS_FILE)
basis_array_sha = sha256_array(basis)

require(
    basis_file_sha == EXPECTED_BASIS_SHA,
    f"ORCA basis FILE SHA mismatch: {basis_file_sha}"
)

# Keep the file-level SHA as the canonical artifact identity used downstream.
basis_sha = basis_file_sha

gram = basis.T @ basis

require(
    np.allclose(gram, np.eye(2, dtype=np.float32), atol=2e-6),
    "ORCA basis orthogonality failure"
)

print("BASIS_SHAPE:", basis.shape)
print("BASIS_FILE_SHA256:", basis_file_sha)
print("BASIS_ARRAY_BYTES_SHA256:", basis_array_sha)
print("BASIS_INTEGRITY=PASS")


# ============================================================
# 10. INITIALIZE FROZEN PROJECTOR AND CHECK FORMULA PARITY
# ============================================================

print("\n========== 6. ORCA PROJECTOR ==========")

projector = ORCAProjector(
    k=2,
    normalize_output=True
)

# Use the already verified global basis. Never refit it here.
projector.basis = basis
projector.input_dim = EXPECTED_DIM

sample = np.asarray(dense_vectors[:10], dtype=np.float32)

manual = sample - (sample @ basis) @ basis.T
manual_norm = np.linalg.norm(manual, axis=1, keepdims=True)
manual = (
    manual / np.maximum(manual_norm, 1e-12)
).astype(np.float32)

projected_sample = projector.transform_responses(sample)

projection_max_diff = float(
    np.max(np.abs(manual - projected_sample))
)

require(
    projection_max_diff < 1e-6,
    f"ORCA projection parity failure: {projection_max_diff}"
)

print("PROJECTION_MAX_DIFF:", projection_max_diff)
print("PROJECTOR_PARITY=PASS")


# ============================================================
# 11. BUILD DENSE INDEX
# ============================================================

print("\n========== 7. BUILD DENSE INDEX ==========")

dense_retriever = FAISSRetriever(normalize=False)

dense_retriever.build(
    embeddings=valid_dense_vectors,
    documents=texts,
    document_ids=uids
)

require(
    dense_retriever.index.ntotal == EXPECTED_VALID_N,
    "Dense FAISS size mismatch"
)

print("DENSE_INDEX_BUILD=PASS")


# ============================================================
# 12. BUILD ORCA INDEX
# ============================================================

print("\n========== 8. BUILD ORCA INDEX ==========")

# Project only the same filtered rows used by the Dense index.
orca_vectors = projector.transform_responses(valid_dense_vectors)

require(
    orca_vectors.shape == (EXPECTED_VALID_N, EXPECTED_DIM),
    f"ORCA embedding shape mismatch: {orca_vectors.shape}"
)

require(
    np.isfinite(orca_vectors).all(),
    "Invalid ORCA embeddings"
)

orca_norms = np.linalg.norm(orca_vectors, axis=1)

require(
    np.allclose(orca_norms, 1.0, atol=1e-5),
    "ORCA vectors are not normalized"
)

orca_retriever = FAISSRetriever(normalize=False)

orca_retriever.build(
    embeddings=orca_vectors,
    documents=texts,
    document_ids=uids
)

require(
    orca_retriever.index.ntotal == EXPECTED_VALID_N,
    "ORCA FAISS size mismatch"
)

print("ORCA_INDEX_BUILD=PASS")


# ============================================================
# 13. DENSE / ORCA RETRIEVAL PARITY
# ============================================================

print("\n========== 9. RETRIEVAL PARITY ==========")

queries = np.load(
    QUERY_EMBEDDING_FILE,
    mmap_mode="r",
    allow_pickle=False
)

require(
    queries.shape == (EXPECTED_QUERY_N, EXPECTED_DIM),
    f"Query embedding shape mismatch: {queries.shape}"
)

positions = [0, len(queries) // 2, len(queries) - 1]

for pos in positions:
    q = np.asarray(queries[pos], dtype=np.float32)

    dense_diff = verify_search(
        dense_retriever,
        valid_dense_vectors,
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
# 14. SAVE INDEXES AND ROW MAPPINGS
# ============================================================

print("\n========== 10. SAVE INDEXES ==========")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

dense_retriever.metadata.update({
    "retrieval_mode": "dense_bge",
    "encoder": "BAAI/bge-large-zh-v1.5",
    "corpus": "TRAIN_UNIQUE_RESPONSES",
    "response_uid_aligned": True,
    "original_corpus_size": EXPECTED_N,
    "corpus_size": EXPECTED_VALID_N,
    "excluded_response_count": len(excluded_responses),
    "excluded_gold_query_count": len(excluded_gold_queries),
    "row_mapping_file": "row_mapping.jsonl",
    "normalization": "L2"
})

orca_retriever.metadata.update({
    "retrieval_mode": "orca_k2",
    "encoder": "BAAI/bge-large-zh-v1.5",
    "orca_algorithm": "raw_svd",
    "orca_k": 2,
    "orca_basis_sha256": basis_sha,
    "corpus": "TRAIN_UNIQUE_RESPONSES",
    "response_uid_aligned": True,
    "original_corpus_size": EXPECTED_N,
    "corpus_size": EXPECTED_VALID_N,
    "excluded_response_count": len(excluded_responses),
    "excluded_gold_query_count": len(excluded_gold_queries),
    "row_mapping_file": "row_mapping.jsonl",
    "normalization": "L2"
})

dense_retriever.save(DENSE_BUILD)
orca_retriever.save(ORCA_BUILD)

# Preserve the mapping from FAISS row -> original embedding row -> response UID.
row_mapping = [
    {
        "faiss_row": int(faiss_row),
        "original_embedding_row": int(original_row),
        "response_uid": str(uid)
    }
    for faiss_row, (original_row, uid) in enumerate(
        zip(valid_embedding_rows, uids)
    )
]

require(
    len(row_mapping) == EXPECTED_VALID_N,
    "Row mapping count mismatch"
)

write_jsonl(DENSE_BUILD / "row_mapping.jsonl", row_mapping)
write_jsonl(ORCA_BUILD / "row_mapping.jsonl", row_mapping)

write_jsonl(
    EXCLUDED_RESPONSES_BUILD,
    excluded_responses
)

write_jsonl(
    EXCLUDED_GOLD_QUERIES_BUILD,
    excluded_gold_queries
)

print("DENSE_SAVE=PASS")
print("ORCA_SAVE=PASS")
print("ROW_MAPPING_ROWS:", len(row_mapping))
print("EXCLUDED_RESPONSE_AUDIT_FILE=PASS")
print("EXCLUDED_GOLD_AUDIT_FILE=PASS")


# ============================================================
# 15. RELOAD AND POST-SAVE VALIDATION
# ============================================================

print("\n========== 11. RELOAD AND POST-SAVE VALIDATION ==========")

dense_loaded = FAISSRetriever.load(DENSE_BUILD)
orca_loaded = FAISSRetriever.load(ORCA_BUILD)

require(
    dense_loaded.index.ntotal == EXPECTED_VALID_N,
    "Reloaded Dense index size mismatch"
)

require(
    orca_loaded.index.ntotal == EXPECTED_VALID_N,
    "Reloaded ORCA index size mismatch"
)

require(
    dense_loaded.document_ids == uids,
    "Reloaded Dense UID order mismatch"
)

require(
    orca_loaded.document_ids == uids,
    "Reloaded ORCA UID order mismatch"
)

require(
    dense_loaded.documents == texts,
    "Reloaded Dense document order mismatch"
)

require(
    orca_loaded.documents == texts,
    "Reloaded ORCA document order mismatch"
)

for pos in positions:
    q = np.asarray(queries[pos], dtype=np.float32)

    verify_search(
        dense_loaded,
        valid_dense_vectors,
        q,
        "DENSE_RELOADED"
    )

    q_orca = projector.transform_queries(q)

    verify_search(
        orca_loaded,
        orca_vectors,
        q_orca,
        "ORCA_RELOADED"
    )

print("DENSE_RELOAD=PASS")
print("ORCA_RELOAD=PASS")
print("UID_ALIGNMENT=PASS")
print("POST_SAVE_RETRIEVAL_PARITY=PASS")

# Promote staged artifacts only after both indexes reload and pass parity.
DENSE_BUILD.replace(DENSE_OUTPUT)
ORCA_BUILD.replace(ORCA_OUTPUT)
EXCLUDED_RESPONSES_BUILD.replace(EXCLUDED_RESPONSES_OUTPUT)
EXCLUDED_GOLD_QUERIES_BUILD.replace(EXCLUDED_GOLD_QUERIES_OUTPUT)

print("ARTIFACT_PROMOTION=PASS")


# ============================================================
# 16. FINAL MANIFEST AND ARTIFACT HASHES
# ============================================================

print("\n========== 12. BUILD MANIFEST ==========")

build_manifest = {
    "artifact": "LongPsychAgent Dual Retrieval Indexes V1",
    "status": "PASS",
    "scope": "TRAIN_UNIQUE_RESPONSES",
    "encoder": "BAAI/bge-large-zh-v1.5",
    "embedding_dimension": EXPECTED_DIM,
    "dtype": "float32",
    "normalization": "L2",
    "similarity": "inner_product",

    "original_corpus_size": EXPECTED_N,
    "valid_corpus_size": EXPECTED_VALID_N,
    "excluded_response_count": len(excluded_responses),
    "excluded_responses_file": str(EXCLUDED_RESPONSES_OUTPUT),
    "excluded_responses_sha256": sha256_file(EXCLUDED_RESPONSES_OUTPUT),

    "query_count": EXPECTED_QUERY_N,
    "excluded_gold_query_count": len(excluded_gold_queries),
    "excluded_gold_queries_file": str(EXCLUDED_GOLD_QUERIES_OUTPUT),
    "excluded_gold_queries_sha256": sha256_file(
        EXCLUDED_GOLD_QUERIES_OUTPUT
    ),

    "excluded_response_records": excluded_responses,
    "excluded_gold_query_records": excluded_gold_queries,

    "response_source_sha256": response_source_sha,
    "query_gold_map_sha256": query_gold_map_sha,
    "response_index_sha256": response_index_sha,
    "dense_embedding_sha256": dense_embedding_sha,
    "query_embedding_sha256": query_embedding_sha,

    "dense_index": str(DENSE_OUTPUT),
    "dense_index_size": int(dense_loaded.index.ntotal),
    "dense_artifact_sha256": hash_directory(DENSE_OUTPUT),

    "orca_index": str(ORCA_OUTPUT),
    "orca_index_size": int(orca_loaded.index.ntotal),
    "orca_k": 2,
    "orca_algorithm": "raw_svd",
    "orca_basis_sha256": basis_sha,
    "orca_basis_file_sha256": basis_file_sha,
    "orca_basis_array_bytes_sha256": basis_array_sha,
    "orca_artifact_sha256": hash_directory(ORCA_OUTPUT),

    "row_mapping_count": len(row_mapping),
    "row_mapping_semantics": {
        "faiss_row": "zero-based row in the built FAISS index",
        "original_embedding_row": "row in the frozen original dense embedding matrix",
        "response_uid": "canonical response identifier"
    },

    "projection_parity": "PASS",
    "retrieval_parity": "PASS",
    "post_save_reload_parity": "PASS",
    "uid_alignment": "PASS"
}

manifest_tmp = BUILD_MANIFEST_OUTPUT.with_suffix(".json.tmp")

with manifest_tmp.open("w", encoding="utf-8") as f:
    json.dump(
        build_manifest,
        f,
        ensure_ascii=False,
        indent=2
    )

os.replace(manifest_tmp, BUILD_MANIFEST_OUTPUT)

print("BUILD_MANIFEST=PASS")

print("\n========== FINAL RESULT ==========")
print("ORIGINAL_CORPUS:", EXPECTED_N)
print("EXCLUDED_EMPTY_RESPONSES:", len(excluded_responses))
print("VALID_CORPUS:", EXPECTED_VALID_N)
print("EXCLUDED_GOLD_QUERIES:", len(excluded_gold_queries))
print("DENSE_CORPUS:", dense_loaded.index.ntotal)
print("ORCA_CORPUS:", orca_loaded.index.ntotal)
print("ORCA_K: 2")
print("DENSE_RELOAD=PASS")
print("ORCA_RELOAD=PASS")
print("UID_ALIGNMENT=PASS")
print("DUAL_INDEX_BUILD=PASS")
