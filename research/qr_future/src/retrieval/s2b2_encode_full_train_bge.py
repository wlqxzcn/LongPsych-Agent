import json
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = ROOT / "models/BAAI/bge-large-zh-v1___5"

QUERY_FILE = ROOT / "data/canonical/fsar_s2/train_query_gold_map_v1.jsonl"
RESPONSE_FILE = ROOT / "data/canonical/fsar_s2/train_unique_responses_v1.jsonl"

CACHE = ROOT / "cache/embeddings"
CACHE.mkdir(parents=True, exist_ok=True)

QUERY_OUT = CACHE / "fsar_train_queries_bge_v1.npy"
RESPONSE_OUT = CACHE / "fsar_train_unique_responses_bge_v1.npy"

QUERY_INDEX_OUT = CACHE / "fsar_train_queries_bge_index_v1.jsonl"
RESPONSE_INDEX_OUT = CACHE / "fsar_train_unique_responses_bge_index_v1.jsonl"

MANIFEST_OUT = CACHE / "fsar_s2b2_full_embedding_manifest_v1.json"

QUERY_TMP = CACHE / "fsar_train_queries_bge_v1.tmp.npy"
RESPONSE_TMP = CACHE / "fsar_train_unique_responses_bge_v1.tmp.npy"

QUERY_INDEX_TMP = CACHE / "fsar_train_queries_bge_index_v1.tmp.jsonl"
RESPONSE_INDEX_TMP = CACHE / "fsar_train_unique_responses_bge_index_v1.tmp.jsonl"


EXPECTED_QUERY_ROWS = 62889
EXPECTED_RESPONSE_ROWS = 61502
EXPECTED_DIM = 1024
BATCH_SIZE = 64

EXPECTED_QUERY_SHA = (
    "3662cef13545533dbda2a65474d03421"
    "cb64115491c655542333766d06092f3d"
)

EXPECTED_RESPONSE_SHA = (
    "6ed2a5807c7a3bdcb30b7b6b9472651"
    "29b19925a2cf7e8872fa36231b4cab25e"
)


def sha256_file(path, block_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_length_stats(tokenizer, texts, batch_size=512):
    lengths = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )

        lengths.extend(len(x) for x in encoded["input_ids"])

    arr = np.asarray(lengths, dtype=np.int32)

    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "gt_512_n": int((arr > 512).sum()),
        "gt_512_rate": float((arr > 512).mean())
    }


print("=" * 74)
print("FSAR PHASE S2-B2 FULL TRAIN BGE EMBEDDING")
print("=" * 74)


print()
print("===== 1. OUTPUT PROTECTION =====")

formal_outputs = [
    QUERY_OUT,
    RESPONSE_OUT,
    QUERY_INDEX_OUT,
    RESPONSE_INDEX_OUT,
    MANIFEST_OUT
]

existing = [str(p) for p in formal_outputs if p.exists()]

if existing:
    raise RuntimeError(
        "Formal S2-B2 outputs already exist; refusing overwrite:\n"
        + "\n".join(existing)
    )

for p in [
    QUERY_TMP,
    RESPONSE_TMP,
    QUERY_INDEX_TMP,
    RESPONSE_INDEX_TMP
]:
    if p.exists():
        p.unlink()

print("FORMAL_OUTPUT_COLLISION = 0")
print("OUTPUT_PROTECTION = PASS")


print()
print("===== 2. INPUT PROVENANCE =====")

query_sha = sha256_file(QUERY_FILE)
response_sha = sha256_file(RESPONSE_FILE)

print("query_input_sha256 =", query_sha)
print("response_input_sha256 =", response_sha)

assert query_sha == EXPECTED_QUERY_SHA
assert response_sha == EXPECTED_RESPONSE_SHA

print("INPUT_SHA_GATE = PASS")


print()
print("===== 3. INPUT STRUCTURE =====")

query_rows = load_jsonl(QUERY_FILE)
response_rows = load_jsonl(RESPONSE_FILE)

assert len(query_rows) == EXPECTED_QUERY_ROWS
assert len(response_rows) == EXPECTED_RESPONSE_ROWS

query_ids = [r["query_id"] for r in query_rows]
response_uids = [r["response_uid"] for r in response_rows]

assert len(query_ids) == len(set(query_ids))
assert len(response_uids) == len(set(response_uids))

query_texts = [r["current_user"] for r in query_rows]
response_texts = [r["response_text"] for r in response_rows]

response_uid_set = set(response_uids)

missing_gold = sum(
    r["gold_response_uid"] not in response_uid_set
    for r in query_rows
)

assert missing_gold == 0

print("query_rows =", len(query_rows))
print("response_rows =", len(response_rows))
print("duplicate_query_id = 0")
print("duplicate_response_uid = 0")
print("missing_gold_response_uid = 0")
print("INPUT_STRUCTURE = PASS")


print()
print("===== 4. MODEL AUDIT =====")

assert torch.cuda.is_available()

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))

model = SentenceTransformer(
    str(MODEL_PATH),
    device="cuda"
)

model.eval()

for p in model.parameters():
    p.requires_grad = False

assert model.max_seq_length == 512
assert model.get_sentence_embedding_dimension() == 1024

pooling = None
normalize_found = False
transformer_module = None

for module in model._modules.values():
    name = module.__class__.__name__

    if name == "Transformer":
        transformer_module = module
    elif name == "Pooling":
        pooling = module
    elif name == "Normalize":
        normalize_found = True

assert transformer_module is not None
assert pooling is not None
assert normalize_found

assert pooling.pooling_mode_cls_token is True
assert pooling.pooling_mode_mean_tokens is False
assert pooling.pooling_mode_max_tokens is False

print("max_seq_length =", model.max_seq_length)
print(
    "sentence_embedding_dimension =",
    model.get_sentence_embedding_dimension()
)
print("pooling_cls =", pooling.pooling_mode_cls_token)
print("pooling_mean =", pooling.pooling_mode_mean_tokens)
print("pooling_max =", pooling.pooling_mode_max_tokens)
print("normalize_module =", normalize_found)
print("MODEL_AUDIT = PASS")


print()
print("===== 5. TOKEN LENGTH AUDIT =====")

tokenizer = transformer_module.tokenizer

query_token_stats = token_length_stats(tokenizer, query_texts)
response_token_stats = token_length_stats(tokenizer, response_texts)

print("query_token_stats =", query_token_stats)
print("response_token_stats =", response_token_stats)

print("TOKEN_LENGTH_AUDIT_ROLE = DESCRIPTIVE_ONLY")
print("MAX_SEQ_LENGTH_REMAINS_FROZEN = 512")


print()
print("===== 6. QUERY ENCODING =====")

with torch.inference_mode():
    query_emb = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

query_emb = np.asarray(query_emb, dtype=np.float32)

assert query_emb.shape == (EXPECTED_QUERY_ROWS, EXPECTED_DIM)
assert query_emb.dtype == np.float32
assert np.isfinite(query_emb).all()

query_norms = np.linalg.norm(query_emb, axis=1)

assert np.allclose(query_norms, 1.0, atol=1e-5)

np.save(QUERY_TMP, query_emb, allow_pickle=False)

print("query_shape =", query_emb.shape)
print("query_dtype =", query_emb.dtype)

print(
    "query_norm_min_mean_max =",
    float(query_norms.min()),
    float(query_norms.mean()),
    float(query_norms.max())
)

print("QUERY_ENCODING = PASS")


print()
print("===== 7. RESPONSE ENCODING =====")

with torch.inference_mode():
    response_emb = model.encode(
        response_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

response_emb = np.asarray(response_emb, dtype=np.float32)

assert response_emb.shape == (EXPECTED_RESPONSE_ROWS, EXPECTED_DIM)
assert response_emb.dtype == np.float32
assert np.isfinite(response_emb).all()

response_norms = np.linalg.norm(response_emb, axis=1)

assert np.allclose(response_norms, 1.0, atol=1e-5)

np.save(RESPONSE_TMP, response_emb, allow_pickle=False)

print("response_shape =", response_emb.shape)
print("response_dtype =", response_emb.dtype)

print(
    "response_norm_min_mean_max =",
    float(response_norms.min()),
    float(response_norms.mean()),
    float(response_norms.max())
)

print("RESPONSE_ENCODING = PASS")


print()
print("===== 8. INDEX CONSTRUCTION =====")

with QUERY_INDEX_TMP.open("w", encoding="utf-8") as f:
    for i, row in enumerate(query_rows):
        out = {
            "embedding_row": i,
            "query_id": row["query_id"],
            "dialogue_id": row["dialogue_id"],
            "target_pair_index": row["target_pair_index"],
            "gold_response_uid": row["gold_response_uid"]
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


with RESPONSE_INDEX_TMP.open("w", encoding="utf-8") as f:
    for i, row in enumerate(response_rows):
        out = {
            "embedding_row": i,
            "response_uid": row["response_uid"],
            "occurrence_count": row["occurrence_count"]
        }
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


query_index_rows = load_jsonl(QUERY_INDEX_TMP)
response_index_rows = load_jsonl(RESPONSE_INDEX_TMP)

assert len(query_index_rows) == EXPECTED_QUERY_ROWS
assert len(response_index_rows) == EXPECTED_RESPONSE_ROWS

for i, row in enumerate(query_index_rows):
    assert row["embedding_row"] == i
    assert row["query_id"] == query_rows[i]["query_id"]
    assert row["gold_response_uid"] == query_rows[i]["gold_response_uid"]

for i, row in enumerate(response_index_rows):
    assert row["embedding_row"] == i
    assert row["response_uid"] == response_rows[i]["response_uid"]

print("query_index_rows =", len(query_index_rows))
print("response_index_rows =", len(response_index_rows))
print("INDEX_ALIGNMENT = PASS")


print()
print("===== 9. CACHE RELOAD + SPOT CHECK =====")

q_reload = np.load(
    QUERY_TMP,
    mmap_mode="r",
    allow_pickle=False
)

r_reload = np.load(
    RESPONSE_TMP,
    mmap_mode="r",
    allow_pickle=False
)

assert q_reload.shape == (EXPECTED_QUERY_ROWS, EXPECTED_DIM)
assert r_reload.shape == (EXPECTED_RESPONSE_ROWS, EXPECTED_DIM)

q_positions = [
    0,
    EXPECTED_QUERY_ROWS // 2,
    EXPECTED_QUERY_ROWS - 1
]

r_positions = [
    0,
    EXPECTED_RESPONSE_ROWS // 2,
    EXPECTED_RESPONSE_ROWS - 1
]

with torch.inference_mode():
    q_check = model.encode(
        [query_texts[i] for i in q_positions],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    r_check = model.encode(
        [response_texts[i] for i in r_positions],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

q_check = np.asarray(q_check, dtype=np.float32)
r_check = np.asarray(r_check, dtype=np.float32)

q_cached = np.asarray(q_reload[q_positions])
r_cached = np.asarray(r_reload[r_positions])

q_spot_max_diff = float(
    np.max(np.abs(q_check - q_cached))
)

r_spot_max_diff = float(
    np.max(np.abs(r_check - r_cached))
)

print(
    "query_spotcheck_max_abs_diff =",
    q_spot_max_diff
)

print(
    "response_spotcheck_max_abs_diff =",
    r_spot_max_diff
)

assert q_spot_max_diff < 1e-4
assert r_spot_max_diff < 1e-4

print("CACHE_RELOAD = PASS")
print("CACHE_SPOTCHECK = PASS")


print()
print("===== 10. FORMAL ARTIFACT PROMOTION =====")

os.replace(QUERY_TMP, QUERY_OUT)
os.replace(RESPONSE_TMP, RESPONSE_OUT)
os.replace(QUERY_INDEX_TMP, QUERY_INDEX_OUT)
os.replace(RESPONSE_INDEX_TMP, RESPONSE_INDEX_OUT)

print("FORMAL_ARTIFACT_PROMOTION = PASS")


print()
print("===== 11. ARTIFACT SHA256 =====")

artifact_hashes = {}

for p in [
    QUERY_OUT,
    RESPONSE_OUT,
    QUERY_INDEX_OUT,
    RESPONSE_INDEX_OUT
]:
    h = sha256_file(p)
    artifact_hashes[p.name] = h
    print(h, p)


manifest = {
    "artifact":
        "FSAR S2-B2 Full TRAIN BGE Embedding Cache V1",

    "status": "PASS",
    "scope": "TRAIN_ONLY",

    "encoder": "BAAI/bge-large-zh-v1.5",
    "model_path": str(MODEL_PATH),

    "query_input_sha256": query_sha,
    "response_input_sha256": response_sha,

    "query_rows": EXPECTED_QUERY_ROWS,
    "response_rows": EXPECTED_RESPONSE_ROWS,

    "embedding_dimension": EXPECTED_DIM,
    "dtype": "float32",
    "runtime_batch_size": BATCH_SIZE,

    "max_seq_length": 512,
    "pooling": "CLS",
    "normalize_module": True,
    "normalize_embeddings_argument": True,

    "query_token_stats": query_token_stats,
    "response_token_stats": response_token_stats,

    "query_embedding_shape": list(query_emb.shape),
    "response_embedding_shape": list(response_emb.shape),

    "query_norm": {
        "min": float(query_norms.min()),
        "mean": float(query_norms.mean()),
        "max": float(query_norms.max())
    },

    "response_norm": {
        "min": float(response_norms.min()),
        "mean": float(response_norms.mean()),
        "max": float(response_norms.max())
    },

    "query_spotcheck_max_abs_diff": q_spot_max_diff,
    "response_spotcheck_max_abs_diff": r_spot_max_diff,

    "artifact_sha256": artifact_hashes,

    "future_state_supervision_used": False,
    "hard_negative_mining_performed": False,
    "retrieval_metrics_computed": False,

    "dev_payload_used": False,
    "test_payload_used": False
}

MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("PHASE_S2B2_FULL_EMBEDDING_CACHE = PASS")
print("MANIFEST =", MANIFEST_OUT)
