import json
import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]

QUERY_EMB_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_queries_bge_v1.npy"
)

RESPONSE_EMB_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
)

QUERY_INDEX_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_queries_bge_index_v1.jsonl"
)

RESPONSE_INDEX_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"
)

RESULT_DIR = ROOT / "results/phase_s2"

NPZ_OUT = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_arrays_v1.npz"
)

JSONL_OUT = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_rows_v1.jsonl"
)

SUMMARY_OUT = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_summary_v1.json"
)

NPZ_TMP = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_arrays_v1.tmp.npz"
)

JSONL_TMP = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_rows_v1.tmp.jsonl"
)

SUMMARY_TMP = (
    RESULT_DIR
    / "s2c_train_dense_gold_rank_summary_v1.tmp.json"
)


EXPECTED_QUERY_EMB_SHA = (
    "18aa1ed637bef3980775c10d96d1cde6"
    "112863b0c5761e2e69fb7ab89c981005"
)

EXPECTED_RESPONSE_EMB_SHA = (
    "7a4ee8d462adce73174c853bf21b7581"
    "32f5243020fa6253b1c01a99973bc96f"
)

EXPECTED_QUERY_INDEX_SHA = (
    "7d3659d38668f653289fd649fef186cd"
    "2f0a1003338af88ff6c805c23960de75"
)

EXPECTED_RESPONSE_INDEX_SHA = (
    "fd8263c8fd00b4d8d23b915c7f241fb"
    "84d1d2ffc7b24c747bf34e4a9b8e47274"
)

EXPECTED_QUERY_ROWS = 62889
EXPECTED_RESPONSE_ROWS = 61502
EXPECTED_DIM = 1024

QUERY_BATCH_SIZE = 1024
CANDIDATE_CHUNK_SIZE = 16384

RECALL_CUTOFFS = [
    1,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000
]

RANK_QUANTILES = [
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99
]


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


print("=" * 76)
print("FSAR PHASE S2-C TRAIN DENSE GOLD-RANK DIAGNOSTIC")
print("=" * 76)


# ============================================================
# 1. Output protection
# ============================================================

print()
print("===== 1. OUTPUT PROTECTION =====")

formal_outputs = [
    NPZ_OUT,
    JSONL_OUT,
    SUMMARY_OUT
]

existing = [
    str(p)
    for p in formal_outputs
    if p.exists()
]

if existing:
    raise RuntimeError(
        "Formal S2-C outputs already exist; refusing overwrite:\n"
        + "\n".join(existing)
    )

for p in [
    NPZ_TMP,
    JSONL_TMP,
    SUMMARY_TMP
]:
    if p.exists():
        p.unlink()

print("FORMAL_OUTPUT_COLLISION = 0")
print("OUTPUT_PROTECTION = PASS")


# ============================================================
# 2. Frozen artifact provenance
# ============================================================

print()
print("===== 2. FROZEN INPUT SHA =====")

actual_hashes = {
    "query_embedding":
        sha256_file(QUERY_EMB_FILE),

    "response_embedding":
        sha256_file(RESPONSE_EMB_FILE),

    "query_index":
        sha256_file(QUERY_INDEX_FILE),

    "response_index":
        sha256_file(RESPONSE_INDEX_FILE)
}

for k, v in actual_hashes.items():
    print(k, "=", v)


assert (
    actual_hashes["query_embedding"]
    == EXPECTED_QUERY_EMB_SHA
)

assert (
    actual_hashes["response_embedding"]
    == EXPECTED_RESPONSE_EMB_SHA
)

assert (
    actual_hashes["query_index"]
    == EXPECTED_QUERY_INDEX_SHA
)

assert (
    actual_hashes["response_index"]
    == EXPECTED_RESPONSE_INDEX_SHA
)

print("FROZEN_INPUT_SHA_GATE = PASS")


# ============================================================
# 3. Load cache/index
# ============================================================

print()
print("===== 3. CACHE STRUCTURE =====")

queries = np.load(
    QUERY_EMB_FILE,
    mmap_mode="r",
    allow_pickle=False
)

responses = np.load(
    RESPONSE_EMB_FILE,
    mmap_mode="r",
    allow_pickle=False
)

query_index = load_jsonl(
    QUERY_INDEX_FILE
)

response_index = load_jsonl(
    RESPONSE_INDEX_FILE
)


assert queries.shape == (
    EXPECTED_QUERY_ROWS,
    EXPECTED_DIM
)

assert responses.shape == (
    EXPECTED_RESPONSE_ROWS,
    EXPECTED_DIM
)

assert queries.dtype == np.float32
assert responses.dtype == np.float32

assert len(query_index) == EXPECTED_QUERY_ROWS
assert len(response_index) == EXPECTED_RESPONSE_ROWS


response_uid_to_row = {}

for i, row in enumerate(response_index):

    assert row["embedding_row"] == i

    uid = row["response_uid"]

    assert uid not in response_uid_to_row

    response_uid_to_row[uid] = i


gold_response_rows = np.empty(
    EXPECTED_QUERY_ROWS,
    dtype=np.int32
)

missing_gold = 0

for i, row in enumerate(query_index):

    assert row["embedding_row"] == i

    uid = row["gold_response_uid"]

    if uid not in response_uid_to_row:
        missing_gold += 1
        continue

    gold_response_rows[i] = (
        response_uid_to_row[uid]
    )


assert missing_gold == 0

print("query_shape =", queries.shape)
print("response_shape =", responses.shape)
print("query_index_rows =", len(query_index))
print("response_index_rows =", len(response_index))
print("gold_mapping_missing =", missing_gold)
print("CACHE_STRUCTURE = PASS")


# ============================================================
# 4. Numerical configuration
# ============================================================

print()
print("===== 4. NUMERICAL CONFIGURATION =====")

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

device = torch.device("cuda")

print("torch =", torch.__version__)
print("torch_cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))
print(
    "allow_tf32 =",
    torch.backends.cuda.matmul.allow_tf32
)
print(
    "float32_matmul_precision =",
    torch.get_float32_matmul_precision()
)
print("query_batch_size =", QUERY_BATCH_SIZE)
print(
    "candidate_chunk_size =",
    CANDIDATE_CHUNK_SIZE
)

print("NUMERICAL_CONFIGURATION = PASS")


# ============================================================
# 5. Move candidate universe to GPU
# ============================================================

print()
print("===== 5. RESPONSE CACHE TO GPU =====")

response_tensor = torch.from_numpy(
    np.asarray(responses)
).to(
    device=device,
    dtype=torch.float32
)

assert response_tensor.shape == (
    EXPECTED_RESPONSE_ROWS,
    EXPECTED_DIM
)

print(
    "response_gpu_shape =",
    tuple(response_tensor.shape)
)

print(
    "allocated_gpu_MB =",
    round(
        torch.cuda.memory_allocated()
        / 1024**2,
        2
    )
)

print("RESPONSE_CACHE_GPU = PASS")


# ============================================================
# 6. Exact gold-rank computation
# ============================================================

print()
print("===== 6. EXACT GOLD-RANK COMPUTATION =====")

ranks = np.empty(
    EXPECTED_QUERY_ROWS,
    dtype=np.int32
)

tie_counts = np.empty(
    EXPECTED_QUERY_ROWS,
    dtype=np.int32
)

gold_similarities = np.empty(
    EXPECTED_QUERY_ROWS,
    dtype=np.float32
)


num_query_batches = math.ceil(
    EXPECTED_QUERY_ROWS
    / QUERY_BATCH_SIZE
)

num_candidate_chunks = math.ceil(
    EXPECTED_RESPONSE_ROWS
    / CANDIDATE_CHUNK_SIZE
)

print(
    "query_batches =",
    num_query_batches
)

print(
    "candidate_chunks_per_batch =",
    num_candidate_chunks
)


start_time = time.time()


with torch.inference_mode():

    for batch_id, q_start in enumerate(
        range(
            0,
            EXPECTED_QUERY_ROWS,
            QUERY_BATCH_SIZE
        )
    ):

        q_end = min(
            q_start + QUERY_BATCH_SIZE,
            EXPECTED_QUERY_ROWS
        )

        q_np = np.asarray(
            queries[q_start:q_end]
        )

        q = torch.from_numpy(
            q_np
        ).to(
            device=device,
            dtype=torch.float32
        )

        gold_idx_np = gold_response_rows[
            q_start:q_end
        ]

        gold_idx = torch.from_numpy(
            gold_idx_np.astype(
                np.int64,
                copy=False
            )
        ).to(device=device)

        gold_response_vecs = (
            response_tensor[gold_idx]
        )

        gold_sim = torch.sum(
            q * gold_response_vecs,
            dim=1
        )

        greater_count = torch.zeros(
            q.shape[0],
            dtype=torch.int64,
            device=device
        )

        equal_count = torch.zeros(
            q.shape[0],
            dtype=torch.int64,
            device=device
        )


        for c_start in range(
            0,
            EXPECTED_RESPONSE_ROWS,
            CANDIDATE_CHUNK_SIZE
        ):

            c_end = min(
                c_start + CANDIDATE_CHUNK_SIZE,
                EXPECTED_RESPONSE_ROWS
            )

            candidate_chunk = (
                response_tensor[c_start:c_end]
            )

            sims = torch.matmul(
                q,
                candidate_chunk.T
            )

            # Force the factual gold cell to use
            # exactly the precomputed gold score.
            mask = (
                (gold_idx >= c_start)
                &
                (gold_idx < c_end)
            )

            if bool(mask.any()):

                row_idx = torch.nonzero(
                    mask,
                    as_tuple=False
                ).squeeze(1)

                local_gold_idx = (
                    gold_idx[row_idx]
                    - c_start
                )

                sims[
                    row_idx,
                    local_gold_idx
                ] = gold_sim[row_idx]


            greater_count += torch.sum(
                sims > gold_sim[:, None],
                dim=1
            )

            equal_count += torch.sum(
                sims == gold_sim[:, None],
                dim=1
            )


        batch_ranks = (
            greater_count + 1
        )

        assert bool(
            torch.all(batch_ranks >= 1)
        )

        assert bool(
            torch.all(
                batch_ranks
                <= EXPECTED_RESPONSE_ROWS
            )
        )

        assert bool(
            torch.all(equal_count >= 1)
        )


        ranks[q_start:q_end] = (
            batch_ranks
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        tie_counts[q_start:q_end] = (
            equal_count
            .cpu()
            .numpy()
            .astype(np.int32)
        )

        gold_similarities[
            q_start:q_end
        ] = (
            gold_sim
            .cpu()
            .numpy()
            .astype(np.float32)
        )


        if (
            batch_id == 0
            or
            (batch_id + 1) % 10 == 0
            or
            batch_id + 1 == num_query_batches
        ):

            elapsed = (
                time.time()
                - start_time
            )

            completed = q_end

            rate = completed / elapsed

            remaining = (
                EXPECTED_QUERY_ROWS
                - completed
            )

            eta = (
                remaining / rate
                if rate > 0
                else float("nan")
            )

            print(
                f"batch "
                f"{batch_id + 1}/"
                f"{num_query_batches}"
                f" | queries "
                f"{completed}/"
                f"{EXPECTED_QUERY_ROWS}"
                f" | elapsed_s="
                f"{elapsed:.1f}"
                f" | eta_s="
                f"{eta:.1f}"
            )


elapsed_total = time.time() - start_time

assert np.isfinite(
    gold_similarities
).all()

assert ranks.min() >= 1
assert ranks.max() <= EXPECTED_RESPONSE_ROWS
assert tie_counts.min() >= 1


print(
    "rank_min =",
    int(ranks.min())
)

print(
    "rank_max =",
    int(ranks.max())
)

print(
    "gold_similarity_min =",
    float(gold_similarities.min())
)

print(
    "gold_similarity_max =",
    float(gold_similarities.max())
)

print(
    "tie_count_min =",
    int(tie_counts.min())
)

print(
    "tie_count_max =",
    int(tie_counts.max())
)

print(
    "queries_with_extra_ties =",
    int(
        np.sum(
            tie_counts > 1
        )
    )
)

print(
    "computation_seconds =",
    round(elapsed_total, 3)
)

print("EXACT_GOLD_RANK_COMPUTATION = PASS")


# ============================================================
# 7. Frozen descriptive statistics
# ============================================================

print()
print("===== 7. DENSE GOLD-RANK SUMMARY =====")


recall_at = {}

for k in RECALL_CUTOFFS:

    value = float(
        np.mean(
            ranks <= k
        )
    )

    recall_at[str(k)] = value

    print(
        f"R@{k} = {value:.8f}"
    )


rank_quantile_values = {}

for qv in RANK_QUANTILES:

    value = int(
        np.quantile(
            ranks,
            qv,
            method="higher"
        )
    )

    key = f"{qv:.2f}"

    rank_quantile_values[key] = value

    print(
        f"rank_q{qv:.2f} = {value}"
    )


gold_similarity_summary = {
    "min":
        float(
            gold_similarities.min()
        ),

    "p25":
        float(
            np.quantile(
                gold_similarities,
                0.25
            )
        ),

    "median":
        float(
            np.median(
                gold_similarities
            )
        ),

    "mean":
        float(
            np.mean(
                gold_similarities
            )
        ),

    "p75":
        float(
            np.quantile(
                gold_similarities,
                0.75
            )
        ),

    "p90":
        float(
            np.quantile(
                gold_similarities,
                0.90
            )
        ),

    "p95":
        float(
            np.quantile(
                gold_similarities,
                0.95
            )
        ),

    "p99":
        float(
            np.quantile(
                gold_similarities,
                0.99
            )
        ),

    "max":
        float(
            gold_similarities.max()
        )
}


print(
    "gold_similarity_summary =",
    gold_similarity_summary
)

print(
    "queries_with_extra_ties =",
    int(np.sum(tie_counts > 1))
)

print(
    "max_tie_count =",
    int(tie_counts.max())
)

print("DENSE_GOLD_RANK_SUMMARY = COMPLETE")


# ============================================================
# 8. Write temporary result artifacts
# ============================================================

print()
print("===== 8. RESULT ARTIFACT CONSTRUCTION =====")


np.savez_compressed(
    NPZ_TMP,
    gold_response_row=gold_response_rows,
    gold_rank=ranks,
    gold_similarity=gold_similarities,
    tie_count=tie_counts
)


with JSONL_TMP.open(
    "w",
    encoding="utf-8"
) as f:

    for i, qrow in enumerate(query_index):

        out = {
            "embedding_row":
                i,

            "query_id":
                qrow["query_id"],

            "dialogue_id":
                qrow["dialogue_id"],

            "target_pair_index":
                qrow["target_pair_index"],

            "gold_response_uid":
                qrow["gold_response_uid"],

            "gold_response_row":
                int(
                    gold_response_rows[i]
                ),

            "gold_similarity":
                float(
                    gold_similarities[i]
                ),

            "gold_rank":
                int(
                    ranks[i]
                ),

            "tie_count":
                int(
                    tie_counts[i]
                )
        }

        f.write(
            json.dumps(
                out,
                ensure_ascii=False
            )
            + "\n"
        )


summary = {
    "artifact":
        "FSAR S2-C TRAIN Dense Gold-Rank Diagnostic V1",

    "status":
        "PASS",

    "scope":
        "TRAIN_ONLY",

    "query_rows":
        EXPECTED_QUERY_ROWS,

    "response_candidates":
        EXPECTED_RESPONSE_ROWS,

    "embedding_dimension":
        EXPECTED_DIM,

    "similarity":
        "COSINE_AS_DOT_PRODUCT_OF_UNIT_NORMALIZED_EMBEDDINGS",

    "rank_definition":
        "1 + number of candidates with similarity strictly greater than gold similarity",

    "rank_quantile_method":
        "higher",

    "recall_at":
        recall_at,

    "rank_quantiles":
        rank_quantile_values,

    "rank_min":
        int(ranks.min()),

    "rank_max":
        int(ranks.max()),

    "gold_similarity":
        gold_similarity_summary,

    "tie_diagnostics": {
        "queries_with_extra_ties":
            int(
                np.sum(
                    tie_counts > 1
                )
            ),

        "rate_with_extra_ties":
            float(
                np.mean(
                    tie_counts > 1
                )
            ),

        "max_tie_count":
            int(
                tie_counts.max()
            )
    },

    "computation": {
        "device":
            torch.cuda.get_device_name(0),

        "query_batch_size":
            QUERY_BATCH_SIZE,

        "candidate_chunk_size":
            CANDIDATE_CHUNK_SIZE,

        "tf32":
            False,

        "float32_matmul_precision":
            torch.get_float32_matmul_precision(),

        "seconds":
            elapsed_total
    },

    "input_sha256":
        actual_hashes,

    "hard_negative_selection_performed":
        False,

    "hard_negative_depth_frozen":
        False,

    "reranker_training_performed":
        False,

    "future_state_supervision_used":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False
}


SUMMARY_TMP.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# Validate row count before promotion
jsonl_count = 0

with JSONL_TMP.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:
        if line.strip():
            jsonl_count += 1


assert jsonl_count == EXPECTED_QUERY_ROWS

print(
    "result_jsonl_rows =",
    jsonl_count
)

print("RESULT_ARTIFACT_CONSTRUCTION = PASS")


# ============================================================
# 9. Promote formal results
# ============================================================

print()
print("===== 9. FORMAL ARTIFACT PROMOTION =====")

os.replace(
    NPZ_TMP,
    NPZ_OUT
)

os.replace(
    JSONL_TMP,
    JSONL_OUT
)

os.replace(
    SUMMARY_TMP,
    SUMMARY_OUT
)

print("FORMAL_ARTIFACT_PROMOTION = PASS")


# ============================================================
# 10. SHA
# ============================================================

print()
print("===== 10. RESULT SHA256 =====")

for p in [
    NPZ_OUT,
    JSONL_OUT,
    SUMMARY_OUT
]:

    print(
        sha256_file(p),
        p
    )


print()
print(
    "HARD_NEGATIVE_SELECTION_PERFORMED = NO"
)

print(
    "DEV_PAYLOAD_USED = NO"
)

print(
    "TEST_PAYLOAD_USED = NO"
)

print()
print(
    "PHASE_S2C_DENSE_GOLD_RANK_DIAGNOSTIC = PASS"
)
