import json
import hashlib
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]

Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

QI_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_index_v1.jsonl"
RI_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"

S2C_FILE = ROOT / "results/phase_s2/s2c_train_dense_gold_rank_arrays_v1.npz"

OUT_DIR = ROOT / "cache/hard_negatives"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NPZ_OUT = OUT_DIR / "fsar_train_dense_top100_reservoir_v1.npz"
MANIFEST_OUT = OUT_DIR / "fsar_train_dense_top100_reservoir_manifest_v1.json"

NPZ_TMP = OUT_DIR / "fsar_train_dense_top100_reservoir_v1.tmp.npz"
MANIFEST_TMP = OUT_DIR / "fsar_train_dense_top100_reservoir_manifest_v1.tmp.json"


NQ = 62889
NR = 61502
DIM = 1024

DEPTH = 100
RAW_TOPK = 102
QUERY_BATCH = 512


EXPECTED_Q_SHA = "18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005"
EXPECTED_R_SHA = "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f"
EXPECTED_QI_SHA = "7d3659d38668f653289fd649fef186cd2f0a1003338af88ff6c805c23960de75"
EXPECTED_RI_SHA = "fd8263c8fd00b4d8d23b915c7f241fb84d1d2ffc7b24c747bf34e4a9b8e47274"
EXPECTED_S2C_SHA = "04b346fb61ad655850c67033000d3096f674459cbd826571880fdd907b4d686f"


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            x = f.read(block)
            if not x:
                break
            h.update(x)
    return h.hexdigest()


def load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


print("=" * 78)
print("FSAR PHASE S2-D1 MATERIALIZE DENSE TOP-100 RESERVOIR")
print("=" * 78)


print()
print("===== 1. OUTPUT PROTECTION =====")

if NPZ_OUT.exists() or MANIFEST_OUT.exists():
    raise RuntimeError(
        "Formal S2-D1 output already exists; refusing overwrite."
    )

for p in [NPZ_TMP, MANIFEST_TMP]:
    if p.exists():
        p.unlink()

print("OUTPUT_PROTECTION = PASS")


print()
print("===== 2. FROZEN INPUT SHA =====")

actual_sha = {
    "query_embedding": sha256_file(Q_FILE),
    "response_embedding": sha256_file(R_FILE),
    "query_index": sha256_file(QI_FILE),
    "response_index": sha256_file(RI_FILE),
    "s2c_rank_arrays": sha256_file(S2C_FILE)
}

for k, v in actual_sha.items():
    print(k, "=", v)

assert actual_sha["query_embedding"] == EXPECTED_Q_SHA
assert actual_sha["response_embedding"] == EXPECTED_R_SHA
assert actual_sha["query_index"] == EXPECTED_QI_SHA
assert actual_sha["response_index"] == EXPECTED_RI_SHA
assert actual_sha["s2c_rank_arrays"] == EXPECTED_S2C_SHA

print("FROZEN_INPUT_SHA_GATE = PASS")


print()
print("===== 3. LOAD STRUCTURE =====")

Q = np.load(Q_FILE, mmap_mode="r", allow_pickle=False)
R = np.load(R_FILE, mmap_mode="r", allow_pickle=False)

QI = load_jsonl(QI_FILE)
RI = load_jsonl(RI_FILE)

S2C = np.load(S2C_FILE, allow_pickle=False)

gold_rows = np.asarray(
    S2C["gold_response_row"],
    dtype=np.int32
)

gold_sims = np.asarray(
    S2C["gold_similarity"],
    dtype=np.float32
)

gold_ranks = np.asarray(
    S2C["gold_rank"],
    dtype=np.int32
)

assert Q.shape == (NQ, DIM)
assert R.shape == (NR, DIM)
assert len(QI) == NQ
assert len(RI) == NR

uid_to_row = {
    row["response_uid"]: i
    for i, row in enumerate(RI)
}

response_dialogues = [
    set()
    for _ in range(NR)
]

for row in QI:
    r = uid_to_row[row["gold_response_uid"]]
    response_dialogues[r].add(row["dialogue_id"])

occurrence_count = np.asarray(
    [int(x["occurrence_count"]) for x in RI],
    dtype=np.int16
)

print("query_rows =", NQ)
print("response_rows =", NR)
print("reservoir_depth =", DEPTH)
print("STRUCTURE = PASS")


print()
print("===== 4. CUDA =====")

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

device = torch.device("cuda")

R_gpu = torch.from_numpy(
    np.array(R, dtype=np.float32, copy=True)
).to(device)

print("gpu =", torch.cuda.get_device_name(0))
print("allow_tf32 =", torch.backends.cuda.matmul.allow_tf32)
print("CUDA_SETUP = PASS")


print()
print("===== 5. MATERIALIZE TOP-100 NON-GOLD =====")

candidate_rows = np.empty(
    (NQ, DEPTH),
    dtype=np.int32
)

candidate_scores = np.empty(
    (NQ, DEPTH),
    dtype=np.float32
)

same_dialogue = np.empty(
    (NQ, DEPTH),
    dtype=np.bool_
)

candidate_occurrence = np.empty(
    (NQ, DEPTH),
    dtype=np.int16
)

boundary_ties = np.zeros(
    NQ,
    dtype=np.bool_
)


with torch.inference_mode():

    batch_id = 0

    for start in range(0, NQ, QUERY_BATCH):

        end = min(start + QUERY_BATCH, NQ)

        q = torch.from_numpy(
            np.array(
                Q[start:end],
                dtype=np.float32,
                copy=True
            )
        ).to(device)

        sims = torch.matmul(q, R_gpu.T)

        values, indices = torch.topk(
            sims,
            k=RAW_TOPK,
            dim=1,
            largest=True,
            sorted=True
        )

        values = values.cpu().numpy()
        indices = indices.cpu().numpy()

        for local in range(end - start):

            i = start + local
            gold = int(gold_rows[i])

            keep = indices[local] != gold

            neg_idx = indices[local][keep]
            neg_scores = values[local][keep]

            assert len(neg_idx) >= DEPTH + 1

            selected_idx = neg_idx[:DEPTH]
            selected_scores = neg_scores[:DEPTH]

            assert gold not in selected_idx
            assert len(np.unique(selected_idx)) == DEPTH

            candidate_rows[i] = selected_idx
            candidate_scores[i] = selected_scores

            boundary_ties[i] = (
                neg_scores[DEPTH - 1]
                == neg_scores[DEPTH]
            )

            did = QI[i]["dialogue_id"]

            same_dialogue[i] = np.fromiter(
                (
                    did in response_dialogues[int(r)]
                    for r in selected_idx
                ),
                dtype=np.bool_,
                count=DEPTH
            )

            candidate_occurrence[i] = (
                occurrence_count[selected_idx]
            )

        batch_id += 1

        if (
            batch_id == 1
            or batch_id % 20 == 0
            or end == NQ
        ):
            print(
                f"batch={batch_id} "
                f"queries={end}/{NQ}"
            )


assert candidate_rows.shape == (NQ, DEPTH)
assert candidate_scores.shape == (NQ, DEPTH)

assert np.all(
    candidate_scores[:, :-1]
    >= candidate_scores[:, 1:]
)

print("TOP100_MATERIALIZATION = PASS")


print()
print("===== 6. RESERVOIR AUDIT =====")

flat_rows = candidate_rows.reshape(-1)

reuse = np.bincount(
    flat_rows,
    minlength=NR
)

unique_used = int(
    np.sum(reuse > 0)
)

coverage = float(
    unique_used / NR
)

max_reuse = int(
    reuse.max()
)

same_exposures = int(
    same_dialogue.sum()
)

same_exposure_rate = float(
    same_dialogue.mean()
)

same_query_n = int(
    np.sum(
        same_dialogue.any(axis=1)
    )
)

same_query_rate = float(
    same_query_n / NQ
)

boundary_tie_n = int(
    boundary_ties.sum()
)

rank100_scores = candidate_scores[:, 99]

margin100 = (
    gold_sims
    - rank100_scores
)

fraction_gold_above = float(
    np.mean(
        margin100 > 0
    )
)


print("unique_candidates_used =", unique_used)
print("candidate_identity_coverage =", coverage)
print("max_candidate_reuse =", max_reuse)

print(
    "rank100_similarity_median =",
    float(np.median(rank100_scores))
)

print(
    "gold_minus_rank100_margin_median =",
    float(np.median(margin100))
)

print(
    "fraction_gold_above_rank100 =",
    fraction_gold_above
)

print(
    "same_dialogue_exposures =",
    same_exposures
)

print(
    "same_dialogue_exposure_rate =",
    same_exposure_rate
)

print(
    "queries_with_same_dialogue_candidate =",
    same_query_n
)

print(
    "same_dialogue_query_rate =",
    same_query_rate
)

print(
    "rank100_boundary_tie_queries =",
    boundary_tie_n
)

print(
    "rank100_boundary_tie_rate =",
    float(boundary_tie_n / NQ)
)

print("RESERVOIR_AUDIT = PASS")


print()
print("===== 7. SAVE FORMAL RESERVOIR =====")

np.savez_compressed(
    NPZ_TMP,
    candidate_response_row=candidate_rows,
    candidate_similarity=candidate_scores,
    same_dialogue=same_dialogue,
    candidate_occurrence_count=candidate_occurrence,
    gold_response_row=gold_rows,
    gold_similarity=gold_sims,
    gold_dense_rank=gold_ranks,
    rank100_boundary_tie=boundary_ties
)


manifest = {
    "artifact":
        "FSAR S2-D1 TRAIN Dense Top-100 Hard-Negative Reservoir V1",

    "status": "PASS",

    "scope": "TRAIN_ONLY",

    "query_rows": NQ,
    "candidate_universe_size": NR,

    "reservoir_depth": DEPTH,

    "candidate_unit":
        "UNIQUE_NORMALIZED_RESPONSE",

    "gold_excluded": True,

    "same_dialogue_excluded": False,

    "negative_future_state_labels_created": False,

    "reservoir_shape":
        [NQ, DEPTH],

    "unique_candidates_used":
        unique_used,

    "candidate_identity_coverage":
        coverage,

    "max_candidate_reuse":
        max_reuse,

    "rank100_similarity_median":
        float(
            np.median(rank100_scores)
        ),

    "gold_minus_rank100_margin_median":
        float(
            np.median(margin100)
        ),

    "fraction_gold_above_rank100":
        fraction_gold_above,

    "same_dialogue": {
        "exposures":
            same_exposures,

        "exposure_rate":
            same_exposure_rate,

        "queries_with_at_least_one":
            same_query_n,

        "query_rate":
            same_query_rate
    },

    "boundary_ties": {
        "rank100_queries":
            boundary_tie_n,

        "rank100_rate":
            float(
                boundary_tie_n / NQ
            )
    },

    "formal_training_negative_count":
        "NOT_YET_FROZEN",

    "negative_sampling_strategy":
        "NOT_YET_FROZEN",

    "reranker_training_performed":
        False,

    "future_state_supervision_used":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256":
        actual_sha
}


MANIFEST_TMP.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


os.replace(
    NPZ_TMP,
    NPZ_OUT
)

os.replace(
    MANIFEST_TMP,
    MANIFEST_OUT
)

print("FORMAL_RESERVOIR_SAVED = PASS")


print()
print("===== 8. SHA256 =====")

print(
    sha256_file(NPZ_OUT),
    NPZ_OUT
)

print(
    sha256_file(MANIFEST_OUT),
    MANIFEST_OUT
)


print()
print("FORMAL_TRAINING_NEGATIVE_COUNT = NOT_YET_FROZEN")
print("NEGATIVE_SAMPLING_STRATEGY = NOT_YET_FROZEN")
print("FUTURE_STATE_LABELS_FOR_NEGATIVES = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S2D1_TOP100_RESERVOIR = PASS")
