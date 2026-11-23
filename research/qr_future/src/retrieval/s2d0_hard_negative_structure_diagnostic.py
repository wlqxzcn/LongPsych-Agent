import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]

Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

QI_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_index_v1.jsonl"
RI_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"

S2C_FILE = ROOT / "results/phase_s2/s2c_train_dense_gold_rank_arrays_v1.npz"

SUMMARY_OUT = (
    ROOT
    / "results/phase_s2/s2d0_hard_negative_structure_summary_v1.json"
)

DETAIL_OUT = (
    ROOT
    / "results/phase_s2/s2d0_hard_negative_structure_detail_v1.npz"
)

SUMMARY_TMP = Path(str(SUMMARY_OUT) + ".tmp")
DETAIL_TMP = (
    ROOT
    / "results/phase_s2/s2d0_hard_negative_structure_detail_v1.tmp.npz"
)


EXPECTED_Q_SHA = (
    "18aa1ed637bef3980775c10d96d1cde6"
    "112863b0c5761e2e69fb7ab89c981005"
)

EXPECTED_R_SHA = (
    "7a4ee8d462adce73174c853bf21b7581"
    "32f5243020fa6253b1c01a99973bc96f"
)

EXPECTED_QI_SHA = (
    "7d3659d38668f653289fd649fef186cd"
    "2f0a1003338af88ff6c805c23960de75"
)

EXPECTED_RI_SHA = (
    "fd8263c8fd00b4d8d23b915c7f241fb"
    "84d1d2ffc7b24c747bf34e4a9b8e47274"
)

EXPECTED_S2C_SHA = (
    "04b346fb61ad655850c67033000d3096"
    "f674459cbd826571880fdd907b4d686f"
)

NQ = 62889
NR = 61502
DIM = 1024

CUTOFFS = [10, 20, 50, 100, 200, 500, 1000]

TOP_NON_GOLD = 1001
RAW_TOPK = 1002

QUERY_BATCH = 512


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
print("FSAR PHASE S2-D0 HARD-NEGATIVE STRUCTURE DIAGNOSTIC")
print("=" * 78)


# ============================================================
# 1. Output protection
# ============================================================

print()
print("===== 1. OUTPUT PROTECTION =====")

if SUMMARY_OUT.exists() or DETAIL_OUT.exists():
    raise RuntimeError(
        "Formal S2-D0 output already exists; refusing overwrite."
    )

for p in [SUMMARY_TMP, DETAIL_TMP]:
    if p.exists():
        p.unlink()

print("OUTPUT_PROTECTION = PASS")


# ============================================================
# 2. Frozen input provenance
# ============================================================

print()
print("===== 2. FROZEN INPUT SHA =====")

actual = {
    "query_embedding": sha256_file(Q_FILE),
    "response_embedding": sha256_file(R_FILE),
    "query_index": sha256_file(QI_FILE),
    "response_index": sha256_file(RI_FILE),
    "s2c_rank_arrays": sha256_file(S2C_FILE)
}

for k, v in actual.items():
    print(k, "=", v)

assert actual["query_embedding"] == EXPECTED_Q_SHA
assert actual["response_embedding"] == EXPECTED_R_SHA
assert actual["query_index"] == EXPECTED_QI_SHA
assert actual["response_index"] == EXPECTED_RI_SHA
assert actual["s2c_rank_arrays"] == EXPECTED_S2C_SHA

print("FROZEN_INPUT_SHA_GATE = PASS")


# ============================================================
# 3. Load frozen artifacts
# ============================================================

print()
print("===== 3. STRUCTURE =====")

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

assert Q.shape == (NQ, DIM)
assert R.shape == (NR, DIM)
assert len(QI) == NQ
assert len(RI) == NR
assert gold_rows.shape == (NQ,)
assert gold_sims.shape == (NQ,)

response_occurrence = np.asarray(
    [int(x["occurrence_count"]) for x in RI],
    dtype=np.int32
)

# response row -> dialogues in which this exact normalized response occurs
uid_to_row = {
    x["response_uid"]: i
    for i, x in enumerate(RI)
}

response_dialogues = [
    set()
    for _ in range(NR)
]

for qrow in QI:
    rr = uid_to_row[qrow["gold_response_uid"]]
    response_dialogues[rr].add(qrow["dialogue_id"])

print("query_rows =", NQ)
print("response_rows =", NR)
print("STRUCTURE = PASS")


# ============================================================
# 4. CUDA setup
# ============================================================

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
print("query_batch =", QUERY_BATCH)
print("raw_topk =", RAW_TOPK)
print("CUDA_SETUP = PASS")


# ============================================================
# 5. Diagnostics storage
# ============================================================

rank_score = {
    k: np.empty(NQ, dtype=np.float32)
    for k in CUTOFFS
}

rank_margin = {
    k: np.empty(NQ, dtype=np.float32)
    for k in CUTOFFS
}

same_dialogue_count = {
    k: np.zeros(NQ, dtype=np.int16)
    for k in CUTOFFS
}

reuse_counts = {
    k: np.zeros(NR, dtype=np.int64)
    for k in CUTOFFS
}

weighted_occurrence_sum = {
    k: 0
    for k in CUTOFFS
}

exposure_total = {
    k: 0
    for k in CUTOFFS
}

boundary_ties = np.zeros(
    NQ,
    dtype=np.bool_
)


# ============================================================
# 6. Exact Top-1000 non-gold diagnosis
# ============================================================

print()
print("===== 5. TOP-1000 NON-GOLD DIAGNOSTIC =====")

with torch.inference_mode():

    for b, start in enumerate(range(0, NQ, QUERY_BATCH)):

        end = min(start + QUERY_BATCH, NQ)

        q = torch.from_numpy(
            np.array(
                Q[start:end],
                dtype=np.float32,
                copy=True
            )
        ).to(device)

        sims = torch.matmul(
            q,
            R_gpu.T
        )

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

            global_i = start + local
            gold = int(gold_rows[global_i])

            keep = indices[local] != gold

            neg_idx = indices[local][keep]
            neg_val = values[local][keep]

            assert len(neg_idx) >= TOP_NON_GOLD

            neg_idx = neg_idx[:TOP_NON_GOLD]
            neg_val = neg_val[:TOP_NON_GOLD]

            assert gold not in neg_idx[:1000]

            # Top-1000 boundary tie diagnostic:
            # compare non-gold rank 1000 and 1001.
            boundary_ties[global_i] = (
                neg_val[999] == neg_val[1000]
            )

            dialogue_id = QI[global_i]["dialogue_id"]

            top1000_rows = neg_idx[:1000]

            same_flags = np.fromiter(
                (
                    dialogue_id in response_dialogues[int(r)]
                    for r in top1000_rows
                ),
                dtype=np.bool_,
                count=1000
            )

            for k in CUTOFFS:

                kth_score = float(neg_val[k - 1])

                rank_score[k][global_i] = kth_score

                rank_margin[k][global_i] = (
                    float(gold_sims[global_i])
                    - kth_score
                )

                rows_k = neg_idx[:k]

                reuse_counts[k] += np.bincount(
                    rows_k,
                    minlength=NR
                )

                same_dialogue_count[k][global_i] = int(
                    same_flags[:k].sum()
                )

                weighted_occurrence_sum[k] += int(
                    response_occurrence[rows_k].sum()
                )

                exposure_total[k] += k


        if (
            b == 0
            or
            (b + 1) % 20 == 0
            or
            end == NQ
        ):
            print(
                f"batch={b + 1} "
                f"queries={end}/{NQ}"
            )


print("TOP1000_DIAGNOSTIC_COMPUTATION = PASS")


# ============================================================
# 7. Aggregate
# ============================================================

print()
print("===== 6. CUTOFF SUMMARY =====")

cutoff_summary = {}

for k in CUTOFFS:

    counts = reuse_counts[k]

    active = counts[counts > 0]

    unique_candidates = int(
        np.sum(counts > 0)
    )

    top_reuse = np.sort(counts)[-20:][::-1]

    same_total = int(
        same_dialogue_count[k].sum()
    )

    total_exposures = NQ * k

    summary = {
        "negative_similarity": {
            "mean": float(rank_score[k].mean()),
            "median": float(np.median(rank_score[k])),
            "p25": float(np.quantile(rank_score[k], 0.25)),
            "p75": float(np.quantile(rank_score[k], 0.75))
        },

        "gold_minus_kth_negative_margin": {
            "mean": float(rank_margin[k].mean()),
            "median": float(np.median(rank_margin[k])),
            "fraction_gold_above_kth_negative":
                float(np.mean(rank_margin[k] > 0)),
            "fraction_gold_below_kth_negative":
                float(np.mean(rank_margin[k] < 0))
        },

        "candidate_reuse": {
            "unique_candidates_used": unique_candidates,
            "unique_candidate_rate":
                float(unique_candidates / NR),
            "max_reuse_count":
                int(counts.max()),
            "median_reuse_among_used":
                float(np.median(active)),
            "top20_reuse_counts":
                [int(x) for x in top_reuse]
        },

        "same_dialogue": {
            "exposures": same_total,
            "rate":
                float(same_total / total_exposures),
            "queries_with_at_least_one":
                int(
                    np.sum(
                        same_dialogue_count[k] > 0
                    )
                ),
            "query_rate_with_at_least_one":
                float(
                    np.mean(
                        same_dialogue_count[k] > 0
                    )
                )
        },

        "source_response_frequency": {
            "mean_occurrence_count_per_exposure":
                float(
                    weighted_occurrence_sum[k]
                    / exposure_total[k]
                )
        }
    }

    cutoff_summary[str(k)] = summary

    print()
    print(f"--- K={k} ---")
    print(
        "negative_similarity_median =",
        summary["negative_similarity"]["median"]
    )
    print(
        "margin_median =",
        summary[
            "gold_minus_kth_negative_margin"
        ]["median"]
    )
    print(
        "fraction_gold_above_kth_negative =",
        summary[
            "gold_minus_kth_negative_margin"
        ]["fraction_gold_above_kth_negative"]
    )
    print(
        "unique_candidates_used =",
        unique_candidates
    )
    print(
        "max_reuse_count =",
        int(counts.max())
    )
    print(
        "same_dialogue_exposure_rate =",
        summary["same_dialogue"]["rate"]
    )
    print(
        "same_dialogue_query_rate =",
        summary[
            "same_dialogue"
        ]["query_rate_with_at_least_one"]
    )


boundary_tie_n = int(boundary_ties.sum())

print()
print(
    "top1000_boundary_tie_queries =",
    boundary_tie_n
)

print(
    "top1000_boundary_tie_rate =",
    float(boundary_tie_n / NQ)
)


# ============================================================
# 8. Save
# ============================================================

summary = {
    "artifact":
        "FSAR S2-D0 TRAIN Hard-Negative Structure Diagnostic V1",

    "status":
        "PASS",

    "scope":
        "TRAIN_ONLY",

    "diagnostic_cutoffs":
        CUTOFFS,

    "diagnostic_max_non_gold_rank":
        1000,

    "cutoff_summary":
        cutoff_summary,

    "top1000_boundary_ties": {
        "queries":
            boundary_tie_n,
        "rate":
            float(boundary_tie_n / NQ)
    },

    "formal_training_negative_selection":
        False,

    "hard_negative_depth_frozen":
        False,

    "hard_negative_count_frozen":
        False,

    "future_state_supervision_used":
        False,

    "reranker_training_performed":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256":
        actual
}


SUMMARY_TMP.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


np.savez_compressed(
    DETAIL_TMP,
    boundary_ties=boundary_ties,
    **{
        f"k{k}_score": rank_score[k]
        for k in CUTOFFS
    },
    **{
        f"k{k}_margin": rank_margin[k]
        for k in CUTOFFS
    },
    **{
        f"k{k}_same_dialogue_count":
            same_dialogue_count[k]
        for k in CUTOFFS
    }
)


os.replace(
    SUMMARY_TMP,
    SUMMARY_OUT
)

os.replace(
    DETAIL_TMP,
    DETAIL_OUT
)


print()
print("===== 7. FINAL GATE =====")
print("FORMAL_TRAINING_NEGATIVE_SELECTION = NO")
print("HARD_NEGATIVE_DEPTH_FROZEN = NO")
print("HARD_NEGATIVE_COUNT_FROZEN = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")
print()
print("PHASE_S2D0_HARD_NEGATIVE_STRUCTURE_DIAGNOSTIC = PASS")
