import json
import hashlib
import numpy as np
import torch

from pathlib import Path
from src.models.s12c_matched_ablation_models import (
    QRFutureFullMatchedModel,
)

ROOT = Path.cwd()
DEVICE = "cuda"
DIM = 1024
FOLDS = 5
CHUNK = 1024

OUT = ROOT / (
    "results/phase_s12c/"
    "step_2f16ai_qr_full_tie_audit_v1.json"
)

assert not OUT.exists(), "OUTPUT_COLLISION"

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

# ------------------------------------------------
# 1. Frozen inputs
# ------------------------------------------------

protocol = json.loads(
    (ROOT / "protocols/fsar_s12c_external_test_evaluation_v1.json")
    .read_text()
)

pred = np.load(
    ROOT / "results/phase_s12c/external_test_v1/"
    "s12c_external_test_ranking_predictions_v1.npz",
    allow_pickle=False
)

D = ROOT / "results/phase_s12c/external_test_v1"

Q = np.load(
    D / "test_all_clean_query_embeddings_bge_v1.npy",
    mmap_mode="r"
)

G = np.load(
    D / "test_all_clean_gold_embeddings_bge_v1.npy",
    mmap_mode="r"
)

R = np.load(
    ROOT / "cache/embeddings/"
    "fsar_train_unique_responses_bge_v1.npy",
    mmap_mode="r"
)

assert Q.shape == (18073, DIM)
assert G.shape == (18073, DIM)
assert R.shape == (61502, DIM)

# ------------------------------------------------
# 2. Three anomalous queries
# ------------------------------------------------

target_ids = [
    "psydial_1344_7",
    "psydial_393_25",
    "psydial_955_69",
]

id_map = {
    str(s): i for i, s in enumerate(pred["sample_ids"])
}

indices = np.array([id_map[s] for s in target_ids])

official = pred["rank_QR_FUTURE_FULL"][indices]
gold_idx = pred["gold_match_idx"][indices]

q = torch.tensor(
    np.array(Q[indices], dtype=np.float32),
    device=DEVICE
)

g = torch.tensor(
    np.array(G[indices], dtype=np.float32),
    device=DEVICE
)

print("STEP_2F16AJ_START")
print("QUERIES =", target_ids)
print("OFFICIAL_RANKS =", official.tolist())
print("GOLD_MATCH_IDX =", gold_idx.tolist())

# ------------------------------------------------
# 3. Load exact frozen QR-FUTURE-FULL checkpoints
# ------------------------------------------------

items = sorted(
    [
        x for x in protocol["checkpoint_manifest"]
        if x["condition"] == "QR_FUTURE_FULL"
    ],
    key=lambda x: int(x["fold"])
)

assert len(items) == FOLDS

direct_folds = []
full_folds = []
component_folds = []

direct_gold_folds = []
full_gold_folds = []
component_gold_folds = []

# ------------------------------------------------
# 4. Calculate three score implementations
# ------------------------------------------------

for fold, item in enumerate(items):

    path = ROOT / item["path"]
    assert sha(path) == item["sha256"]

    ck = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    assert ck["condition"] == "QR_FUTURE_FULL"
    assert int(ck["fold"]) == fold

    model = QRFutureFullMatchedModel().to(DEVICE)
    model.load_state_dict(
        ck["model_state_dict"],
        strict=True
    )
    model.eval()

    ranker = model.ranker

    with torch.inference_mode():

        zq = ranker.adapt_query(q)
        zg = ranker.adapt_response(g)

        w = ranker.rank_head.weight.detach().float().reshape(-1)

        wq = w[:DIM]
        wr = w[DIM:2*DIM]
        wd = w[2*DIM:3*DIM]
        wp = w[3*DIM:4*DIM]

        bias = (
            ranker.rank_head.bias.detach().float()
            if ranker.rank_head.bias is not None
            else torch.zeros(1, device=DEVICE)
        )

        # Gold scores
        direct_gold = ranker.score_from_adapted(
            zq[:, None, :],
            zg[:, None, :]
        ).squeeze(1)

        qterm = (zq * wq).sum(-1)
        gr = (zg * wr).sum(-1)
        gd = (torch.abs(zq-zg) * wd).sum(-1)
        gp = (zq * zg * wp).sum(-1)

        full_gold = qterm + gr + gd + gp + bias
        component_gold = gr + gd + gp

        direct_gold_folds.append(
            direct_gold.cpu().numpy()
        )
        full_gold_folds.append(
            full_gold.cpu().numpy()
        )
        component_gold_folds.append(
            component_gold.cpu().numpy()
        )

        direct_parts = []
        full_parts = []
        component_parts = []

        for start in range(0, len(R), CHUNK):

            end = min(start + CHUNK, len(R))

            rb = torch.tensor(
                np.array(R[start:end], dtype=np.float32),
                device=DEVICE
            )

            zr = ranker.adapt_response(rb)

            nq = len(zq)
            nr = len(zr)

            zqe = zq[:, None, :].expand(-1, nr, -1)
            zre = zr[None, :, :].expand(nq, -1, -1)

            # Actual model scoring
            direct = ranker.score_from_adapted(zqe, zre)

            # Explicit decomposition
            response = (zr * wr).sum(-1)[None, :]
            absolute = (
                torch.abs(zqe-zre) * wd
            ).sum(-1)
            product = (zqe * zre * wp).sum(-1)

            full = (
                qterm[:, None] +
                response +
                absolute +
                product +
                bias
            )

            component = response + absolute + product

            direct_parts.append(direct.cpu().numpy())
            full_parts.append(full.cpu().numpy())
            component_parts.append(component.cpu().numpy())

    direct_folds.append(
        np.concatenate(direct_parts, axis=1)
    )
    full_folds.append(
        np.concatenate(full_parts, axis=1)
    )
    component_folds.append(
        np.concatenate(component_parts, axis=1)
    )

    del model, ck
    torch.cuda.empty_cache()

    print("FOLD", fold, "PASS", flush=True)

# ------------------------------------------------
# 5. Compare aggregation strategies
# ------------------------------------------------

def aggregate(arrays, mode):

    stack = np.stack(arrays, axis=0).astype(np.float32)

    if mode == "incremental_float32":
        result = np.zeros_like(stack[0])
        for x in arrays:
            result += x / np.float32(FOLDS)
        return result

    if mode == "sum_then_divide_float32":
        return (
            np.sum(stack, axis=0, dtype=np.float32)
            / np.float32(FOLDS)
        )

    if mode == "mean_float32":
        return np.mean(stack, axis=0, dtype=np.float32)

    if mode == "mean_float64":
        return np.mean(stack.astype(np.float64), axis=0)

    raise ValueError(mode)

modes = [
    "incremental_float32",
    "sum_then_divide_float32",
    "mean_float32",
    "mean_float64",
]

score_sources = {
    "DIRECT_MODEL": (direct_folds, direct_gold_folds),
    "MANUAL_FULL": (full_folds, full_gold_folds),
    "COMPONENT_ONLY": (
        component_folds,
        component_gold_folds
    ),
}

results = {}

for source_name, (score_folds, gold_folds) in score_sources.items():

    print("\nSCORE_SOURCE =", source_name)

    results[source_name] = {}

    for mode in modes:

        scores = aggregate(score_folds, mode)
        gold = aggregate(gold_folds, mode)

        records = []

        for i in range(len(indices)):

            mi = int(gold_idx[i])

            valid = np.ones(len(R), dtype=bool)

            if mi >= 0:
                valid[mi] = False

            negative = scores[i, valid]
            gs = gold[i]

            rank_ge = int(
                1 + np.count_nonzero(negative >= gs)
            )

            rank_gt = int(
                1 + np.count_nonzero(negative > gs)
            )

            above = negative[negative > gs]
            below = negative[negative < gs]

            record = {
                "sample_id": target_ids[i],
                "official_rank": int(official[i]),
                "rank_ge": rank_ge,
                "rank_gt": rank_gt,
                "rank_diff": rank_ge - int(official[i]),
                "exact_ties": int(
                    np.count_nonzero(negative == gs)
                ),
                "near_ties_1e-7": int(
                    np.count_nonzero(
                        np.abs(negative-gs) <= 1e-7
                    )
                ),
                "near_ties_1e-6": int(
                    np.count_nonzero(
                        np.abs(negative-gs) <= 1e-6
                    )
                ),
                "nearest_above_gap": (
                    float(np.min(above-gs))
                    if len(above) else None
                ),
                "nearest_below_gap": (
                    float(gs-np.max(below))
                    if len(below) else None
                ),
                "gold_score": float(gs),
            }

            records.append(record)
            print(mode, record)

        results[source_name][mode] = records

# ------------------------------------------------
# 6. Final audit
# ------------------------------------------------

for source_name, modes_data in results.items():
    for mode, records in modes_data.items():

        matched = sum(
            x["rank_ge"] == x["official_rank"]
            for x in records
        )

        print(
            "OFFICIAL_MATCH",
            source_name,
            mode,
            matched,
            "/ 3"
        )

report = {
    "status": "COMPLETED",
    "scope": "QR_FUTURE_FULL_THREE_QUERY_NUMERICAL_AUDIT",
    "sample_ids": target_ids,
    "official_ranks": official.tolist(),
    "results": results,
    "training": False,
    "checkpoint_modification": False,
    "external_prediction_modification": False,
}

OUT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nSTEP_2F16AJ_COMPLETE")
print("OUTPUT =", OUT)
