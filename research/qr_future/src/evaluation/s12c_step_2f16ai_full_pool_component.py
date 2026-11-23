#!/usr/bin/env python3
from pathlib import Path
import json
import csv
import hashlib
import numpy as np
import torch

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RFutureFullMatchedModel,
    RFutureNoProductMatchedModel,
    QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

ROOT = Path.cwd()
DIM = 1024
FOLDS = 5
SEED = 20261108
Q_BATCH = 8
R_CHUNK = 1024

OUT = ROOT / (
    "results/phase_s12c/"
    "step_2f16ai_full_pool_component_v1"
)

if OUT.exists():
    raise RuntimeError("OUTPUT_EXISTS: refuse overwrite")

OUT.mkdir(parents=True)

device = torch.device("cuda")
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False

CLASSES = {
    "RANK_ONLY": RankOnlyFullMatchedModel,
    "R_FUTURE_FULL": RFutureFullMatchedModel,
    "R_FUTURE_NO_PRODUCT": RFutureNoProductMatchedModel,
    "QR_FUTURE_FULL": QRFutureFullMatchedModel,
    "QR_FUTURE_NO_PRODUCT": QRFutureNoProductMatchedModel,
}

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

# --------------------------------------------------
# 1. Frozen inputs and official predictions
# --------------------------------------------------

protocol_path = (
    ROOT / "protocols/fsar_s12c_external_test_evaluation_v1.json"
)

protocol = json.loads(
    protocol_path.read_text(encoding="utf-8")
)

assert protocol["status"] == "FROZEN"

pred_path = (
    ROOT / "results/phase_s12c/external_test_v1/"
    "s12c_external_test_ranking_predictions_v1.npz"
)

pred = np.load(pred_path, allow_pickle=False)

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

match = pred["gold_match_idx"]

assert Q.shape == (18073, DIM)
assert G.shape == (18073, DIM)
assert R.shape == (61502, DIM)

# --------------------------------------------------
# 2. Same deterministic 64-query sample
# --------------------------------------------------

rng = np.random.default_rng(SEED)

matched = np.flatnonzero(match >= 0)
unmatched = np.flatnonzero(match < 0)

qidx = np.concatenate([
    rng.choice(matched, 32, replace=False),
    rng.choice(unmatched, 32, replace=False)
])

rng.shuffle(qidx)

match_sample = match[qidx]

q = torch.as_tensor(
    np.asarray(Q[qidx], dtype=np.float32),
    device=device
)

g = torch.as_tensor(
    np.asarray(G[qidx], dtype=np.float32),
    device=device
)

NQ = len(qidx)
NR = len(R)

print("STEP_2F16AI_START")
print("QUERIES =", NQ)
print("MATCHED =", int((match_sample >= 0).sum()))
print("UNMATCHED =", int((match_sample < 0).sum()))
print("FULL_RESPONSE_POOL =", NR)

# --------------------------------------------------
# 3. Checkpoint manifest
# --------------------------------------------------

checkpoint_manifest = protocol["checkpoint_manifest"]

assert len(checkpoint_manifest) == 25

summary = {}
query_records = []
reconstruction_errors = {}

# --------------------------------------------------
# 4. Full-pool decomposition
# --------------------------------------------------

for condition, ModelClass in CLASSES.items():

    print("\nCONDITION =", condition, flush=True)

    has_product = "NO_PRODUCT" not in condition

    terms = [
        "RESPONSE",
        "ABS_DIFF",
        "RESPONSE_PLUS_ABS",
        "FULL",
    ]

    if has_product:
        terms.append("PRODUCT")

    neg = {
        name: np.zeros(
            (NQ, NR), dtype=np.float32
        )
        for name in terms
    }

    gold_scores = {
        name: np.zeros(NQ, dtype=np.float32)
        for name in terms
    }

    items = sorted(
        [
            x for x in checkpoint_manifest
            if x["condition"] == condition
        ],
        key=lambda x: int(x["fold"])
    )

    assert len(items) == FOLDS

    max_reconstruction_error = 0.0

    for fold, item in enumerate(items):

        path = ROOT / item["path"]

        assert sha(path) == item["sha256"], (
            "CHECKPOINT_SHA_FAILURE",
            condition,
            fold
        )

        ck = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        assert ck["condition"] == condition
        assert int(ck["fold"]) == fold

        model = ModelClass().to(device)

        model.load_state_dict(
            ck["model_state_dict"],
            strict=True
        )

        model.eval()

        ranker = model.ranker

        with torch.inference_mode():

            zq = ranker.adapt_query(q)
            zg = ranker.adapt_response(g)

            w = (
                ranker.rank_head.weight
                .detach()
                .float()
                .reshape(-1)
            )

            wq = w[:DIM]
            wr = w[DIM:2*DIM]
            wd = w[2*DIM:3*DIM]

            wp = (
                w[3*DIM:4*DIM]
                if has_product else None
            )

            bias = (
                ranker.rank_head.bias.detach().float()
                if ranker.rank_head.bias is not None
                else torch.zeros(1, device=device)
            )

            # Gold decomposition
            gr = (zg * wr).sum(-1)

            gd = (
                torch.abs(zq - zg) * wd
            ).sum(-1)

            if has_product:
                gp = (
                    zq * zg * wp
                ).sum(-1)
            else:
                gp = torch.zeros_like(gr)

            gv = {
                "RESPONSE": gr,
                "ABS_DIFF": gd,
                "RESPONSE_PLUS_ABS": gr + gd,
                "FULL": gr + gd + gp,
            }

            if has_product:
                gv["PRODUCT"] = gp

            for name in terms:
                gold_scores[name] += (
                    gv[name].cpu().numpy() / FOLDS
                )

            # Full response library, chunk by chunk
            for r0 in range(0, NR, R_CHUNK):

                r1 = min(r0 + R_CHUNK, NR)

                rb = torch.from_numpy(
                    np.array(
                        R[r0:r1],
                        dtype=np.float32,
                        copy=True
                    )
                ).to(device)

                zr = ranker.adapt_response(rb)

                response = (zr * wr).sum(-1)

                response_np = response.cpu().numpy()

                neg["RESPONSE"][:, r0:r1] += (
                    response_np[None, :] / FOLDS
                )

                for q0 in range(0, NQ, Q_BATCH):

                    q1 = min(q0 + Q_BATCH, NQ)

                    zqb = zq[q0:q1]

                    diff = torch.abs(
                        zqb[:, None, :] -
                        zr[None, :, :]
                    )

                    abs_score = (
                        diff * wd
                    ).sum(-1)

                    if has_product:
                        product_score = (
                            zqb * wp
                        ) @ zr.T
                    else:
                        product_score = torch.zeros_like(
                            abs_score
                        )

                    abs_np = abs_score.cpu().numpy()

                    product_np = product_score.cpu().numpy()

                    response_block = response_np[None, :]

                    neg["ABS_DIFF"][q0:q1, r0:r1] += (
                        abs_np / FOLDS
                    )

                    neg["RESPONSE_PLUS_ABS"][q0:q1, r0:r1] += (
                        (response_block + abs_np) / FOLDS
                    )

                    neg["FULL"][q0:q1, r0:r1] += (
                        (
                            response_block +
                            abs_np +
                            product_np
                        ) / FOLDS
                    )

                    if has_product:
                        neg["PRODUCT"][q0:q1, r0:r1] += (
                            product_np / FOLDS
                        )

                # Architecture decomposition audit
                if r0 == 0:

                    nq_check = min(4, NQ)
                    nr_check = min(32, len(zr))

                    direct = ranker.score_from_adapted(
                        zq[:nq_check, None, :].expand(
                            -1, nr_check, -1
                        ),
                        zr[:nr_check][None, :, :].expand(
                            nq_check, -1, -1
                        )
                    )

                    q_term = (
                        zq[:nq_check] * wq
                    ).sum(-1)[:, None]

                    r_term = (
                        zr[:nr_check] * wr
                    ).sum(-1)[None, :]

                    d_term = (
                        torch.abs(
                            zq[:nq_check, None, :] -
                            zr[:nr_check][None, :, :]
                        ) * wd
                    ).sum(-1)

                    if has_product:
                        p_term = (
                            zq[:nq_check, None, :] *
                            zr[:nr_check][None, :, :] *
                            wp
                        ).sum(-1)
                    else:
                        p_term = torch.zeros_like(d_term)

                    reconstructed = (
                        q_term + r_term + d_term +
                        p_term + bias
                    )

                    err = (
                        direct - reconstructed
                    ).abs().max().item()

                    max_reconstruction_error = max(
                        max_reconstruction_error, err
                    )

        print(
            f"FOLD_{fold}=PASS",
            flush=True
        )

        del model, ck

        torch.cuda.empty_cache()

    reconstruction_errors[condition] = (
        max_reconstruction_error
    )

    print(
        "DECOMPOSITION_MAX_ABS_ERROR =",
        max_reconstruction_error
    )

    # --------------------------------------------------
    # 5. Exact full-pool ranking
    # --------------------------------------------------

    condition_summary = {}

    for name in terms:

        scores = neg[name]
        gold = gold_scores[name]

        ge = scores >= gold[:, None]

        # Remove duplicate Gold from negative pool
        for i, mi in enumerate(match_sample):
            if mi >= 0:
                ge[i, int(mi)] = False

        nneg = NR - (match_sample >= 0).astype(int)

        ranks = 1 + ge.sum(axis=1)

        pairwise_rates = (
            ge.sum(axis=1) / nneg
        )

        median_margin = []
        p99_margin = []
        z_margins = []

        for i, mi in enumerate(match_sample):

            valid = np.ones(NR, dtype=bool)

            if mi >= 0:
                valid[int(mi)] = False

            row = scores[i, valid]

            med = np.median(row)
            p99 = np.quantile(row, 0.99)

            mean = np.mean(row)
            std = np.std(row)

            median_margin.append(gold[i] - med)
            p99_margin.append(gold[i] - p99)

            z_margins.append(
                (gold[i] - mean) / max(std, 1e-12)
            )

        result = {
            "queries": int(NQ),
            "negative_pool": int(NR),
            "median_rank": float(np.median(ranks)),
            "mean_rank": float(np.mean(ranks)),
            "MRR": float(np.mean(1.0 / ranks)),
            "R1": float(np.mean(ranks <= 1)),
            "R5": float(np.mean(ranks <= 5)),
            "R10": float(np.mean(ranks <= 10)),
            "R20": float(np.mean(ranks <= 20)),
            "pairwise_ge_gold_micro": float(
                ge.sum() / nneg.sum()
            ),
            "pairwise_ge_gold_macro": float(
                np.mean(pairwise_rates)
            ),
            "gold_minus_candidate_median": float(
                np.median(median_margin)
            ),
            "gold_minus_candidate_p99": float(
                np.median(p99_margin)
            ),
            "gold_zscore_median": float(
                np.median(z_margins)
            ),
        }

        # Matched/unmatched stratification
        for group, mask in {
            "MATCHED": match_sample >= 0,
            "UNMATCHED": match_sample < 0,
        }.items():

            if mask.any():

                result[group] = {
                    "n": int(mask.sum()),
                    "median_rank": float(
                        np.median(ranks[mask])
                    ),
                    "MRR": float(
                        np.mean(1.0 / ranks[mask])
                    ),
                    "pairwise_ge_gold": float(
                        np.mean(pairwise_rates[mask])
                    )
                }

        condition_summary[name] = result

        for i in range(NQ):

            valid = np.ones(NR, dtype=bool)

            mi = int(match_sample[i])

            if mi >= 0:
                valid[mi] = False

            row = scores[i, valid]

            query_records.append({
                "condition": condition,
                "component": name,
                "sample_id": str(pred["sample_ids"][qidx[i]]),
                "matched": int(mi >= 0),
                "rank": int(ranks[i]),
                "rank_percentile": float(
                    (ranks[i] - 1) / nneg[i]
                ),
                "pairwise_ge_gold": float(
                    pairwise_rates[i]
                ),
                "gold_score": float(gold[i]),
                "negative_median": float(np.median(row)),
                "gold_minus_median": float(
                    gold[i] - np.median(row)
                ),
                "gold_minus_p99": float(
                    gold[i] - np.quantile(row, 0.99)
                )
            })

        print(
            name,
            "MEDIAN_RANK=", result["median_rank"],
            "PAIRWISE_GE=", result["pairwise_ge_gold_micro"],
            "MRR=", result["MRR"],
            flush=True
        )

        if name == "FULL":

            official = pred[
                "rank_" + condition
            ][qidx]

            mismatch = np.abs(
                ranks.astype(np.int64) -
                official.astype(np.int64)
            )

            result["official_rank_exact_match"] = bool(
                np.array_equal(ranks, official)
            )

            result["official_rank_mismatch_count"] = int(
                np.sum(mismatch != 0)
            )

            result["official_rank_max_abs_diff"] = int(
                mismatch.max()
            )

            print(
                "OFFICIAL_RANK_EXACT_MATCH =",
                result["official_rank_exact_match"]
            )

    summary[condition] = condition_summary

print("\nRECONSTRUCTION AUDIT")
for condition, error in reconstruction_errors.items():
    print(condition, error)

# --------------------------------------------------
# 6. Save diagnostic outputs
# --------------------------------------------------

result = {
    "artifact": "S12C Full Pool Component Diagnostic V1",
    "status": "COMPLETED",
    "scope": "EXPLORATORY_EXTERNAL_TEST_DIAGNOSTIC",
    "queries": 64,
    "matched": int((match_sample >= 0).sum()),
    "unmatched": int((match_sample < 0).sum()),
    "response_pool": NR,
    "checkpoint_folds": 5,
    "checkpoint_selection": "FROZEN_EXISTING_BEST_V1",
    "score_aggregation": "MEAN_OF_FIVE_RAW_SCORES",
    "components": summary,
    "decomposition_max_abs_error": reconstruction_errors,
    "training": False,
    "checkpoint_modification": False,
    "test_time_fitting": False,
}

(OUT / "summary.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

with (OUT / "per_query.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(query_records[0].keys())
    )

    writer.writeheader()
    writer.writerows(query_records)

print("\nSTEP_2F16AI_COMPLETE")
print("OUTPUT =", OUT)
print("TRAINING = NO")
print("CHECKPOINT_MODIFICATION = NO")
print("EXTERNAL_PREDICTIONS_MODIFIED = NO")
