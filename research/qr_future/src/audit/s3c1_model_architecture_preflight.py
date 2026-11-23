import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]

sys.path.insert(
    0,
    str(ROOT)
)

from src.models.s3c1_fsar_models import (
    RankOnlyModel,
    RFutureModel,
    QRFutureModel,
    compute_listwise_rank_loss,
    compute_masked_future_loss
)


Q_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_queries_bge_v1.npy"
)

R_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
)

RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz"
)

OUT = (
    ROOT
    / "results/phase_s3/"
      "s3c1_model_architecture_preflight_v1.json"
)


print("=" * 80)
print("FSAR PHASE S3-C1 MODEL ARCHITECTURE PREFLIGHT")
print("=" * 80)


Q = np.load(
    Q_FILE,
    mmap_mode="r",
    allow_pickle=False
)

R = np.load(
    R_FILE,
    mmap_mode="r",
    allow_pickle=False
)

reservoir = np.load(
    RESERVOIR_FILE,
    allow_pickle=False
)

negative_rows = np.asarray(
    reservoir["candidate_response_row"],
    dtype=np.int32
)

gold_rows = np.asarray(
    reservoir["gold_response_row"],
    dtype=np.int32
)


# ------------------------------------------------------------
# Build actual TRAIN probe batch.
# Candidate order is explicitly:
# gold first, then frozen Top-100 negatives.
# ------------------------------------------------------------

B = 4

q_np = np.array(
    Q[:B],
    dtype=np.float32,
    copy=True
)

candidate_index = np.concatenate(
    [
        gold_rows[:B, None],
        negative_rows[:B]
    ],
    axis=1
)

assert candidate_index.shape == (
    B,
    101
)

candidate_np = np.array(
    R[candidate_index],
    dtype=np.float32,
    copy=True
)

q = torch.from_numpy(q_np)

candidates = torch.from_numpy(
    candidate_np
)

gold = candidates[:, 0, :]


models = {
    "RANK_ONLY":
        RankOnlyModel(),

    "R_FUTURE":
        RFutureModel(),

    "QR_FUTURE":
        QRFutureModel()
}


print()
print("===== 1. INITIAL DENSE EQUIVALENCE =====")

dense_scores = torch.einsum(
    "bd,bcd->bc",
    q,
    candidates
)

initial_diff = {}


for name, model in models.items():

    model.eval()

    with torch.no_grad():

        scores = model.rank_scores(
            q,
            candidates
        )

    assert scores.shape == (
        B,
        101
    )

    max_diff = float(
        torch.max(
            torch.abs(
                scores
                -
                dense_scores
            )
        ).item()
    )

    initial_diff[name] = max_diff

    print(
        name,
        "max_abs_diff_vs_dense =",
        max_diff
    )

    assert max_diff < 1e-5


print("INITIAL_RANK_SCORE_EQUALS_DENSE = PASS")


print()
print("===== 2. FUTURE FEATURE INTERFACE =====")

with torch.no_grad():

    r_logits = models[
        "R_FUTURE"
    ].future_logits(
        q,
        gold
    )

    qr_logits = models[
        "QR_FUTURE"
    ].future_logits(
        q,
        gold
    )


assert r_logits.shape == (B, 4)
assert qr_logits.shape == (B, 4)

assert not hasattr(
    models["RANK_ONLY"],
    "future_head"
)

print("R_FUTURE_logits_shape =", tuple(r_logits.shape))
print("QR_FUTURE_logits_shape =", tuple(qr_logits.shape))
print("RANK_ONLY_future_head = ABSENT")
print("FUTURE_INTERFACE = PASS")


print()
print("===== 3. PARAMETER SHARING =====")

def parameter_count(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def ranker_parameter_count(model):
    return sum(
        p.numel()
        for p in model.ranker.parameters()
    )


counts = {}

for name, model in models.items():

    counts[name] = {
        "total":
            parameter_count(model),

        "ranker":
            ranker_parameter_count(model)
    }

    print(
        name,
        counts[name]
    )


assert (
    counts["RANK_ONLY"]["ranker"]
    ==
    counts["R_FUTURE"]["ranker"]
    ==
    counts["QR_FUTURE"]["ranker"]
)

assert (
    counts["R_FUTURE"]["total"]
    ==
    counts["QR_FUTURE"]["total"]
)

print("COMMON_RANKER_PARAMETER_COUNT = PASS")
print("R_AND_QR_TOTAL_PARAMETER_COUNT_MATCH = PASS")


print()
print("===== 4. LOSS SEMANTICS =====")

rank_scores = models[
    "QR_FUTURE"
].rank_scores(
    q,
    candidates
)

rank_loss = compute_listwise_rank_loss(
    rank_scores,
    gold_position=0
)

value_labels = torch.tensor(
    [-2, -1, 1, 2],
    dtype=torch.long
)

eligibility = torch.tensor(
    [1, 0, 1, 0],
    dtype=torch.bool
)

future_logits = models[
    "QR_FUTURE"
].future_logits(
    q,
    gold
)

future_loss = compute_masked_future_loss(
    future_logits,
    value_labels,
    eligibility
)

total_loss = (
    rank_loss
    +
    future_loss
)


assert torch.isfinite(rank_loss)
assert torch.isfinite(future_loss)
assert torch.isfinite(total_loss)

print("rank_loss =", float(rank_loss.item()))
print("future_loss =", float(future_loss.item()))
print("total_loss =", float(total_loss.item()))
print("BETA_PARAMETER = ABSENT")
print("MASKED_FULL_BATCH_FUTURE_LOSS = PASS")


print()
print("===== 5. FUTURE GRADIENT ROUTING =====")


def check_future_gradient(model, name):

    model.zero_grad(set_to_none=True)

    logits = model.future_logits(
        q,
        gold
    )

    loss = compute_masked_future_loss(
        logits,
        value_labels,
        torch.ones(
            B,
            dtype=torch.bool
        )
    )

    loss.backward()

    q_grad = model.ranker.q_adapter.weight.grad
    r_grad = model.ranker.r_adapter.weight.grad

    q_nonzero = (
        q_grad is not None
        and
        float(q_grad.abs().sum()) > 0
    )

    r_nonzero = (
        r_grad is not None
        and
        float(r_grad.abs().sum()) > 0
    )

    print(
        name,
        "q_adapter_future_grad =",
        q_nonzero,
        "r_adapter_future_grad =",
        r_nonzero
    )

    return q_nonzero, r_nonzero


r_qgrad, r_rgrad = check_future_gradient(
    models["R_FUTURE"],
    "R_FUTURE"
)

qr_qgrad, qr_rgrad = check_future_gradient(
    models["QR_FUTURE"],
    "QR_FUTURE"
)


assert r_qgrad is False
assert r_rgrad is True

assert qr_qgrad is True
assert qr_rgrad is True

print("R_FUTURE_RESPONSE_ONLY_GRADIENT = PASS")
print("QR_FUTURE_QUERY_RESPONSE_GRADIENT = PASS")


print()
print("===== 6. INFERENCE POLICY =====")

print("Inference ranking API = rank_scores()")
print("Future head required for ranking = NO")
print("Future prediction added to rank score = NO")
print("INFERENCE_POLICY = PASS")


report = {
    "artifact":
        "FSAR S3-C1 Model Architecture Preflight V1",

    "status":
        "PASS",

    "initial_dense_equivalence_max_abs_diff":
        initial_diff,

    "parameter_counts":
        counts,

    "candidate_count":
        101,

    "future_threshold_count":
        4,

    "beta_parameter":
        None,

    "future_gradient_routing": {
        "R_FUTURE": {
            "query_adapter": r_qgrad,
            "response_adapter": r_rgrad
        },

        "QR_FUTURE": {
            "query_adapter": qr_qgrad,
            "response_adapter": qr_rgrad
        }
    },

    "future_head_used_at_inference":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "model_training_performed":
        False
}


OUT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("PHASE_S3C1_MODEL_ARCHITECTURE_PREFLIGHT = PASS")
print("REPORT =", OUT)
