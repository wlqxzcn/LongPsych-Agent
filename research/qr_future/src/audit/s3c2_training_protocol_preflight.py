import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s3c1_fsar_models import (
    RankOnlyModel,
    RFutureModel,
    QRFutureModel,
    compute_listwise_rank_loss,
    compute_masked_future_loss
)


Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

MASTER_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3a_ranking_future_master_index_v1.jsonl"
)

FOLD_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3b_internal_fold_assignment_v1.npy"
)

MASK_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3c0_future_loss_eligibility_by_fold_v1.npz"
)

RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/"
      "fsar_train_dense_top100_reservoir_v1.npz"
)

OUT = (
    ROOT
    / "results/phase_s3/"
      "s3c2_training_protocol_preflight_v1.json"
)

BATCH_SIZE = 128
FOLD = 0
MODEL_SEED = 42
LR = 1e-4


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print("=" * 80)
print("FSAR PHASE S3-C2 TRAINING PROTOCOL PREFLIGHT")
print("=" * 80)

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

device = torch.device("cuda")

print()
print("===== 1. LOAD FROZEN TRAIN ARTIFACTS =====")

Q = np.load(Q_FILE, mmap_mode="r", allow_pickle=False)
R = np.load(R_FILE, mmap_mode="r", allow_pickle=False)

master = load_jsonl(MASTER_FILE)

fold_id = np.load(
    FOLD_FILE,
    allow_pickle=False
)

mask_data = np.load(
    MASK_FILE,
    allow_pickle=False
)

future_eligible = np.asarray(
    mask_data["future_loss_eligible_by_fold"],
    dtype=np.bool_
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

assert Q.shape == (62889, 1024)
assert R.shape == (61502, 1024)
assert len(master) == 62889
assert fold_id.shape == (62889,)
assert future_eligible.shape == (5, 62889)
assert negative_rows.shape == (62889, 100)

print("FROZEN_TRAIN_ARTIFACTS = PASS")


print()
print("===== 2. BUILD ACTUAL TRAIN BATCH =====")

train_rows = np.where(
    fold_id != FOLD
)[0]

batch_rows = train_rows[:BATCH_SIZE]

assert len(batch_rows) == BATCH_SIZE

query_np = np.array(
    Q[batch_rows],
    dtype=np.float32,
    copy=True
)

gold_idx = gold_rows[batch_rows]

negative_idx = negative_rows[batch_rows]

candidate_idx = np.concatenate(
    [
        gold_idx[:, None],
        negative_idx
    ],
    axis=1
)

assert candidate_idx.shape == (BATCH_SIZE, 101)

for i in range(BATCH_SIZE):
    assert candidate_idx[i, 0] == gold_idx[i]
    assert gold_idx[i] not in candidate_idx[i, 1:]
    assert len(np.unique(candidate_idx[i])) == 101

candidate_np = np.array(
    R[candidate_idx],
    dtype=np.float32,
    copy=True
)

q = torch.from_numpy(
    query_np
).to(device)

candidates = torch.from_numpy(
    candidate_np
).to(device)

gold = candidates[:, 0, :]

value_label_np = np.full(
    BATCH_SIZE,
    -2,
    dtype=np.int64
)

future_available_np = np.zeros(
    BATCH_SIZE,
    dtype=np.bool_
)

for j, row_idx in enumerate(batch_rows):
    row = master[int(row_idx)]

    if row["future_value_available"]:
        value_label_np[j] = int(
            row["future_value_label"]
        )

        future_available_np[j] = True


eligibility_np = future_eligible[
    FOLD,
    batch_rows
]

assert np.all(
    ~eligibility_np
    |
    future_available_np
)

value_labels = torch.from_numpy(
    value_label_np
).to(device)

eligibility = torch.from_numpy(
    eligibility_np
).to(device)

print("batch_size =", BATCH_SIZE)
print("candidates_per_query =", candidate_idx.shape[1])
print("future_available_in_batch =", int(future_available_np.sum()))
print("future_eligible_in_batch =", int(eligibility_np.sum()))
print("ACTUAL_BATCH_CONSTRUCTION = PASS")


print()
print("===== 3. IDENTICAL MODEL INITIALIZATION =====")

models = {}

for name, cls in [
    ("RANK_ONLY", RankOnlyModel),
    ("R_FUTURE", RFutureModel),
    ("QR_FUTURE", QRFutureModel)
]:
    seed_all(MODEL_SEED)
    models[name] = cls().to(device)


ranker_states = {
    name: model.ranker.state_dict()
    for name, model in models.items()
}

for key in ranker_states["RANK_ONLY"]:

    a = ranker_states["RANK_ONLY"][key]
    b = ranker_states["R_FUTURE"][key]
    c = ranker_states["QR_FUTURE"][key]

    assert torch.equal(a, b)
    assert torch.equal(a, c)


r_future_state = models["R_FUTURE"].future_head.state_dict()
qr_future_state = models["QR_FUTURE"].future_head.state_dict()

for key in r_future_state:
    assert torch.equal(
        r_future_state[key],
        qr_future_state[key]
    )

print("COMMON_RANKER_INITIALIZATION = IDENTICAL")
print("R_QR_FUTURE_HEAD_INITIALIZATION = IDENTICAL")
print("INITIALIZATION_FAIRNESS = PASS")


print()
print("===== 4. OPTIMIZER AUDIT =====")

optimizers = {}

for name, model in models.items():

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    optimizers[name] = optimizer

    pg = optimizer.param_groups[0]

    assert pg["lr"] == LR
    assert pg["betas"] == (0.9, 0.999)
    assert pg["eps"] == 1e-8
    assert pg["weight_decay"] == 0.0


print("optimizer = Adam")
print("learning_rate =", LR)
print("betas =", (0.9, 0.999))
print("eps =", 1e-8)
print("weight_decay =", 0.0)
print("OPTIMIZER_AUDIT = PASS")


print()
print("===== 5. FORWARD/BACKWARD MEMORY PREFLIGHT =====")

results = {}

for name, model in models.items():

    model.train()
    model.zero_grad(set_to_none=True)

    torch.cuda.reset_peak_memory_stats()

    scores = model.rank_scores(
        q,
        candidates
    )

    rank_loss = compute_listwise_rank_loss(
        scores,
        gold_position=0
    )

    if name == "RANK_ONLY":

        future_loss = torch.tensor(
            0.0,
            device=device
        )

        total_loss = rank_loss

    else:

        logits = model.future_logits(
            q,
            gold
        )

        future_loss = compute_masked_future_loss(
            logits,
            value_labels,
            eligibility
        )

        total_loss = (
            rank_loss
            +
            future_loss
        )

    total_loss.backward()

    peak_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    assert torch.isfinite(rank_loss)
    assert torch.isfinite(total_loss)

    results[name] = {
        "rank_loss":
            float(rank_loss.item()),

        "future_loss":
            float(future_loss.item()),

        "total_loss":
            float(total_loss.item()),

        "peak_gpu_memory_mb":
            float(peak_mb)
    }

    print(
        name,
        results[name]
    )

    # Explicitly no optimizer.step()
    model.zero_grad(set_to_none=True)


print("OPTIMIZER_STEP_EXECUTED = NO")
print("FORWARD_BACKWARD_PREFLIGHT = PASS")


print()
print("===== 6. INITIAL RANKING EQUIVALENCE =====")

with torch.no_grad():

    dense_scores = torch.einsum(
        "bd,bcd->bc",
        q,
        candidates
    )

    for name, model in models.items():

        scores = model.rank_scores(
            q,
            candidates
        )

        max_diff = float(
            torch.max(
                torch.abs(
                    scores - dense_scores
                )
            ).item()
        )

        print(
            name,
            "max_abs_diff_vs_dense =",
            max_diff
        )

        assert max_diff < 1e-5


print("INITIAL_RANKING_EQUIVALENCE = PASS")


report = {
    "artifact":
        "FSAR S3-C2 Training Protocol Preflight V1",

    "status":
        "PASS",

    "fold_probe":
        FOLD,

    "batch_size":
        BATCH_SIZE,

    "candidates_per_query":
        101,

    "optimizer":
        "Adam",

    "learning_rate":
        LR,

    "max_epochs":
        20,

    "checkpoint_metric":
        "validation_ranking_cross_entropy",

    "model_seed":
        MODEL_SEED,

    "initialization_fairness":
        True,

    "forward_backward":
        results,

    "optimizer_step_executed":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "formal_training_started":
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
print("FORMAL_PARAMETER_UPDATE = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S3C2_TRAINING_PROTOCOL_PREFLIGHT = PASS")
print("REPORT =", OUT)
