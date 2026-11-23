import json
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s3c1_fsar_models import (
    RankOnlyModel,
    compute_listwise_rank_loss
)


# ============================================================
# Paths
# ============================================================

Q_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_queries_bge_v1.npy"
)

R_FILE = (
    ROOT
    / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
)

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

RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/"
      "fsar_train_dense_top100_reservoir_v1.npz"
)

MODEL_CODE = (
    ROOT
    / "src/models/s3c1_fsar_models.py"
)

ARCH_PROTOCOL = (
    ROOT
    / "protocols/fsar_s3c1_model_architecture_and_loss_v1.json"
)

TRAIN_PROTOCOL = (
    ROOT
    / "protocols/fsar_s3c2_training_and_checkpoint_protocol_v1.json"
)

S4_PROTOCOL = (
    ROOT
    / "protocols/fsar_s4_rank_only_training_v1.json"
)


RESULT_DIR = ROOT / "results/phase_s4"

CHECKPOINT_DIR = (
    RESULT_DIR
    / "checkpoints/rank_only"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CURVE_OUT = (
    RESULT_DIR
    / "s4_rank_only_training_curve_v1.jsonl"
)

OOF_OUT = (
    RESULT_DIR
    / "s4_rank_only_oof_scores_v1.npz"
)

SUMMARY_OUT = (
    RESULT_DIR
    / "s4_rank_only_training_summary_v1.json"
)

CURVE_TMP = Path(str(CURVE_OUT) + ".tmp")

OOF_TMP = (
    RESULT_DIR
    / "s4_rank_only_oof_scores_v1.tmp.npz"
)

SUMMARY_TMP = Path(str(SUMMARY_OUT) + ".tmp")


# ============================================================
# Frozen constants
# ============================================================

N = 62889
NR = 61502
DIM = 1024
DEPTH = 100
CANDIDATES = 101
N_FOLDS = 5

BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_SEED = 42
LR = 1e-4


EXPECTED_SHA = {
    "query_embedding":
        "18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005",

    "response_embedding":
        "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f",

    "master":
        "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",

    "fold":
        "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",

    "reservoir":
        "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",

    "model_code":
        "7dbd9dcdb1fb382ff031e61eb51058016367be24fe89ab8c95bc9b5fe1d8d755",

    "architecture_protocol":
        "8938e309b0131b2f9961c116c895415c3be227ebba7e3df65f653ee674accf2c",

    "training_protocol":
        "e30ca60f61d12ac0b291a473a656eed326b24cee1d324f06c21623bfc4d0d51e"
}


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
    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_tmp_path(fold):
    return (
        CHECKPOINT_DIR
        / f"rank_only_fold{fold}_best_v1.tmp.pt"
    )


def checkpoint_final_path(fold):
    return (
        CHECKPOINT_DIR
        / f"rank_only_fold{fold}_best_v1.pt"
    )


print("=" * 82)
print("FSAR PHASE S4 RANK-ONLY 5-FOLD FORMAL TRAINING")
print("=" * 82)


# ============================================================
# 1. Output protection
# ============================================================

print()
print("===== 1. OUTPUT PROTECTION =====")

formal_outputs = [
    CURVE_OUT,
    OOF_OUT,
    SUMMARY_OUT
]

formal_outputs += [
    checkpoint_final_path(f)
    for f in range(N_FOLDS)
]

existing = [
    str(p)
    for p in formal_outputs
    if p.exists()
]

if existing:
    raise RuntimeError(
        "Formal S4 outputs already exist; refusing overwrite:\n"
        + "\n".join(existing)
    )

for p in [
    CURVE_TMP,
    OOF_TMP,
    SUMMARY_TMP
]:
    if p.exists():
        p.unlink()

for f in range(N_FOLDS):
    p = checkpoint_tmp_path(f)

    if p.exists():
        p.unlink()

print("FORMAL_OUTPUT_COLLISION = 0")
print("OUTPUT_PROTECTION = PASS")


# ============================================================
# 2. Frozen provenance
# ============================================================

print()
print("===== 2. FROZEN INPUT SHA =====")

input_paths = {
    "query_embedding": Q_FILE,
    "response_embedding": R_FILE,
    "master": MASTER_FILE,
    "fold": FOLD_FILE,
    "reservoir": RESERVOIR_FILE,
    "model_code": MODEL_CODE,
    "architecture_protocol": ARCH_PROTOCOL,
    "training_protocol": TRAIN_PROTOCOL
}

actual_sha = {}

for key, path in input_paths.items():

    actual_sha[key] = sha256_file(path)

    print(
        key,
        "=",
        actual_sha[key]
    )

    assert actual_sha[key] == EXPECTED_SHA[key]


s4_protocol_sha = sha256_file(
    S4_PROTOCOL
)

print(
    "s4_protocol =",
    s4_protocol_sha
)

print("FROZEN_INPUT_SHA_GATE = PASS")


# ============================================================
# 3. Load formal TRAIN artifacts
# ============================================================

print()
print("===== 3. TRAIN STRUCTURE =====")

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

master = load_jsonl(
    MASTER_FILE
)

fold_id = np.load(
    FOLD_FILE,
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


assert Q.shape == (N, DIM)
assert R.shape == (NR, DIM)

assert len(master) == N

assert fold_id.shape == (N,)

assert negative_rows.shape == (
    N,
    DEPTH
)

assert gold_rows.shape == (N,)

assert set(
    fold_id.tolist()
) == {0, 1, 2, 3, 4}

assert np.all(
    negative_rows
    != gold_rows[:, None]
)


print("query_shape =", Q.shape)
print("response_shape =", R.shape)
print("ranking_rows =", N)
print("reservoir_shape =", negative_rows.shape)
print("fold_shape =", fold_id.shape)
print("gold_in_negative_reservoir = 0")
print("TRAIN_STRUCTURE = PASS")


# ============================================================
# 4. CUDA numerical policy
# ============================================================

print()
print("===== 4. CUDA NUMERICS =====")

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision(
    "highest"
)

device = torch.device("cuda")

print(
    "gpu =",
    torch.cuda.get_device_name(0)
)

print(
    "allow_tf32 =",
    torch.backends.cuda.matmul.allow_tf32
)

print(
    "float32_matmul_precision =",
    torch.get_float32_matmul_precision()
)

print("CUDA_NUMERICS = PASS")


# ============================================================
# 5. Batch constructor
# ============================================================

def build_batch(rows):

    rows = np.asarray(
        rows,
        dtype=np.int64
    )

    q_np = np.array(
        Q[rows],
        dtype=np.float32,
        copy=True
    )

    gold_idx = gold_rows[rows]

    neg_idx = negative_rows[rows]

    candidate_idx = np.concatenate(
        [
            gold_idx[:, None],
            neg_idx
        ],
        axis=1
    )

    assert candidate_idx.shape == (
        len(rows),
        CANDIDATES
    )

    candidate_np = np.array(
        R[candidate_idx],
        dtype=np.float32,
        copy=True
    )

    q = torch.from_numpy(
        q_np
    ).to(device)

    candidates = torch.from_numpy(
        candidate_np
    ).to(device)

    return q, candidates


# ============================================================
# 6. Validation CE
# ============================================================

def evaluate_validation_ce(
    model,
    rows
):

    model.eval()

    loss_sum = 0.0
    count = 0

    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            BATCH_SIZE
        ):

            batch_rows = rows[
                start:
                start + BATCH_SIZE
            ]

            q, candidates = build_batch(
                batch_rows
            )

            scores = model.rank_scores(
                q,
                candidates
            )

            loss = (
                compute_listwise_rank_loss(
                    scores,
                    gold_position=0
                )
            )

            b = len(batch_rows)

            loss_sum += (
                float(loss.item())
                * b
            )

            count += b


    assert count == len(rows)

    return loss_sum / count


# ============================================================
# 7. OOF scorer
# ============================================================

def score_validation_rows(
    model,
    rows,
    oof_scores,
    oof_nll
):

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            BATCH_SIZE
        ):

            batch_rows = rows[
                start:
                start + BATCH_SIZE
            ]

            q, candidates = build_batch(
                batch_rows
            )

            scores = model.rank_scores(
                q,
                candidates
            )

            nll = (
                -scores[:, 0]
                +
                torch.logsumexp(
                    scores,
                    dim=1
                )
            )

            scores_np = (
                scores
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            nll_np = (
                nll
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            oof_scores[
                batch_rows
            ] = scores_np

            oof_nll[
                batch_rows
            ] = nll_np


# ============================================================
# 8. Formal 5-fold training
# ============================================================

print()
print("===== 5. FORMAL 5-FOLD TRAINING =====")
print("FORMAL_PARAMETER_UPDATE = YES_FROM_THIS_POINT")


oof_scores = np.full(
    (N, CANDIDATES),
    np.nan,
    dtype=np.float32
)

oof_nll = np.full(
    N,
    np.nan,
    dtype=np.float32
)


fold_summaries = {}

curve_handle = CURVE_TMP.open(
    "w",
    encoding="utf-8"
)


for fold in range(N_FOLDS):

    print()
    print("=" * 72)
    print(f"FOLD {fold}")
    print("=" * 72)

    train_rows = np.where(
        fold_id != fold
    )[0]

    valid_rows = np.where(
        fold_id == fold
    )[0]


    train_dialogues = {
        str(master[i]["dialogue_id"])
        for i in train_rows
    }

    valid_dialogues = {
        str(master[i]["dialogue_id"])
        for i in valid_rows
    }

    assert not (
        train_dialogues
        &
        valid_dialogues
    )


    print(
        "training_rows =",
        len(train_rows)
    )

    print(
        "validation_rows =",
        len(valid_rows)
    )

    print(
        "training_dialogues =",
        len(train_dialogues)
    )

    print(
        "validation_dialogues =",
        len(valid_dialogues)
    )


    # --------------------------------------------------------
    # Identical frozen initialization
    # --------------------------------------------------------

    seed_all(MODEL_SEED)

    model = RankOnlyModel().to(
        device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )


    # Initial check only; epoch zero cannot be selected.
    probe_rows = valid_rows[
        :min(
            128,
            len(valid_rows)
        )
    ]

    with torch.no_grad():

        q_probe, c_probe = build_batch(
            probe_rows
        )

        initial_scores = model.rank_scores(
            q_probe,
            c_probe
        )

        dense_scores = torch.einsum(
            "bd,bcd->bc",
            q_probe,
            c_probe
        )

        initial_diff = float(
            torch.max(
                torch.abs(
                    initial_scores
                    -
                    dense_scores
                )
            ).item()
        )

    assert initial_diff < 1e-5

    print(
        "initial_max_abs_diff_vs_dense =",
        initial_diff
    )


    best_epoch = None
    best_val_ce = None
    best_train_ce = None

    fold_start = time.time()


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        epoch_seed = (
            42000
            +
            fold * 100
            +
            epoch
        )

        rng = np.random.default_rng(
            epoch_seed
        )

        shuffled_train = rng.permutation(
            train_rows
        )


        model.train()

        train_loss_sum = 0.0
        train_count = 0

        epoch_start = time.time()


        for start in range(
            0,
            len(shuffled_train),
            BATCH_SIZE
        ):

            batch_rows = shuffled_train[
                start:
                start + BATCH_SIZE
            ]

            q, candidates = build_batch(
                batch_rows
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            scores = model.rank_scores(
                q,
                candidates
            )


            rank_loss = (
                compute_listwise_rank_loss(
                    scores,
                    gold_position=0
                )
            )


            assert torch.isfinite(
                rank_loss
            )


            rank_loss.backward()

            optimizer.step()


            b = len(batch_rows)

            train_loss_sum += (
                float(rank_loss.item())
                * b
            )

            train_count += b


        assert train_count == len(
            train_rows
        )

        train_ce = (
            train_loss_sum
            /
            train_count
        )


        val_ce = evaluate_validation_ce(
            model,
            valid_rows
        )


        epoch_seconds = (
            time.time()
            -
            epoch_start
        )


        record = {
            "condition":
                "RANK_ONLY",

            "fold":
                fold,

            "epoch":
                epoch,

            "epoch_shuffle_seed":
                epoch_seed,

            "training_rows":
                int(
                    len(train_rows)
                ),

            "validation_rows":
                int(
                    len(valid_rows)
                ),

            "train_ranking_ce":
                float(train_ce),

            "validation_ranking_ce":
                float(val_ce),

            "epoch_seconds":
                float(epoch_seconds)
        }


        curve_handle.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )

        curve_handle.flush()


        improved = (
            best_val_ce is None
            or
            val_ce < best_val_ce
        )


        if improved:

            best_val_ce = float(
                val_ce
            )

            best_train_ce = float(
                train_ce
            )

            best_epoch = int(
                epoch
            )


            state_cpu = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict().items()
            }


            checkpoint = {
                "artifact":
                    "FSAR S4 Rank-Only Best Checkpoint V1",

                "condition":
                    "RANK_ONLY",

                "fold":
                    fold,

                "best_epoch":
                    best_epoch,

                "best_validation_ranking_ce":
                    best_val_ce,

                "training_ranking_ce_at_best_epoch":
                    best_train_ce,

                "model_seed":
                    MODEL_SEED,

                "model_state_dict":
                    state_cpu,

                "dev_payload_used":
                    False,

                "test_payload_used":
                    False
            }


            torch.save(
                checkpoint,
                checkpoint_tmp_path(
                    fold
                )
            )


        print(
            f"fold={fold} "
            f"epoch={epoch:02d}/20 "
            f"train_ce={train_ce:.8f} "
            f"val_ce={val_ce:.8f} "
            f"best_epoch={best_epoch} "
            f"best_val_ce={best_val_ce:.8f} "
            f"seconds={epoch_seconds:.2f}"
        )


    assert best_epoch is not None

    assert checkpoint_tmp_path(
        fold
    ).exists()


    # --------------------------------------------------------
    # Reload selected checkpoint
    # --------------------------------------------------------

    best_checkpoint = torch.load(
        checkpoint_tmp_path(fold),
        map_location=device,
        weights_only=False
    )

    assert (
        best_checkpoint["fold"]
        == fold
    )

    assert (
        best_checkpoint["best_epoch"]
        == best_epoch
    )


    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )


    reloaded_val_ce = (
        evaluate_validation_ce(
            model,
            valid_rows
        )
    )


    assert abs(
        reloaded_val_ce
        -
        best_val_ce
    ) < 1e-6


    # Formal OOF scores from selected checkpoint.
    score_validation_rows(
        model,
        valid_rows,
        oof_scores,
        oof_nll
    )


    fold_seconds = (
        time.time()
        -
        fold_start
    )


    fold_summary = {
        "fold":
            fold,

        "training_rows":
            int(
                len(train_rows)
            ),

        "validation_rows":
            int(
                len(valid_rows)
            ),

        "training_dialogues":
            len(train_dialogues),

        "validation_dialogues":
            len(valid_dialogues),

        "initial_max_abs_diff_vs_dense":
            initial_diff,

        "best_epoch":
            best_epoch,

        "best_validation_ranking_ce":
            best_val_ce,

        "training_ranking_ce_at_best_epoch":
            best_train_ce,

        "reloaded_validation_ranking_ce":
            float(
                reloaded_val_ce
            ),

        "fold_seconds":
            float(
                fold_seconds
            )
    }


    fold_summaries[
        str(fold)
    ] = fold_summary


    print()
    print(
        "FOLD_SELECTED_CHECKPOINT =",
        best_epoch
    )

    print(
        "FOLD_BEST_VALIDATION_RANKING_CE =",
        best_val_ce
    )

    print(
        "CHECKPOINT_RELOAD_VALIDATION_CE =",
        reloaded_val_ce
    )

    print(
        "FOLD_OOF_SCORING = PASS"
    )


    del optimizer
    del model

    torch.cuda.empty_cache()


curve_handle.close()


# ============================================================
# 9. OOF integrity
# ============================================================

print()
print("===== 6. OOF INTEGRITY =====")

assert np.isfinite(
    oof_scores
).all()

assert np.isfinite(
    oof_nll
).all()

assert oof_scores.shape == (
    N,
    CANDIDATES
)

assert oof_nll.shape == (N,)


oof_mean_ce = float(
    oof_nll.mean()
)


weighted_fold_ce = 0.0

for f in range(N_FOLDS):

    n_valid = fold_summaries[
        str(f)
    ]["validation_rows"]

    weighted_fold_ce += (
        fold_summaries[
            str(f)
        ][
            "best_validation_ranking_ce"
        ]
        *
        n_valid
    )

weighted_fold_ce /= N


assert abs(
    oof_mean_ce
    -
    weighted_fold_ce
) < 1e-5


print(
    "oof_scores_shape =",
    oof_scores.shape
)

print(
    "oof_nll_shape =",
    oof_nll.shape
)

print(
    "oof_nonfinite_scores =",
    int(
        np.sum(
            ~np.isfinite(
                oof_scores
            )
        )
    )
)

print(
    "oof_nonfinite_nll =",
    int(
        np.sum(
            ~np.isfinite(
                oof_nll
            )
        )
    )
)

print(
    "oof_mean_ranking_ce =",
    oof_mean_ce
)

print(
    "weighted_fold_validation_ce =",
    weighted_fold_ce
)

print("OOF_INTEGRITY = PASS")


# ============================================================
# 10. Save temporary OOF + summary
# ============================================================

np.savez_compressed(
    OOF_TMP,
    candidate_scores=oof_scores,
    ranking_ce=oof_nll,
    fold_id=fold_id,
    gold_position=np.zeros(
        N,
        dtype=np.int8
    )
)


summary = {
    "artifact":
        "FSAR S4 Rank-Only 5-Fold Formal Training V1",

    "status":
        "PASS",

    "condition":
        "RANK_ONLY",

    "ranking_rows":
        N,

    "folds":
        N_FOLDS,

    "candidate_count":
        CANDIDATES,

    "batch_size":
        BATCH_SIZE,

    "max_epochs":
        MAX_EPOCHS,

    "optimizer":
        "Adam",

    "learning_rate":
        LR,

    "checkpoint_selection":
        "MIN_VALIDATION_RANKING_CROSS_ENTROPY",

    "fold_summary":
        fold_summaries,

    "oof_mean_ranking_ce":
        oof_mean_ce,

    "retrieval_metrics_computed":
        False,

    "future_supervision_used":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256":
        actual_sha,

    "s4_protocol_sha256":
        s4_protocol_sha
}


SUMMARY_TMP.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# 11. Promote checkpoints and formal outputs
# ============================================================

print()
print("===== 7. FORMAL ARTIFACT PROMOTION =====")


for f in range(N_FOLDS):

    os.replace(
        checkpoint_tmp_path(f),
        checkpoint_final_path(f)
    )


os.replace(
    OOF_TMP,
    OOF_OUT
)

os.replace(
    CURVE_TMP,
    CURVE_OUT
)

os.replace(
    SUMMARY_TMP,
    SUMMARY_OUT
)


print("FORMAL_ARTIFACT_PROMOTION = PASS")


# ============================================================
# 12. Final SHA
# ============================================================

print()
print("===== 8. FINAL ARTIFACT SHA256 =====")

artifact_paths = [
    checkpoint_final_path(f)
    for f in range(N_FOLDS)
]

artifact_paths += [
    CURVE_OUT,
    OOF_OUT,
    SUMMARY_OUT
]


artifact_sha = {}

for p in artifact_paths:

    h = sha256_file(p)

    artifact_sha[
        str(
            p.relative_to(ROOT)
        )
    ] = h

    print(
        h,
        p
    )


print()
print("FORMAL_RETRIEVAL_METRICS_COMPUTED = NO")
print("FUTURE_SUPERVISION_USED = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print(
    "PHASE_S4_RANK_ONLY_FORMAL_TRAINING = PASS"
)
