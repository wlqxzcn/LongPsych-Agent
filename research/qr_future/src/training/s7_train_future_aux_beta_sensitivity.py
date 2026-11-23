import json
import hashlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s3c1_fsar_models import (
    RFutureModel,
    QRFutureModel,
    compute_listwise_rank_loss,
    compute_masked_future_loss,
)


# ============================================================
# Frozen paths
# ============================================================

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

MODEL_CODE = ROOT / "src/models/s3c1_fsar_models.py"

ARCH_PROTOCOL = (
    ROOT
    / "protocols/fsar_s3c1_model_architecture_and_loss_v1.json"
)

TRAIN_PROTOCOL = (
    ROOT
    / "protocols/fsar_s3c2_training_and_checkpoint_protocol_v1.json"
)

S5_PROTOCOL = (
    ROOT
    / "protocols/fsar_s5_future_aux_training_v1.json"
)

S7_PROTOCOL = (
    ROOT
    / "protocols/fsar_s7_future_aux_beta_sensitivity_v1.json"
)

S4_SUMMARY = (
    ROOT
    / "results/phase_s4/s4_rank_only_training_summary_v1.json"
)

S4_OOF = (
    ROOT
    / "results/phase_s4/s4_rank_only_oof_scores_v1.npz"
)


BETA_RAW = os.environ.get("FSAR_S7_BETA")
if BETA_RAW is None:
    raise RuntimeError("Set FSAR_S7_BETA to 2, 4, or 8.")
try:
    BETA = float(BETA_RAW)
except ValueError as exc:
    raise RuntimeError("FSAR_S7_BETA must be 2, 4, or 8.") from exc
if BETA not in (2.0, 4.0, 8.0):
    raise RuntimeError("Allowed beta values are 2, 4, and 8.")
BETA_TAG = f"beta_{int(BETA)}"
RESULT_DIR = ROOT / "results/phase_s7" / BETA_TAG
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_ROOT = RESULT_DIR / "checkpoints"


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


CONDITIONS = {
    "R_FUTURE": RFutureModel,
    "QR_FUTURE": QRFutureModel,
}


EXPECTED_SHA = {
    "query_embedding":
        "18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005",

    "response_embedding":
        "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f",

    "master":
        "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",

    "fold":
        "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",

    "future_mask":
        "84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d",

    "reservoir":
        "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",

    "model_code":
        "7dbd9dcdb1fb382ff031e61eb51058016367be24fe89ab8c95bc9b5fe1d8d755",

    "architecture_protocol":
        "8938e309b0131b2f9961c116c895415c3be227ebba7e3df65f653ee674accf2c",

    "training_protocol":
        "e30ca60f61d12ac0b291a473a656eed326b24cee1d324f06c21623bfc4d0d51e",

    "s5_protocol":
        "33dd9bfd795300a73e1450a9278d06ae75809a992f504eaffe96fc28f97e0882",
    "s7_protocol": "6f3924f74a1b3168ea6b75f7f2383ce41b0b2715c236b66ec53f38341edcb23d",

    "s4_summary":
        "be30581e5804c9b7368ed3f23037f79dc78b53659e3ca9abde63a2e4387de6d0",

    "s4_oof":
        "fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4",
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


def condition_slug(condition):
    return condition.lower()


def checkpoint_dir(condition):
    d = CHECKPOINT_ROOT / condition_slug(condition)
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint_tmp(condition, fold):
    return (
        checkpoint_dir(condition)
        / f"{condition_slug(condition)}_fold{fold}_best_v1.tmp.pt"
    )


def checkpoint_final(condition, fold):
    return (
        checkpoint_dir(condition)
        / f"{condition_slug(condition)}_fold{fold}_best_v1.pt"
    )


def curve_tmp(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_training_curve_v1.tmp.jsonl"
    )


def curve_final(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_training_curve_v1.jsonl"
    )


def oof_tmp(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_oof_scores_v1.tmp.npz"
    )


def oof_final(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_oof_scores_v1.npz"
    )


def summary_tmp(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_training_summary_v1.tmp.json"
    )


def summary_final(condition):
    return (
        RESULT_DIR
        / f"s7_beta{int(BETA)}_{condition_slug(condition)}_training_summary_v1.json"
    )


print("=" * 84)
print(f"FSAR PHASE S7 BETA={BETA:g} R-FUTURE / QR-FUTURE 5-FOLD EXPLORATORY TRAINING")
print("=" * 84)


# ============================================================
# 1. Output protection
# ============================================================

print()
print("===== 1. OUTPUT PROTECTION =====")

formal_outputs = []

for condition in CONDITIONS:
    formal_outputs.extend(
        [
            curve_final(condition),
            oof_final(condition),
            summary_final(condition),
        ]
    )

    formal_outputs.extend(
        checkpoint_final(condition, f)
        for f in range(N_FOLDS)
    )


existing = [
    str(p)
    for p in formal_outputs
    if p.exists()
]

if existing:
    raise RuntimeError(
        "Formal S5 outputs already exist; refusing overwrite:\n"
        + "\n".join(existing)
    )


# Old incomplete TEMP artifacts may be deleted safely before
# a fresh formal run because they were never promoted.
for condition in CONDITIONS:

    for p in [
        curve_tmp(condition),
        oof_tmp(condition),
        summary_tmp(condition),
    ]:
        if p.exists():
            p.unlink()

    for fold in range(N_FOLDS):
        p = checkpoint_tmp(condition, fold)
        if p.exists():
            p.unlink()


print("FORMAL_OUTPUT_COLLISION = 0")
print("TEMP_OUTPUTS_RESET = PASS")
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
    "future_mask": MASK_FILE,
    "reservoir": RESERVOIR_FILE,
    "model_code": MODEL_CODE,
    "architecture_protocol": ARCH_PROTOCOL,
    "training_protocol": TRAIN_PROTOCOL,
    "s5_protocol": S5_PROTOCOL,
    "s7_protocol": S7_PROTOCOL,
    "s4_summary": S4_SUMMARY,
    "s4_oof": S4_OOF,
}

actual_sha = {}

for key, path in input_paths.items():

    actual_sha[key] = sha256_file(path)

    print(key, "=", actual_sha[key])

    assert actual_sha[key] == EXPECTED_SHA[key]


print("FROZEN_INPUT_SHA_GATE = PASS")


# ============================================================
# 3. Load structures
# ============================================================

print()
print("===== 3. TRAIN STRUCTURE =====")

Q = np.load(
    Q_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

R = np.load(
    R_FILE,
    mmap_mode="r",
    allow_pickle=False,
)

master = load_jsonl(MASTER_FILE)

fold_id = np.load(
    FOLD_FILE,
    allow_pickle=False,
)

mask_data = np.load(
    MASK_FILE,
    allow_pickle=False,
)

future_eligible = np.asarray(
    mask_data["future_loss_eligible_by_fold"],
    dtype=np.bool_,
)

reservoir = np.load(
    RESERVOIR_FILE,
    allow_pickle=False,
)

negative_rows = np.asarray(
    reservoir["candidate_response_row"],
    dtype=np.int32,
)

gold_rows = np.asarray(
    reservoir["gold_response_row"],
    dtype=np.int32,
)


future_available = np.asarray(
    [
        bool(x["future_value_available"])
        for x in master
    ],
    dtype=np.bool_,
)

future_labels = np.full(
    N,
    -2,
    dtype=np.int64,
)

for i, row in enumerate(master):
    if row["future_value_available"]:
        future_labels[i] = int(
            row["future_value_label"]
        )


assert Q.shape == (N, DIM)
assert R.shape == (NR, DIM)
assert len(master) == N
assert fold_id.shape == (N,)
assert future_eligible.shape == (N_FOLDS, N)
assert negative_rows.shape == (N, DEPTH)
assert gold_rows.shape == (N,)
assert int(future_available.sum()) == 17458

assert np.all(
    negative_rows
    != gold_rows[:, None]
)

print("query_shape =", Q.shape)
print("response_shape =", R.shape)
print("ranking_rows =", N)
print("reservoir_shape =", negative_rows.shape)
print("future_value_rows =", int(future_available.sum()))
print("TRAIN_STRUCTURE = PASS")


# ============================================================
# 4. CUDA policy
# ============================================================

print()
print("===== 4. CUDA NUMERICS =====")

assert torch.cuda.is_available()

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

device = torch.device("cuda")

print("gpu =", torch.cuda.get_device_name(0))
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
# Batch construction
# ============================================================

def build_batch(rows, validation_fold):

    rows = np.asarray(
        rows,
        dtype=np.int64,
    )

    q_np = np.array(
        Q[rows],
        dtype=np.float32,
        copy=True,
    )

    gold_idx = gold_rows[rows]

    neg_idx = negative_rows[rows]

    candidate_idx = np.concatenate(
        [
            gold_idx[:, None],
            neg_idx,
        ],
        axis=1,
    )

    assert candidate_idx.shape == (
        len(rows),
        CANDIDATES,
    )

    candidate_np = np.array(
        R[candidate_idx],
        dtype=np.float32,
        copy=True,
    )

    q = torch.from_numpy(
        q_np
    ).to(device)

    candidates = torch.from_numpy(
        candidate_np
    ).to(device)

    labels = torch.from_numpy(
        future_labels[rows]
    ).to(device)

    eligibility = torch.from_numpy(
        future_eligible[
            validation_fold,
            rows,
        ]
    ).to(device)

    return (
        q,
        candidates,
        labels,
        eligibility,
    )


def evaluate_validation_ce(
    model,
    rows,
    validation_fold,
):

    model.eval()

    loss_sum = 0.0
    count = 0

    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            BATCH_SIZE,
        ):

            batch_rows = rows[
                start:start + BATCH_SIZE
            ]

            q, candidates, _, _ = build_batch(
                batch_rows,
                validation_fold,
            )

            scores = model.rank_scores(
                q,
                candidates,
            )

            loss = compute_listwise_rank_loss(
                scores,
                gold_position=0,
            )

            b = len(batch_rows)

            loss_sum += (
                float(loss.item()) * b
            )

            count += b

    assert count == len(rows)

    return loss_sum / count


def write_oof_rows(
    model,
    rows,
    validation_fold,
    oof_scores,
    oof_nll,
    oof_future_logits,
):

    model.eval()

    with torch.inference_mode():

        for start in range(
            0,
            len(rows),
            BATCH_SIZE,
        ):

            batch_rows = rows[
                start:start + BATCH_SIZE
            ]

            q, candidates, _, _ = build_batch(
                batch_rows,
                validation_fold,
            )

            scores = model.rank_scores(
                q,
                candidates,
            )

            nll = (
                -scores[:, 0]
                + torch.logsumexp(
                    scores,
                    dim=1,
                )
            )

            future_logits = model.future_logits(
                q,
                candidates[:, 0, :],
            )

            oof_scores[
                batch_rows
            ] = (
                scores
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            oof_nll[
                batch_rows
            ] = (
                nll
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            oof_future_logits[
                batch_rows
            ] = (
                future_logits
                .cpu()
                .numpy()
                .astype(np.float32)
            )


# ============================================================
# 5. Formal training
# ============================================================

print()
print("===== 5. FORMAL FUTURE-AUX TRAINING =====")
print("FORMAL_PARAMETER_UPDATE = YES_FROM_THIS_POINT")


condition_runtime_summaries = {}


for condition, model_cls in CONDITIONS.items():

    print()
    print("#" * 78)
    print("CONDITION =", condition)
    print("#" * 78)

    oof_scores = np.full(
        (N, CANDIDATES),
        np.nan,
        dtype=np.float32,
    )

    oof_nll = np.full(
        N,
        np.nan,
        dtype=np.float32,
    )

    oof_future_logits = np.full(
        (N, 4),
        np.nan,
        dtype=np.float32,
    )

    fold_summaries = {}

    curve_handle = curve_tmp(
        condition
    ).open(
        "w",
        encoding="utf-8",
    )


    for fold in range(N_FOLDS):

        print()
        print("=" * 72)
        print(
            f"{condition} FOLD {fold}"
        )
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


        eligible_n = int(
            future_eligible[
                fold,
                train_rows,
            ].sum()
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

        print(
            "future_eligible_training_rows =",
            eligible_n
        )


        # ----------------------------------------------------
        # Frozen identical initialization
        # ----------------------------------------------------

        seed_all(MODEL_SEED)

        model = model_cls().to(
            device
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )


        probe_rows = valid_rows[
            :min(
                128,
                len(valid_rows),
            )
        ]

        with torch.no_grad():

            (
                q_probe,
                c_probe,
                _,
                _,
            ) = build_batch(
                probe_rows,
                fold,
            )

            initial_scores = (
                model.rank_scores(
                    q_probe,
                    c_probe,
                )
            )

            dense_scores = torch.einsum(
                "bd,bcd->bc",
                q_probe,
                c_probe,
            )

            initial_diff = float(
                torch.max(
                    torch.abs(
                        initial_scores
                        - dense_scores
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
        best_train_rank_ce = None
        best_train_future_loss = None
        best_train_total_loss = None

        fold_start = time.time()


        for epoch in range(
            1,
            MAX_EPOCHS + 1,
        ):

            epoch_seed = (
                42000
                + fold * 100
                + epoch
            )

            rng = np.random.default_rng(
                epoch_seed
            )

            shuffled_train = (
                rng.permutation(
                    train_rows
                )
            )


            model.train()

            rank_sum = 0.0
            future_sum = 0.0
            total_sum = 0.0
            train_count = 0

            epoch_start = time.time()


            for start in range(
                0,
                len(shuffled_train),
                BATCH_SIZE,
            ):

                batch_rows = (
                    shuffled_train[
                        start:
                        start + BATCH_SIZE
                    ]
                )

                (
                    q,
                    candidates,
                    value_labels,
                    eligibility,
                ) = build_batch(
                    batch_rows,
                    fold,
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                scores = model.rank_scores(
                    q,
                    candidates,
                )

                rank_loss = (
                    compute_listwise_rank_loss(
                        scores,
                        gold_position=0,
                    )
                )


                future_logits = (
                    model.future_logits(
                        q,
                        candidates[:, 0, :],
                    )
                )


                future_loss = (
                    compute_masked_future_loss(
                        future_logits,
                        value_labels,
                        eligibility,
                    )
                )


                total_loss = (
                    rank_loss
                    + BETA * future_loss
                )


                assert torch.isfinite(
                    rank_loss
                )

                assert torch.isfinite(
                    future_loss
                )

                assert torch.isfinite(
                    total_loss
                )


                total_loss.backward()
                optimizer.step()


                b = len(batch_rows)

                rank_sum += (
                    float(rank_loss.item())
                    * b
                )

                future_sum += (
                    float(future_loss.item())
                    * b
                )

                total_sum += (
                    float(total_loss.item())
                    * b
                )

                train_count += b


            assert (
                train_count
                == len(train_rows)
            )


            train_rank_ce = (
                rank_sum
                / train_count
            )

            train_future_loss = (
                future_sum
                / train_count
            )

            train_total_loss = (
                total_sum
                / train_count
            )


            val_ce = (
                evaluate_validation_ce(
                    model,
                    valid_rows,
                    fold,
                )
            )


            epoch_seconds = (
                time.time()
                - epoch_start
            )


            record = {
                "condition":
                    condition,

                "beta": BETA,

                "fold":
                    fold,

                "epoch":
                    epoch,

                "epoch_shuffle_seed":
                    epoch_seed,

                "training_rows":
                    int(len(train_rows)),

                "validation_rows":
                    int(len(valid_rows)),

                "future_eligible_training_rows":
                    eligible_n,

                "train_ranking_ce":
                    float(train_rank_ce),

                "train_future_loss":
                    float(train_future_loss),

                "train_total_loss":
                    float(train_total_loss),

                "validation_ranking_ce":
                    float(val_ce),

                "epoch_seconds":
                    float(epoch_seconds),
            }


            curve_handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
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

                best_epoch = int(epoch)
                best_val_ce = float(val_ce)

                best_train_rank_ce = float(
                    train_rank_ce
                )

                best_train_future_loss = float(
                    train_future_loss
                )

                best_train_total_loss = float(
                    train_total_loss
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
                        f"FSAR S7 beta={BETA:g} {condition} Best Checkpoint V1",

                    "condition":
                        condition,

                    "beta": BETA,

                    "fold":
                        fold,

                    "best_epoch":
                        best_epoch,

                    "best_validation_ranking_ce":
                        best_val_ce,

                    "training_ranking_ce_at_best_epoch":
                        best_train_rank_ce,

                    "training_future_loss_at_best_epoch":
                        best_train_future_loss,

                    "training_total_loss_at_best_epoch":
                        best_train_total_loss,

                    "model_seed":
                        MODEL_SEED,

                    "model_state_dict":
                        state_cpu,

                    "dev_payload_used":
                        False,

                    "test_payload_used":
                        False,
                }


                torch.save(
                    checkpoint,
                    checkpoint_tmp(
                        condition,
                        fold,
                    ),
                )


            print(
                f"{condition} "
                f"fold={fold} "
                f"epoch={epoch:02d}/20 "
                f"train_rank_ce={train_rank_ce:.8f} "
                f"train_future={train_future_loss:.8f} "
                f"train_total={train_total_loss:.8f} "
                f"val_rank_ce={val_ce:.8f} "
                f"best_epoch={best_epoch} "
                f"best_val_ce={best_val_ce:.8f} "
                f"seconds={epoch_seconds:.2f}"
            )


        assert best_epoch is not None

        assert checkpoint_tmp(
            condition,
            fold,
        ).exists()


        # ----------------------------------------------------
        # Reload frozen selected checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_tmp(
                condition,
                fold,
            ),
            map_location=device,
            weights_only=False,
        )

        assert (
            checkpoint["condition"]
            == condition
        )

        assert (
            checkpoint["fold"]
            == fold
        )

        assert (
            checkpoint["best_epoch"]
            == best_epoch
        )


        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )


        reloaded_val_ce = (
            evaluate_validation_ce(
                model,
                valid_rows,
                fold,
            )
        )


        assert abs(
            reloaded_val_ce
            - best_val_ce
        ) < 1e-6


        write_oof_rows(
            model,
            valid_rows,
            fold,
            oof_scores,
            oof_nll,
            oof_future_logits,
        )


        fold_seconds = (
            time.time()
            - fold_start
        )


        fold_summaries[str(fold)] = {
            "fold":
                fold,

            "training_rows":
                int(len(train_rows)),

            "validation_rows":
                int(len(valid_rows)),

            "training_dialogues":
                len(train_dialogues),

            "validation_dialogues":
                len(valid_dialogues),

            "future_eligible_training_rows":
                eligible_n,

            "initial_max_abs_diff_vs_dense":
                initial_diff,

            "best_epoch":
                best_epoch,

            "best_validation_ranking_ce":
                best_val_ce,

            "training_ranking_ce_at_best_epoch":
                best_train_rank_ce,

            "training_future_loss_at_best_epoch":
                best_train_future_loss,

            "training_total_loss_at_best_epoch":
                best_train_total_loss,

            "reloaded_validation_ranking_ce":
                float(reloaded_val_ce),

            "fold_seconds":
                float(fold_seconds),
        }


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


    # ========================================================
    # Condition OOF audit
    # ========================================================

    print()
    print(
        f"===== {condition} OOF INTEGRITY ====="
    )

    assert np.isfinite(
        oof_scores
    ).all()

    assert np.isfinite(
        oof_nll
    ).all()

    assert np.isfinite(
        oof_future_logits
    ).all()


    oof_mean_ce = float(
        oof_nll.mean()
    )


    weighted_fold_ce = 0.0

    for fold in range(N_FOLDS):

        n_valid = (
            fold_summaries[
                str(fold)
            ]["validation_rows"]
        )

        weighted_fold_ce += (
            fold_summaries[
                str(fold)
            ][
                "best_validation_ranking_ce"
            ]
            * n_valid
        )


    weighted_fold_ce /= N


    assert abs(
        oof_mean_ce
        - weighted_fold_ce
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
        "oof_future_logits_shape =",
        oof_future_logits.shape
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


    # ========================================================
    # Save temporary outputs only
    # ========================================================

    np.savez_compressed(
        oof_tmp(condition),
        candidate_scores=oof_scores,
        ranking_ce=oof_nll,
        future_logits=oof_future_logits,
        fold_id=fold_id,
        future_available=future_available,
        future_value_label=future_labels,
        gold_position=np.zeros(
            N,
            dtype=np.int8,
        ),
    )


    summary = {
        "artifact":
            f"FSAR S7 beta={BETA:g} {condition} 5-Fold Formal Training V1",

        "status":
            "PASS",

        "condition":
            condition,

        "beta": BETA,

        "ranking_rows":
            N,

        "candidate_count":
            CANDIDATES,

        "future_value_rows":
            17458,

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

        "future_metrics_computed":
            False,

        "formal_outputs_promoted":
            False,

        "dev_payload_used":
            False,

        "test_payload_used":
            False,

        "input_sha256":
            actual_sha,
    }


    summary_tmp(
        condition
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    condition_runtime_summaries[
        condition
    ] = {
        "oof_mean_ranking_ce":
            oof_mean_ce,

        "weighted_fold_validation_ce":
            weighted_fold_ce,

        "fold_summary":
            fold_summaries,
    }


# ============================================================
# 6. Cross-condition completion gate
# ============================================================

print()
print("===== 6. CROSS-CONDITION COMPLETION GATE =====")

for condition in CONDITIONS:

    assert curve_tmp(
        condition
    ).exists()

    assert oof_tmp(
        condition
    ).exists()

    assert summary_tmp(
        condition
    ).exists()

    for fold in range(N_FOLDS):
        assert checkpoint_tmp(
            condition,
            fold,
        ).exists()


print("R_FUTURE_COMPLETE = PASS")
print("QR_FUTURE_COMPLETE = PASS")
print("CROSS_CONDITION_COMPLETION_GATE = PASS")


# ============================================================
# 7. Atomic formal promotion
# ============================================================

print()
print("===== 7. FORMAL ARTIFACT PROMOTION =====")

for condition in CONDITIONS:

    for fold in range(N_FOLDS):

        os.replace(
            checkpoint_tmp(
                condition,
                fold,
            ),
            checkpoint_final(
                condition,
                fold,
            ),
        )

    os.replace(
        curve_tmp(condition),
        curve_final(condition),
    )

    os.replace(
        oof_tmp(condition),
        oof_final(condition),
    )

    # update promotion flag before formal write
    summary = json.loads(
        summary_tmp(
            condition
        ).read_text(
            encoding="utf-8"
        )
    )

    summary[
        "formal_outputs_promoted"
    ] = True

    summary_tmp(
        condition
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        summary_tmp(condition),
        summary_final(condition),
    )


print("FORMAL_ARTIFACT_PROMOTION = PASS")


# ============================================================
# 8. Final SHA
# ============================================================

print()
print("===== 8. FINAL ARTIFACT SHA256 =====")

for condition in CONDITIONS:

    print()
    print("CONDITION =", condition)

    for fold in range(N_FOLDS):

        p = checkpoint_final(
            condition,
            fold,
        )

        print(
            sha256_file(p),
            p
        )

    for p in [
        curve_final(condition),
        oof_final(condition),
        summary_final(condition),
    ]:

        print(
            sha256_file(p),
            p
        )


print()
print("FORMAL_RETRIEVAL_METRICS_COMPUTED = NO")
print("FORMAL_FUTURE_METRICS_COMPUTED = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print(
    "PHASE_S5_FUTURE_AUX_FORMAL_TRAINING = PASS"
)
