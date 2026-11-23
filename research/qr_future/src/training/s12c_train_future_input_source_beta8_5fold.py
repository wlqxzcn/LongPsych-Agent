import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s12c_future_input_source_models import (
    CONDITIONS,
    DIM,
)
from src.models.s12c_matched_ablation_models import (
    compute_listwise_rank_loss,
    compute_masked_future_loss,
)

# ============================================================
# FROZEN PATHS
# ============================================================

Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

MASTER_FILE = (
    ROOT / "data/supervision/fsar_s3/"
    "fsar_s3a_ranking_future_master_index_v1.jsonl"
)

FOLD_FILE = (
    ROOT / "data/supervision/fsar_s3/"
    "fsar_s3b_internal_fold_assignment_v1.npy"
)

MASK_FILE = (
    ROOT / "data/supervision/fsar_s3/"
    "fsar_s3c0_future_loss_eligibility_by_fold_v1.npz"
)

RESERVOIR_FILE = (
    ROOT / "cache/hard_negatives/"
    "fsar_train_dense_top100_reservoir_v1.npz"
)

MODEL_FILE = ROOT / "src/models/s12c_future_input_source_models.py"
BASE_MODEL = ROOT / "src/models/s12c_matched_ablation_models.py"
AUDIT_FILE = ROOT / "src/evaluation/s12c_future_input_source_architecture_audit.py"

ARCH_PROTOCOL = ROOT / "protocols/fsar_s12c_matched_ablation_v1.json"
TRAIN_PROTOCOL = ROOT / "protocols/fsar_s3c2_training_and_checkpoint_protocol_v1.json"
S5_PROTOCOL = ROOT / "protocols/fsar_s5_future_aux_training_v1.json"
S7_PROTOCOL = ROOT / "protocols/fsar_s7_future_aux_beta_sensitivity_v1.json"

S4_SUMMARY = ROOT / "results/phase_s4/s4_rank_only_training_summary_v1.json"
S4_OOF = ROOT / "results/phase_s4/s4_rank_only_oof_scores_v1.npz"

PROTOCOL = ROOT / "protocols/fsar_s12c_future_input_source_v1.json"

OUT_ROOT = ROOT / "results/phase_s12c/future_input_source_v1"
CHECKPOINT_ROOT = OUT_ROOT / "checkpoints"
CURVE_ROOT = OUT_ROOT / "curves"
OOF_ROOT = OUT_ROOT / "oof"
SUMMARY_ROOT = OUT_ROOT / "summaries"

# ============================================================
# FROZEN CONSTANTS
# ============================================================

N = 62889
NR = 61502
DEPTH = 100
CANDIDATES = 101
N_FOLDS = 5

BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_SEED = 42
LR = 1e-4
BETA = 8.0

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

    "model":
        "67ff3cc1a7af8248aef45ebe01e85b3b98d6000257aefe40c9f68250eb8d6c36",

    "base_model":
        "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c",

    "audit":
        "f4e8dd75f233db97b15fbb073162481b49fc6ac67e6a539bbbf8633074de9d83",

    "architecture_protocol":
        "2e954b30aa028b997d0528fb3657dbe634e8c0043aa0c1c7bda4e819a329bce1",

    "training_protocol":
        "e30ca60f61d12ac0b291a473a656eed326b24cee1d324f06c21623bfc4d0d51e",

    "s5_protocol":
        "33dd9bfd795300a73e1450a9278d06ae75809a992f504eaffe96fc28f97e0882",

    "s7_protocol":
        "6f3924f74a1b3168ea6b75f7f2383ce41b0b2715c236b66ec53f38341edcb23d",

    "s4_summary":
        "be30581e5804c9b7368ed3f23037f79dc78b53659e3ca9abde63a2e4387de6d0",

    "s4_oof":
        "fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4",
}

INPUTS = {
    "query_embedding": Q_FILE,
    "response_embedding": R_FILE,
    "master": MASTER_FILE,
    "fold": FOLD_FILE,
    "future_mask": MASK_FILE,
    "reservoir": RESERVOIR_FILE,
    "model": MODEL_FILE,
    "base_model": BASE_MODEL,
    "audit": AUDIT_FILE,
    "architecture_protocol": ARCH_PROTOCOL,
    "training_protocol": TRAIN_PROTOCOL,
    "s5_protocol": S5_PROTOCOL,
    "s7_protocol": S7_PROTOCOL,
    "s4_summary": S4_SUMMARY,
    "s4_oof": S4_OOF,
}


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def verify_inputs():
    actual = {}

    for key, path in INPUTS.items():

        if not path.is_file():
            raise RuntimeError(f"MISSING INPUT: {path}")

        got = sha256_file(path)
        expected = EXPECTED_SHA[key]

        print(key, "=", got)

        if got != expected:
            raise RuntimeError(
                f"SHA256 MISMATCH: {key}\n"
                f"EXPECTED={expected}\n"
                f"ACTUAL={got}"
            )

        actual[key] = got

    print("FROZEN_INPUT_SHA_GATE = PASS")
    return actual


def artifact_paths():
    result = {}

    for condition in CONDITIONS:

        slug = condition.lower()

        result[condition] = {
            "curve_tmp": (
                CURVE_ROOT /
                f"s12c_beta8_{slug}_training_curve_v1.tmp.jsonl"
            ),
            "curve_final": (
                CURVE_ROOT /
                f"s12c_beta8_{slug}_training_curve_v1.jsonl"
            ),
            "oof_tmp": (
                OOF_ROOT /
                f"s12c_beta8_{slug}_oof_scores_v1.tmp.npz"
            ),
            "oof_final": (
                OOF_ROOT /
                f"s12c_beta8_{slug}_oof_scores_v1.npz"
            ),
            "summary_tmp": (
                SUMMARY_ROOT /
                f"s12c_beta8_{slug}_training_summary_v1.tmp.json"
            ),
            "summary_final": (
                SUMMARY_ROOT /
                f"s12c_beta8_{slug}_training_summary_v1.json"
            ),
            "checkpoint_tmp": [
                CHECKPOINT_ROOT / slug /
                f"{slug}_fold{f}_best_v1.tmp.pt"
                for f in range(N_FOLDS)
            ],
            "checkpoint_final": [
                CHECKPOINT_ROOT / slug /
                f"{slug}_fold{f}_best_v1.pt"
                for f in range(N_FOLDS)
            ],
        }

    return result


def check_output_collisions():
    paths = artifact_paths()
    collisions = []

    for condition, items in paths.items():

        for key, value in items.items():

            if isinstance(value, list):
                collisions.extend(
                    str(p) for p in value if p.exists()
                )
            elif value.exists():
                collisions.append(str(value))

    if OUT_ROOT.exists():
        unexpected = [
            str(p) for p in OUT_ROOT.rglob("*")
            if p.is_file()
        ]

        collisions.extend(
            p for p in unexpected
            if p not in collisions
        )

    if collisions:
        raise RuntimeError(
            "OUTPUT COLLISION. REFUSING OVERWRITE:\n"
            + "\n".join(collisions)
        )


# ============================================================
# STEP 1: PREPARE AND FREEZE PROTOCOL
# ============================================================

def prepare():

    print("=" * 76)
    print("S12C FUTURE INPUT SOURCE: PREPARATION")
    print("=" * 76)

    if PROTOCOL.exists():
        raise RuntimeError(
            f"Protocol already exists: {PROTOCOL}"
        )

    if PROTOCOL.with_suffix(".json.tmp").exists():
        raise RuntimeError("Protocol temporary file exists.")

    check_output_collisions()

    input_sha = verify_inputs()

    protocol = {
        "project": "PsychResponse_FSAR",

        "artifact":
            "FSAR S12C Future Input Source Ablation Protocol V1",

        "phase": "S12C_FUTURE_INPUT_SOURCE_ABLATION",

        "status": "FROZEN_BEFORE_TRAINING",

        "scope": "TRAIN_INTERNAL_5FOLD_OOF_ONLY",

        "purpose":
            "Disentangle future-state predictive information "
            "from current query, gold response, and their "
            "joint concatenated representation.",

        "training_script":
            "src/training/s12c_train_future_input_source_beta8_5fold.py",

        "training_script_sha256":
            sha256_file(Path(__file__).resolve()),

        "source_sha256": input_sha,

        "ranking_architecture": {
            "type": "FULL_MATCHED",
            "features": [
                "z_q",
                "z_r",
                "abs(z_q-z_r)",
                "z_q*z_r",
            ],
            "initial_ranking_function":
                "-L1(z_q,z_r)/sqrt(1024)",
        },

        "future_head": {
            "input_dimension": 2048,
            "output_dimension": 4,
            "ordinal_thresholds": [-2, -1, 0, 1],
            "parameter_count": 8196,
        },

        "conditions": {
            "R_ONLY": {
                "model_class": "ROnlyFullMatchedModel",
                "future_input": "[z_r; zeros_like(z_r)]",
            },
            "Q_ONLY": {
                "model_class": "QOnlyFullMatchedModel",
                "future_input": "[zeros_like(z_q); z_q]",
            },
            "QR_ADD": {
                "model_class": "QRAddFullMatchedModel",
                "future_input": "[z_r; z_q]",
                "operation": "CONCATENATION_NOT_ELEMENTWISE_ADDITION",
            },
        },

        "initialization": {
            "seed": MODEL_SEED,
            "identical_initialization": True,
            "identical_parameter_count": True,
            "parameter_count_per_model": 2109444,
            "initial_ranking_function_matched": True,
        },

        "data": {
            "ranking_rows": N,
            "unique_responses": NR,
            "candidate_count": CANDIDATES,
            "future_available_rows": 17458,
            "dialogues": 1666,
            "future_available_dialogues": 1653,
            "split_unit": "dialogue_id",
            "folds": N_FOLDS,
        },

        "future_supervision": {
            "gold_response_only": True,
            "negative_future_labels": False,
            "fold_specific_eligibility_mask": True,
            "loss": "MEAN_MASKED_ORDINAL_BCE",
            "denominator": "FULL_QUERY_BATCH",
        },

        "training": {
            "beta": BETA,
            "optimizer": "Adam",
            "learning_rate": LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "model_seed": MODEL_SEED,
            "epoch_shuffle_seed":
                "42000 + fold_id * 100 + epoch_id",
            "mixed_precision": False,
            "tf32": False,
        },

        "checkpoint_selection": {
            "metric": "VALIDATION_RANKING_CROSS_ENTROPY",
            "direction": "MINIMIZE",
            "tie_break": "EARLIEST_EPOCH",
            "future_metrics_used": False,
        },

        "evaluation": {
            "oof_candidate_scores_saved": True,
            "oof_future_logits_saved": True,
            "retrieval_metrics_computed_during_training": False,
            "future_metrics_computed_during_training": False,
            "dev_payload_used": False,
            "test_payload_used": False,
            "interpretation": "EXPLORATORY_COMPONENT_ABLATION",
        },

        "output_root":
            "results/phase_s12c/future_input_source_v1",

        "overwrite_existing": False,
        "atomic_promotion": True,
    }

    tmp = PROTOCOL.with_suffix(".json.tmp")

    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)

    write_json(tmp, protocol)
    os.replace(tmp, PROTOCOL)

    print("OUTPUT_COLLISION = 0")
    print("FROZEN_INPUT_SHA_GATE = PASS")
    print("PROTOCOL_FROZEN = PASS")
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL))
    print("TRAINING = NO")
    print("PROTOCOL =", PROTOCOL)


# ============================================================
# STEP 2: LOAD TRAINING DATA
# ============================================================

def load_data():

    global Q, R, master, fold_id
    global future_eligible, negative_rows
    global gold_rows, future_available, future_labels

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
        [bool(x["future_value_available"]) for x in master],
        dtype=np.bool_,
    )

    future_labels = np.full(
        N,
        -2,
        dtype=np.int64,
    )

    for i, row in enumerate(master):
        if row["future_value_available"]:
            future_labels[i] = int(row["future_value_label"])

    assert Q.shape == (N, DIM)
    assert R.shape == (NR, DIM)
    assert len(master) == N
    assert fold_id.shape == (N,)
    assert future_eligible.shape == (N_FOLDS, N)
    assert negative_rows.shape == (N, DEPTH)
    assert gold_rows.shape == (N,)
    assert int(future_available.sum()) == 17458

    assert np.all(
        (fold_id >= 0) & (fold_id < N_FOLDS)
    )

    assert np.all(
        negative_rows != gold_rows[:, None]
    )

    assert np.all(
        (future_labels[future_available] >= -2)
        & (future_labels[future_available] <= 2)
    )

    dialogue_ids = {
        str(x["dialogue_id"]) for x in master
    }

    future_dialogues = {
        str(master[i]["dialogue_id"])
        for i in np.flatnonzero(future_available)
    }

    assert len(dialogue_ids) == 1666
    assert len(future_dialogues) == 1653

    print("query_shape =", Q.shape)
    print("response_shape =", R.shape)
    print("ranking_rows =", N)
    print("reservoir_shape =", negative_rows.shape)
    print("future_value_rows =", int(future_available.sum()))
    print("dialogues =", len(dialogue_ids))
    print("future_available_dialogues =", len(future_dialogues))
    print("TRAIN_STRUCTURE = PASS")


# ============================================================
# BATCH CONSTRUCTION
# ============================================================

def build_batch(rows, validation_fold, device):

    rows = np.asarray(rows, dtype=np.int64)

    q_np = np.array(
        Q[rows],
        dtype=np.float32,
        copy=True,
    )

    candidate_idx = np.concatenate(
        [
            gold_rows[rows, None],
            negative_rows[rows],
        ],
        axis=1,
    )

    assert candidate_idx.shape == (len(rows), CANDIDATES)

    candidate_np = np.array(
        R[candidate_idx],
        dtype=np.float32,
        copy=True,
    )

    q = torch.from_numpy(q_np).to(device)

    candidates = torch.from_numpy(candidate_np).to(device)

    labels = torch.from_numpy(
        future_labels[rows]
    ).to(device)

    eligibility = torch.from_numpy(
        future_eligible[validation_fold, rows]
    ).to(device)

    return q, candidates, labels, eligibility


def evaluate_validation_ce(model, rows, fold, device):

    model.eval()

    loss_sum = 0.0
    count = 0

    with torch.inference_mode():

        for start in range(0, len(rows), BATCH_SIZE):

            batch_rows = rows[start:start + BATCH_SIZE]

            q, candidates, _, _ = build_batch(
                batch_rows, fold, device
            )

            scores = model.rank_scores(q, candidates)

            loss = compute_listwise_rank_loss(
                scores,
                gold_position=0,
            )

            b = len(batch_rows)
            loss_sum += float(loss.item()) * b
            count += b

    assert count == len(rows)

    return loss_sum / count


def write_oof_rows(
    model,
    rows,
    fold,
    device,
    oof_scores,
    oof_nll,
    oof_future_logits,
):

    model.eval()

    with torch.inference_mode():

        for start in range(0, len(rows), BATCH_SIZE):

            batch_rows = rows[start:start + BATCH_SIZE]

            q, candidates, _, _ = build_batch(
                batch_rows, fold, device
            )

            scores = model.rank_scores(q, candidates)

            nll = (
                -scores[:, 0]
                + torch.logsumexp(scores, dim=1)
            )

            future_logits = model.future_logits(
                q,
                candidates[:, 0, :],
            )

            oof_scores[batch_rows] = (
                scores.cpu().numpy().astype(np.float32)
            )

            oof_nll[batch_rows] = (
                nll.cpu().numpy().astype(np.float32)
            )

            oof_future_logits[batch_rows] = (
                future_logits.cpu().numpy().astype(np.float32)
            )


# ============================================================
# STEP 3: INITIALIZATION REPRODUCIBILITY
# ============================================================

def audit_initialization():

    reference = None

    for condition, model_cls in CONDITIONS.items():

        seed_all(MODEL_SEED)

        model = model_cls()

        state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        if reference is None:
            reference = state
        else:

            assert set(state) == set(reference)

            for key in reference:
                assert torch.equal(
                    state[key],
                    reference[key],
                ), f"Initialization mismatch: {condition}/{key}"

        count = sum(
            p.numel() for p in model.parameters()
        )

        assert count == 2109444

        print(
            condition,
            "INITIALIZATION = PASS",
            "PARAMETERS =", count,
        )

        del model

    print("CROSS_CONDITION_INITIALIZATION = PASS")


# ============================================================
# STEP 4: FORMAL TRAINING
# ============================================================

def train(protocol):

    print("=" * 84)
    print("S12C STEP 2F-8C: FUTURE INPUT SOURCE TRAINING")
    print("=" * 84)

    protocol_sha = sha256_file(PROTOCOL)

    print("PROTOCOL_SHA256 =", protocol_sha)

    actual = verify_inputs()

    assert actual == protocol["source_sha256"]

    assert (
        protocol["training_script_sha256"]
        == sha256_file(Path(__file__).resolve())
    )

    check_output_collisions()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training.")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")

    print("GPU =", torch.cuda.get_device_name(0))
    print("CUDA_NUMERICS = PASS")

    load_data()
    audit_initialization()

    for d in [
        CHECKPOINT_ROOT,
        CURVE_ROOT,
        OOF_ROOT,
        SUMMARY_ROOT,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    paths = artifact_paths()

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

        with paths[condition]["curve_tmp"].open(
            "w", encoding="utf-8"
        ) as curve_handle:

            for fold in range(N_FOLDS):

                print()
                print("=" * 72)
                print(f"{condition} FOLD {fold}")
                print("=" * 72)

                train_rows = np.where(fold_id != fold)[0]
                valid_rows = np.where(fold_id == fold)[0]

                train_dialogues = {
                    str(master[i]["dialogue_id"])
                    for i in train_rows
                }

                valid_dialogues = {
                    str(master[i]["dialogue_id"])
                    for i in valid_rows
                }

                assert not (train_dialogues & valid_dialogues)

                eligible_n = int(
                    future_eligible[fold, train_rows].sum()
                )

                print("training_rows =", len(train_rows))
                print("validation_rows =", len(valid_rows))
                print("training_dialogues =", len(train_dialogues))
                print("validation_dialogues =", len(valid_dialogues))
                print("future_eligible_training_rows =", eligible_n)

                seed_all(MODEL_SEED)

                model = model_cls().to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=LR,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0.0,
                )

                # Verify initial ranking function.
                probe_rows = valid_rows[:min(128, len(valid_rows))]

                with torch.no_grad():

                    q_probe, c_probe, _, _ = build_batch(
                        probe_rows, fold, device
                    )

                    z_q = model.ranker.adapt_query(q_probe)
                    z_r = model.ranker.adapt_response(c_probe)

                    expected_scores = (
                        -torch.abs(
                            z_q[:, None, :] - z_r
                        ).sum(dim=-1) / (DIM ** 0.5)
                    )

                    actual_scores = model.rank_scores(
                        q_probe, c_probe
                    )

                    initial_diff = float(
                        torch.max(
                            torch.abs(actual_scores - expected_scores)
                        ).item()
                    )

                    assert initial_diff < 1e-5

                print("initial_ranking_max_abs_diff =", initial_diff)

                best_epoch = None
                best_val_ce = None
                best_train_rank_ce = None
                best_train_future_loss = None
                best_train_total_loss = None

                fold_start = time.time()

                for epoch in range(1, MAX_EPOCHS + 1):

                    epoch_seed = 42000 + fold * 100 + epoch

                    rng = np.random.default_rng(epoch_seed)

                    shuffled_train = rng.permutation(train_rows)

                    model.train()

                    rank_sum = 0.0
                    future_sum = 0.0
                    total_sum = 0.0
                    train_count = 0

                    epoch_start = time.time()

                    for start in range(
                        0, len(shuffled_train), BATCH_SIZE
                    ):

                        batch_rows = shuffled_train[
                            start:start + BATCH_SIZE
                        ]

                        q, candidates, labels, eligibility = build_batch(
                            batch_rows, fold, device
                        )

                        optimizer.zero_grad(set_to_none=True)

                        scores = model.rank_scores(q, candidates)

                        rank_loss = compute_listwise_rank_loss(
                            scores,
                            gold_position=0,
                        )

                        future_logits = model.future_logits(
                            q,
                            candidates[:, 0, :],
                        )

                        future_loss = compute_masked_future_loss(
                            future_logits,
                            labels,
                            eligibility,
                        )

                        total_loss = rank_loss + BETA * future_loss

                        assert torch.isfinite(rank_loss)
                        assert torch.isfinite(future_loss)
                        assert torch.isfinite(total_loss)

                        total_loss.backward()
                        optimizer.step()

                        b = len(batch_rows)

                        rank_sum += float(rank_loss.item()) * b
                        future_sum += float(future_loss.item()) * b
                        total_sum += float(total_loss.item()) * b

                        train_count += b

                    assert train_count == len(train_rows)

                    train_rank_ce = rank_sum / train_count
                    train_future_loss = future_sum / train_count
                    train_total_loss = total_sum / train_count

                    val_ce = evaluate_validation_ce(
                        model,
                        valid_rows,
                        fold,
                        device,
                    )

                    epoch_seconds = time.time() - epoch_start

                    record = {
                        "condition": condition,
                        "beta": BETA,
                        "fold": fold,
                        "epoch": epoch,
                        "epoch_shuffle_seed": epoch_seed,
                        "training_rows": int(len(train_rows)),
                        "validation_rows": int(len(valid_rows)),
                        "future_eligible_training_rows": eligible_n,
                        "train_ranking_ce": train_rank_ce,
                        "train_future_loss": train_future_loss,
                        "train_total_loss": train_total_loss,
                        "validation_ranking_ce": val_ce,
                        "epoch_seconds": epoch_seconds,
                    }

                    curve_handle.write(
                        json.dumps(record) + "\n"
                    )
                    curve_handle.flush()

                    improved = (
                        best_val_ce is None
                        or val_ce < best_val_ce
                    )

                    if improved:

                        best_epoch = epoch
                        best_val_ce = float(val_ce)

                        best_train_rank_ce = float(train_rank_ce)
                        best_train_future_loss = float(train_future_loss)
                        best_train_total_loss = float(train_total_loss)

                        state_cpu = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }

                        checkpoint = {
                            "artifact": (
                                f"FSAR S12C Future Input Source "
                                f"{condition} Checkpoint V1"
                            ),
                            "condition": condition,
                            "beta": BETA,
                            "fold": fold,
                            "best_epoch": best_epoch,
                            "best_validation_ranking_ce": best_val_ce,
                            "training_ranking_ce_at_best_epoch":
                                best_train_rank_ce,
                            "training_future_loss_at_best_epoch":
                                best_train_future_loss,
                            "training_total_loss_at_best_epoch":
                                best_train_total_loss,
                            "model_seed": MODEL_SEED,
                            "model_state_dict": state_cpu,
                            "dev_payload_used": False,
                            "test_payload_used": False,
                        }

                        torch.save(
                            checkpoint,
                            paths[condition]["checkpoint_tmp"][fold],
                        )

                    print(
                        f"{condition} "
                        f"fold={fold} "
                        f"epoch={epoch:02d}/{MAX_EPOCHS} "
                        f"rank_ce={train_rank_ce:.8f} "
                        f"future={train_future_loss:.8f} "
                        f"val_ce={val_ce:.8f} "
                        f"best_epoch={best_epoch} "
                        f"seconds={epoch_seconds:.2f}",
                        flush=True,
                    )

                assert best_epoch is not None

                checkpoint = torch.load(
                    paths[condition]["checkpoint_tmp"][fold],
                    map_location=device,
                    weights_only=False,
                )

                assert checkpoint["condition"] == condition
                assert checkpoint["fold"] == fold
                assert checkpoint["best_epoch"] == best_epoch

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

                reloaded_val_ce = evaluate_validation_ce(
                    model,
                    valid_rows,
                    fold,
                    device,
                )

                assert abs(reloaded_val_ce - best_val_ce) < 1e-6

                write_oof_rows(
                    model,
                    valid_rows,
                    fold,
                    device,
                    oof_scores,
                    oof_nll,
                    oof_future_logits,
                )

                fold_seconds = time.time() - fold_start

                fold_summaries[str(fold)] = {
                    "fold": fold,
                    "training_rows": int(len(train_rows)),
                    "validation_rows": int(len(valid_rows)),
                    "training_dialogues": len(train_dialogues),
                    "validation_dialogues": len(valid_dialogues),
                    "future_eligible_training_rows": eligible_n,
                    "initial_ranking_max_abs_diff": initial_diff,
                    "best_epoch": best_epoch,
                    "best_validation_ranking_ce": best_val_ce,
                    "training_ranking_ce_at_best_epoch":
                        best_train_rank_ce,
                    "training_future_loss_at_best_epoch":
                        best_train_future_loss,
                    "training_total_loss_at_best_epoch":
                        best_train_total_loss,
                    "reloaded_validation_ranking_ce":
                        float(reloaded_val_ce),
                    "fold_seconds": fold_seconds,
                }

                print("FOLD_OOF_SCORING = PASS")

                del optimizer
                del model
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # OOF integrity
        # ----------------------------------------------------

        assert np.isfinite(oof_scores).all()
        assert np.isfinite(oof_nll).all()
        assert np.isfinite(oof_future_logits).all()

        oof_mean_ce = float(oof_nll.mean())

        weighted_fold_ce = sum(
            fold_summaries[str(f)]["best_validation_ranking_ce"]
            * fold_summaries[str(f)]["validation_rows"]
            for f in range(N_FOLDS)
        ) / N

        assert abs(oof_mean_ce - weighted_fold_ce) < 1e-5

        print("OOF_INTEGRITY = PASS")
        print("OOF_MEAN_RANKING_CE =", oof_mean_ce)

        # ----------------------------------------------------
        # Temporary outputs
        # ----------------------------------------------------

        np.savez_compressed(
            paths[condition]["oof_tmp"],
            candidate_scores=oof_scores,
            ranking_ce=oof_nll,
            future_logits=oof_future_logits,
            fold_id=fold_id,
            future_available=future_available,
            future_value_label=future_labels,
            gold_position=np.zeros(N, dtype=np.int8),
        )

        summary = {
            "artifact": (
                f"FSAR S12C Future Input Source "
                f"{condition} Training Summary V1"
            ),
            "status": "PASS",
            "condition": condition,
            "beta": BETA,
            "ranking_rows": N,
            "candidate_count": CANDIDATES,
            "future_value_rows": 17458,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": LR,
            "checkpoint_selection":
                "MIN_VALIDATION_RANKING_CROSS_ENTROPY",
            "fold_summary": fold_summaries,
            "oof_mean_ranking_ce": oof_mean_ce,
            "weighted_fold_validation_ce": weighted_fold_ce,
            "retrieval_metrics_computed": False,
            "future_metrics_computed": False,
            "formal_outputs_promoted": False,
            "dev_payload_used": False,
            "test_payload_used": False,
            "protocol_sha256": protocol_sha,
            "training_script_sha256": sha256_file(
                Path(__file__).resolve()
            ),
            "input_sha256": actual,
        }

        write_json(
            paths[condition]["summary_tmp"],
            summary,
        )

        print(condition, "TEMP_OUTPUTS = PASS")

    # ========================================================
    # CROSS-CONDITION COMPLETION GATE
    # ========================================================

    print()
    print("===== CROSS-CONDITION COMPLETION GATE =====")

    for condition in CONDITIONS:

        p = paths[condition]

        assert p["curve_tmp"].is_file()
        assert p["oof_tmp"].is_file()
        assert p["summary_tmp"].is_file()

        for f in range(N_FOLDS):
            assert p["checkpoint_tmp"][f].is_file()

        print(condition, "COMPLETE = PASS")

    print("CROSS_CONDITION_COMPLETION_GATE = PASS")

    # ========================================================
    # ATOMIC PROMOTION
    # ========================================================

    print()
    print("===== FORMAL ARTIFACT PROMOTION =====")

    for condition in CONDITIONS:

        p = paths[condition]

        for f in range(N_FOLDS):
            os.replace(
                p["checkpoint_tmp"][f],
                p["checkpoint_final"][f],
            )

        os.replace(p["curve_tmp"], p["curve_final"])
        os.replace(p["oof_tmp"], p["oof_final"])

        summary = json.loads(
            p["summary_tmp"].read_text(encoding="utf-8")
        )

        summary["formal_outputs_promoted"] = True

        write_json(p["summary_tmp"], summary)

        os.replace(
            p["summary_tmp"],
            p["summary_final"],
        )

        print(condition, "PROMOTION = PASS")

    print("FORMAL_ARTIFACT_PROMOTION = PASS")

    # ========================================================
    # FINAL HASHES
    # ========================================================

    print()
    print("===== FINAL ARTIFACT SHA256 =====")

    for condition in CONDITIONS:

        p = paths[condition]

        print()
        print("CONDITION =", condition)

        for checkpoint in p["checkpoint_final"]:
            print(sha256_file(checkpoint), checkpoint)

        for key in [
            "curve_final",
            "oof_final",
            "summary_final",
        ]:
            file_path = p[key]
            print(sha256_file(file_path), file_path)

    print()
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("RETRIEVAL_METRICS_COMPUTED = NO")
    print("FUTURE_METRICS_COMPUTED = NO")
    print("STEP_2F8C_FORMAL_TRAINING = PASS")


# ============================================================
# ENTRY
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--prepare", action="store_true")
    group.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.prepare:
        prepare()
        return

    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    if protocol.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("Protocol is not frozen.")

    train(protocol)


if __name__ == "__main__":
    main()
