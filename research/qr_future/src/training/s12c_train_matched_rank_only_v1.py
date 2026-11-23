#!/usr/bin/env python3
"""
FSAR S12C matched Rank-Only supplementary training.

Two conditions:
  RANK_ONLY_FULL_MATCHED
  RANK_ONLY_NO_PRODUCT_MATCHED

This script reuses the frozen S12C model classes and ranking-only objective.
It does NOT read Dev/Test payloads or future-state labels/masks.

Two-stage execution:
  python src/training/s12c_train_matched_rank_only_v1.py --prepare
  python src/training/s12c_train_matched_rank_only_v1.py --train
"""

from __future__ import annotations

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

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RankOnlyNoProductMatchedModel,
    compute_listwise_rank_loss,
)

# ---------------------------------------------------------------------
# Frozen inputs
# ---------------------------------------------------------------------

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
RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/"
      "fsar_train_dense_top100_reservoir_v1.npz"
)
MODEL_FILE = ROOT / "src/models/s12c_matched_ablation_models.py"
ARCH_PROTOCOL = ROOT / "protocols/fsar_s12c_matched_ablation_v1.json"

PROTOCOL_FILE = (
    ROOT / "protocols/fsar_s12c_matched_rank_only_training_v1.json"
)
OUT_ROOT = ROOT / "results/phase_s12c/rank_only_matched_v1"

# SHA256 values inherited from the frozen S12C protocol.
EXPECTED_SHA = {
    "query_embedding": "18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005",
    "response_embedding": "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f",
    "master": "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",
    "fold": "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",
    "reservoir": "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",
    "model_code": "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c",
    "architecture_protocol": "2e954b30aa028b997d0528fb3657dbe634e8c0043aa0c1c7bda4e819a329bce1",
}

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
    "RANK_ONLY_FULL_MATCHED": RankOnlyFullMatchedModel,
    "RANK_ONLY_NO_PRODUCT_MATCHED": RankOnlyNoProductMatchedModel,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def atomic_json(path: Path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_slug(condition: str) -> str:
    return condition.lower()


def checkpoint_dir(condition: str) -> Path:
    return OUT_ROOT / "checkpoints" / condition_slug(condition)


def checkpoint_tmp(condition: str, fold: int) -> Path:
    return checkpoint_dir(condition) / (
        f"{condition_slug(condition)}_fold{fold}_best_v1.tmp.pt"
    )


def checkpoint_final(condition: str, fold: int) -> Path:
    return checkpoint_dir(condition) / (
        f"{condition_slug(condition)}_fold{fold}_best_v1.pt"
    )


def curve_tmp(condition: str) -> Path:
    return OUT_ROOT / "training_curves" / (
        f"s12c_{condition_slug(condition)}_training_curve_v1.tmp.jsonl"
    )


def curve_final(condition: str) -> Path:
    return OUT_ROOT / "training_curves" / (
        f"s12c_{condition_slug(condition)}_training_curve_v1.jsonl"
    )


def oof_tmp(condition: str) -> Path:
    return OUT_ROOT / "oof" / (
        f"s12c_{condition_slug(condition)}_oof_scores_v1.tmp.npz"
    )


def oof_final(condition: str) -> Path:
    return OUT_ROOT / "oof" / (
        f"s12c_{condition_slug(condition)}_oof_scores_v1.npz"
    )


def summary_tmp(condition: str) -> Path:
    return OUT_ROOT / "summaries" / (
        f"s12c_{condition_slug(condition)}_training_summary_v1.tmp.json"
    )


def summary_final(condition: str) -> Path:
    return OUT_ROOT / "summaries" / (
        f"s12c_{condition_slug(condition)}_training_summary_v1.json"
    )


def load_and_validate_inputs():
    print("=" * 84)
    print("S12C MATCHED RANK-ONLY INPUT PREFLIGHT")
    print("=" * 84)

    paths = {
        "query_embedding": Q_FILE,
        "response_embedding": R_FILE,
        "master": MASTER_FILE,
        "fold": FOLD_FILE,
        "reservoir": RESERVOIR_FILE,
        "model_code": MODEL_FILE,
        "architecture_protocol": ARCH_PROTOCOL,
    }

    actual_sha = {}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        actual_sha[key] = actual
        expected = EXPECTED_SHA[key]
        print(f"{key}: {actual}  expected_match={actual == expected}")
        if actual != expected:
            raise RuntimeError(f"Frozen input SHA mismatch: {key}")

    q = np.load(Q_FILE, mmap_mode="r", allow_pickle=False)
    r = np.load(R_FILE, mmap_mode="r", allow_pickle=False)
    fold_id = np.load(FOLD_FILE, allow_pickle=False)
    master = load_jsonl(MASTER_FILE)

    reservoir = np.load(RESERVOIR_FILE, allow_pickle=False)
    if "candidate_response_row" not in reservoir:
        raise RuntimeError("Reservoir missing candidate_response_row")
    if "gold_response_row" not in reservoir:
        raise RuntimeError("Reservoir missing gold_response_row")

    negative_rows = np.asarray(
        reservoir["candidate_response_row"], dtype=np.int32
    )
    gold_rows = np.asarray(
        reservoir["gold_response_row"], dtype=np.int32
    )

    if q.shape != (N, DIM) or q.dtype != np.float32:
        raise RuntimeError(f"Query embedding mismatch: {q.shape}, {q.dtype}")
    if r.shape != (NR, DIM) or r.dtype != np.float32:
        raise RuntimeError(f"Response embedding mismatch: {r.shape}, {r.dtype}")
    if len(master) != N:
        raise RuntimeError(f"Master row count mismatch: {len(master)}")
    if fold_id.shape != (N,):
        raise RuntimeError(f"Fold shape mismatch: {fold_id.shape}")
    if negative_rows.shape != (N, DEPTH):
        raise RuntimeError(f"Negative reservoir shape mismatch: {negative_rows.shape}")
    if gold_rows.shape != (N,):
        raise RuntimeError(f"Gold row shape mismatch: {gold_rows.shape}")

    if not np.array_equal(np.unique(fold_id), np.arange(N_FOLDS)):
        raise RuntimeError(f"Unexpected fold IDs: {np.unique(fold_id)}")
    if np.any(gold_rows < 0) or np.any(gold_rows >= NR):
        raise RuntimeError("Gold response row out of range")
    if np.any(negative_rows < 0) or np.any(negative_rows >= NR):
        raise RuntimeError("Negative response row out of range")
    if np.any(negative_rows == gold_rows[:, None]):
        raise RuntimeError("Gold response appears in negative reservoir")
    if any(len(np.unique(row)) != DEPTH for row in negative_rows):
        raise RuntimeError("Duplicate negative response within a query")

    dialogue_ids = np.asarray(
        [str(row["dialogue_id"]) for row in master], dtype=str
    )
    if len(dialogue_ids) != N:
        raise RuntimeError("Dialogue ID alignment failure")

    dialogue_fold = {}
    for did, fid in zip(dialogue_ids, fold_id):
        previous = dialogue_fold.setdefault(did, int(fid))
        if previous != int(fid):
            raise RuntimeError(f"Dialogue split across folds: {did}")

    if len(dialogue_fold) != 1666:
        raise RuntimeError(f"Dialogue count mismatch: {len(dialogue_fold)}")

    for fold in range(N_FOLDS):
        valid_rows = np.flatnonzero(fold_id == fold)
        train_rows = np.flatnonzero(fold_id != fold)
        if not len(valid_rows) or not len(train_rows):
            raise RuntimeError(f"Empty train/validation fold: {fold}")
        print(
            f"fold={fold} train_rows={len(train_rows)} "
            f"valid_rows={len(valid_rows)} "
            f"train_dialogues={len(set(dialogue_ids[train_rows]))} "
            f"valid_dialogues={len(set(dialogue_ids[valid_rows]))}"
        )

    print("query_shape =", q.shape)
    print("response_shape =", r.shape)
    print("reservoir_shape =", negative_rows.shape)
    print("ranking_rows =", N)
    print("dialogues =", len(dialogue_fold))
    print("future_labels_read = NO")
    print("future_mask_read = NO")
    print("dev_payload_read = NO")
    print("test_payload_read = NO")
    print("INPUT_PREFLIGHT = PASS")

    return q, r, fold_id.astype(np.int64), dialogue_ids, negative_rows, gold_rows, actual_sha


def build_batch(rows, Q, R, negative_rows, gold_rows, device):
    rows = np.asarray(rows, dtype=np.int64)

    q_np = np.array(Q[rows], dtype=np.float32, copy=True)
    candidate_idx = np.concatenate(
        [gold_rows[rows, None], negative_rows[rows]], axis=1
    )
    if candidate_idx.shape != (len(rows), CANDIDATES):
        raise RuntimeError(f"Candidate shape mismatch: {candidate_idx.shape}")

    candidate_np = np.array(R[candidate_idx], dtype=np.float32, copy=True)

    q = torch.from_numpy(q_np).to(device)
    candidates = torch.from_numpy(candidate_np).to(device)
    return q, candidates


def evaluate_validation_ce(model, rows, Q, R, negative_rows, gold_rows, device):
    model.eval()
    loss_sum = 0.0
    count = 0

    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            q, candidates = build_batch(
                batch_rows, Q, R, negative_rows, gold_rows, device
            )
            scores = model.rank_scores(q, candidates)
            loss = compute_listwise_rank_loss(scores, gold_position=0)
            b = len(batch_rows)
            loss_sum += float(loss.item()) * b
            count += b

    if count != len(rows):
        raise RuntimeError("Validation row accounting mismatch")
    return loss_sum / count


def check_initialization(device, Q, R, negative_rows, gold_rows, fold_id):
    print("\n===== MATCHED INITIALIZATION CHECK =====")
    fold = 0
    probe_rows = np.flatnonzero(fold_id == fold)[:128]
    q, candidates = build_batch(
        probe_rows, Q, R, negative_rows, gold_rows, device
    )

    initial_scores = {}
    expected_scores = None

    for name, model_cls in CONDITIONS.items():
        seed_all(MODEL_SEED)
        model = model_cls().to(device)
        model.eval()

        with torch.inference_mode():
            zq = model.ranker.adapt_query(q)
            zr = model.ranker.adapt_response(candidates)
            expected = -torch.abs(zq[:, None, :] - zr).sum(-1) / (DIM ** 0.5)
            actual = model.rank_scores(q, candidates)
            diff = float((actual - expected).abs().max().item())

        if diff > 1e-6:
            raise RuntimeError(
                f"{name} initial score differs from negative-L1: {diff}"
            )

        if expected_scores is None:
            expected_scores = actual.detach().clone()
        else:
            cross_diff = float((actual - expected_scores).abs().max().item())
            if cross_diff > 1e-6:
                raise RuntimeError(
                    f"Full/No-Product initial ranking mismatch: {cross_diff}"
                )

        initial_scores[name] = diff
        print(name, "rank_head_shape=", tuple(model.ranker.rank_head.weight.shape),
              "initial_max_abs_diff=", diff)
        del model

    print("INITIAL_RANKING_FUNCTION = -L1(z_q,z_r)/sqrt(1024)")
    print("FULL_VS_NO_PRODUCT_INITIAL_MATCH = PASS")
    print("INITIALIZATION_CHECK = PASS")
    return initial_scores


def make_protocol(actual_sha, initial_diffs):
    return {
        "artifact": "FSAR S12C Matched Rank-Only Supplementary Training Protocol V1",
        "project": "PsychResponse_FSAR",
        "phase": "S12C_MATCHED_RANK_ONLY_SUPPLEMENT",
        "status": "FROZEN_BEFORE_TRAINING",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "purpose": (
            "Train architecture-matched ranking-only baselines for the S12C "
            "FULL and NO_PRODUCT ranking architectures."
        ),
        "conditions": {
            "RANK_ONLY_FULL_MATCHED": {
                "model_class": "RankOnlyFullMatchedModel",
                "rank_input_dimension": 4096,
                "product_interaction": True,
            },
            "RANK_ONLY_NO_PRODUCT_MATCHED": {
                "model_class": "RankOnlyNoProductMatchedModel",
                "rank_input_dimension": 3072,
                "product_interaction": False,
            },
        },
        "initialization": {
            "query_adapter": "zero",
            "response_adapter": "zero",
            "rank_head_weights": "zero except absolute-difference block",
            "absolute_difference_weights": "-1/sqrt(1024)",
            "initial_ranking_function": "-L1(z_q,z_r)/sqrt(1024)",
            "initial_ranking_function_matched": True,
            "initial_max_abs_diff": initial_diffs,
            "parameter_count_matched": False,
        },
        "training": {
            "ranking_rows": N,
            "candidate_count": CANDIDATES,
            "folds": N_FOLDS,
            "split_unit": "dialogue_id",
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "seed": MODEL_SEED,
            "epoch_shuffle_seed": "42000 + fold * 100 + epoch",
            "optimizer": "Adam",
            "learning_rate": LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "loss": "listwise ranking cross entropy only",
            "checkpoint_selection": "MIN_VALIDATION_RANKING_CROSS_ENTROPY",
            "future_supervision_used": False,
        },
        "input_sha256": actual_sha,
        "source": {
            "training_script_sha256": sha256_file(Path(__file__)),
            "model_code_sha256": actual_sha["model_code"],
            "architecture_protocol_sha256": actual_sha["architecture_protocol"],
        },
        "data_access": {
            "train_embeddings_used": True,
            "train_reservoir_used": True,
            "future_labels_read": False,
            "future_mask_read": False,
            "dev_payload_used": False,
            "test_payload_used": False,
        },
        "outputs": {
            "root": str(OUT_ROOT.relative_to(ROOT)),
            "checkpoint_subdirectories": [
                "checkpoints/rank_only_full_matched",
                "checkpoints/rank_only_no_product_matched",
            ],
            "oof_subdirectory": "oof",
            "curve_subdirectory": "training_curves",
            "summary_subdirectory": "summaries",
        },
        "overwrite_policy": "REFUSE_IF_PROTOCOL_OR_OUTPUT_EXISTS",
    }


def prepare():
    if PROTOCOL_FILE.exists():
        raise RuntimeError(f"Protocol already exists; refusing overwrite: {PROTOCOL_FILE}")
    if OUT_ROOT.exists():
        raise RuntimeError(f"Output directory already exists; refusing overwrite: {OUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")

    Q, R, fold_id, dialogue_ids, negative_rows, gold_rows, actual_sha = (
        load_and_validate_inputs()
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    print("GPU =", torch.cuda.get_device_name(0))

    initial_diffs = check_initialization(
        device, Q, R, negative_rows, gold_rows, fold_id
    )
    protocol = make_protocol(actual_sha, initial_diffs)
    atomic_json(PROTOCOL_FILE, protocol)

    print("\nRANK_ONLY_PROTOCOL_FROZEN = PASS")
    print("PROTOCOL =", PROTOCOL_FILE)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_FILE))
    print("TRAINING_STARTED = NO")
    print("FILES_MODIFIED = only the new supplementary protocol")


def verify_frozen_protocol(actual_sha):
    if not PROTOCOL_FILE.is_file():
        raise RuntimeError(
            f"Frozen protocol missing. Run --prepare first: {PROTOCOL_FILE}"
        )

    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_TRAINING":
        raise RuntimeError("Rank-only protocol status mismatch")
    if protocol.get("input_sha256") != actual_sha:
        raise RuntimeError("Frozen input hashes changed after protocol preparation")
    if protocol.get("source", {}).get("training_script_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Training script changed after protocol freeze")
    if OUT_ROOT.exists():
        raise RuntimeError(f"Output directory already exists; refusing overwrite: {OUT_ROOT}")
    return protocol


def write_oof_rows(model, rows, Q, R, negative_rows, gold_rows, device,
                   oof_scores, oof_nll):
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            q, candidates = build_batch(
                batch_rows, Q, R, negative_rows, gold_rows, device
            )
            scores = model.rank_scores(q, candidates)
            nll = -scores[:, 0] + torch.logsumexp(scores, dim=1)

            oof_scores[batch_rows] = (
                scores.detach().cpu().numpy().astype(np.float32)
            )
            oof_nll[batch_rows] = (
                nll.detach().cpu().numpy().astype(np.float32)
            )


def train_condition(condition, model_cls, Q, R, fold_id, dialogue_ids,
                    negative_rows, gold_rows, device):
    print("\n" + "#" * 84)
    print("CONDITION =", condition)
    print("#" * 84)

    oof_scores = np.full((N, CANDIDATES), np.nan, dtype=np.float32)
    oof_nll = np.full(N, np.nan, dtype=np.float32)
    fold_summaries = {}

    curve_path = curve_tmp(condition)
    with curve_path.open("w", encoding="utf-8") as curve_handle:
        for fold in range(N_FOLDS):
            print("\n===== FOLD", fold, "=====")
            train_rows = np.flatnonzero(fold_id != fold)
            valid_rows = np.flatnonzero(fold_id == fold)

            train_dialogues = set(dialogue_ids[train_rows])
            valid_dialogues = set(dialogue_ids[valid_rows])
            if train_dialogues & valid_dialogues:
                raise RuntimeError(f"Dialogue leakage in fold {fold}")

            seed_all(MODEL_SEED)
            model = model_cls().to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            )

            # Confirm this fold starts from the frozen negative-L1 function.
            probe_rows = valid_rows[:min(128, len(valid_rows))]
            q_probe, c_probe = build_batch(
                probe_rows, Q, R, negative_rows, gold_rows, device
            )
            with torch.inference_mode():
                zq = model.ranker.adapt_query(q_probe)
                zr = model.ranker.adapt_response(c_probe)
                expected_initial = (
                    -torch.abs(zq[:, None, :] - zr).sum(-1) / (DIM ** 0.5)
                )
                actual_initial = model.rank_scores(q_probe, c_probe)
                initial_diff = float(
                    (actual_initial - expected_initial).abs().max().item()
                )
            if initial_diff > 1e-6:
                raise RuntimeError(
                    f"{condition} fold={fold} initialization mismatch: {initial_diff}"
                )

            best_epoch = None
            best_val_ce = None
            best_train_rank_ce = None
            fold_start = time.time()

            for epoch in range(1, MAX_EPOCHS + 1):
                epoch_seed = 42000 + fold * 100 + epoch
                rng = np.random.default_rng(epoch_seed)
                shuffled_train = rng.permutation(train_rows)

                model.train()
                rank_sum = 0.0
                train_count = 0
                epoch_start = time.time()

                for start in range(0, len(shuffled_train), BATCH_SIZE):
                    batch_rows = shuffled_train[start:start + BATCH_SIZE]
                    q, candidates = build_batch(
                        batch_rows, Q, R, negative_rows, gold_rows, device
                    )

                    optimizer.zero_grad(set_to_none=True)
                    scores = model.rank_scores(q, candidates)
                    rank_loss = compute_listwise_rank_loss(
                        scores, gold_position=0
                    )
                    if not torch.isfinite(rank_loss):
                        raise RuntimeError(
                            f"Non-finite rank loss: {condition}, fold={fold}, epoch={epoch}"
                        )

                    rank_loss.backward()
                    optimizer.step()

                    b = len(batch_rows)
                    rank_sum += float(rank_loss.item()) * b
                    train_count += b

                if train_count != len(train_rows):
                    raise RuntimeError("Training row accounting mismatch")

                train_rank_ce = rank_sum / train_count
                val_ce = evaluate_validation_ce(
                    model, valid_rows, Q, R, negative_rows, gold_rows, device
                )
                epoch_seconds = time.time() - epoch_start

                record = {
                    "condition": condition,
                    "fold": fold,
                    "epoch": epoch,
                    "epoch_shuffle_seed": epoch_seed,
                    "training_rows": int(len(train_rows)),
                    "validation_rows": int(len(valid_rows)),
                    "train_ranking_ce": float(train_rank_ce),
                    "validation_ranking_ce": float(val_ce),
                    "epoch_seconds": float(epoch_seconds),
                }
                curve_handle.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                curve_handle.flush()

                improved = best_val_ce is None or val_ce < best_val_ce
                if improved:
                    best_epoch = int(epoch)
                    best_val_ce = float(val_ce)
                    best_train_rank_ce = float(train_rank_ce)

                    state_cpu = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
                    checkpoint = {
                        "artifact": (
                            f"FSAR S12C {condition} Best Checkpoint V1"
                        ),
                        "condition": condition,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_validation_ranking_ce": best_val_ce,
                        "training_ranking_ce_at_best_epoch": best_train_rank_ce,
                        "model_seed": MODEL_SEED,
                        "model_state_dict": state_cpu,
                        "initial_ranking_function": (
                            "-L1(z_q,z_r)/sqrt(1024)"
                        ),
                        "future_supervision_used": False,
                        "dev_payload_used": False,
                        "test_payload_used": False,
                    }
                    torch.save(checkpoint, checkpoint_tmp(condition, fold))

                print(
                    f"{condition} fold={fold} epoch={epoch:02d}/{MAX_EPOCHS} "
                    f"train_rank_ce={train_rank_ce:.8f} "
                    f"val_rank_ce={val_ce:.8f} "
                    f"best_epoch={best_epoch} "
                    f"best_val_ce={best_val_ce:.8f} "
                    f"seconds={epoch_seconds:.2f}",
                    flush=True,
                )

            if best_epoch is None or not checkpoint_tmp(condition, fold).is_file():
                raise RuntimeError(f"Missing best checkpoint: {condition}, fold={fold}")

            checkpoint = torch.load(
                checkpoint_tmp(condition, fold),
                map_location=device,
                weights_only=False,
            )
            if checkpoint.get("condition") != condition:
                raise RuntimeError("Checkpoint condition mismatch")
            if checkpoint.get("fold") != fold:
                raise RuntimeError("Checkpoint fold mismatch")
            if checkpoint.get("best_epoch") != best_epoch:
                raise RuntimeError("Checkpoint best epoch mismatch")

            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            reloaded_val_ce = evaluate_validation_ce(
                model, valid_rows, Q, R, negative_rows, gold_rows, device
            )
            if abs(reloaded_val_ce - best_val_ce) >= 1e-6:
                raise RuntimeError(
                    f"Reloaded validation CE mismatch: {reloaded_val_ce} vs {best_val_ce}"
                )

            write_oof_rows(
                model, valid_rows, Q, R, negative_rows, gold_rows, device,
                oof_scores, oof_nll,
            )

            fold_seconds = time.time() - fold_start
            fold_summaries[str(fold)] = {
                "fold": fold,
                "training_rows": int(len(train_rows)),
                "validation_rows": int(len(valid_rows)),
                "training_dialogues": len(train_dialogues),
                "validation_dialogues": len(valid_dialogues),
                "initial_max_abs_diff_vs_negative_scaled_l1": initial_diff,
                "best_epoch": best_epoch,
                "best_validation_ranking_ce": best_val_ce,
                "training_ranking_ce_at_best_epoch": best_train_rank_ce,
                "reloaded_validation_ranking_ce": float(reloaded_val_ce),
                "fold_seconds": float(fold_seconds),
            }

            print(
                f"{condition} fold={fold} RELOAD_VALIDATION_CE_PASS "
                f"OOF_ROWS={len(valid_rows)}",
                flush=True,
            )

            del optimizer
            del model
            torch.cuda.empty_cache()

    if not np.isfinite(oof_scores).all():
        raise RuntimeError(f"{condition}: non-finite OOF scores")
    if not np.isfinite(oof_nll).all():
        raise RuntimeError(f"{condition}: non-finite OOF ranking CE")

    oof_mean_ce = float(oof_nll.mean())
    weighted_fold_ce = sum(
        fold_summaries[str(f)]["best_validation_ranking_ce"]
        * fold_summaries[str(f)]["validation_rows"]
        for f in range(N_FOLDS)
    ) / N

    if abs(oof_mean_ce - weighted_fold_ce) >= 1e-5:
        raise RuntimeError(
            f"{condition}: OOF CE does not match weighted validation CE: "
            f"{oof_mean_ce} vs {weighted_fold_ce}"
        )

    np.savez_compressed(
        oof_tmp(condition),
        candidate_scores=oof_scores,
        ranking_ce=oof_nll,
        fold_id=fold_id.astype(np.int8),
        gold_position=np.zeros(N, dtype=np.int8),
    )

    summary = {
        "artifact": f"FSAR S12C {condition} 5-Fold Matched Rank-Only Training V1",
        "status": "PASS",
        "condition": condition,
        "ranking_rows": N,
        "candidate_count": CANDIDATES,
        "folds": N_FOLDS,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": 0.0,
        "checkpoint_selection": "MIN_VALIDATION_RANKING_CROSS_ENTROPY",
        "initial_ranking_function": "-L1(z_q,z_r)/sqrt(1024)",
        "initial_ranking_function_matched_to_s12c_future_models": True,
        "future_supervision_used": False,
        "dev_payload_used": False,
        "test_payload_used": False,
        "fold_summary": fold_summaries,
        "oof_mean_ranking_ce": oof_mean_ce,
        "weighted_fold_validation_ce": weighted_fold_ce,
        "input_sha256": EXPECTED_SHA,
        "model_code_sha256": EXPECTED_SHA["model_code"],
        "architecture_protocol_sha256": EXPECTED_SHA["architecture_protocol"],
    }
    atomic_json(summary_tmp(condition), summary)

    print(
        f"{condition} OOF_INTEGRITY=PASS "
        f"oof_mean_ranking_ce={oof_mean_ce:.8f}",
        flush=True,
    )


def train():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if OUT_ROOT.exists():
        raise RuntimeError(f"Output directory already exists; refusing overwrite: {OUT_ROOT}")

    Q, R, fold_id, dialogue_ids, negative_rows, gold_rows, actual_sha = (
        load_and_validate_inputs()
    )
    protocol = verify_frozen_protocol(actual_sha)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    print("\nGPU =", torch.cuda.get_device_name(0))
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_FILE))
    print("PROTOCOL_STATUS =", protocol["status"])
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("FUTURE_SUPERVISION_USED = NO")

    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUT_ROOT / "checkpoints").mkdir()
    (OUT_ROOT / "training_curves").mkdir()
    (OUT_ROOT / "oof").mkdir()
    (OUT_ROOT / "summaries").mkdir()

    for condition in CONDITIONS:
        checkpoint_dir(condition).mkdir(parents=True, exist_ok=False)

    for condition, model_cls in CONDITIONS.items():
        train_condition(
            condition, model_cls, Q, R, fold_id, dialogue_ids,
            negative_rows, gold_rows, device,
        )

    # Cross-condition completion gate before any final promotion.
    print("\n===== CROSS-CONDITION COMPLETION GATE =====")
    for condition in CONDITIONS:
        for fold in range(N_FOLDS):
            if not checkpoint_tmp(condition, fold).is_file():
                raise RuntimeError(f"Missing temporary checkpoint: {condition}, fold={fold}")
        for path in (curve_tmp(condition), oof_tmp(condition), summary_tmp(condition)):
            if not path.is_file():
                raise RuntimeError(f"Missing temporary artifact: {path}")
        with np.load(oof_tmp(condition), allow_pickle=False) as z:
            if z["candidate_scores"].shape != (N, CANDIDATES):
                raise RuntimeError(f"OOF score shape mismatch: {condition}")
            if z["ranking_ce"].shape != (N,):
                raise RuntimeError(f"OOF CE shape mismatch: {condition}")
            if not np.isfinite(z["candidate_scores"]).all():
                raise RuntimeError(f"OOF score non-finite: {condition}")
            if not np.isfinite(z["ranking_ce"]).all():
                raise RuntimeError(f"OOF CE non-finite: {condition}")
        print(condition, "COMPLETE = PASS")

    # Promote all outputs only after both conditions pass the gate.
    print("\n===== ATOMIC ARTIFACT PROMOTION =====")
    for condition in CONDITIONS:
        for fold in range(N_FOLDS):
            os.replace(
                checkpoint_tmp(condition, fold),
                checkpoint_final(condition, fold),
            )
        os.replace(curve_tmp(condition), curve_final(condition))
        os.replace(oof_tmp(condition), oof_final(condition))
        os.replace(summary_tmp(condition), summary_final(condition))

    manifest = {
        "artifact": "FSAR S12C Matched Rank-Only Supplementary Run Manifest V1",
        "status": "PASS",
        "protocol": str(PROTOCOL_FILE.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "training_script_sha256": sha256_file(Path(__file__)),
        "conditions": list(CONDITIONS.keys()),
        "folds": list(range(N_FOLDS)),
        "dev_payload_used": False,
        "test_payload_used": False,
        "future_supervision_used": False,
        "outputs": {},
    }
    for condition in CONDITIONS:
        manifest["outputs"][condition] = {
            "checkpoints": [
                {
                    "path": str(checkpoint_final(condition, f).relative_to(ROOT)),
                    "sha256": sha256_file(checkpoint_final(condition, f)),
                }
                for f in range(N_FOLDS)
            ],
            "curve": {
                "path": str(curve_final(condition).relative_to(ROOT)),
                "sha256": sha256_file(curve_final(condition)),
            },
            "oof": {
                "path": str(oof_final(condition).relative_to(ROOT)),
                "sha256": sha256_file(oof_final(condition)),
            },
            "summary": {
                "path": str(summary_final(condition).relative_to(ROOT)),
                "sha256": sha256_file(summary_final(condition)),
            },
        }

    atomic_json(OUT_ROOT / "run_manifest_v1.json", manifest)

    print("\n===== FINAL OUTPUTS =====")
    print("S12C_MATCHED_RANK_ONLY_TRAINING = PASS")
    print("OUTPUT_ROOT =", OUT_ROOT)
    print("MANIFEST =", OUT_ROOT / "run_manifest_v1.json")
    for condition in CONDITIONS:
        print(condition, "SUMMARY =", summary_final(condition))
        print(condition, "OOF =", oof_final(condition))
        print(condition, "CHECKPOINTS =", checkpoint_dir(condition))
    print("EXISTING_S4_S12C_ARTIFACTS_MODIFIED = NO")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prepare",
        action="store_true",
        help="Read-only input audit, initialization check, and freeze supplementary protocol",
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Train the two matched Rank-Only conditions using the frozen protocol",
    )
    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        train()


if __name__ == "__main__":
    main()
