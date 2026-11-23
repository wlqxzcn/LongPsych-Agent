#!/usr/bin/env python3
"""
ORCA-FSAR S12C six-condition, fold-specific 5-fold training.

Two-stage workflow:
  python src/training/s12c_train_orca_fsar_six_condition_5fold.py --prepare
  python src/training/s12c_train_orca_fsar_six_condition_5fold.py --train
  python src/training/s12c_train_orca_fsar_six_condition_5fold.py --train --resume

--prepare is CPU-only. It audits frozen inputs, all five ORCA projection
artifacts, model initialization, and freezes a protocol.
--train requires CUDA and never reads Dev/Test payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RankOnlyNoProductMatchedModel,
    RFutureFullMatchedModel,
    RFutureNoProductMatchedModel,
    QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
    compute_listwise_rank_loss,
    compute_masked_future_loss,
)

# ---------------------------------------------------------------------
# Frozen source inputs
# ---------------------------------------------------------------------

RAW_Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
RAW_R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

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

MODEL_CODE = ROOT / "src/models/s12c_matched_ablation_models.py"
SOURCE_RANK_ONLY_TRAINER_FILE = (
    ROOT / "src/training/s12c_train_matched_rank_only_v1.py"
)
SOURCE_FUTURE_MATCHED_TRAINER_FILE = (
    ROOT / "src/training/s12c_train_matched_ablation_beta8_5fold.py"
)
ORCA_PROJECTION_SCRIPT = (
    ROOT / "src/evaluation/s12c_step_2f17d_fold_projection.py"
)
ORCA_PROJECTION_AUDIT_SCRIPT = (
    ROOT / "src/evaluation/s12c_step_2f17e_projection_audit.py"
)
ARCH_PROTOCOL = ROOT / "protocols/fsar_s12c_matched_ablation_v1.json"
TRAIN_PROTOCOL = (
    ROOT / "protocols/fsar_s3c2_training_and_checkpoint_protocol_v1.json"
)
S5_PROTOCOL = ROOT / "protocols/fsar_s5_future_aux_training_v1.json"
S7_PROTOCOL = ROOT / "protocols/fsar_s7_future_aux_beta_sensitivity_v1.json"
S4_SUMMARY = ROOT / "results/phase_s4/s4_rank_only_training_summary_v1.json"
S4_OOF = ROOT / "results/phase_s4/s4_rank_only_oof_scores_v1.npz"

# ---------------------------------------------------------------------
# Frozen ORCA artifacts
# ---------------------------------------------------------------------

ORCA_BASIS_ROOT = (
    ROOT / "results/phase_s12c/step_2f17c_orca_fold_basis_v1"
)
ORCA_PROJECTION_ROOT = (
    ROOT / "results/phase_s12c/step_2f17d_orca_fold_projection_v1"
)
ORCA_BASIS_SUMMARY = ORCA_BASIS_ROOT / "summary.json"
ORCA_PROJECTION_SUMMARY = ORCA_PROJECTION_ROOT / "summary.json"
ORCA_INTEGRITY_AUDIT = ORCA_PROJECTION_ROOT / "integrity_audit_v1.json"

# New, isolated output. This directory must not be used by prior S12C runs.
OUT_ROOT = (
    ROOT / "results/phase_s12c/step_2f18_orca_fsar_training_v1"
)
PROTOCOL_FILE = OUT_ROOT / "training_protocol_v1.json"
RUN_MANIFEST = OUT_ROOT / "run_manifest.json"

# ---------------------------------------------------------------------
# Frozen hashes inherited from S12C and ORCA projection audits
# ---------------------------------------------------------------------

EXPECTED_BASE_SHA = {
    "raw_query_embedding": (
        "18aa1ed637bef3980775c10d96d1cde6112863b0c5761e2e69fb7ab89c981005"
    ),
    "raw_response_embedding": (
        "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f"
    ),
    "master": (
        "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be"
    ),
    "fold": (
        "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9"
    ),
    "future_mask": (
        "84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d"
    ),
    "reservoir": (
        "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0"
    ),
    "model_code": (
        "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c"
    ),
    "architecture_protocol": (
        "2e954b30aa028b997d0528fb3657dbe634e8c0043aa0c1c7bda4e819a329bce1"
    ),
    "training_protocol": (
        "e30ca60f61d12ac0b291a473a656eed326b24cee1d324f06c21623bfc4d0d51e"
    ),
    "s5_protocol": (
        "33dd9bfd795300a73e1450a9278d06ae75809a992f504eaffe96fc28f97e0882"
    ),
    "s7_protocol": (
        "6f3924f74a1b3168ea6b75f7f2383ce41b0b2715c236b66ec53f38341edcb23d"
    ),
    "s4_summary": (
        "be30581e5804c9b7368ed3f23037f79dc78b53659e3ca9abde63a2e4387de6d0"
    ),
    "s4_oof": (
        "fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4"
    ),
    "source_rank_only_trainer": (
        "aa2ac1f739367515094554753c0babe4de36d79919cb40ffeaf5a0a09acde930"
    ),
    "source_future_matched_trainer": (
        "5730ccc5fc49f9a91c134a2ffa6bc06c374cb4bb0b3618c12a2a45b9768f908d"
    ),
    "orca_projection_script": (
        "9b3791d50a6efa03ddea532ae674749d10472ffef1d329bba0d26d8e4290aa9d"
    ),
    "orca_projection_audit_script": (
        "1bb76ed3876df49b87084d4b9cf4aa6c8fa6642ce1920c38283f66a7ac94c391"
    ),
    "orca_projection_summary": (
        "46803dd0677ec44c85e49f2b16a29757310d1f25c0c8fc3e441a5dc18ea3ddb7"
    ),
    "orca_integrity_audit": (
        "f82d0c1edebfcdad4e4ec5b09d0c9d8bfe31f6b104d937310aca3f6adf233958"
    ),
}

# Set below from the frozen ORCA basis summary included with the source bundle.
EXPECTED_ORCA_BASIS_SUMMARY_SHA = "cf28193f57cba9bd25fe06d2259ab0138c4c2cd73436ff056ff5189aef70c17f"

# ---------------------------------------------------------------------
# Frozen experiment constants
# ---------------------------------------------------------------------

N = 62889
NR = 61502
DIM = 1024
DEPTH = 100
CANDIDATES = 101
N_FOLDS = 5
N_FUTURE_AVAILABLE = 17458
EXPECTED_FOLD_ROWS = {
    0: 12814,
    1: 12970,
    2: 12543,
    3: 12004,
    4: 12558,
}
EXPECTED_FOLD_DIALOGUES = {
    0: 335,
    1: 333,
    2: 332,
    3: 332,
    4: 334,
}

BATCH_SIZE = 128
MAX_EPOCHS = 20
MODEL_SEED = 42
LR = 1e-4
BETA = 8.0

CONDITIONS = {
    "RANK_ONLY_FULL_MATCHED": {
        "model_cls": RankOnlyFullMatchedModel,
        "uses_future": False,
        "beta": None,
    },
    "RANK_ONLY_NO_PRODUCT_MATCHED": {
        "model_cls": RankOnlyNoProductMatchedModel,
        "uses_future": False,
        "beta": None,
    },
    "R_FUTURE_FULL": {
        "model_cls": RFutureFullMatchedModel,
        "uses_future": True,
        "beta": BETA,
    },
    "R_FUTURE_NO_PRODUCT": {
        "model_cls": RFutureNoProductMatchedModel,
        "uses_future": True,
        "beta": BETA,
    },
    "QR_FUTURE_FULL": {
        "model_cls": QRFutureFullMatchedModel,
        "uses_future": True,
        "beta": BETA,
    },
    "QR_FUTURE_NO_PRODUCT": {
        "model_cls": QRFutureNoProductMatchedModel,
        "uses_future": True,
        "beta": BETA,
    },
}

METADATA_FILES = {
    "master": MASTER_FILE,
    "fold": FOLD_FILE,
    "future_mask": MASK_FILE,
    "reservoir": RESERVOIR_FILE,
    "model_code": MODEL_CODE,
    "source_rank_only_trainer": SOURCE_RANK_ONLY_TRAINER_FILE,
    "source_future_matched_trainer": SOURCE_FUTURE_MATCHED_TRAINER_FILE,
    "orca_projection_script": ORCA_PROJECTION_SCRIPT,
    "orca_projection_audit_script": ORCA_PROJECTION_AUDIT_SCRIPT,
    "architecture_protocol": ARCH_PROTOCOL,
    "training_protocol": TRAIN_PROTOCOL,
    "s5_protocol": S5_PROTOCOL,
    "s7_protocol": S7_PROTOCOL,
    "s4_summary": S4_SUMMARY,
    "s4_oof": S4_OOF,
    "orca_basis_summary": ORCA_BASIS_SUMMARY,
    "orca_projection_summary": ORCA_PROJECTION_SUMMARY,
    "orca_integrity_audit": ORCA_INTEGRITY_AUDIT,
}

# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_slug(condition: str) -> str:
    return condition.lower()


def fold_projection_paths(fold: int) -> tuple[Path, Path, Path, Path]:
    fold_dir = ORCA_PROJECTION_ROOT / f"fold{fold}"
    query = fold_dir / "query_orca_k2_v1.npy"
    response = fold_dir / "response_orca_k2_v1.npy"
    manifest = fold_dir / "manifest.json"
    basis = ORCA_BASIS_ROOT / f"fold{fold}_orca_basis_k2_v1.npy"
    return query, response, manifest, basis


def condition_dir(condition: str) -> Path:
    return OUT_ROOT / "conditions" / condition_slug(condition)


def checkpoint_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "checkpoints" / f"fold{fold}_best_v1.pt"


def curve_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "curves" / f"fold{fold}_training_curve_v1.jsonl"


def fold_oof_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "oof_folds" / f"fold{fold}_oof_v1.npz"


def fold_summary_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "fold_summaries" / f"fold{fold}_summary_v1.json"


def fold_marker_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "fold_markers" / f"fold{fold}_complete_v1.json"


def condition_oof_path(condition: str) -> Path:
    return condition_dir(condition) / f"{condition_slug(condition)}_oof_scores_v1.npz"


def condition_summary_path(condition: str) -> Path:
    return condition_dir(condition) / f"{condition_slug(condition)}_training_summary_v1.json"


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def check_expected_hash(name: str, path: Path, expected: str) -> str:
    assert_file(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"FROZEN_SHA_MISMATCH: {name}: actual={actual}, expected={expected}"
        )
    print(f"HASH_PASS: {name}")
    return actual


def read_projection_summaries():
    basis_summary = json.loads(ORCA_BASIS_SUMMARY.read_text(encoding="utf-8"))
    projection_summary = json.loads(
        ORCA_PROJECTION_SUMMARY.read_text(encoding="utf-8")
    )
    integrity = json.loads(ORCA_INTEGRITY_AUDIT.read_text(encoding="utf-8"))

    if basis_summary.get("status") != "COMPLETED":
        raise RuntimeError("ORCA basis summary is not COMPLETED")
    if basis_summary.get("k") != 2:
        raise RuntimeError("ORCA basis dimension is not k=2")
    if basis_summary.get("source_selection") != "FROZEN_STEP_2F17B3":
        raise RuntimeError("ORCA source selection provenance mismatch")
    if projection_summary.get("status") != "COMPLETED":
        raise RuntimeError("ORCA projection summary is not COMPLETED")
    if integrity.get("status") != "COMPLETED" or integrity.get("folds_passed") != 5:
        raise RuntimeError("ORCA projection integrity audit is not 5/5 PASS")

    return basis_summary, projection_summary, integrity



def _array_content_sha256(
    path: Path,
    chunk_size: int = 64 * 1024 * 1024,
) -> str:
    """Hash contiguous NumPy array contents, excluding the .npy header."""
    arr = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if not arr.flags.c_contiguous:
        raise RuntimeError(
            f"ORCA projection is not C-contiguous: {path}"
        )

    digest = hashlib.sha256()
    view = memoryview(arr).cast("B")

    for start in range(0, view.nbytes, chunk_size):
        digest.update(view[start:start + chunk_size])

    return digest.hexdigest()


def validate_orca_artifacts(
    verify_large_hashes: bool = True,
) -> tuple[dict, dict, dict, dict]:
    basis_summary, projection_summary, integrity = read_projection_summaries()

    basis_summary_sha = sha256_file(ORCA_BASIS_SUMMARY)
    if (
        EXPECTED_ORCA_BASIS_SUMMARY_SHA != "BASIS_SUMMARY_SHA_PLACEHOLDER"
        and basis_summary_sha != EXPECTED_ORCA_BASIS_SUMMARY_SHA
    ):
        raise RuntimeError(
            "ORCA basis summary SHA mismatch: "
            f"{basis_summary_sha} != {EXPECTED_ORCA_BASIS_SUMMARY_SHA}"
        )
    print("HASH_PASS: orca_basis_summary")

    fold_hashes = {}
    for fold in range(N_FOLDS):
        query_path, response_path, manifest_path, basis_path = (
            fold_projection_paths(fold)
        )
        for p in (query_path, response_path, manifest_path, basis_path):
            assert_file(p)

        basis_meta = basis_summary["fold_results"][str(fold)]
        projection_meta = projection_summary["folds"][str(fold)]
        expected_basis = basis_meta["basis_sha256"]
        expected_q = projection_meta["query"]["array_sha256"]
        expected_r = projection_meta["response"]["array_sha256"]
        if basis_meta.get("source_count") != 5000 or basis_meta.get("k") != 2:
            raise RuntimeError(f"ORCA basis provenance mismatch fold={fold}")
        if basis_meta.get("source_mode") != "affective_user":
            raise RuntimeError(f"ORCA source mode mismatch fold={fold}")
        if projection_meta.get("basis_sha256") != expected_basis:
            raise RuntimeError(f"ORCA projection basis linkage mismatch fold={fold}")

        basis_hash = hashlib.sha256(np.ascontiguousarray(np.load(basis_path, allow_pickle=False)).tobytes(order="C")).hexdigest()
        if basis_hash != expected_basis:
            raise RuntimeError(f"ORCA basis hash mismatch fold={fold}")
        query_hash = _array_content_sha256(query_path) if verify_large_hashes else None
        response_hash = _array_content_sha256(response_path) if verify_large_hashes else None
        if verify_large_hashes:
            if query_hash != expected_q:
                raise RuntimeError(f"ORCA query projection hash mismatch fold={fold}")
            if response_hash != expected_r:
                raise RuntimeError(f"ORCA response projection hash mismatch fold={fold}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("basis_sha256") != expected_basis:
            raise RuntimeError(f"ORCA manifest basis hash mismatch fold={fold}")
        if manifest.get("query", {}).get("array_sha256") != expected_q:
            raise RuntimeError(f"ORCA manifest query hash mismatch fold={fold}")
        if manifest.get("response", {}).get("array_sha256") != expected_r:
            raise RuntimeError(f"ORCA manifest response hash mismatch fold={fold}")
        if manifest.get("projection_method") != "x - (xB)B.T":
            raise RuntimeError(f"ORCA projection method mismatch fold={fold}")
        if manifest.get("post_projection_normalization") != "L2":
            raise RuntimeError(f"ORCA normalization mismatch fold={fold}")
        integrity_fold = integrity.get("results", {}).get(str(fold), {})
        if integrity_fold.get("status") != "PASS":
            raise RuntimeError(f"ORCA integrity fold status is not PASS: {fold}")
        if not integrity_fold.get("basis_hash_pass", False):
            raise RuntimeError(f"ORCA integrity basis hash did not pass: {fold}")
        if integrity_fold.get("query", {}).get("sha256") != expected_q:
            raise RuntimeError(f"ORCA integrity query hash mismatch: {fold}")
        if integrity_fold.get("response", {}).get("sha256") != expected_r:
            raise RuntimeError(f"ORCA integrity response hash mismatch: {fold}")
        formula_errors = [
            float(integrity_fold.get("query_formula_max_abs_diff", float("inf"))),
            float(integrity_fold.get("response_formula_max_abs_diff", float("inf"))),
        ]
        if max(formula_errors) >= 2e-6:
            raise RuntimeError(f"ORCA formula audit tolerance exceeded fold={fold}")

        q = np.load(query_path, mmap_mode="r", allow_pickle=False)
        r = np.load(response_path, mmap_mode="r", allow_pickle=False)
        b = np.load(basis_path, allow_pickle=False)

        if q.shape != (N, DIM) or q.dtype != np.float32:
            raise RuntimeError(f"ORCA query shape/dtype mismatch fold={fold}")
        if r.shape != (NR, DIM) or r.dtype != np.float32:
            raise RuntimeError(f"ORCA response shape/dtype mismatch fold={fold}")
        if b.shape != (DIM, 2) or b.dtype != np.float32:
            raise RuntimeError(f"ORCA basis shape/dtype mismatch fold={fold}")
        if list(q.shape) != manifest["query"]["shape"]:
            raise RuntimeError(f"ORCA manifest query shape mismatch fold={fold}")
        if list(r.shape) != manifest["response"]["shape"]:
            raise RuntimeError(f"ORCA manifest response shape mismatch fold={fold}")
        if os.path.getsize(query_path) != manifest["query"]["file_size"]:
            raise RuntimeError(f"ORCA manifest query file size mismatch fold={fold}")
        if os.path.getsize(response_path) != manifest["response"]["file_size"]:
            raise RuntimeError(f"ORCA manifest response file size mismatch fold={fold}")

        gram_error = float(
            np.max(np.abs(b.T @ b - np.eye(2, dtype=np.float32)))
        )
        if gram_error > 2e-5:
            raise RuntimeError(f"ORCA basis orthogonality failure fold={fold}")

        rng = np.random.default_rng(21700 + fold)
        qi = np.sort(rng.choice(N, size=128, replace=False))
        ri = np.sort(rng.choice(NR, size=128, replace=False))
        q_norm_error = float(
            np.max(np.abs(np.linalg.norm(np.asarray(q[qi]), axis=1) - 1.0))
        )
        r_norm_error = float(
            np.max(np.abs(np.linalg.norm(np.asarray(r[ri]), axis=1) - 1.0))
        )
        if max(q_norm_error, r_norm_error) > 2e-5:
            raise RuntimeError(f"ORCA projection norm failure fold={fold}")

        fold_hashes[str(fold)] = {
            "basis_sha256": basis_hash,
            "query_projection_sha256": expected_q,
            "response_projection_sha256": expected_r,
            "manifest_sha256": sha256_file(manifest_path),
            "basis_orthogonality_max_abs_error": gram_error,
            "query_sample_max_norm_error": q_norm_error,
            "response_sample_max_norm_error": r_norm_error,
            "query_shape": list(q.shape),
            "response_shape": list(r.shape),
        }
        print(
            f"ORCA_FOLD_{fold}_PASS "
            f"basis={basis_hash[:12]} "
            f"query={expected_q[:12]} "
            f"response={expected_r[:12]}"
        )
        del q, r, b

    return basis_summary, projection_summary, integrity, fold_hashes


def load_and_validate_base_inputs(verify_hashes: bool = True):
    paths = {
        "raw_query_embedding": RAW_Q_FILE,
        "raw_response_embedding": RAW_R_FILE,
        **METADATA_FILES,
    }

    actual_sha = {}
    for key, path in paths.items():
        assert_file(path)
        if verify_hashes:
            expected = EXPECTED_BASE_SHA.get(key)
            if expected is not None:
                actual_sha[key] = check_expected_hash(key, path, expected)
            else:
                actual_sha[key] = sha256_file(path)
        else:
            actual_sha[key] = sha256_file(path)

    raw_q = np.load(RAW_Q_FILE, mmap_mode="r", allow_pickle=False)
    raw_r = np.load(RAW_R_FILE, mmap_mode="r", allow_pickle=False)
    fold_id = np.load(FOLD_FILE, allow_pickle=False)
    master = load_jsonl(MASTER_FILE)

    mask_data = np.load(MASK_FILE, allow_pickle=False)
    if "future_loss_eligible_by_fold" not in mask_data:
        raise RuntimeError("Future mask missing future_loss_eligible_by_fold")
    if "future_available" not in mask_data:
        raise RuntimeError("Future mask missing future_available")
    if "future_label" not in mask_data:
        raise RuntimeError("Future mask missing future_label")
    future_eligible = np.asarray(
        mask_data["future_loss_eligible_by_fold"], dtype=np.bool_
    )
    future_available = np.asarray(mask_data["future_available"], dtype=np.bool_)
    mask_future_labels = np.asarray(mask_data["future_label"], dtype=np.int64)
    mask_fold_id = np.asarray(mask_data["fold_id"], dtype=np.int64)
    mask_gold_rows = np.asarray(mask_data["gold_response_row"], dtype=np.int64)

    reservoir = np.load(RESERVOIR_FILE, allow_pickle=False)
    if "candidate_response_row" not in reservoir or "gold_response_row" not in reservoir:
        raise RuntimeError("Reservoir schema mismatch")
    negative_rows = np.asarray(
        reservoir["candidate_response_row"], dtype=np.int32
    )
    gold_rows = np.asarray(reservoir["gold_response_row"], dtype=np.int32)

    if raw_q.shape != (N, DIM) or raw_q.dtype != np.float32:
        raise RuntimeError(f"Raw query embedding mismatch: {raw_q.shape}, {raw_q.dtype}")
    if raw_r.shape != (NR, DIM) or raw_r.dtype != np.float32:
        raise RuntimeError(
            f"Raw response embedding mismatch: {raw_r.shape}, {raw_r.dtype}"
        )
    if len(master) != N:
        raise RuntimeError(f"Master row count mismatch: {len(master)}")
    if fold_id.shape != (N,):
        raise RuntimeError(f"Fold assignment shape mismatch: {fold_id.shape}")
    if future_eligible.shape != (N_FOLDS, N):
        raise RuntimeError(f"Future eligibility shape mismatch: {future_eligible.shape}")
    if future_available.shape != (N,) or mask_future_labels.shape != (N,):
        raise RuntimeError("Future label/mask row alignment mismatch")
    if mask_fold_id.shape != (N,) or not np.array_equal(mask_fold_id, fold_id):
        raise RuntimeError("Future mask fold_id alignment mismatch")
    if mask_gold_rows.shape != (N,) or not np.array_equal(mask_gold_rows, gold_rows):
        raise RuntimeError("Future mask gold_response_row alignment mismatch")
    if negative_rows.shape != (N, DEPTH) or gold_rows.shape != (N,):
        raise RuntimeError("Reservoir row shape mismatch")

    if not np.array_equal(np.unique(fold_id), np.arange(N_FOLDS)):
        raise RuntimeError(f"Unexpected fold values: {np.unique(fold_id)}")
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
    if len(set(dialogue_ids)) != 1666:
        raise RuntimeError("Dialogue count mismatch")
    dialogue_fold = {}
    for did, fid in zip(dialogue_ids, fold_id):
        previous = dialogue_fold.setdefault(did, int(fid))
        if previous != int(fid):
            raise RuntimeError(f"Dialogue split across folds: {did}")

    if int(future_available.sum()) != N_FUTURE_AVAILABLE:
        raise RuntimeError(
            f"Future available count mismatch: {int(future_available.sum())}"
        )
    master_future_available = np.asarray(
        [bool(row["future_value_available"]) for row in master],
        dtype=np.bool_,
    )
    future_labels = np.full(N, -2, dtype=np.int64)
    for row_index, row in enumerate(master):
        if row["future_value_available"]:
            future_labels[row_index] = int(row["future_value_label"])
    if not np.array_equal(future_available, master_future_available):
        raise RuntimeError("Future availability mask does not match frozen master")
    if not np.array_equal(
        mask_future_labels[future_available],
        future_labels[future_available],
    ):
        raise RuntimeError("Available future labels do not match frozen master")
    if np.any(future_eligible & ~future_available[None, :]):
        raise RuntimeError("Future eligibility includes unavailable future labels")
    for fold in range(N_FOLDS):
        heldout = fold_id == fold
        if np.any(future_eligible[fold, heldout]):
            raise RuntimeError(
                f"Future eligibility leaks into heldout fold {fold}"
            )

    # The frozen S3A master index is the training ranking/future index.
    # Unlike the upstream target-row table, this master index is not required
    # to carry a per-row `split` field. The exact master SHA256 is checked
    # above, and train-only membership is established by the frozen source
    # provenance and the independent upstream row-alignment audit.
    split_presence = ["split" in row for row in master]
    if any(split_presence) and not all(split_presence):
        raise RuntimeError("Inconsistent master schema: split field is only partially present")
    if all(split_presence):
        invalid_split = [
            (i, row.get("split"))
            for i, row in enumerate(master)
            if str(row.get("split", "")).strip().lower() not in {"train", "training"}
        ]
        if invalid_split:
            raise RuntimeError(
                f"Non-train target found in master index: {invalid_split[:10]}"
            )
        print("MASTER_SPLIT_FIELD=PASS")
    else:
        print("MASTER_SPLIT_FIELD=ABSENT")
        print("MASTER_TRAIN_ONLY_PROVENANCE=FROZEN_SHA_AND_UPSTREAM_AUDIT")

    for i, row in enumerate(master):
        if bool(row.get("future_value_available")) != bool(future_available[i]):
            raise RuntimeError(f"Future availability alignment mismatch at row {i}")
        if future_available[i]:
            if int(row["future_value_label"]) != int(future_labels[i]):
                raise RuntimeError(f"Future label alignment mismatch at row {i}")

    fold_counts = {
        str(f): int(np.sum(fold_id == f)) for f in range(N_FOLDS)
    }
    dialogue_counts = {
        str(f): len(set(dialogue_ids[fold_id == f])) for f in range(N_FOLDS)
    }
    for fold in range(N_FOLDS):
        if fold_counts[str(fold)] != EXPECTED_FOLD_ROWS[fold]:
            raise RuntimeError(f"Frozen fold row count mismatch: fold={fold}")
        if dialogue_counts[str(fold)] != EXPECTED_FOLD_DIALOGUES[fold]:
            raise RuntimeError(f"Frozen fold dialogue count mismatch: fold={fold}")
    valid_future_labels = future_labels[future_available]
    if not np.isin(valid_future_labels, np.asarray([-2, -1, 0, 1, 2])).all():
        raise RuntimeError("Future ordinal labels outside frozen [-2, 2] range")

    print("BASE_INPUT_ALIGNMENT=PASS")
    print("QUERY_ROWS =", N)
    print("RESPONSE_ROWS =", NR)
    print("DIALOGUES =", len(set(dialogue_ids)))
    print("FUTURE_AVAILABLE =", int(future_available.sum()))
    print("FOLD_COUNTS =", fold_counts)
    print("DIALOGUE_COUNTS =", dialogue_counts)
    print("DEV_PAYLOAD_READ = NO")
    print("TEST_PAYLOAD_READ = NO")

    return {
        "actual_sha": actual_sha,
        "fold_id": fold_id.astype(np.int64),
        "master": master,
        "dialogue_ids": dialogue_ids,
        "negative_rows": negative_rows,
        "gold_rows": gold_rows,
        "future_eligible": future_eligible,
        "future_available": future_available,
        "future_labels": future_labels,
        "fold_counts": fold_counts,
        "dialogue_counts": dialogue_counts,
    }


def build_batch(
    rows: np.ndarray,
    query_projection: np.ndarray,
    response_projection: np.ndarray,
    negative_rows: np.ndarray,
    gold_rows: np.ndarray,
    device: torch.device,
    future_labels: np.ndarray | None = None,
    future_eligible: np.ndarray | None = None,
):
    rows = np.asarray(rows, dtype=np.int64)
    q_np = np.array(query_projection[rows], dtype=np.float32, copy=True)
    candidate_idx = np.concatenate(
        [gold_rows[rows, None], negative_rows[rows]], axis=1
    )
    if candidate_idx.shape != (len(rows), CANDIDATES):
        raise RuntimeError(f"Candidate shape mismatch: {candidate_idx.shape}")
    candidate_np = np.array(
        response_projection[candidate_idx], dtype=np.float32, copy=True
    )
    q = torch.from_numpy(q_np).to(device)
    candidates = torch.from_numpy(candidate_np).to(device)

    if future_labels is None or future_eligible is None:
        return q, candidates, None, None

    labels = torch.from_numpy(
        future_labels[rows].copy()
    ).to(device)
    eligibility = torch.from_numpy(
        future_eligible[rows].copy()
    ).to(device)
    return q, candidates, labels, eligibility


def evaluate_validation_ce(
    model,
    rows,
    query_projection,
    response_projection,
    negative_rows,
    gold_rows,
    device,
):
    model.eval()
    loss_sum = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start:start + BATCH_SIZE]
            q, candidates, _, _ = build_batch(
                batch_rows,
                query_projection,
                response_projection,
                negative_rows,
                gold_rows,
                device,
            )
            scores = model.rank_scores(q, candidates)
            loss = compute_listwise_rank_loss(scores, gold_position=0)
            b = len(batch_rows)
            loss_sum += float(loss.item()) * b
            count += b
    if count != len(rows):
        raise RuntimeError("Validation row accounting mismatch")
    return loss_sum / count


def initial_model_check(
    query_projection,
    response_projection,
    negative_rows,
    gold_rows,
):
    device = torch.device("cpu")
    probe_rows = np.arange(8, dtype=np.int64)
    q, candidates, _, _ = build_batch(
        probe_rows,
        query_projection,
        response_projection,
        negative_rows,
        gold_rows,
        device,
    )
    reference = None
    results = {}
    for condition, spec in CONDITIONS.items():
        seed_all(MODEL_SEED)
        model = spec["model_cls"]().to(device)
        model.eval()
        with torch.inference_mode():
            zq = model.ranker.adapt_query(q)
            zr = model.ranker.adapt_response(candidates)
            expected = -torch.abs(zq[:, None, :] - zr).sum(-1) / (DIM ** 0.5)
            actual = model.rank_scores(q, candidates)
            diff = float((actual - expected).abs().max().item())
            if diff > 1e-6:
                raise RuntimeError(
                    f"{condition} initial negative-L1 mismatch: {diff}"
                )
            if reference is None:
                reference = actual.detach().clone()
                cross_diff = 0.0
            else:
                cross_diff = float((actual - reference).abs().max().item())
                if cross_diff > 1e-6:
                    raise RuntimeError(
                        f"{condition} initial ranking differs across conditions: "
                        f"{cross_diff}"
                    )
            if spec["uses_future"]:
                logits = model.future_logits(q, candidates[:, 0, :])
                if logits.shape != (len(probe_rows), 4):
                    raise RuntimeError(
                        f"{condition} future logit shape mismatch: {tuple(logits.shape)}"
                    )
                if not torch.isfinite(logits).all():
                    raise RuntimeError(f"{condition} future logits are non-finite")
        results[condition] = {
            "initial_negative_l1_max_abs_error": diff,
            "cross_condition_initial_score_max_abs_error": cross_diff,
            "future_logits_shape": (
                [len(probe_rows), 4] if spec["uses_future"] else None
            ),
        }
        del model
    print("CPU_INITIALIZATION_CHECK=PASS")
    for condition, result in results.items():
        print(
            condition,
            "initial_error=",
            result["initial_negative_l1_max_abs_error"],
            "cross_condition_error=",
            result["cross_condition_initial_score_max_abs_error"],
        )
    return results


def collect_preflight():
    print("=" * 88)
    print("STEP 2F18I: ORCA-FSAR SIX-CONDITION CPU PREFLIGHT")
    print("=" * 88)
    base = load_and_validate_base_inputs(verify_hashes=True)

    for key, expected in EXPECTED_BASE_SHA.items():
        if base["actual_sha"].get(key) != expected:
            raise RuntimeError(f"Base hash not frozen: {key}")

    basis_summary, projection_summary, integrity, fold_hashes = (
        validate_orca_artifacts(verify_large_hashes=True)
    )

    init_results = {}
    for fold in range(N_FOLDS):
        q_path, r_path, _, _ = fold_projection_paths(fold)
        q_fold = np.load(q_path, mmap_mode="r", allow_pickle=False)
        r_fold = np.load(r_path, mmap_mode="r", allow_pickle=False)
        init_results[str(fold)] = initial_model_check(
            q_fold,
            r_fold,
            base["negative_rows"],
            base["gold_rows"],
        )
        del q_fold, r_fold

    actual_sha = dict(base["actual_sha"])
    actual_sha["orca_basis_summary"] = sha256_file(ORCA_BASIS_SUMMARY)
    actual_sha["orca_projection_summary"] = sha256_file(ORCA_PROJECTION_SUMMARY)
    actual_sha["orca_integrity_audit"] = sha256_file(ORCA_INTEGRITY_AUDIT)

    protocol = {
        "artifact": "ORCA-FSAR S12C Six-Condition Five-Fold Training Protocol V1",
        "project": "PsychResponse_FSAR",
        "status": "FROZEN_BEFORE_ANY_PARAMETER_UPDATE",
        "scope": "TRAIN_INTERNAL_5FOLD_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_script": str(Path(__file__).resolve()),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "model_code_sha256": actual_sha["model_code"],
        "training_logic_sources": {
            "rank_only_trainer_sha256": actual_sha["source_rank_only_trainer"],
            "future_matched_ablation_trainer_sha256": actual_sha["source_future_matched_trainer"],
            "orca_projection_script_sha256": actual_sha["orca_projection_script"],
            "orca_projection_audit_script_sha256": actual_sha["orca_projection_audit_script"],
        },
        "input_sha256": actual_sha,
        "orca": {
            "method": "FOLD_SPECIFIC_ORCA_PROJECTION",
            "k": 2,
            "projection_formula": "z = normalize(x - (xB_f)B_f.T)",
            "basis_source": "FROZEN_STEP_2F17B3",
            "basis_summary_sha256": actual_sha["orca_basis_summary"],
            "projection_summary_sha256": actual_sha["orca_projection_summary"],
            "integrity_audit_sha256": actual_sha["orca_integrity_audit"],
            "fold_artifacts": fold_hashes,
            "same_fold_projection_for_train_and_validation": True,
            "original_bge_embeddings_modified": False,
        },
        "data": {
            "ranking_rows": N,
            "unique_responses": NR,
            "dialogues": 1666,
            "folds": N_FOLDS,
            "candidate_count": CANDIDATES,
            "gold_position": 0,
            "hard_negatives": DEPTH,
            "future_available_rows": N_FUTURE_AVAILABLE,
            "fold_row_counts": base["fold_counts"],
            "fold_dialogue_counts": base["dialogue_counts"],
            "split_unit": "dialogue",
        },
        "conditions": {
            condition: {
                "model_class": spec["model_cls"].__name__,
                "uses_future_supervision": spec["uses_future"],
                "beta": spec["beta"],
            }
            for condition, spec in CONDITIONS.items()
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LR,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "batch_size_queries": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "mixed_precision": False,
            "tf32": False,
            "gradient_accumulation": 1,
            "gradient_clipping": None,
            "lr_scheduler": None,
            "model_initialization_seed": MODEL_SEED,
            "epoch_shuffle_seed_formula": "42000 + fold_id * 100 + epoch_id",
            "same_initialization_seed_across_conditions": True,
            "same_query_order_within_fold_across_conditions": True,
            "future_beta": BETA,
        },
        "loss": {
            "rank_only": "LISTWISE_SOFTMAX_CROSS_ENTROPY",
            "future": (
                "MEAN_LISTWISE_RANK_CE + 8 * "
                "MEAN_OVER_FULL_QUERY_BATCH(MASK * MEAN_4_THRESHOLD_BCE)"
            ),
            "future_labels": "STRICT_DUAL_CONSENSUS_VALUE_ONLY",
            "future_loss_on_gold_pair_only": True,
            "future_loss_on_negatives": False,
            "future_oversampling": False,
            "future_class_weight": False,
            "ordinal_thresholds": [-2, -1, 0, 1],
            "fold_specific_future_eligibility_mask": True,
        },
        "checkpoint_selection": {
            "metric": "VALIDATION_RANKING_CROSS_ENTROPY",
            "direction": "MINIMIZE",
            "eligible_epochs": [1, MAX_EPOCHS],
            "tie_break": "EARLIEST_EPOCH",
            "future_validation_loss_used": False,
            "retrieval_metric_used": False,
        },
        "oof": {
            "ranking_scores_saved": True,
            "ranking_ce_saved": True,
            "future_logits_saved_for_future_conditions": True,
            "fold_id_saved": True,
        },
        "evaluation_policy": {
            "dev_payload_used": False,
            "test_payload_used": False,
            "retrieval_metrics_computed_during_training": False,
            "future_metrics_computed_during_training": False,
        },
        "preflight": {
            "base_input_alignment": "PASS",
            "orca_five_fold_projection_hashes": "PASS",
            "orca_five_fold_projection_norms": "PASS",
            "orca_basis_orthogonality": "PASS",
            "cpu_initialization_check": "PASS",
            "initialization_results": init_results,
            "cuda_used": False,
            "parameter_updates": False,
        },
        "output": {
            "root": str(OUT_ROOT),
            "overwrite_existing": False,
            "atomic_artifacts": True,
            "fold_level_resume": True,
        },
    }
    return protocol


def prepare():
    if OUT_ROOT.exists():
        raise RuntimeError(
            f"Output directory already exists; refusing overwrite: {OUT_ROOT}"
        )
    protocol = collect_preflight()
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    atomic_json(PROTOCOL_FILE, protocol)
    print("=" * 88)
    print("STEP_2F18I_PREPARE_COMPLETE")
    print("PROTOCOL =", PROTOCOL_FILE)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_FILE))
    print("CUDA_USED = NO")
    print("TRAINING = NO")
    print("PARAMETER_UPDATES = NO")
    print("FILES_MODIFIED = only new ORCA-FSAR protocol/output directory")
    print("=" * 88)


def verify_frozen_protocol():
    assert_file(PROTOCOL_FILE)
    protocol = json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_ANY_PARAMETER_UPDATE":
        raise RuntimeError("ORCA-FSAR protocol status mismatch")
    if protocol.get("training_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Training script changed after protocol freeze")

    # Re-run all frozen hashes and row/projection checks before training.
    fresh = collect_preflight()
    if fresh["input_sha256"] != protocol.get("input_sha256"):
        raise RuntimeError("Frozen input SHA values changed after prepare")
    if fresh["orca"]["fold_artifacts"] != protocol.get("orca", {}).get("fold_artifacts"):
        raise RuntimeError("Frozen ORCA fold artifacts changed after prepare")
    return protocol


def atomic_jsonl_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_oof_fold(
    model,
    rows,
    query_projection,
    response_projection,
    negative_rows,
    gold_rows,
    device,
    uses_future,
):
    scores_out = np.full((len(rows), CANDIDATES), np.nan, dtype=np.float32)
    nll_out = np.full(len(rows), np.nan, dtype=np.float32)
    future_out = (
        np.full((len(rows), 4), np.nan, dtype=np.float32)
        if uses_future
        else None
    )

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            local_rows = rows[start:start + BATCH_SIZE]
            q, candidates, _, _ = build_batch(
                local_rows,
                query_projection,
                response_projection,
                negative_rows,
                gold_rows,
                device,
            )
            scores = model.rank_scores(q, candidates)
            nll = -scores[:, 0] + torch.logsumexp(scores, dim=1)
            scores_out[start:start + len(local_rows)] = (
                scores.detach().cpu().numpy().astype(np.float32)
            )
            nll_out[start:start + len(local_rows)] = (
                nll.detach().cpu().numpy().astype(np.float32)
            )
            if uses_future:
                logits = model.future_logits(q, candidates[:, 0, :])
                future_out[start:start + len(local_rows)] = (
                    logits.detach().cpu().numpy().astype(np.float32)
                )

    if not np.isfinite(scores_out).all() or not np.isfinite(nll_out).all():
        raise RuntimeError("Non-finite OOF ranking output")
    if future_out is not None and not np.isfinite(future_out).all():
        raise RuntimeError("Non-finite OOF future logits")
    return scores_out, nll_out, future_out


def validate_completed_fold(condition: str, fold: int, protocol: dict) -> dict:
    marker_path = fold_marker_path(condition, fold)
    assert_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "FOLD_COMPLETE":
        raise RuntimeError(f"Resume marker status mismatch: {condition}, fold={fold}")
    if marker.get("condition") != condition or marker.get("fold") != fold:
        raise RuntimeError(f"Resume marker identity mismatch: {condition}, fold={fold}")
    if marker.get("protocol_sha256") != sha256_file(PROTOCOL_FILE):
        raise RuntimeError(f"Resume marker protocol mismatch: {condition}, fold={fold}")

    paths = {
        "checkpoint": checkpoint_path(condition, fold),
        "curve": curve_path(condition, fold),
        "oof": fold_oof_path(condition, fold),
        "summary": fold_summary_path(condition, fold),
    }
    for key, path in paths.items():
        assert_file(path)
        actual = sha256_file(path)
        if actual != marker.get("artifact_sha256", {}).get(key):
            raise RuntimeError(
                f"Resume artifact hash mismatch: {condition}, fold={fold}, {key}"
            )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError(f"Resume fold summary not PASS: {condition}, fold={fold}")
    if summary.get("condition") != condition or summary.get("fold") != fold:
        raise RuntimeError(f"Resume fold summary identity mismatch: {condition}, fold={fold}")
    return summary


def load_completed_fold_oof(condition: str, fold: int):
    path = fold_oof_path(condition, fold)
    with np.load(path, allow_pickle=False) as data:
        rows = np.asarray(data["row_indices"], dtype=np.int64)
        scores = np.asarray(data["candidate_scores"], dtype=np.float32)
        nll = np.asarray(data["ranking_ce"], dtype=np.float32)
        future = (
            np.asarray(data["future_logits"], dtype=np.float32)
            if "future_logits" in data.files
            else None
        )
    return rows, scores, nll, future


def train_one_fold(
    condition: str,
    fold: int,
    spec: dict,
    base: dict,
    protocol: dict,
):
    print("\n" + "=" * 84)
    print(f"CONDITION={condition} FOLD={fold}")
    print("=" * 84, flush=True)

    query_path, response_path, _, _ = fold_projection_paths(fold)
    query_projection = np.load(query_path, mmap_mode="r", allow_pickle=False)
    response_projection = np.load(response_path, mmap_mode="r", allow_pickle=False)

    fold_id = base["fold_id"]
    master = base["master"]
    dialogue_ids = base["dialogue_ids"]
    negative_rows = base["negative_rows"]
    gold_rows = base["gold_rows"]
    future_eligible = base["future_eligible"]
    future_labels = base["future_labels"]

    train_rows = np.flatnonzero(fold_id != fold)
    valid_rows = np.flatnonzero(fold_id == fold)

    train_dialogues = set(dialogue_ids[train_rows])
    valid_dialogues = set(dialogue_ids[valid_rows])
    if train_dialogues & valid_dialogues:
        raise RuntimeError(f"Dialogue leakage in fold {fold}")

    eligible_n = int(future_eligible[fold, train_rows].sum()) if spec["uses_future"] else 0

    seed_all(MODEL_SEED)
    model = spec["model_cls"]().to(torch.device("cuda"))
    device = torch.device("cuda")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    probe_rows = valid_rows[:min(128, len(valid_rows))]
    q_probe, c_probe, _, _ = build_batch(
        probe_rows,
        query_projection,
        response_projection,
        negative_rows,
        gold_rows,
        device,
    )
    with torch.inference_mode():
        zq = model.ranker.adapt_query(q_probe)
        zr = model.ranker.adapt_response(c_probe)
        expected_initial = -torch.abs(zq[:, None, :] - zr).sum(-1) / (DIM ** 0.5)
        actual_initial = model.rank_scores(q_probe, c_probe)
        initial_diff = float((actual_initial - expected_initial).abs().max().item())
    if initial_diff > 1e-6:
        raise RuntimeError(
            f"{condition} fold={fold} initial ranking mismatch: {initial_diff}"
        )

    best_epoch = None
    best_val_ce = None
    best_train_rank_ce = None
    best_train_future_loss = None
    best_train_total_loss = None
    fold_start = time.time()

    curve_final = curve_path(condition, fold)
    curve_tmp = curve_final.with_name(curve_final.name + ".tmp")
    curve_final.parent.mkdir(parents=True, exist_ok=True)
    if curve_final.exists() or curve_tmp.exists():
        raise RuntimeError(
            f"Unexpected curve artifact before fold training: {curve_final}"
        )

    try:
        with curve_tmp.open("w", encoding="utf-8") as curve_handle:
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

                for start in range(0, len(shuffled_train), BATCH_SIZE):
                    batch_rows = shuffled_train[start:start + BATCH_SIZE]
                    if spec["uses_future"]:
                        q, candidates, labels, eligibility = build_batch(
                            batch_rows,
                            query_projection,
                            response_projection,
                            negative_rows,
                            gold_rows,
                            device,
                            future_labels=future_labels,
                            future_eligible=future_eligible[fold],
                        )
                    else:
                        q, candidates, _, _ = build_batch(
                            batch_rows,
                            query_projection,
                            response_projection,
                            negative_rows,
                            gold_rows,
                            device,
                        )
                        labels = eligibility = None

                    optimizer.zero_grad(set_to_none=True)
                    scores = model.rank_scores(q, candidates)
                    rank_loss = compute_listwise_rank_loss(
                        scores, gold_position=0
                    )

                    if spec["uses_future"]:
                        future_logits = model.future_logits(
                            q, candidates[:, 0, :]
                        )
                        future_loss = compute_masked_future_loss(
                            future_logits, labels, eligibility
                        )
                        total_loss = rank_loss + BETA * future_loss
                    else:
                        future_loss = torch.zeros((), device=device)
                        total_loss = rank_loss

                    if not (
                        torch.isfinite(rank_loss)
                        and torch.isfinite(future_loss)
                        and torch.isfinite(total_loss)
                    ):
                        raise RuntimeError(
                            f"Non-finite loss: {condition}, fold={fold}, epoch={epoch}"
                        )

                    total_loss.backward()
                    optimizer.step()

                    b = len(batch_rows)
                    rank_sum += float(rank_loss.item()) * b
                    future_sum += float(future_loss.item()) * b
                    total_sum += float(total_loss.item()) * b
                    train_count += b

                if train_count != len(train_rows):
                    raise RuntimeError("Training row accounting mismatch")

                train_rank_ce = rank_sum / train_count
                train_future_loss = future_sum / train_count
                train_total_loss = total_sum / train_count
                val_ce = evaluate_validation_ce(
                    model,
                    valid_rows,
                    query_projection,
                    response_projection,
                    negative_rows,
                    gold_rows,
                    device,
                )
                epoch_seconds = time.time() - epoch_start

                record = {
                    "condition": condition,
                    "fold": fold,
                    "epoch": epoch,
                    "epoch_shuffle_seed": epoch_seed,
                    "training_rows": int(len(train_rows)),
                    "validation_rows": int(len(valid_rows)),
                    "future_eligible_training_rows": eligible_n,
                    "train_ranking_ce": float(train_rank_ce),
                    "train_future_loss": float(train_future_loss),
                    "train_total_loss": float(train_total_loss),
                    "validation_ranking_ce": float(val_ce),
                    "epoch_seconds": float(epoch_seconds),
                }
                atomic_jsonl_record(curve_handle, record)

                improved = best_val_ce is None or val_ce < best_val_ce
                if improved:
                    best_epoch = int(epoch)
                    best_val_ce = float(val_ce)
                    best_train_rank_ce = float(train_rank_ce)
                    best_train_future_loss = float(train_future_loss)
                    best_train_total_loss = float(train_total_loss)

                    state_cpu = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    checkpoint = {
                        "artifact": f"ORCA-FSAR {condition} Best Checkpoint V1",
                        "condition": condition,
                        "fold": fold,
                        "best_epoch": best_epoch,
                        "best_validation_ranking_ce": best_val_ce,
                        "training_ranking_ce_at_best_epoch": best_train_rank_ce,
                        "training_future_loss_at_best_epoch": best_train_future_loss,
                        "training_total_loss_at_best_epoch": best_train_total_loss,
                        "model_seed": MODEL_SEED,
                        "beta": spec["beta"],
                        "model_state_dict": state_cpu,
                        "input_space": "ORCA_K2_FOLD_SPECIFIC",
                        "orca_fold": fold,
                        "orca_basis_sha256": protocol["orca"]["fold_artifacts"][str(fold)]["basis_sha256"],
                        "orca_query_projection_sha256": protocol["orca"]["fold_artifacts"][str(fold)]["query_projection_sha256"],
                        "orca_response_projection_sha256": protocol["orca"]["fold_artifacts"][str(fold)]["response_projection_sha256"],
                        "future_supervision_used": bool(spec["uses_future"]),
                        "dev_payload_used": False,
                        "test_payload_used": False,
                    }
                    atomic_torch_save(checkpoint_path(condition, fold), checkpoint)

                print(
                    f"{condition} fold={fold} epoch={epoch:02d}/{MAX_EPOCHS} "
                    f"train_rank_ce={train_rank_ce:.8f} "
                    f"train_future={train_future_loss:.8f} "
                    f"val_rank_ce={val_ce:.8f} "
                    f"best_epoch={best_epoch} best_val={best_val_ce:.8f} "
                    f"seconds={epoch_seconds:.2f}",
                    flush=True,
                )
        os.replace(curve_tmp, curve_final)
    except Exception:
        # Leave the fold unmarked. --train --resume will safely restart this fold.
        raise

    if best_epoch is None or not checkpoint_path(condition, fold).is_file():
        raise RuntimeError(f"Missing best checkpoint: {condition}, fold={fold}")

    checkpoint = torch.load(
        checkpoint_path(condition, fold),
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("condition") != condition or checkpoint.get("fold") != fold:
        raise RuntimeError("Checkpoint identity mismatch")
    if checkpoint.get("input_space") != "ORCA_K2_FOLD_SPECIFIC":
        raise RuntimeError("Checkpoint input-space marker mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    reloaded_val_ce = evaluate_validation_ce(
        model,
        valid_rows,
        query_projection,
        response_projection,
        negative_rows,
        gold_rows,
        device,
    )
    if abs(reloaded_val_ce - best_val_ce) >= 1e-6:
        raise RuntimeError(
            f"Reloaded validation CE mismatch: {reloaded_val_ce} vs {best_val_ce}"
        )

    scores, nll, future_logits = write_oof_fold(
        model,
        valid_rows,
        query_projection,
        response_projection,
        negative_rows,
        gold_rows,
        device,
        spec["uses_future"],
    )

    oof_payload = {
        "row_indices": valid_rows.astype(np.int32),
        "candidate_scores": scores,
        "ranking_ce": nll,
        "fold_id": np.full(len(valid_rows), fold, dtype=np.int8),
        "gold_position": np.zeros(len(valid_rows), dtype=np.int8),
    }
    if spec["uses_future"]:
        oof_payload["future_logits"] = future_logits
        oof_payload["future_available"] = base["future_available"][valid_rows]
        oof_payload["future_value_label"] = base["future_labels"][valid_rows]

    atomic_npz(fold_oof_path(condition, fold), **oof_payload)

    fold_summary = {
        "artifact": "ORCA-FSAR Fold Training Summary V1",
        "status": "PASS",
        "condition": condition,
        "fold": fold,
        "input_space": "ORCA_K2_FOLD_SPECIFIC",
        "training_rows": int(len(train_rows)),
        "validation_rows": int(len(valid_rows)),
        "training_dialogues": len(train_dialogues),
        "validation_dialogues": len(valid_dialogues),
        "future_eligible_training_rows": eligible_n,
        "initial_max_abs_diff_vs_negative_scaled_l1": initial_diff,
        "best_epoch": best_epoch,
        "best_validation_ranking_ce": best_val_ce,
        "training_ranking_ce_at_best_epoch": best_train_rank_ce,
        "training_future_loss_at_best_epoch": best_train_future_loss,
        "training_total_loss_at_best_epoch": best_train_total_loss,
        "reloaded_validation_ranking_ce": float(reloaded_val_ce),
        "oof_rows": int(len(valid_rows)),
        "fold_seconds": float(time.time() - fold_start),
        "checkpoint_sha256": sha256_file(checkpoint_path(condition, fold)),
        "oof_sha256": sha256_file(fold_oof_path(condition, fold)),
        "curve_sha256": sha256_file(curve_final),
    }
    atomic_json(fold_summary_path(condition, fold), fold_summary)

    marker = {
        "status": "FOLD_COMPLETE",
        "condition": condition,
        "fold": fold,
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "artifact_sha256": {
            "checkpoint": sha256_file(checkpoint_path(condition, fold)),
            "curve": sha256_file(curve_final),
            "oof": sha256_file(fold_oof_path(condition, fold)),
            "summary": sha256_file(fold_summary_path(condition, fold)),
        },
    }
    atomic_json(fold_marker_path(condition, fold), marker)

    del optimizer, model, query_projection, response_projection
    torch.cuda.empty_cache()

    print(
        f"FOLD_COMPLETE_PASS condition={condition} fold={fold} "
        f"best_epoch={best_epoch} val_ce={best_val_ce:.8f}",
        flush=True,
    )
    return fold_summary


def aggregate_condition(condition: str, spec: dict, base: dict):
    all_scores = np.full((N, CANDIDATES), np.nan, dtype=np.float32)
    all_nll = np.full(N, np.nan, dtype=np.float32)
    all_future = (
        np.full((N, 4), np.nan, dtype=np.float32)
        if spec["uses_future"]
        else None
    )
    fold_summaries = {}

    for fold in range(N_FOLDS):
        summary = validate_completed_fold(condition, fold, {})
        fold_summaries[str(fold)] = summary
        rows, scores, nll, future = load_completed_fold_oof(condition, fold)
        if len(rows) != len(np.unique(rows)):
            raise RuntimeError(f"Duplicate OOF rows: {condition}, fold={fold}")
        if not np.all(base["fold_id"][rows] == fold):
            raise RuntimeError(f"OOF fold alignment mismatch: {condition}, fold={fold}")
        if scores.shape != (len(rows), CANDIDATES) or nll.shape != (len(rows),):
            raise RuntimeError(f"OOF shape mismatch: {condition}, fold={fold}")
        if np.isfinite(all_nll[rows]).any():
            raise RuntimeError(f"Overlapping OOF assignment: {condition}, fold={fold}")
        all_scores[rows] = scores
        all_nll[rows] = nll
        if spec["uses_future"]:
            if future is None or future.shape != (len(rows), 4):
                raise RuntimeError(f"Future OOF shape mismatch: {condition}, fold={fold}")
            all_future[rows] = future
        elif future is not None:
            raise RuntimeError(f"Unexpected future logits for {condition}")

    if not np.isfinite(all_scores).all() or not np.isfinite(all_nll).all():
        raise RuntimeError(f"Incomplete OOF arrays: {condition}")
    if all_future is not None and not np.isfinite(all_future).all():
        raise RuntimeError(f"Incomplete future OOF arrays: {condition}")

    weighted_fold_ce = sum(
        fold_summaries[str(f)]["best_validation_ranking_ce"]
        * fold_summaries[str(f)]["validation_rows"]
        for f in range(N_FOLDS)
    ) / N
    oof_mean_ce = float(all_nll.mean())
    if abs(oof_mean_ce - weighted_fold_ce) >= 1e-5:
        raise RuntimeError(
            f"OOF CE mismatch {condition}: {oof_mean_ce} vs {weighted_fold_ce}"
        )

    payload = {
        "candidate_scores": all_scores,
        "ranking_ce": all_nll,
        "fold_id": base["fold_id"].astype(np.int8),
        "gold_position": np.zeros(N, dtype=np.int8),
    }
    if spec["uses_future"]:
        payload["future_logits"] = all_future
        payload["future_available"] = base["future_available"]
        payload["future_value_label"] = base["future_labels"]

    atomic_npz(condition_oof_path(condition), **payload)

    summary = {
        "artifact": f"ORCA-FSAR {condition} Five-Fold Training Summary V1",
        "status": "PASS",
        "condition": condition,
        "input_space": "ORCA_K2_FOLD_SPECIFIC",
        "beta": spec["beta"],
        "ranking_rows": N,
        "unique_responses": NR,
        "candidate_count": CANDIDATES,
        "folds": N_FOLDS,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "Adam",
        "learning_rate": LR,
        "weight_decay": 0.0,
        "checkpoint_selection": "MIN_VALIDATION_RANKING_CROSS_ENTROPY",
        "future_supervision_used": bool(spec["uses_future"]),
        "future_available_rows": N_FUTURE_AVAILABLE if spec["uses_future"] else 0,
        "fold_summary": fold_summaries,
        "oof_mean_ranking_ce": oof_mean_ce,
        "weighted_fold_validation_ce": weighted_fold_ce,
        "retrieval_metrics_computed": False,
        "future_metrics_computed": False,
        "dev_payload_used": False,
        "test_payload_used": False,
        "input_sha256": json.loads(PROTOCOL_FILE.read_text())["input_sha256"],
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "oof_sha256": sha256_file(condition_oof_path(condition)),
    }
    atomic_json(condition_summary_path(condition), summary)
    print(
        f"CONDITION_OOF_PASS {condition} "
        f"mean_ce={oof_mean_ce:.8f} weighted_val_ce={weighted_fold_ce:.8f}",
        flush=True,
    )
    return summary


def train(resume: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for --train. Keep the current no-GPU mode for --prepare."
        )

    protocol = verify_frozen_protocol()
    if not resume:
        existing = []
        for condition in CONDITIONS:
            for fold in range(N_FOLDS):
                candidates = [
                    checkpoint_path(condition, fold),
                    curve_path(condition, fold),
                    fold_oof_path(condition, fold),
                    fold_summary_path(condition, fold),
                    fold_marker_path(condition, fold),
                ]
                existing.extend(str(p) for p in candidates if p.exists())
        if existing:
            raise RuntimeError(
                "Training artifacts already exist; use --train --resume:\n"
                + "\n".join(existing[:30])
            )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")

    print("=" * 88)
    print("STEP 2F18I: ORCA-FSAR SIX-CONDITION FORMAL TRAINING")
    print("GPU =", torch.cuda.get_device_name(0))
    print("CUDA_NUMERICS: TF32=OFF FLOAT32_PRECISION=highest")
    print("RESUME =", resume)
    print("=" * 88, flush=True)

    base = load_and_validate_base_inputs(verify_hashes=True)
    # Projection artifacts were re-hashed by verify_frozen_protocol().
    run_manifest = {
        "artifact": "ORCA-FSAR Six-Condition Training Run Manifest V1",
        "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "completed_folds": {},
        "conditions": list(CONDITIONS.keys()),
    }
    if RUN_MANIFEST.exists():
        old = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != run_manifest["protocol_sha256"]:
            raise RuntimeError("Existing run manifest protocol mismatch")
        run_manifest = old
        run_manifest["status"] = "RUNNING"
        run_manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(RUN_MANIFEST, run_manifest)

    condition_summaries = {}
    for condition, spec in CONDITIONS.items():
        print("\n" + "#" * 88)
        print("START_CONDITION =", condition)
        print("#" * 88, flush=True)

        for fold in range(N_FOLDS):
            marker = fold_marker_path(condition, fold)
            if marker.exists():
                if not resume:
                    raise RuntimeError(
                        f"Completed fold already exists; rerun with --resume: {condition}, {fold}"
                    )
                fold_summary = validate_completed_fold(condition, fold, protocol)
                print(
                    f"RESUME_SKIP_COMPLETE_FOLD condition={condition} fold={fold}",
                    flush=True,
                )
            else:
                # If an interrupted fold left uncommitted files, remove only those
                # artifacts for this exact condition/fold before deterministic restart.
                for p in [
                    checkpoint_path(condition, fold),
                    curve_path(condition, fold),
                    curve_path(condition, fold).with_name(
                        curve_path(condition, fold).name + ".tmp"
                    ),
                    fold_oof_path(condition, fold),
                    fold_oof_path(condition, fold).with_name(
                        fold_oof_path(condition, fold).name + ".tmp"
                    ),
                    fold_summary_path(condition, fold),
                ]:
                    if p.exists():
                        p.unlink()
                fold_summary = train_one_fold(
                    condition, fold, spec, base, protocol
                )

            run_manifest.setdefault("completed_folds", {}).setdefault(
                condition, {}
            )[str(fold)] = {
                "status": "PASS",
                "summary_sha256": sha256_file(fold_summary_path(condition, fold)),
                "marker_sha256": sha256_file(fold_marker_path(condition, fold)),
            }
            atomic_json(RUN_MANIFEST, run_manifest)

        condition_summaries[condition] = aggregate_condition(
            condition, spec, base
        )

    run_manifest["status"] = "COMPLETED"
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    run_manifest["condition_summary_sha256"] = {
        c: sha256_file(condition_summary_path(c)) for c in CONDITIONS
    }
    run_manifest["condition_oof_sha256"] = {
        c: sha256_file(condition_oof_path(c)) for c in CONDITIONS
    }
    atomic_json(RUN_MANIFEST, run_manifest)

    final_report = {
        "artifact": "ORCA-FSAR Six-Condition Training Final Report V1",
        "status": "COMPLETED",
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "run_manifest_sha256": sha256_file(RUN_MANIFEST),
        "conditions": condition_summaries,
        "training_completed": True,
        "parameter_updates_performed": True,
        "dev_payload_used": False,
        "test_payload_used": False,
        "retrieval_metrics_computed": False,
        "future_metrics_computed": False,
    }
    atomic_json(OUT_ROOT / "final_report_v1.json", final_report)

    print("=" * 88)
    print("STEP_2F18I_TRAINING_COMPLETE")
    print("CONDITIONS =", len(CONDITIONS))
    print("FOLDS_PER_CONDITION =", N_FOLDS)
    print("CHECKPOINTS =", len(CONDITIONS) * N_FOLDS)
    print("OOF_COMPLETE = YES")
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("RETRIEVAL_METRICS_COMPUTED = NO")
    print("FUTURE_METRICS_COMPUTED = NO")
    print("OUTPUT =", OUT_ROOT)
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prepare",
        action="store_true",
        help="CPU-only audit and freeze ORCA-FSAR training protocol",
    )
    group.add_argument(
        "--train",
        action="store_true",
        help="Run six-condition ORCA-FSAR training; requires CUDA",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume by validating and skipping completed condition-fold units",
    )
    args = parser.parse_args()

    if args.resume and not args.train:
        parser.error("--resume is valid only with --train")
    if args.prepare:
        prepare()
    else:
        train(resume=args.resume)


if __name__ == "__main__":
    main()
