#!/usr/bin/env python3
"""
ORCA-FSAR S12C Dev-only Top-100 matched evaluation.

This evaluator fixes the input-space mismatch in the earlier
s12c_dev_top100_matched_evaluation.py:

  * each fold model is paired with its own frozen ORCA-k=2 basis;
  * raw Dev query, Gold response, and negative-candidate embeddings are
    projected with that fold's basis before scoring;
  * five fold-specific raw score matrices are averaged before ranking;
  * no training, checkpoint modification/reselection, or Test payload use.

Workflow on the AutoDL project server:
  python src/evaluation/s12c_step_2f18s_orca_fsar_dev_evaluation.py --prepare
  python src/evaluation/s12c_step_2f18s_orca_fsar_dev_evaluation.py --preflight
  python src/evaluation/s12c_step_2f18s_orca_fsar_dev_evaluation.py --evaluate

--prepare: CPU-only data/artifact audit and immutable protocol freeze.
--preflight: CPU-only checkpoint, model, and ORCA projection parity checks.
--evaluate: CUDA-only Dev ranking inference and dialogue-cluster bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from longpsych_agent.config import settings

SOURCE_ROOT = settings.source_root
sys.path.insert(0, str(SOURCE_ROOT))

import src.evaluation.s9_dev_top100_reranking_diagnostic as s9

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RankOnlyNoProductMatchedModel,
    RFutureFullMatchedModel,
    RFutureNoProductMatchedModel,
    QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

# ---------------------------------------------------------------------
# Frozen Dev data and upstream provenance
# ---------------------------------------------------------------------

DIM = 1024
N_DEV = 9070
N_DEV_DIALOGUES = 239
N_RESPONSES = 61502
N_LONG20 = 4375
N_GOLD_MATCHED = 250
TOP_K = 100
N_CANDIDATES = TOP_K + 1
N_FOLDS = 5
BATCH_SIZE = 32
N_BOOT = 5000
BOOT_SEED = 20261108

S9_PRED = ROOT / "results/phase_s9/s9_dev_top100_reranking_predictions_v1.npz"
S11D_PRED = ROOT / "results/phase_s11d/s11d_full_library_boundary_predictions_v2.npz"
S9_SCRIPT = ROOT / "src/evaluation/s9_dev_top100_reranking_diagnostic.py"
S8_SCRIPT = ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py"

RAW_Q_FILE = ROOT / "cache/embeddings/fsar_train_queries_bge_v1.npy"
RAW_R_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

MODEL_FILE = ROOT / "src/models/s12c_matched_ablation_models.py"
TRAIN_SCRIPT = ROOT / "src/training/s12c_train_orca_fsar_six_condition_5fold.py"

TRAIN_ROOT = ROOT / "results/phase_s12c/step_2f18_orca_fsar_training_v1"
TRAIN_PROTOCOL = TRAIN_ROOT / "training_protocol_v1.json"
TRAIN_RUN_MANIFEST = TRAIN_ROOT / "run_manifest.json"

ORCA_BASIS_ROOT = ROOT / "results/phase_s12c/step_2f17c_orca_fold_basis_v1"
ORCA_PROJECTION_ROOT = ROOT / "results/phase_s12c/step_2f17d_orca_fold_projection_v1"
ORCA_BASIS_SUMMARY = ORCA_BASIS_ROOT / "summary.json"
ORCA_PROJECTION_SUMMARY = ORCA_PROJECTION_ROOT / "summary.json"
ORCA_INTEGRITY_AUDIT = ORCA_PROJECTION_ROOT / "integrity_audit_v1.json"

OUT_ROOT = ROOT / "results/phase_s12c/step_2f18s_orca_fsar_dev_evaluation_v1"
PROTOCOL_FILE = OUT_ROOT / "dev_evaluation_protocol_v1.json"
PRED_FILE = OUT_ROOT / "dev_predictions_v1.npz"
SUMMARY_FILE = OUT_ROOT / "dev_summary_v1.json"
REPORT_FILE = OUT_ROOT / "dev_report_v1.txt"
RUN_MANIFEST_FILE = OUT_ROOT / "run_manifest.json"

CLASSES = {
    "RANK_ONLY_FULL_MATCHED": RankOnlyFullMatchedModel,
    "RANK_ONLY_NO_PRODUCT_MATCHED": RankOnlyNoProductMatchedModel,
    "R_FUTURE_FULL": RFutureFullMatchedModel,
    "R_FUTURE_NO_PRODUCT": RFutureNoProductMatchedModel,
    "QR_FUTURE_FULL": QRFutureFullMatchedModel,
    "QR_FUTURE_NO_PRODUCT": QRFutureNoProductMatchedModel,
}
CONDITIONS = list(CLASSES)

FUTURE_CONDITIONS = {
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
}

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]

CONTRASTS = [
    ("RANK_ONLY_FULL_MATCHED", "DENSE_BGE"),
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("RANK_ONLY_NO_PRODUCT_MATCHED", "DENSE_BGE"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
    ("R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL"),
    ("QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL"),
]

# ---------------------------------------------------------------------
# Paths and basic helpers
# ---------------------------------------------------------------------


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def array_content_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_text_atomic(path: Path, content: str) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def condition_dir(condition: str) -> Path:
    return TRAIN_ROOT / "conditions" / condition.lower()


def checkpoint_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "checkpoints" / f"fold{fold}_best_v1.pt"


def fold_summary_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "fold_summaries" / f"fold{fold}_summary_v1.json"


def fold_marker_path(condition: str, fold: int) -> Path:
    return condition_dir(condition) / "fold_markers" / f"fold{fold}_complete_v1.json"


def condition_summary_path(condition: str) -> Path:
    return condition_dir(condition) / f"{condition.lower()}_training_summary_v1.json"


def condition_oof_path(condition: str) -> Path:
    return condition_dir(condition) / f"{condition.lower()}_oof_scores_v1.npz"


def basis_path(fold: int) -> Path:
    return ORCA_BASIS_ROOT / f"fold{fold}_orca_basis_k2_v1.npy"


def projection_paths(fold: int) -> tuple[Path, Path, Path]:
    fold_dir = ORCA_PROJECTION_ROOT / f"fold{fold}"
    return (
        fold_dir / "query_orca_k2_v1.npy",
        fold_dir / "response_orca_k2_v1.npy",
        fold_dir / "manifest.json",
    )


def source_files() -> list[Path]:
    paths = [
        S9_PRED,
        S11D_PRED,
        S9_SCRIPT,
        S8_SCRIPT,
        RAW_Q_FILE,
        RAW_R_FILE,
        MODEL_FILE,
        TRAIN_SCRIPT,
        TRAIN_PROTOCOL,
        TRAIN_RUN_MANIFEST,
        ORCA_BASIS_SUMMARY,
        ORCA_PROJECTION_SUMMARY,
        ORCA_INTEGRITY_AUDIT,
    ]
    for fold in range(N_FOLDS):
        paths.append(basis_path(fold))
        paths.append(projection_paths(fold)[2])
    for condition in CONDITIONS:
        paths.extend([condition_summary_path(condition), condition_oof_path(condition)])
        for fold in range(N_FOLDS):
            paths.extend([fold_summary_path(condition, fold), fold_marker_path(condition, fold)])
    return paths


def require_file(path: Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")


def load_json(path: Path) -> dict:
    require_file(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ---------------------------------------------------------------------
# Dev data loading and frozen alignment audit
# ---------------------------------------------------------------------


def load_dev_data() -> dict:
    print("LOADING FROZEN DEV DATA", flush=True)

    s8_data = s9.read_s8_data()
    if len(s8_data) != 8:
        raise RuntimeError("Unexpected S8 data-loader signature")
    _, _, dev, _, _, _, match_idx, _ = s8_data

    q, gold, responses = s9.load_embeddings()

    with np.load(S9_PRED, allow_pickle=False) as z:
        p = {key: z[key] for key in z.files}
    with np.load(S11D_PRED, allow_pickle=False) as z:
        p11 = {key: z[key] for key in z.files}

    sample_ids = np.asarray([str(row["sample_id"]) for row in dev], dtype=str)
    dialogue_ids = np.asarray([str(row["dialogue_id"]) for row in dev], dtype=str)
    match_idx = np.asarray(match_idx, dtype=np.int32)

    q = np.asarray(q, dtype=np.float32)
    gold = np.asarray(gold, dtype=np.float32)
    responses = np.asarray(responses, dtype=np.float32)
    # The Top-100 row indices and the ORCA fold bases are both defined over
    # this exact frozen training response pool. Shape equality alone is not
    # enough: verify row order and embedding values against the trainer input.
    train_responses = np.load(RAW_R_FILE, mmap_mode="r", allow_pickle=False)
    if train_responses.shape != responses.shape or train_responses.dtype != np.float32:
        raise RuntimeError("Training/Dev response-pool shape or dtype mismatch")
    response_pool_max_abs_diff = 0.0
    for response_start in range(0, N_RESPONSES, 2048):
        response_end = min(response_start + 2048, N_RESPONSES)
        block_diff = np.abs(
            responses[response_start:response_end]
            - np.asarray(train_responses[response_start:response_end])
        )
        response_pool_max_abs_diff = max(
            response_pool_max_abs_diff, float(block_diff.max())
        )
    if response_pool_max_abs_diff > 1e-6:
        raise RuntimeError(
            "S9 response pool does not align with ORCA-FSAR training response pool: "
            f"max_abs_diff={response_pool_max_abs_diff:.3e}"
        )
    candidates = np.asarray(p["candidate_response_rows"], dtype=np.int32)
    long_mask = np.asarray(p["dev_long20_mask"], dtype=bool)
    dense_ranks = np.asarray(p["ranks__DENSE_BGE"], dtype=np.int32)
    dense_candidate_scores = np.asarray(p["candidate_dense_scores"], dtype=np.float32)
    dense_gold_scores = np.asarray(p["gold_dense_scores"], dtype=np.float32)

    if len(dev) != N_DEV:
        raise RuntimeError(f"Dev row count mismatch: {len(dev)} != {N_DEV}")
    if q.shape != (N_DEV, DIM) or gold.shape != (N_DEV, DIM):
        raise RuntimeError(f"Dev query/Gold shape mismatch: {q.shape}, {gold.shape}")
    if responses.shape != (N_RESPONSES, DIM):
        raise RuntimeError(f"Response embedding shape mismatch: {responses.shape}")
    if candidates.shape != (N_DEV, TOP_K):
        raise RuntimeError(f"Top-100 candidate shape mismatch: {candidates.shape}")
    if long_mask.shape != (N_DEV,) or dense_ranks.shape != (N_DEV,):
        raise RuntimeError("Dev mask/rank shape mismatch")
    if dense_candidate_scores.shape != (N_DEV, TOP_K):
        raise RuntimeError("Dense candidate-score shape mismatch")
    if dense_gold_scores.shape != (N_DEV,):
        raise RuntimeError("Dense Gold-score shape mismatch")
    if match_idx.shape != (N_DEV,):
        raise RuntimeError("Gold-match index shape mismatch")

    if len(np.unique(dialogue_ids)) != N_DEV_DIALOGUES:
        raise RuntimeError(
            f"Dev dialogue count mismatch: {len(np.unique(dialogue_ids))} "
            f"!= {N_DEV_DIALOGUES}"
        )
    if int(long_mask.sum()) != N_LONG20:
        raise RuntimeError(f"Long20 count mismatch: {int(long_mask.sum())} != {N_LONG20}")
    if int((match_idx >= 0).sum()) != N_GOLD_MATCHED:
        raise RuntimeError(
            f"Gold-matched count mismatch: {(match_idx >= 0).sum()} != {N_GOLD_MATCHED}"
        )

    if not np.array_equal(sample_ids, p["sample_ids"]):
        raise RuntimeError("S9 prediction sample_id alignment mismatch")
    if not np.array_equal(dialogue_ids, p["dialogue_ids"]):
        raise RuntimeError("S9 prediction dialogue_id alignment mismatch")
    if not np.array_equal(match_idx, p["gold_match_idx"]):
        raise RuntimeError("S9 prediction Gold-match alignment mismatch")

    if not np.array_equal(sample_ids, p11["sample_ids"]):
        raise RuntimeError("S11D prediction sample_id alignment mismatch")
    if not np.array_equal(dialogue_ids, p11["dialogue_ids"]):
        raise RuntimeError("S11D prediction dialogue_id alignment mismatch")
    if not np.array_equal(match_idx, p11["gold_match_idx"]):
        raise RuntimeError("S11D prediction Gold-match alignment mismatch")

    if not np.array_equal(candidates, p11["s9_top100_response_rows"].astype(np.int32)):
        raise RuntimeError("S9/S11D Top-100 candidate identity mismatch")
    if np.any(candidates < 0) or np.any(candidates >= N_RESPONSES):
        raise RuntimeError("Candidate response row is out of range")
    if np.any(match_idx < -1) or np.any(match_idx >= N_RESPONSES):
        raise RuntimeError("Gold-match response row is out of range")

    for row_index, row in enumerate(candidates):
        if len(np.unique(row)) != TOP_K:
            raise RuntimeError(f"Duplicate Top-100 candidates at Dev row {row_index}")
        if match_idx[row_index] >= 0 and match_idx[row_index] in row:
            raise RuntimeError(f"Gold identity leakage at Dev row {row_index}")

    if not all(
        np.isfinite(x).all()
        for x in (q, gold, responses, dense_candidate_scores, dense_gold_scores)
    ):
        raise RuntimeError("Non-finite value found in frozen Dev inputs")
    if np.any(dense_ranks < 1) or np.any(dense_ranks > N_CANDIDATES):
        raise RuntimeError("Dense rank outside [1, 101]")

    # Independently reconstruct the frozen Dense rank using the same tie rule.
    dense_rank_recomputed = 1 + (
        dense_candidate_scores >= dense_gold_scores[:, None]
    ).sum(axis=1)
    if not np.array_equal(dense_rank_recomputed.astype(np.int32), dense_ranks):
        raise RuntimeError("Stored Dense ranks do not match frozen Dense scores")

    print("DEV_QUERIES =", len(dev))
    print("DEV_DIALOGUES =", len(np.unique(dialogue_ids)))
    print("LONG20_QUERIES =", int(long_mask.sum()))
    print("GOLD_MATCHED =", int((match_idx >= 0).sum()))
    print("GOLD_UNMATCHED =", int((match_idx < 0).sum()))
    print("TOP100_CANDIDATES = PASS")
    print("S9_S11D_COHORT_ALIGNMENT = PASS")
    print("DENSE_RANK_RECOMPUTATION = PASS")

    return {
        "q": q,
        "gold": gold,
        "responses": responses,
        "candidates": candidates,
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "long_mask": long_mask,
        "match_idx": match_idx,
        "dense_ranks": dense_ranks,
        "dense_candidate_scores": dense_candidate_scores,
        "dense_gold_scores": dense_gold_scores,
        "response_pool_max_abs_diff_vs_training": response_pool_max_abs_diff,
    }

# ---------------------------------------------------------------------
# ORCA basis and training-run provenance audit
# ---------------------------------------------------------------------


def load_and_validate_orca_artifacts(train_protocol: dict) -> dict:
    orca = train_protocol.get("orca", {})
    if orca.get("method") != "FOLD_SPECIFIC_ORCA_PROJECTION":
        raise RuntimeError("Training protocol does not declare fold-specific ORCA")
    if int(orca.get("k", -1)) != 2:
        raise RuntimeError("Training protocol ORCA k is not 2")
    if orca.get("projection_formula") != "z = normalize(x - (xB_f)B_f.T)":
        raise RuntimeError("Unexpected frozen ORCA projection formula")
    if orca.get("same_fold_projection_for_train_and_validation") is not True:
        raise RuntimeError("Training protocol does not confirm fold-matched projection")

    basis_summary = load_json(ORCA_BASIS_SUMMARY)
    projection_summary = load_json(ORCA_PROJECTION_SUMMARY)
    integrity = load_json(ORCA_INTEGRITY_AUDIT)

    if basis_summary.get("status") != "COMPLETED" or basis_summary.get("k") != 2:
        raise RuntimeError("ORCA basis summary status/k mismatch")
    if basis_summary.get("source_selection") != "FROZEN_STEP_2F17B3":
        raise RuntimeError("ORCA basis source-selection provenance mismatch")
    if projection_summary.get("status") != "COMPLETED":
        raise RuntimeError("ORCA projection summary is not COMPLETED")
    if integrity.get("status") != "COMPLETED" or integrity.get("folds_passed") != 5:
        raise RuntimeError("ORCA projection integrity audit is not 5/5 PASS")

    expected_summary_hashes = {
        "basis_summary_sha256": sha256_file(ORCA_BASIS_SUMMARY),
        "projection_summary_sha256": sha256_file(ORCA_PROJECTION_SUMMARY),
        "integrity_audit_sha256": sha256_file(ORCA_INTEGRITY_AUDIT),
    }
    for key, actual in expected_summary_hashes.items():
        if orca.get(key) != actual:
            raise RuntimeError(f"ORCA provenance hash mismatch: {key}")

    fold_artifacts = orca.get("fold_artifacts", {})
    if set(fold_artifacts) != {str(f) for f in range(N_FOLDS)}:
        raise RuntimeError("ORCA fold artifact coverage mismatch")

    bases = {}
    basis_records = {}
    for fold in range(N_FOLDS):
        key = str(fold)
        b_path = basis_path(fold)
        q_path, r_path, manifest_path = projection_paths(fold)
        for path in (b_path, q_path, r_path, manifest_path):
            require_file(path)

        artifact = fold_artifacts[key]
        basis = np.load(b_path, allow_pickle=False)
        if basis.shape != (DIM, 2) or basis.dtype != np.float32:
            raise RuntimeError(f"ORCA basis shape/dtype mismatch for fold {fold}")
        if not np.isfinite(basis).all():
            raise RuntimeError(f"Non-finite ORCA basis for fold {fold}")

        basis_hash = array_content_sha256(basis)
        if basis_hash != artifact.get("basis_sha256"):
            raise RuntimeError(f"ORCA basis content SHA mismatch for fold {fold}")

        basis_meta = basis_summary.get("fold_results", {}).get(key, {})
        projection_meta = projection_summary.get("folds", {}).get(key, {})
        if basis_meta.get("source_count") != 5000 or basis_meta.get("k") != 2:
            raise RuntimeError(f"ORCA basis source metadata mismatch for fold {fold}")
        if basis_meta.get("source_mode") != "affective_user":
            raise RuntimeError(f"ORCA source mode mismatch for fold {fold}")
        if basis_meta.get("basis_sha256") != basis_hash:
            raise RuntimeError(f"ORCA basis summary SHA mismatch for fold {fold}")
        if projection_meta.get("basis_sha256") != basis_hash:
            raise RuntimeError(f"ORCA projection basis linkage mismatch for fold {fold}")

        manifest = load_json(manifest_path)
        if sha256_file(manifest_path) != artifact.get("manifest_sha256"):
            raise RuntimeError(f"ORCA projection manifest SHA mismatch for fold {fold}")
        if manifest.get("basis_sha256") != basis_hash:
            raise RuntimeError(f"ORCA manifest basis SHA mismatch for fold {fold}")
        if manifest.get("projection_method") != "x - (xB)B.T":
            raise RuntimeError(f"ORCA projection method mismatch for fold {fold}")
        if manifest.get("post_projection_normalization") != "L2":
            raise RuntimeError(f"ORCA post-projection normalization mismatch for fold {fold}")
        if manifest.get("query", {}).get("array_sha256") != artifact.get("query_projection_sha256"):
            raise RuntimeError(f"ORCA query projection provenance mismatch for fold {fold}")
        if manifest.get("response", {}).get("array_sha256") != artifact.get("response_projection_sha256"):
            raise RuntimeError(f"ORCA response projection provenance mismatch for fold {fold}")

        integrity_fold = integrity.get("results", {}).get(key, {})
        if integrity_fold.get("status") != "PASS":
            raise RuntimeError(f"ORCA integrity fold {fold} is not PASS")
        if integrity_fold.get("basis_hash_pass") is not True:
            raise RuntimeError(f"ORCA basis integrity flag failed for fold {fold}")
        if integrity_fold.get("query", {}).get("sha256") != artifact.get("query_projection_sha256"):
            raise RuntimeError(f"ORCA query integrity hash mismatch for fold {fold}")
        if integrity_fold.get("response", {}).get("sha256") != artifact.get("response_projection_sha256"):
            raise RuntimeError(f"ORCA response integrity hash mismatch for fold {fold}")
        formula_error = max(
            float(integrity_fold.get("query_formula_max_abs_diff", float("inf"))),
            float(integrity_fold.get("response_formula_max_abs_diff", float("inf"))),
        )
        if formula_error >= 2e-6:
            raise RuntimeError(f"ORCA formula audit tolerance exceeded for fold {fold}")

        orth_error = float(np.max(np.abs(basis.T @ basis - np.eye(2, dtype=np.float32))))
        if orth_error > 2e-5:
            raise RuntimeError(f"ORCA basis orthogonality failure for fold {fold}")

        bases[fold] = basis.astype(np.float32, copy=False)
        basis_records[key] = {
            "path": str(b_path.relative_to(ROOT)),
            "basis_sha256": basis_hash,
            "query_projection_sha256": artifact["query_projection_sha256"],
            "response_projection_sha256": artifact["response_projection_sha256"],
            "manifest_sha256": artifact["manifest_sha256"],
            "orthogonality_max_abs_error": orth_error,
        }
        print(f"ORCA_FOLD_{fold}_BASIS_PASS sha256={basis_hash[:16]}")

    return {
        "bases": bases,
        "basis_records": basis_records,
        "basis_summary_sha256": expected_summary_hashes["basis_summary_sha256"],
        "projection_summary_sha256": expected_summary_hashes["projection_summary_sha256"],
        "integrity_audit_sha256": expected_summary_hashes["integrity_audit_sha256"],
    }


def load_and_validate_training_run() -> tuple[dict, dict, list[dict]]:
    train_protocol = load_json(TRAIN_PROTOCOL)
    run_manifest = load_json(TRAIN_RUN_MANIFEST)

    if train_protocol.get("status") != "FROZEN_BEFORE_ANY_PARAMETER_UPDATE":
        raise RuntimeError("ORCA-FSAR training protocol status mismatch")
    if run_manifest.get("status") != "COMPLETED":
        raise RuntimeError("ORCA-FSAR training run is not COMPLETED")
    train_protocol_sha = sha256_file(TRAIN_PROTOCOL)
    if run_manifest.get("protocol_sha256") != train_protocol_sha:
        raise RuntimeError("Training run/protocol SHA mismatch")
    if train_protocol.get("training_script_sha256") != sha256_file(TRAIN_SCRIPT):
        raise RuntimeError("Frozen ORCA-FSAR training script SHA mismatch")
    if train_protocol.get("model_code_sha256") != sha256_file(MODEL_FILE):
        raise RuntimeError("Frozen model-code SHA mismatch")
    frozen_inputs = train_protocol.get("input_sha256", {})
    if frozen_inputs.get("raw_query_embedding") != sha256_file(RAW_Q_FILE):
        raise RuntimeError("Frozen raw query embedding SHA mismatch")
    if frozen_inputs.get("raw_response_embedding") != sha256_file(RAW_R_FILE):
        raise RuntimeError("Frozen raw response embedding SHA mismatch")
    if run_manifest.get("training_script_sha256") != sha256_file(TRAIN_SCRIPT):
        raise RuntimeError("Training run manifest script SHA mismatch")
    if set(run_manifest.get("conditions", [])) != set(CONDITIONS):
        raise RuntimeError("Training run condition coverage mismatch")
    if set(train_protocol.get("conditions", {})) != set(CONDITIONS):
        raise RuntimeError("Training protocol condition coverage mismatch")

    orca_info = load_and_validate_orca_artifacts(train_protocol)
    fold_artifacts = train_protocol["orca"]["fold_artifacts"]
    checkpoint_records: list[dict] = []

    for condition in CONDITIONS:
        condition_protocol = train_protocol["conditions"][condition]
        if condition_protocol.get("model_class") != CLASSES[condition].__name__:
            raise RuntimeError(f"Frozen model class mismatch: {condition}")
        expected_future = condition in FUTURE_CONDITIONS
        if bool(condition_protocol.get("uses_future_supervision")) != expected_future:
            raise RuntimeError(f"Future-supervision protocol mismatch: {condition}")
        condition_state = run_manifest.get("completed_folds", {}).get(condition, {})
        if set(condition_state) != {str(f) for f in range(N_FOLDS)}:
            raise RuntimeError(f"Training run fold coverage mismatch: {condition}")

        summary_path = condition_summary_path(condition)
        oof_path = condition_oof_path(condition)
        require_file(summary_path)
        require_file(oof_path)
        if sha256_file(summary_path) != run_manifest.get("condition_summary_sha256", {}).get(condition):
            raise RuntimeError(f"Condition summary SHA mismatch: {condition}")
        if sha256_file(oof_path) != run_manifest.get("condition_oof_sha256", {}).get(condition):
            raise RuntimeError(f"Condition OOF SHA mismatch: {condition}")

        condition_summary = load_json(summary_path)
        if condition_summary.get("status") != "PASS" or condition_summary.get("condition") != condition:
            raise RuntimeError(f"Condition summary status/identity mismatch: {condition}")
        if condition_summary.get("input_space") != "ORCA_K2_FOLD_SPECIFIC":
            raise RuntimeError(f"Condition input-space mismatch: {condition}")

        for fold in range(N_FOLDS):
            key = str(fold)
            marker_path = fold_marker_path(condition, fold)
            summary_fold_path = fold_summary_path(condition, fold)
            ckpt_path = checkpoint_path(condition, fold)
            for path in (marker_path, summary_fold_path, ckpt_path):
                require_file(path)

            marker = load_json(marker_path)
            fold_summary = load_json(summary_fold_path)
            run_fold = condition_state[key]

            if run_fold.get("status") != "PASS":
                raise RuntimeError(f"Run manifest fold status mismatch: {condition}/{fold}")
            if run_fold.get("marker_sha256") != sha256_file(marker_path):
                raise RuntimeError(f"Fold marker SHA mismatch: {condition}/{fold}")
            if run_fold.get("summary_sha256") != sha256_file(summary_fold_path):
                raise RuntimeError(f"Fold summary SHA mismatch: {condition}/{fold}")
            if marker.get("status") != "FOLD_COMPLETE":
                raise RuntimeError(f"Fold marker status mismatch: {condition}/{fold}")
            if marker.get("condition") != condition or int(marker.get("fold", -1)) != fold:
                raise RuntimeError(f"Fold marker identity mismatch: {condition}/{fold}")
            if marker.get("protocol_sha256") != train_protocol_sha:
                raise RuntimeError(f"Fold marker protocol SHA mismatch: {condition}/{fold}")
            if fold_summary.get("status") != "PASS":
                raise RuntimeError(f"Fold summary status mismatch: {condition}/{fold}")
            if fold_summary.get("condition") != condition or int(fold_summary.get("fold", -1)) != fold:
                raise RuntimeError(f"Fold summary identity mismatch: {condition}/{fold}")
            if fold_summary.get("input_space") != "ORCA_K2_FOLD_SPECIFIC":
                raise RuntimeError(f"Fold summary input-space mismatch: {condition}/{fold}")

            checkpoint_sha = sha256_file(ckpt_path)
            marker_ckpt_sha = marker.get("artifact_sha256", {}).get("checkpoint")
            if checkpoint_sha != marker_ckpt_sha:
                raise RuntimeError(f"Checkpoint/marker SHA mismatch: {condition}/{fold}")
            if checkpoint_sha != fold_summary.get("checkpoint_sha256"):
                raise RuntimeError(f"Checkpoint/fold-summary SHA mismatch: {condition}/{fold}")

            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            fold_orca = fold_artifacts[key]
            if ck.get("condition") != condition or int(ck.get("fold", -1)) != fold:
                raise RuntimeError(f"Checkpoint identity mismatch: {condition}/{fold}")
            if ck.get("input_space") != "ORCA_K2_FOLD_SPECIFIC":
                raise RuntimeError(f"Checkpoint input-space mismatch: {condition}/{fold}")
            if int(ck.get("orca_fold", -1)) != fold:
                raise RuntimeError(f"Checkpoint ORCA fold mismatch: {condition}/{fold}")
            if ck.get("orca_basis_sha256") != fold_orca.get("basis_sha256"):
                raise RuntimeError(f"Checkpoint basis SHA mismatch: {condition}/{fold}")
            if ck.get("orca_query_projection_sha256") != fold_orca.get("query_projection_sha256"):
                raise RuntimeError(f"Checkpoint query projection SHA mismatch: {condition}/{fold}")
            if ck.get("orca_response_projection_sha256") != fold_orca.get("response_projection_sha256"):
                raise RuntimeError(f"Checkpoint response projection SHA mismatch: {condition}/{fold}")
            if int(ck.get("best_epoch", -1)) != int(fold_summary.get("best_epoch", -2)):
                raise RuntimeError(f"Checkpoint best-epoch mismatch: {condition}/{fold}")
            if ck.get("dev_payload_used") is not False or ck.get("test_payload_used") is not False:
                raise RuntimeError(f"Training checkpoint provenance flag mismatch: {condition}/{fold}")
            if bool(ck.get("future_supervision_used")) != expected_future:
                raise RuntimeError(f"Checkpoint future-supervision flag mismatch: {condition}/{fold}")

            checkpoint_records.append({
                "condition": condition,
                "fold": fold,
                "path": str(ckpt_path.relative_to(ROOT)),
                "sha256": checkpoint_sha,
                "marker_sha256": sha256_file(marker_path),
                "fold_summary_sha256": sha256_file(summary_fold_path),
                "best_epoch": int(ck["best_epoch"]),
                "orca_basis_sha256": ck["orca_basis_sha256"],
                "orca_query_projection_sha256": ck["orca_query_projection_sha256"],
                "orca_response_projection_sha256": ck["orca_response_projection_sha256"],
            })

    if len(checkpoint_records) != len(CONDITIONS) * N_FOLDS:
        raise RuntimeError("Expected exactly 30 ORCA-FSAR checkpoints")

    return train_protocol, run_manifest, checkpoint_records

# ---------------------------------------------------------------------
# Projection implementation and parity checks
# ---------------------------------------------------------------------


def project_numpy(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    b32 = np.asarray(basis, dtype=np.float32)
    residual = x32 - (x32 @ b32) @ b32.T
    norms = np.linalg.norm(residual, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("ORCA residual has a near-zero norm")
    return (residual / norms).astype(np.float32, copy=False)


def project_torch(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    residual = x - torch.matmul(torch.matmul(x, basis), basis.transpose(0, 1))
    norms = torch.linalg.vector_norm(residual, ord=2, dim=-1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise RuntimeError("ORCA residual has a near-zero norm")
    return residual / norms


def validate_projection_parity(orca_info: dict) -> dict:
    raw_q = np.load(RAW_Q_FILE, mmap_mode="r", allow_pickle=False)
    raw_r = np.load(RAW_R_FILE, mmap_mode="r", allow_pickle=False)
    if raw_q.shape != (62889, DIM) or raw_r.shape != (N_RESPONSES, DIM):
        raise RuntimeError("Frozen training raw embedding shape mismatch")

    results = {}
    for fold in range(N_FOLDS):
        q_path, r_path, _ = projection_paths(fold)
        q_saved = np.load(q_path, mmap_mode="r", allow_pickle=False)
        r_saved = np.load(r_path, mmap_mode="r", allow_pickle=False)
        basis = orca_info["bases"][fold]
        if q_saved.shape != (62889, DIM) or q_saved.dtype != np.float32:
            raise RuntimeError(f"Saved ORCA query projection shape/dtype mismatch: fold={fold}")
        if r_saved.shape != (N_RESPONSES, DIM) or r_saved.dtype != np.float32:
            raise RuntimeError(f"Saved ORCA response projection shape/dtype mismatch: fold={fold}")

        rng = np.random.default_rng(21700 + fold)
        qi = np.sort(rng.choice(raw_q.shape[0], size=128, replace=False))
        ri = np.sort(rng.choice(raw_r.shape[0], size=128, replace=False))

        q_expected = project_numpy(np.asarray(raw_q[qi]), basis)
        r_expected = project_numpy(np.asarray(raw_r[ri]), basis)
        q_diff = float(np.max(np.abs(q_expected - np.asarray(q_saved[qi]))))
        r_diff = float(np.max(np.abs(r_expected - np.asarray(r_saved[ri]))))
        if max(q_diff, r_diff) >= 2e-6:
            raise RuntimeError(
                f"ORCA projection formula parity failed for fold {fold}: "
                f"query={q_diff:.3e}, response={r_diff:.3e}"
            )

        # Torch implementation used by the evaluator must match the same formula.
        q_torch = project_torch(
            torch.from_numpy(np.asarray(raw_q[qi]).copy()),
            torch.from_numpy(basis.copy()),
        ).numpy()
        r_torch = project_torch(
            torch.from_numpy(np.asarray(raw_r[ri]).copy()),
            torch.from_numpy(basis.copy()),
        ).numpy()
        torch_diff = max(
            float(np.max(np.abs(q_torch - q_expected))),
            float(np.max(np.abs(r_torch - r_expected))),
        )
        if torch_diff >= 2e-6:
            raise RuntimeError(f"Torch/NumPy ORCA projection mismatch for fold {fold}")

        results[str(fold)] = {
            "query_saved_projection_max_abs_diff": q_diff,
            "response_saved_projection_max_abs_diff": r_diff,
            "torch_vs_numpy_max_abs_diff": torch_diff,
            "sample_count_each": 128,
        }
        print(
            f"ORCA_PROJECTION_PARITY_FOLD_{fold}_PASS "
            f"saved_q={q_diff:.3e} saved_r={r_diff:.3e} torch={torch_diff:.3e}"
        )

    return results

# ---------------------------------------------------------------------
# Protocol freeze and verification
# ---------------------------------------------------------------------


def build_source_hashes() -> dict:
    hashes = {}
    for path in source_files():
        require_file(path)
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def prepare() -> None:
    if OUT_ROOT.exists():
        raise RuntimeError(
            f"Output directory already exists; refusing overwrite: {OUT_ROOT}"
        )

    data = load_dev_data()
    train_protocol, run_manifest, checkpoint_records = load_and_validate_training_run()
    orca_info = load_and_validate_orca_artifacts(train_protocol)
    projection_parity = validate_projection_parity(orca_info)
    source_hashes = build_source_hashes()

    protocol = {
        "artifact": "ORCA-FSAR S12C Dev Top100 Matched Evaluation Protocol V1",
        "project": "PsychResponse_FSAR",
        "status": "FROZEN_BEFORE_DEV_RANKING_INFERENCE",
        "created_for": "STEP_2F18S",
        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_script_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(TRAIN_PROTOCOL),
        "training_run_manifest_sha256": sha256_file(TRAIN_RUN_MANIFEST),
        "training_script_sha256": sha256_file(TRAIN_SCRIPT),
        "model_code_sha256": sha256_file(MODEL_FILE),
        "dev": {
            "queries": N_DEV,
            "dialogues": N_DEV_DIALOGUES,
            "long20_queries": N_LONG20,
            "gold_matched_queries": N_GOLD_MATCHED,
            "gold_unmatched_queries": N_DEV - N_GOLD_MATCHED,
            "query_shape": [N_DEV, DIM],
            "gold_shape": [N_DEV, DIM],
            "response_shape": [N_RESPONSES, DIM],
            "response_pool_alignment_to_training_max_abs_diff": data["response_pool_max_abs_diff_vs_training"],
            "sample_id_alignment": "S8_DEV_TO_S9_AND_S11D_EXACT",
            "dialogue_id_alignment": "S8_DEV_TO_S9_AND_S11D_EXACT",
            "gold_match_index_alignment": "S8_DEV_TO_S9_AND_S11D_EXACT",
        },
        "candidate_set": {
            "source": "Frozen S9 BGE Top-100 negatives",
            "negative_candidates": TOP_K,
            "gold_candidates": 1,
            "total_candidates": N_CANDIDATES,
            "gold_injected_for_scoring": True,
            "gold_identity_leakage": False,
        },
        "orca": {
            "method": "FOLD_SPECIFIC_ORCA_PROJECTION",
            "k": 2,
            "formula": "z = normalize(x - (xB_f)B_f.T)",
            "basis_source": "FROZEN_STEP_2F17B3",
            "fold_specific_basis": True,
            "query_gold_and_negative_candidates_projected_per_fold": True,
            "projection_parity": projection_parity,
            "fold_artifacts": orca_info["basis_records"],
        },
        "models": {
            "conditions": ["DENSE_BGE"] + CONDITIONS,
            "folds_per_condition": N_FOLDS,
            "checkpoint_count": len(checkpoint_records),
            "checkpoints": checkpoint_records,
            "fold_model_basis_pairing": "MODEL_FOLD_F_USES_ORCA_BASIS_F",
            "score_aggregation": "ARITHMETIC_MEAN_OF_FIVE_RAW_FOLD_SCORES_BEFORE_RANKING",
            "checkpoint_reselection": False,
            "retraining": False,
        },
        "ranking": {
            "tie_rule": "negative_score >= gold_score ranks before Gold",
            "rank_formula": "1 + count(negative_score >= gold_score)",
            "metrics": METRICS,
            "ranking_ce": "logsumexp([gold_score, negative_scores]) - gold_score",
        },
        "statistics": {
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "confidence_level": 0.95,
            "resampling_unit": "dialogue",
            "point_estimator": "unweighted_mean_of_dialogue_level_query_means",
        },
        "source_sha256": source_hashes,
        "training": False,
        "test_payload_used": False,
        "future_metrics_computed": False,
        "dev_ranking_inference_performed": False,
        "overwrite_existing": False,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    write_json_atomic(PROTOCOL_FILE, protocol)
    print("=" * 88)
    print("STEP_2F18S_PREPARE_COMPLETE")
    print("PROTOCOL_FROZEN = PASS")
    print("CHECKPOINTS =", len(checkpoint_records))
    print("DEV_QUERIES =", len(data["q"]))
    print("ORCA_PROJECTION_PARITY = PASS")
    print("PROTOCOL =", PROTOCOL_FILE)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_FILE))
    print("CUDA_USED = NO")
    print("RANKING_INFERENCE = NO")
    print("TRAINING = NO")
    print("TEST_USED = NO")
    print("=" * 88)


def verify_protocol() -> dict:
    require_file(PROTOCOL_FILE)
    protocol = load_json(PROTOCOL_FILE)
    if protocol.get("status") != "FROZEN_BEFORE_DEV_RANKING_INFERENCE":
        raise RuntimeError("Dev evaluation protocol status mismatch")
    if protocol.get("evaluator_script_sha256") != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Evaluator script changed after protocol freeze")

    for relative, expected in protocol.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen source SHA mismatch: {relative}")

    if sha256_file(TRAIN_PROTOCOL) != protocol.get("training_protocol_sha256"):
        raise RuntimeError("Training protocol SHA changed after Dev freeze")
    if sha256_file(TRAIN_RUN_MANIFEST) != protocol.get("training_run_manifest_sha256"):
        raise RuntimeError("Training run manifest SHA changed after Dev freeze")

    for item in protocol["models"]["checkpoints"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Frozen checkpoint SHA mismatch: {path}")

    return protocol

# ---------------------------------------------------------------------
# Model loading and preflight
# ---------------------------------------------------------------------


def validate_model_scorer(model: torch.nn.Module) -> float:
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        q = torch.randn(2, DIM, device=device)
        r = torch.randn(2, 3, DIM, device=device)
        direct = model.rank_scores(q, r)
        zq = model.ranker.adapt_query(q)
        zr = model.ranker.adapt_response(r)
        reference = model.ranker.score_from_adapted(
            zq[:, None, :].expand_as(zr), zr
        )
        return float((direct - reference).abs().max().item())


def load_condition_models(
    condition: str,
    protocol: dict,
    device: torch.device,
) -> list[tuple[int, torch.nn.Module, torch.Tensor]]:
    records = [
        item for item in protocol["models"]["checkpoints"]
        if item["condition"] == condition
    ]
    records.sort(key=lambda item: item["fold"])
    if len(records) != N_FOLDS or [x["fold"] for x in records] != list(range(N_FOLDS)):
        raise RuntimeError(f"Checkpoint fold coverage mismatch: {condition}")

    fold_basis_records = protocol["orca"]["fold_artifacts"]
    models = []
    for item in records:
        fold = int(item["fold"])
        ckpt_path = ROOT / item["path"]
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = CLASSES[condition]()
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device).eval()

        expected_rank_dim = 3072 if "NO_PRODUCT" in condition else 4096
        if model.ranker.rank_head.weight.shape[1] != expected_rank_dim:
            raise RuntimeError(f"Rank-head dimension mismatch: {condition}/{fold}")

        basis_meta = fold_basis_records[str(fold)]
        b = np.load(ROOT / basis_meta["path"], allow_pickle=False)
        if array_content_sha256(b) != basis_meta["basis_sha256"]:
            raise RuntimeError(f"Basis SHA mismatch while loading: fold={fold}")
        basis_gpu = torch.from_numpy(np.asarray(b, dtype=np.float32).copy()).to(device)

        scorer_diff = validate_model_scorer(model)
        if scorer_diff > 1e-5:
            raise RuntimeError(
                f"Model scorer parity failure: {condition}/{fold}, diff={scorer_diff:.3e}"
            )

        if condition in FUTURE_CONDITIONS:
            with torch.inference_mode():
                q_probe = torch.randn(2, DIM, device=device)
                r_probe = torch.randn(2, DIM, device=device)
                logits = model.future_logits(q_probe, r_probe)
            if tuple(logits.shape) != (2, 4) or not torch.isfinite(logits).all():
                raise RuntimeError(f"Future-head shape/value failure: {condition}/{fold}")

        print(
            f"MODEL_LOAD_PASS {condition} fold={fold} "
            f"best_epoch={item['best_epoch']} scorer_diff={scorer_diff:.3e}"
        )
        models.append((fold, model, basis_gpu))

    return models


def preflight() -> None:
    protocol = verify_protocol()
    data = load_dev_data()
    train_protocol, _, _ = load_and_validate_training_run()
    orca_info = load_and_validate_orca_artifacts(train_protocol)
    projection_parity = validate_projection_parity(orca_info)

    # Verify the exact Dev tensors can be projected by every fold basis and remain unit norm.
    for fold in range(N_FOLDS):
        b = torch.from_numpy(orca_info["bases"][fold].copy())
        q_sample = torch.from_numpy(data["q"][:64].copy())
        g_sample = torch.from_numpy(data["gold"][:64].copy())
        c_sample = torch.from_numpy(
            data["responses"][data["candidates"][:64, :4]].copy()
        )
        for name, projected in (
            ("query", project_torch(q_sample, b)),
            ("gold", project_torch(g_sample, b)),
            ("candidate", project_torch(c_sample, b)),
        ):
            norms = torch.linalg.vector_norm(projected, dim=-1)
            if not torch.isfinite(projected).all() or float(torch.max(torch.abs(norms - 1))) > 2e-5:
                raise RuntimeError(f"Dev {name} projection norm failure: fold={fold}")
        print(f"DEV_ORCA_INPUT_FOLD_{fold}_PASS")

    for condition in CONDITIONS:
        models = load_condition_models(condition, protocol, torch.device("cpu"))
        del models

    print("=" * 88)
    print("STEP_2F18S_PREFLIGHT = PASS")
    print("ORCA_PROJECTION_PARITY = PASS")
    print("DEV_INPUT_SPACE_ALIGNMENT = PASS")
    print("CHECKPOINTS = 30/30")
    print("TRAINING = NO")
    print("DEV_RANKING_INFERENCE = NO")
    print("TEST_USED = NO")
    print("=" * 88)

# ---------------------------------------------------------------------
# Dev ranking inference
# ---------------------------------------------------------------------


def metric_matrix(ranks: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.int32)
    return np.column_stack([
        1.0 / ranks,
        ranks <= 1,
        ranks <= 5,
        ranks <= 10,
        ranks <= 20,
    ]).astype(np.float64)


def score_condition(
    condition: str,
    models: list[tuple[int, torch.nn.Module, torch.Tensor]],
    q_gpu: torch.Tensor,
    gold_gpu: torch.Tensor,
    responses_gpu: torch.Tensor,
    candidates: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(candidates)
    ranks = np.empty(n, dtype=np.int32)
    candidate_scores_out = np.empty((n, TOP_K), dtype=np.float32)
    gold_scores_out = np.empty(n, dtype=np.float32)
    ranking_ce_out = np.empty(n, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            q_raw = q_gpu[start:end]
            gold_raw = gold_gpu[start:end]
            candidate_idx = torch.from_numpy(
                candidates[start:end].astype(np.int64, copy=False)
            ).to(device)
            response_raw = responses_gpu[candidate_idx]

            fold_candidate_scores = []
            fold_gold_scores = []

            # Critical correction: every model receives the projection generated
            # by the basis belonging to its own fold.
            for fold, model, basis_gpu in models:
                q_fold = project_torch(q_raw, basis_gpu)
                gold_fold = project_torch(gold_raw, basis_gpu)
                response_fold = project_torch(response_raw, basis_gpu)

                candidate_scores_fold = model.rank_scores(q_fold, response_fold)
                gold_scores_fold = model.rank_scores(
                    q_fold, gold_fold[:, None, :]
                ).squeeze(1)

                fold_candidate_scores.append(candidate_scores_fold)
                fold_gold_scores.append(gold_scores_fold)

            # Average raw scores across the five fold-matched model/basis pairs,
            # then rank once. Do not average bases or fold ranks.
            candidate_scores = torch.stack(fold_candidate_scores, dim=0).mean(dim=0)
            gold_scores = torch.stack(fold_gold_scores, dim=0).mean(dim=0)

            batch_ranks = 1 + (candidate_scores >= gold_scores[:, None]).sum(dim=1)
            all_logits = torch.cat([gold_scores[:, None], candidate_scores], dim=1)
            batch_ce = torch.logsumexp(all_logits, dim=1) - gold_scores

            ranks[start:end] = batch_ranks.cpu().numpy().astype(np.int32)
            candidate_scores_out[start:end] = candidate_scores.cpu().numpy().astype(np.float32)
            gold_scores_out[start:end] = gold_scores.cpu().numpy().astype(np.float32)
            ranking_ce_out[start:end] = batch_ce.cpu().numpy().astype(np.float32)

            if start == 0 or end == n or end % 512 == 0:
                print(f"{condition}: {end}/{n}", flush=True)

    if not (
        np.isfinite(candidate_scores_out).all()
        and np.isfinite(gold_scores_out).all()
        and np.isfinite(ranking_ce_out).all()
    ):
        raise RuntimeError(f"Non-finite ranking output: {condition}")
    if np.any(ranks < 1) or np.any(ranks > N_CANDIDATES):
        raise RuntimeError(f"Rank outside [1, 101]: {condition}")

    return ranks, candidate_scores_out, gold_scores_out, ranking_ce_out


def evaluate_subset(
    ranks: dict[str, np.ndarray],
    dialogue_ids: np.ndarray,
    indices: np.ndarray,
    seed: int,
) -> dict:
    dialogue_subset = dialogue_ids[indices].astype(str)
    unique_dialogues, inverse = np.unique(dialogue_subset, return_inverse=True)
    if len(unique_dialogues) == 0:
        raise RuntimeError("Cannot bootstrap an empty evaluation subset")

    grouped = {}
    counts = np.bincount(inverse, minlength=len(unique_dialogues))
    for name, rank_values in ranks.items():
        values = metric_matrix(rank_values[indices])
        sums = np.zeros((len(unique_dialogues), len(METRICS)), dtype=np.float64)
        np.add.at(sums, inverse, values)
        grouped[name] = sums / counts[:, None]

    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(
        0, len(unique_dialogues), size=(N_BOOT, len(unique_dialogues))
    )

    result = {
        "n_queries": int(len(indices)),
        "n_dialogues": int(len(unique_dialogues)),
        "point_estimator": "unweighted_mean_of_dialogue_level_query_means",
        "models": {},
        "paired_contrasts": {},
    }

    for name, values in grouped.items():
        bootstrap_values = values[bootstrap_indices].mean(axis=1)
        estimate = values.mean(axis=0)
        lower, upper = np.quantile(bootstrap_values, [0.025, 0.975], axis=0)
        result["models"][name] = {
            metric: {
                "estimate": float(estimate[j]),
                "ci95_lower": float(lower[j]),
                "ci95_upper": float(upper[j]),
            }
            for j, metric in enumerate(METRICS)
        }

    for a, b in CONTRASTS:
        delta = grouped[a] - grouped[b]
        bootstrap_delta = delta[bootstrap_indices].mean(axis=1)
        estimate = delta.mean(axis=0)
        lower, upper = np.quantile(bootstrap_delta, [0.025, 0.975], axis=0)
        result["paired_contrasts"][f"{a}_MINUS_{b}"] = {
            metric: {
                "delta": float(estimate[j]),
                "ci95_lower": float(lower[j]),
                "ci95_upper": float(upper[j]),
            }
            for j, metric in enumerate(METRICS)
        }

    return result


def evaluate() -> None:
    protocol = verify_protocol()
    for path in (PRED_FILE, SUMMARY_FILE, REPORT_FILE, RUN_MANIFEST_FILE):
        if path.exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --evaluate")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    print("GPU =", torch.cuda.get_device_name(0))
    print("TF32 = OFF")

    data = load_dev_data()
    q_gpu = torch.from_numpy(data["q"].copy()).to(device)
    gold_gpu = torch.from_numpy(data["gold"].copy()).to(device)
    responses_gpu = torch.from_numpy(data["responses"].copy()).to(device)

    dense_candidate_scores = data["dense_candidate_scores"]
    dense_gold_scores = data["dense_gold_scores"]
    dense_ce = (
        np.logaddexp(
            dense_gold_scores,
            np.logaddexp.reduce(dense_candidate_scores, axis=1),
        ) - dense_gold_scores
    ).astype(np.float32)

    ranks = {"DENSE_BGE": data["dense_ranks"].copy()}
    prediction_arrays = {
        "sample_ids": data["sample_ids"],
        "dialogue_ids": data["dialogue_ids"],
        "dev_long20_mask": data["long_mask"],
        "gold_match_idx": data["match_idx"],
        "candidate_response_rows": data["candidates"],
        "candidate_dense_scores": dense_candidate_scores,
        "gold_dense_scores": dense_gold_scores,
        "ranks__DENSE_BGE": data["dense_ranks"],
        "ranking_ce__DENSE_BGE": dense_ce,
    }

    scorer_diffs = {}
    fold_basis_hashes = protocol["orca"]["fold_artifacts"]

    for condition in CONDITIONS:
        print("\nRUNNING", condition, flush=True)
        models = load_condition_models(condition, protocol, device)
        condition_ranks, candidate_scores, gold_scores, ranking_ce = score_condition(
            condition,
            models,
            q_gpu,
            gold_gpu,
            responses_gpu,
            data["candidates"],
            device,
        )
        ranks[condition] = condition_ranks
        prediction_arrays[f"ranks__{condition}"] = condition_ranks
        prediction_arrays[f"candidate_scores__{condition}"] = candidate_scores
        prediction_arrays[f"gold_scores__{condition}"] = gold_scores
        prediction_arrays[f"ranking_ce__{condition}"] = ranking_ce
        scorer_diffs[condition] = {
            str(fold): float(validate_model_scorer(model))
            for fold, model, _ in models
        }

        print(
            condition,
            "COMPLETE",
            "MRR =", float(np.mean(1.0 / condition_ranks)),
            "R@5 =", float(np.mean(condition_ranks <= 5)),
            "CE =", float(np.mean(ranking_ce)),
            "rank_min =", int(condition_ranks.min()),
            "rank_max =", int(condition_ranks.max()),
        )

        del models
        torch.cuda.empty_cache()

    all_indices = np.arange(N_DEV, dtype=np.int64)
    subsets = {
        "DEV_ALL": all_indices,
        "DEV_LONG20": np.flatnonzero(data["long_mask"]),
        "DEV_GOLD_MATCHED": np.flatnonzero(data["match_idx"] >= 0),
        "DEV_GOLD_UNMATCHED": np.flatnonzero(data["match_idx"] < 0),
    }

    datasets = {}
    for subset_index, (subset_name, indices) in enumerate(subsets.items()):
        if len(indices) == 0:
            raise RuntimeError(f"Empty Dev subset: {subset_name}")
        datasets[subset_name] = evaluate_subset(
            ranks,
            data["dialogue_ids"],
            indices,
            BOOT_SEED + subset_index,
        )

    summary = {
        "artifact": "ORCA-FSAR S12C Dev Top100 Matched Evaluation Summary V1",
        "status": "COMPLETED",
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "training_protocol_sha256": sha256_file(TRAIN_PROTOCOL),
        "training_run_manifest_sha256": sha256_file(TRAIN_RUN_MANIFEST),
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "dev_queries": N_DEV,
        "dev_dialogues": N_DEV_DIALOGUES,
        "candidate_set": {
            "negative_candidates": TOP_K,
            "gold_candidates": 1,
            "total_candidates": N_CANDIDATES,
            "source": "Frozen S9 BGE Top-100 negatives",
        },
        "orca": {
            "method": "FOLD_SPECIFIC_ORCA_PROJECTION",
            "k": 2,
            "formula": "z = normalize(x - (xB_f)B_f.T)",
            "fold_basis_hashes": {
                key: value["basis_sha256"]
                for key, value in fold_basis_hashes.items()
            },
            "query_gold_and_candidates_projected_per_fold": True,
        },
        "ensemble": {
            "folds": N_FOLDS,
            "aggregation": "arithmetic_mean_of_five_raw_fold_scores_before_ranking",
            "fold_model_basis_pairing": "MODEL_FOLD_F_USES_ORCA_BASIS_F",
            "retraining": False,
            "checkpoint_reselection": False,
        },
        "ranking": {
            "tie_rule": "negative score >= Gold score ranks before Gold",
            "metrics": METRICS,
            "ranking_ce_saved": True,
        },
        "statistics": {
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "confidence_level": 0.95,
            "resampling_unit": "dialogue",
            "point_estimator": "unweighted_mean_of_dialogue_level_query_means",
        },
        "model_scorer_max_abs_diff": scorer_diffs,
        "datasets": datasets,
        "training": False,
        "dev_payload_used": True,
        "test_payload_used": False,
        "future_metrics_computed": False,
    }

    # Write each artifact atomically. A completed run manifest is written last.
    pred_tmp = PRED_FILE.with_suffix(PRED_FILE.suffix + ".tmp")
    with pred_tmp.open("wb") as handle:
        np.savez_compressed(handle, **prediction_arrays)
    os.replace(pred_tmp, PRED_FILE)
    write_json_atomic(SUMMARY_FILE, summary)

    report_lines = [
        "ORCA-FSAR S12C DEV TOP100 MATCHED RERANKING EVALUATION",
        "=" * 88,
        "Input correction: fold-specific ORCA-k=2 projection for query, Gold, and negatives",
        "Fold pairing: fold-f model is scored only with fold-f ORCA basis",
        "Ensemble: arithmetic mean of five raw fold scores before ranking",
        "Candidates: frozen S9 BGE Top-100 negatives + one injected Gold",
        "Tie rule: negative score >= Gold ranks before Gold",
        f"Bootstrap: {N_BOOT} paired dialogue-cluster replicates",
        "Training = NO",
        "Checkpoint modification/reselection = NO",
        "Test payload = NO",
        "Future-state metrics = NOT COMPUTED (ranking-only Dev scope)",
    ]

    for subset_name, result in datasets.items():
        report_lines.extend(["", "=" * 88, subset_name, "=" * 88])
        for model_name, metric_values in result["models"].items():
            fields = [
                f"{metric}={value['estimate']:.6f}"
                f"[{value['ci95_lower']:.6f},{value['ci95_upper']:.6f}]"
                for metric, value in metric_values.items()
            ]
            report_lines.append(model_name + ": " + " | ".join(fields))
        report_lines.append("PAIRED CONTRASTS")
        for contrast, metric_values in result["paired_contrasts"].items():
            fields = [
                f"{metric}={value['delta']:+.6f}"
                f"[{value['ci95_lower']:+.6f},{value['ci95_upper']:+.6f}]"
                for metric, value in metric_values.items()
            ]
            report_lines.append(contrast + ": " + " | ".join(fields))

    write_text_atomic(REPORT_FILE, "\n".join(report_lines) + "\n")

    run_manifest = {
        "artifact": "ORCA-FSAR S12C Dev Evaluation Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": sha256_file(PROTOCOL_FILE),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "training_protocol_sha256": sha256_file(TRAIN_PROTOCOL),
        "training_run_manifest_sha256": sha256_file(TRAIN_RUN_MANIFEST),
        "dev_queries": N_DEV,
        "dev_dialogues": N_DEV_DIALOGUES,
        "long20_queries": int(data["long_mask"].sum()),
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "checkpoint_count": len(protocol["models"]["checkpoints"]),
        "predictions_sha256": sha256_file(PRED_FILE),
        "summary_sha256": sha256_file(SUMMARY_FILE),
        "report_sha256": sha256_file(REPORT_FILE),
        "training": False,
        "test_used": False,
        "ranking_inference": True,
        "future_metrics_computed": False,
    }
    write_json_atomic(RUN_MANIFEST_FILE, run_manifest)

    print("\nSTEP_2F18S_DEV_EVALUATION_COMPLETE")
    print("SUMMARY =", SUMMARY_FILE)
    print("PREDICTIONS =", PRED_FILE)
    print("REPORT =", REPORT_FILE)
    print("RUN_MANIFEST =", RUN_MANIFEST_FILE)
    print("DEV_QUERIES =", N_DEV)
    print("CHECKPOINTS = 30/30")
    print("TEST_USED = NO")
    print("TRAINING = NO")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    elif args.preflight:
        preflight()
    elif args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
