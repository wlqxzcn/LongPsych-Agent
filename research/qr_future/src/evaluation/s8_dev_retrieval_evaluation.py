#!/usr/bin/env python3
"""FSAR S8 external Dev retrieval evaluation.

Two-step use:
  python src/evaluation/s8_dev_retrieval_evaluation.py --prepare
  # inspect protocols/fsar_s8_dev_retrieval_evaluation_v1.json
  python src/evaluation/s8_dev_retrieval_evaluation.py --evaluate

Preparation encodes only Dev queries and their Gold answers, then freezes a
protocol with input/checkpoint/code hashes. Evaluation verifies those hashes
before computing retrieval metrics. No Test files are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s3c1_fsar_models import RankOnlyModel, RFutureModel, QRFutureModel

MODEL_PATH = ROOT / "models/BAAI/bge-large-zh-v1___5"
MODEL_CODE = ROOT / "src/models/s3c1_fsar_models.py"
ENCODER_SCRIPT = ROOT / "src/retrieval/s2b2_encode_full_train_bge.py"

TRAIN_RESPONSES = ROOT / "data/canonical/fsar_s2/train_unique_responses_v1.jsonl"
TRAIN_RESPONSE_INDEX = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"
TRAIN_RESPONSE_EMB = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
DEV_ALL = ROOT / "data/splits/psydial_targets_dev_all.jsonl"
DEV_LONG20 = ROOT / "data/splits/psydial_targets_dev_long20.jsonl"

RESULT_DIR = ROOT / "results/phase_s8"
QUERY_EMB_OUT = RESULT_DIR / "dev_all_query_embeddings_bge_v1.npy"
GOLD_EMB_OUT = RESULT_DIR / "dev_all_gold_embeddings_bge_v1.npy"
EMBED_MANIFEST_OUT = RESULT_DIR / "dev_all_embedding_manifest_v1.json"
PROTOCOL_OUT = ROOT / "protocols/fsar_s8_dev_retrieval_evaluation_v1.json"
SUMMARY_OUT = RESULT_DIR / "s8_dev_retrieval_summary_v1.json"
PREDICTIONS_OUT = RESULT_DIR / "s8_dev_retrieval_predictions_v1.npz"
REPORT_OUT = RESULT_DIR / "s8_dev_retrieval_report_v1.txt"

EXPECTED_RESPONSE_EMB_SHA = "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f"
EXPECTED_QUERY_ROWS = 9070
EXPECTED_LONG20_ROWS = 4375
EXPECTED_DEV_ALL_SHA = "8909877897573c0d7fdf517ae53e739f4a68f96c89fb93819282521349562ca5"
EXPECTED_DEV_LONG20_SHA = "6721ffa84c55ab63942f0a173fc4d5747f0190ce4394a57677ba70f98138484d"
EXPECTED_RESPONSE_ROWS = 61502
DIM = 1024
Q_BATCH = 32
C_CHUNK = 2048
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261027
METRIC_NAMES = ["MRR", "R@1", "R@5", "R@10", "R@20"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def all_checkpoint_paths() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    out["RANK_ONLY"] = [
        ROOT / f"results/phase_s4/checkpoints/rank_only/rank_only_fold{fold}_best_v1.pt"
        for fold in range(5)
    ]
    for condition in ("R_FUTURE", "QR_FUTURE"):
        for beta in (1, 2, 4, 8):
            paths = []
            for fold in range(5):
                if beta == 1:
                    p = ROOT / f"results/phase_s5/checkpoints/{condition.lower()}/{condition.lower()}_fold{fold}_best_v1.pt"
                else:
                    p = ROOT / f"results/phase_s7/beta_{beta}/checkpoints/{condition.lower()}/{condition.lower()}_fold{fold}_best_v1.pt"
                paths.append(p)
            out[f"{condition}_BETA{beta}"] = paths
    return out


def encoder_files() -> list[Path]:
    rels = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "modules.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "1_Pooling/config.json",
    ]
    return [MODEL_PATH / x for x in rels]


def input_paths_and_hashes() -> dict[str, str]:
    paths = [
        TRAIN_RESPONSES,
        TRAIN_RESPONSE_INDEX,
        TRAIN_RESPONSE_EMB,
        DEV_ALL,
        DEV_LONG20,
        MODEL_CODE,
        ENCODER_SCRIPT,
        *encoder_files(),
    ]
    for condition_paths in all_checkpoint_paths().values():
        paths.extend(condition_paths)
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError("Required frozen input missing:\n" + "\n".join(missing))
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def load_and_validate_text_data():
    if sha256_file(DEV_ALL) != EXPECTED_DEV_ALL_SHA:
        raise RuntimeError("Frozen dev_all SHA256 mismatch")
    if sha256_file(DEV_LONG20) != EXPECTED_DEV_LONG20_SHA:
        raise RuntimeError("Frozen dev_long20 SHA256 mismatch")
    responses = load_jsonl(TRAIN_RESPONSES)
    response_index = load_jsonl(TRAIN_RESPONSE_INDEX)
    dev = load_jsonl(DEV_ALL)
    long20 = load_jsonl(DEV_LONG20)

    if len(responses) != EXPECTED_RESPONSE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_RESPONSE_ROWS} train responses, got {len(responses)}")
    if len(response_index) != EXPECTED_RESPONSE_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_RESPONSE_ROWS} response index rows, got {len(response_index)}")
    if len(dev) != EXPECTED_QUERY_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_QUERY_ROWS} Dev rows, got {len(dev)}")
    if len(long20) != EXPECTED_LONG20_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_LONG20_ROWS} Dev-long20 rows, got {len(long20)}")

    required_dev = {"sample_id", "dialogue_id", "query", "gold_answer"}
    for name, rows in (("dev_all", dev), ("dev_long20", long20)):
        if not rows:
            raise RuntimeError(f"{name} is empty")
        missing_keys = required_dev - set(rows[0])
        if missing_keys:
            raise RuntimeError(f"{name} missing required keys: {sorted(missing_keys)}")
        if any("future_value_label" in r or "future_value_available" in r for r in rows):
            # The current frozen Dev schema should not include future labels.
            raise RuntimeError(f"Unexpected future-state label fields found in {name}; review protocol before proceeding")

    dev_ids = [str(r["sample_id"]) for r in dev]
    long_ids = [str(r["sample_id"]) for r in long20]
    if len(set(dev_ids)) != len(dev_ids) or len(set(long_ids)) != len(long_ids):
        raise RuntimeError("Duplicate sample_id found in Dev input")
    if not set(long_ids).issubset(set(dev_ids)):
        raise RuntimeError("dev_long20 is not a subset of dev_all by sample_id")

    response_norms = []
    response_uids = []
    for i, row in enumerate(responses):
        if "response_text" not in row or "response_uid" not in row:
            raise RuntimeError(f"Train response row {i} missing response_text/response_uid")
        n = norm_text(row["response_text"])
        if "normalized_response_text" in row and n != norm_text(row["normalized_response_text"]):
            raise RuntimeError(f"Normalization mismatch at train response row {i}")
        response_norms.append(n)
        response_uids.append(str(row["response_uid"]))
        idx = response_index[i]
        if int(idx.get("embedding_row", -1)) != i or str(idx.get("response_uid", "")) != str(row["response_uid"]):
            raise RuntimeError(f"Response embedding index order mismatch at row {i}")
    if len(set(response_norms)) != len(response_norms):
        raise RuntimeError("Train response file is not unique after protocol normalization")
    if len(set(response_uids)) != len(response_uids):
        raise RuntimeError("Duplicate response_uid in train response file")

    train_norm_to_idx = {n: i for i, n in enumerate(response_norms)}
    dev_gold_norms = [norm_text(r["gold_answer"]) for r in dev]
    match_idx = np.array([train_norm_to_idx.get(n, -1) for n in dev_gold_norms], dtype=np.int32)
    unique_gold = set(dev_gold_norms)
    unique_covered = len(unique_gold.intersection(train_norm_to_idx.keys()))
    coverage = {
        "dev_rows": len(dev),
        "gold_row_coverage_n": int(np.sum(match_idx >= 0)),
        "gold_row_coverage_rate": float(np.mean(match_idx >= 0)),
        "unique_gold_n": len(unique_gold),
        "unique_gold_covered_n": int(unique_covered),
        "unique_gold_coverage_rate": float(unique_covered / max(1, len(unique_gold))),
    }
    return responses, response_index, dev, long20, response_norms, train_norm_to_idx, match_idx, coverage


def encode_dev_embeddings(dev: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    from sentence_transformers import SentenceTransformer

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if QUERY_EMB_OUT.exists() or GOLD_EMB_OUT.exists() or EMBED_MANIFEST_OUT.exists():
        raise RuntimeError(
            "S8 embedding output collision. Do not overwrite existing artifacts. "
            f"Inspect: {QUERY_EMB_OUT}, {GOLD_EMB_OUT}, {EMBED_MANIFEST_OUT}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(MODEL_PATH), device=device)
    if model.max_seq_length != 512:
        raise RuntimeError(f"Unexpected BGE max_seq_length={model.max_seq_length}")
    if model.get_sentence_embedding_dimension() != DIM:
        raise RuntimeError(f"Unexpected BGE embedding dimension={model.get_sentence_embedding_dimension()}")

    q_texts = [str(r["query"]) for r in dev]
    g_texts = [str(r["gold_answer"]) for r in dev]
    q_emb = model.encode(
        q_texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    g_emb = model.encode(
        g_texts, batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    if q_emb.shape != (len(dev), DIM) or g_emb.shape != (len(dev), DIM):
        raise RuntimeError(f"Unexpected Dev embedding shapes: q={q_emb.shape}, gold={g_emb.shape}")
    if not np.isfinite(q_emb).all() or not np.isfinite(g_emb).all():
        raise RuntimeError("Non-finite values in Dev embeddings")
    if max(float(np.max(np.abs(np.linalg.norm(q_emb, axis=1) - 1))),
           float(np.max(np.abs(np.linalg.norm(g_emb, axis=1) - 1)))) > 1e-4:
        raise RuntimeError("Dev embeddings are not L2 normalized")

    q_tmp = QUERY_EMB_OUT.with_suffix(".npy.tmp")
    g_tmp = GOLD_EMB_OUT.with_suffix(".npy.tmp")
    with q_tmp.open("wb") as f:
        np.save(f, q_emb)
    with g_tmp.open("wb") as f:
        np.save(f, g_emb)
    os.replace(q_tmp, QUERY_EMB_OUT)
    os.replace(g_tmp, GOLD_EMB_OUT)
    manifest = {
        "artifact": "FSAR S8 Dev Query and Gold BGE Embeddings V1",
        "status": "PASS",
        "scope": "DEV_ONLY",
        "encoder": "BAAI/bge-large-zh-v1.5",
        "model_path": str(MODEL_PATH),
        "max_seq_length": 512,
        "dimension": DIM,
        "normalized": True,
        "query_source_sha256": sha256_file(DEV_ALL),
        "gold_source_sha256": sha256_file(DEV_ALL),
        "query_rows": len(dev),
        "gold_rows": len(dev),
        "query_embedding_sha256": sha256_file(QUERY_EMB_OUT),
        "gold_embedding_sha256": sha256_file(GOLD_EMB_OUT),
        "query_embedding_shape": list(q_emb.shape),
        "gold_embedding_shape": list(g_emb.shape),
        "test_payload_used": False,
    }
    atomic_json(EMBED_MANIFEST_OUT, manifest)
    return q_emb, g_emb


def build_protocol() -> dict[str, Any]:
    _, _, dev, long20, _, _, _, coverage = load_and_validate_text_data()
    if sha256_file(TRAIN_RESPONSE_EMB) != EXPECTED_RESPONSE_EMB_SHA:
        raise RuntimeError("Frozen TRAIN response embedding SHA256 mismatch")
    if not QUERY_EMB_OUT.is_file() or not GOLD_EMB_OUT.is_file() or not EMBED_MANIFEST_OUT.is_file():
        raise RuntimeError("Dev embeddings/manifest missing after preparation")

    checkpoint_spec = {}
    for condition, paths in all_checkpoint_paths().items():
        checkpoint_spec[condition] = [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)} for p in paths
        ]

    return {
        "project": "PsychResponse_FSAR",
        "phase": "S8",
        "artifact": "FSAR S8 External Dev Retrieval Evaluation Protocol V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "EXTERNAL_DEV_RETRIEVAL_ONLY",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evaluation_script_sha256": sha256_file(Path(__file__)),
        "source_and_model_file_sha256": input_paths_and_hashes(),
        "generated_embedding_sha256": {
            str(QUERY_EMB_OUT.relative_to(ROOT)): sha256_file(QUERY_EMB_OUT),
            str(GOLD_EMB_OUT.relative_to(ROOT)): sha256_file(GOLD_EMB_OUT),
            str(EMBED_MANIFEST_OUT.relative_to(ROOT)): sha256_file(EMBED_MANIFEST_OUT),
        },
        "checkpoint_spec": checkpoint_spec,
        "dataset": {
            "dev_all_path": str(DEV_ALL.relative_to(ROOT)),
            "dev_all_rows": len(dev),
            "dev_long20_path": str(DEV_LONG20.relative_to(ROOT)),
            "dev_long20_rows": len(long20),
            "dev_long20_is_subset_of_dev_all_by_sample_id": True,
            "gold_coverage_diagnostic": coverage,
        },
        "candidate_construction": {
            "base_candidate_pool": "61,502 unique normalized TRAIN counselor responses",
            "candidate_identity_normalization": {
                "unicode_normalization": "NFKC",
                "strip_leading_trailing_whitespace": True,
                "collapse_internal_whitespace": True,
                "lowercase": False,
                "remove_punctuation": False,
            },
            "per_query_rule": "Exclude any TRAIN candidate whose normalized text equals this query's Gold answer, then add exactly one encoded Gold answer as the positive candidate.",
            "candidate_count": "61,502 when Gold matches a TRAIN response identity; otherwise 61,503",
            "same_normalized_gold_excluded_from_negatives": True,
            "candidate_pool_is_query_specific_due_to_gold_injection": True,
            "interpretation_limit": "This is Gold-injected candidate-set ranking generalization, not retrieval coverage over the TRAIN response library alone.",
        },
        "conditions": [
            "DENSE_BGE",
            "RANK_ONLY",
            "R_FUTURE_BETA1",
            "R_FUTURE_BETA2",
            "R_FUTURE_BETA4",
            "R_FUTURE_BETA8",
            "QR_FUTURE_BETA1",
            "QR_FUTURE_BETA2",
            "QR_FUTURE_BETA4",
            "QR_FUTURE_BETA8",
        ],
        "ensemble": {
            "folds": 5,
            "aggregation": "Arithmetic mean of the five fold models' raw ranking scores before rank computation",
            "checkpoint_selection": "Use the already promoted best_v1 checkpoint for each fold; no retraining or reselection",
        },
        "metrics": {
            "primary": "Dialogue-macro MRR, R@1, R@5, R@10, R@20",
            "aggregation": "Average query metric within each dialogue, then unweighted mean across dialogues",
            "uncertainty": f"{BOOTSTRAP_REPLICATES} paired cluster bootstrap replicates resampling dialogues with replacement; seed={BOOTSTRAP_SEED}",
            "score_ties": "Conservative deterministic tie rule: all negative candidates with score >= Gold score are ranked before Gold",
            "contrasts": "Paired dialogue-cluster bootstrap contrasts against DENSE_BGE and RANK_ONLY",
        },
        "future_state_metrics": {
            "computed": False,
            "reason": "The frozen Dev target files contain no future_value_label or future_value_available fields.",
        },
        "dev_payload_used": True,
        "test_payload_used": False,
        "training_data_or_checkpoints_modified": False,
        "posthoc_success_gate": False,
        "output_paths": {
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "predictions": str(PREDICTIONS_OUT.relative_to(ROOT)),
            "report": str(REPORT_OUT.relative_to(ROOT)),
        },
    }


def get_model_class(condition: str):
    if condition == "RANK_ONLY":
        return RankOnlyModel
    if condition.startswith("R_FUTURE_"):
        return RFutureModel
    if condition.startswith("QR_FUTURE_"):
        return QRFutureModel
    raise ValueError(condition)


def load_condition_models(condition: str, device: torch.device):
    models = []
    for item in all_checkpoint_paths()[condition]:
        path = item
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise RuntimeError(f"Missing model_state_dict in {path}")
        model = get_model_class(condition)()
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device)
        model.eval()
        if model.ranker.rank_head.weight.shape != (1, 4 * DIM):
            raise RuntimeError(f"Unexpected rank head shape in {path}")
        if model.ranker.rank_head.bias is not None:
            raise RuntimeError(f"Expected bias-free rank head in {path}")
        models.append(model)
    return models


def fast_score_adapted(model, zq: torch.Tensor, zr: torch.Tensor) -> torch.Tensor:
    """Exact algebraic form of rank_head([zq,zr,abs(zq-zr),zq*zr])."""
    w = model.ranker.rank_head.weight[0]
    wq = w[:DIM]
    wr = w[DIM:2 * DIM]
    wa = w[2 * DIM:3 * DIM]
    wp = w[3 * DIM:4 * DIM]
    q_term = (zq * wq).sum(dim=-1, keepdim=True)
    r_term = (zr * wr).sum(dim=-1).unsqueeze(0)
    abs_term = (torch.abs(zq[:, None, :] - zr[None, :, :]) * wa).sum(dim=-1)
    prod_term = torch.matmul(zq * wp, zr.transpose(0, 1))
    return q_term + r_term + abs_term + prod_term


def validate_fast_scorer(model, device: torch.device) -> float:
    q = torch.randn(2, DIM, device=device)
    r = torch.randn(3, DIM, device=device)
    with torch.inference_mode():
        zq = model.ranker.adapt_query(q)
        zr = model.ranker.adapt_response(r)
        fast = fast_score_adapted(model, zq, zr)
        expanded = r.unsqueeze(0).expand(q.shape[0], -1, -1)
        reference = model.rank_scores(q, expanded)
        diff = float(torch.max(torch.abs(fast - reference)).item())
    if diff > 1e-5:
        raise RuntimeError(f"Fast scorer mismatch: max_abs_diff={diff}")
    return diff


def ranks_dense(q_np: np.ndarray, gold_np: np.ndarray, r_gpu: torch.Tensor,
                match_idx: np.ndarray, device: torch.device) -> np.ndarray:
    q_gpu = torch.from_numpy(q_np).to(device)
    gold_gpu = torch.from_numpy(gold_np).to(device)
    with torch.inference_mode():
        gold_scores = (q_gpu * gold_gpu).sum(dim=1)
        rank_counts = np.zeros(len(q_np), dtype=np.int64)
        for qs in range(0, len(q_np), Q_BATCH):
            qe = min(qs + Q_BATCH, len(q_np))
            qb = q_gpu[qs:qe]
            gs = gold_scores[qs:qe]
            local_counts = torch.zeros(qe - qs, dtype=torch.int64, device=device)
            for cs in range(0, r_gpu.shape[0], C_CHUNK):
                ce = min(cs + C_CHUNK, r_gpu.shape[0])
                scores = torch.matmul(qb, r_gpu[cs:ce].T)
                ge = scores >= gs[:, None]
                local_counts += ge.sum(dim=1)
                mi = match_idx[qs:qe]
                in_chunk = (mi >= cs) & (mi < ce)
                if np.any(in_chunk):
                    row_idx = torch.from_numpy(np.flatnonzero(in_chunk)).to(device)
                    col_idx = torch.from_numpy((mi[in_chunk] - cs).astype(np.int64)).to(device)
                    counted = scores[row_idx, col_idx] >= gs[row_idx]
                    local_counts[row_idx] -= counted.to(torch.int64)
            rank_counts[qs:qe] = local_counts.cpu().numpy()
    return (rank_counts + 1).astype(np.int32)


def ranks_learned(q_np: np.ndarray, gold_np: np.ndarray, r_gpu: torch.Tensor,
                  match_idx: np.ndarray, models: list[torch.nn.Module],
                  device: torch.device, condition: str) -> tuple[np.ndarray, float]:
    q_gpu = torch.from_numpy(q_np).to(device)
    gold_gpu = torch.from_numpy(gold_np).to(device)
    zq_folds = []
    zr_folds = []
    gold_scores_folds = []
    max_diff = 0.0
    with torch.inference_mode():
        for model in models:
            diff = validate_fast_scorer(model, device)
            max_diff = max(max_diff, diff)
            zq = model.ranker.adapt_query(q_gpu)
            zg = model.ranker.adapt_response(gold_gpu)
            gold_score = model.ranker.score_from_adapted(zq, zg)
            zr = model.ranker.adapt_response(r_gpu)
            zq_folds.append(zq)
            zr_folds.append(zr)
            gold_scores_folds.append(gold_score)
        gold_scores = torch.stack(gold_scores_folds, dim=0).mean(dim=0)
        rank_counts = np.zeros(len(q_np), dtype=np.int64)
        for qs in range(0, len(q_np), Q_BATCH):
            qe = min(qs + Q_BATCH, len(q_np))
            gs = gold_scores[qs:qe]
            local_counts = torch.zeros(qe - qs, dtype=torch.int64, device=device)
            for cs in range(0, r_gpu.shape[0], C_CHUNK):
                ce = min(cs + C_CHUNK, r_gpu.shape[0])
                score_sum = None
                zr_chunk_by_fold = [zr[cs:ce] for zr in zr_folds]
                for fold_i, model in enumerate(models):
                    score = fast_score_adapted(
                        model,
                        zq_folds[fold_i][qs:qe],
                        zr_chunk_by_fold[fold_i],
                    )
                    score_sum = score if score_sum is None else score_sum + score
                scores = score_sum / len(models)
                ge = scores >= gs[:, None]
                local_counts += ge.sum(dim=1)
                mi = match_idx[qs:qe]
                in_chunk = (mi >= cs) & (mi < ce)
                if np.any(in_chunk):
                    row_idx = torch.from_numpy(np.flatnonzero(in_chunk)).to(device)
                    col_idx = torch.from_numpy((mi[in_chunk] - cs).astype(np.int64)).to(device)
                    counted = scores[row_idx, col_idx] >= gs[row_idx]
                    local_counts[row_idx] -= counted.to(torch.int64)
            rank_counts[qs:qe] = local_counts.cpu().numpy()
    return (rank_counts + 1).astype(np.int32), diff


def query_metric_matrix(ranks: np.ndarray) -> np.ndarray:
    return np.column_stack([
        1.0 / ranks,
        (ranks <= 1).astype(np.float64),
        (ranks <= 5).astype(np.float64),
        (ranks <= 10).astype(np.float64),
        (ranks <= 20).astype(np.float64),
    ])


def dialogue_group_values(ranks: np.ndarray, dialogue_ids: np.ndarray):
    values = query_metric_matrix(ranks)
    unique = np.array(sorted(set(str(x) for x in dialogue_ids)), dtype=str)
    group_values = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    group_to_idx = {d: i for i, d in enumerate(unique.tolist())}
    sums = np.zeros_like(group_values)
    counts = np.zeros(len(unique), dtype=np.int64)
    for i, did in enumerate(dialogue_ids.astype(str)):
        gi = group_to_idx[did]
        sums[gi] += values[i]
        counts[gi] += 1
    group_values = sums / counts[:, None]
    return unique, group_values


def bootstrap_mean_ci(group_values: np.ndarray, rng: np.random.Generator):
    n = group_values.shape[0]
    estimate = group_values.mean(axis=0)
    boot = np.empty((BOOTSTRAP_REPLICATES, group_values.shape[1]), dtype=np.float64)
    for i in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        boot[i] = group_values[idx].mean(axis=0)
    lower = np.quantile(boot, 0.025, axis=0)
    upper = np.quantile(boot, 0.975, axis=0)
    return estimate, lower, upper


def evaluate_metrics(ranks_by_model: dict[str, np.ndarray], dialogue_ids: np.ndarray,
                     sample_ids: np.ndarray, subset_mask: np.ndarray, subset_name: str,
                     rng: np.random.Generator):
    idx = np.flatnonzero(subset_mask)
    dids = dialogue_ids[idx].astype(str)
    result = {"n_queries": int(len(idx)), "n_dialogues": int(len(set(dids))), "models": {}, "contrasts": {}}
    group_by_model = {}
    for name, all_ranks in ranks_by_model.items():
        _, gv = dialogue_group_values(all_ranks[idx], dids)
        group_by_model[name] = gv
        est, lo, hi = bootstrap_mean_ci(gv, rng)
        result["models"][name] = {
            "dialogue_macro": {
                metric: {"estimate": float(est[j]), "ci95_lower": float(lo[j]), "ci95_upper": float(hi[j])}
                for j, metric in enumerate(METRIC_NAMES)
            }
        }
    baselines = ["DENSE_BGE", "RANK_ONLY"]
    for name in ranks_by_model:
        if name in baselines:
            continue
        result["contrasts"][name] = {}
        for base in baselines:
            delta_g = group_by_model[name] - group_by_model[base]
            est, lo, hi = bootstrap_mean_ci(delta_g, rng)
            result["contrasts"][name][f"MINUS_{base}"] = {
                metric: {"delta": float(est[j]), "ci95_lower": float(lo[j]), "ci95_upper": float(hi[j])}
                for j, metric in enumerate(METRIC_NAMES)
            }
    return result


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_OUT.is_file():
        raise RuntimeError(f"Frozen protocol missing: {PROTOCOL_OUT}; run --prepare first")
    protocol = json.loads(PROTOCOL_OUT.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise RuntimeError("Protocol is not frozen before evaluation")
    if protocol.get("evaluation_script_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Evaluation script SHA changed since protocol freeze")
    for rel, expected in protocol["source_and_model_file_sha256"].items():
        p = ROOT / rel
        if not p.is_file() or sha256_file(p) != expected:
            raise RuntimeError(f"Frozen input SHA mismatch: {rel}")
    for rel, expected in protocol["generated_embedding_sha256"].items():
        p = ROOT / rel
        if not p.is_file() or sha256_file(p) != expected:
            raise RuntimeError(f"Generated embedding SHA mismatch: {rel}")
    return protocol


def evaluate():
    protocol = verify_protocol()
    for p in (SUMMARY_OUT, PREDICTIONS_OUT, REPORT_OUT,
              SUMMARY_OUT.with_suffix(SUMMARY_OUT.suffix + ".tmp"),
              PREDICTIONS_OUT.with_suffix(PREDICTIONS_OUT.suffix + ".tmp"),
              REPORT_OUT.with_suffix(REPORT_OUT.suffix + ".tmp")):
        if p.exists():
            raise RuntimeError(f"Evaluation output collision; refusing overwrite: {p}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device)
    if device.type == "cuda":
        print("GPU =", torch.cuda.get_device_name(0))

    _, _, dev, long20, response_norms, _, match_idx, coverage = load_and_validate_text_data()
    q_np = np.load(QUERY_EMB_OUT, allow_pickle=False)
    gold_np = np.load(GOLD_EMB_OUT, allow_pickle=False)
    r_np = np.load(TRAIN_RESPONSE_EMB, mmap_mode="r", allow_pickle=False)
    if q_np.shape != (len(dev), DIM) or gold_np.shape != (len(dev), DIM):
        raise RuntimeError("Dev embedding shape mismatch")
    if r_np.shape != (EXPECTED_RESPONSE_ROWS, DIM):
        raise RuntimeError(f"TRAIN response embedding shape mismatch: {r_np.shape}")
    if r_np.dtype != np.float32:
        raise RuntimeError(f"TRAIN response embedding dtype mismatch: {r_np.dtype}")

    r_gpu = torch.from_numpy(np.array(r_np, dtype=np.float32, copy=True)).to(device)
    dev_dialogue_ids = np.array([str(r["dialogue_id"]) for r in dev], dtype=str)
    dev_sample_ids = np.array([str(r["sample_id"]) for r in dev], dtype=str)
    long_ids = {str(r["sample_id"]) for r in long20}
    long_mask = np.array([sid in long_ids for sid in dev_sample_ids], dtype=bool)
    if int(long_mask.sum()) != len(long20):
        raise RuntimeError("Long20 subset alignment failed")

    ranks_by_model: dict[str, np.ndarray] = {}
    print("\n===== DENSE BASELINE =====")
    ranks_by_model["DENSE_BGE"] = ranks_dense(q_np, gold_np, r_gpu, match_idx, device)
    print("DENSE_BGE rank median =", float(np.median(ranks_by_model["DENSE_BGE"])))

    conditions = [
        "RANK_ONLY",
        "R_FUTURE_BETA1", "R_FUTURE_BETA2", "R_FUTURE_BETA4", "R_FUTURE_BETA8",
        "QR_FUTURE_BETA1", "QR_FUTURE_BETA2", "QR_FUTURE_BETA4", "QR_FUTURE_BETA8",
    ]
    fast_diffs = {}
    for condition in conditions:
        print(f"\n===== {condition} =====", flush=True)
        models = load_condition_models(condition, device)
        ranks, max_diff = ranks_learned(q_np, gold_np, r_gpu, match_idx, models, device, condition)
        name = "RANK_ONLY" if condition == "RANK_ONLY" else condition
        ranks_by_model[name] = ranks
        fast_diffs[name] = max_diff
        print("fast_scorer_max_abs_diff =", max_diff)
        print("rank median =", float(np.median(ranks)), "MRR micro =", float(np.mean(1.0 / ranks)), flush=True)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    all_mask = np.ones(len(dev), dtype=bool)
    metrics = {
        "dev_all": evaluate_metrics(ranks_by_model, dev_dialogue_ids, dev_sample_ids, all_mask, "dev_all", rng),
        "dev_long20": evaluate_metrics(ranks_by_model, dev_dialogue_ids, dev_sample_ids, long_mask, "dev_long20", rng),
    }
    summary = {
        "artifact": "FSAR S8 External Dev Retrieval Evaluation Results V1",
        "status": "PASS",
        "protocol_path": str(PROTOCOL_OUT.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_OUT),
        "scope": "EXTERNAL_DEV_RETRIEVAL_ONLY",
        "candidate_pool": protocol["candidate_construction"],
        "gold_coverage_diagnostic": coverage,
        "fast_scorer_max_abs_diff_by_condition": fast_diffs,
        "datasets": metrics,
        "future_state_metrics_computed": False,
        "dev_payload_used": True,
        "test_payload_used": False,
        "training_artifacts_modified": False,
        "interpretation_note": "Results evaluate ranking on a per-query Gold-injected candidate set. They do not measure strict retrieval coverage over the TRAIN response library alone.",
    }

    report_lines = [
        "FSAR S8 EXTERNAL DEV RETRIEVAL EVALUATION",
        f"protocol_sha256 = {summary['protocol_sha256']}",
        f"dev_all_rows = {len(dev)}; dev_all_dialogues = {len(set(dev_dialogue_ids))}",
        f"dev_long20_rows = {int(long_mask.sum())}; dev_long20_dialogues = {len(set(dev_dialogue_ids[long_mask]))}",
        f"gold_row_coverage_train_library = {coverage['gold_row_coverage_n']}/{coverage['dev_rows']} ({coverage['gold_row_coverage_rate']:.4%})",
        f"gold_unique_coverage_train_library = {coverage['unique_gold_covered_n']}/{coverage['unique_gold_n']} ({coverage['unique_gold_coverage_rate']:.4%})",
        "metrics = dialogue-macro MRR, R@1, R@5, R@10, R@20; 95% dialogue-cluster bootstrap CI",
        "tie_rule = conservative; all negatives with score >= Gold are counted ahead of Gold",
        "future_state_metrics_computed = NO (Dev labels unavailable)",
        "test_payload_used = NO",
        "",
    ]
    for ds_name, ds in metrics.items():
        report_lines.append(f"===== {ds_name} (n={ds['n_queries']}, dialogues={ds['n_dialogues']}) =====")
        for model_name, model_result in ds["models"].items():
            bits = [model_name]
            for metric in METRIC_NAMES:
                m = model_result["dialogue_macro"][metric]
                bits.append(f"{metric}={m['estimate']:.6f} [{m['ci95_lower']:.6f}, {m['ci95_upper']:.6f}]")
            report_lines.append(" | ".join(bits))
        report_lines.append("----- Paired contrasts vs Dense and Rank-only -----")
        for model_name, contrast_dict in ds["contrasts"].items():
            for base, vals in contrast_dict.items():
                bits = [f"{model_name} {base}"]
                for metric in METRIC_NAMES:
                    m = vals[metric]
                    bits.append(f"{metric}={m['delta']:+.6f} [{m['ci95_lower']:+.6f}, {m['ci95_upper']:+.6f}]")
                report_lines.append(" | ".join(bits))
        report_lines.append("")

    summary_tmp = SUMMARY_OUT.with_suffix(SUMMARY_OUT.suffix + ".tmp")
    pred_tmp = PREDICTIONS_OUT.with_suffix(PREDICTIONS_OUT.suffix + ".tmp")
    report_tmp = REPORT_OUT.with_suffix(REPORT_OUT.suffix + ".tmp")
    atomic_json(summary_tmp, summary)
    with pred_tmp.open("wb") as f:
        np.savez_compressed(
            f,
            sample_ids=dev_sample_ids,
            dialogue_ids=dev_dialogue_ids,
            dev_long20_mask=long_mask,
            match_idx=match_idx,
            **{f"ranks__{name}": ranks for name, ranks in ranks_by_model.items()},
        )
    report_tmp.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(summary_tmp, SUMMARY_OUT)
    os.replace(pred_tmp, PREDICTIONS_OUT)
    os.replace(report_tmp, REPORT_OUT)

    print("\n===== S8 OUTPUTS =====")
    print("S8_EVALUATION = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("PREDICTIONS =", PREDICTIONS_OUT)
    print("REPORT =", REPORT_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_OUT))
    for ds_name, ds in metrics.items():
        print(f"\n[{ds_name}]")
        for model_name, model_result in ds["models"].items():
            vals = model_result["dialogue_macro"]
            print(model_name, json.dumps({m: round(vals[m]["estimate"], 6) for m in METRIC_NAMES}))


def prepare():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_OUT.exists():
        raise RuntimeError(f"Protocol already exists; refusing overwrite: {PROTOCOL_OUT}")
    for p in (QUERY_EMB_OUT, GOLD_EMB_OUT, EMBED_MANIFEST_OUT,
              SUMMARY_OUT, PREDICTIONS_OUT, REPORT_OUT):
        if p.exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {p}")
    print("===== S8 PREPARE: READ-ONLY INPUT AUDIT =====")
    if sha256_file(TRAIN_RESPONSE_EMB) != EXPECTED_RESPONSE_EMB_SHA:
        raise RuntimeError("Frozen TRAIN response embedding SHA256 mismatch")
    _, _, dev, long20, _, _, _, coverage = load_and_validate_text_data()
    print("dev_rows =", len(dev))
    print("dev_long20_rows =", len(long20))
    print("gold_coverage =", json.dumps(coverage, ensure_ascii=False))
    print("test_payload_used = NO")
    print("===== ENCODE DEV QUERY AND GOLD TEXTS =====")
    encode_dev_embeddings(dev)
    protocol = build_protocol()
    atomic_json(PROTOCOL_OUT, protocol)
    print("S8_PROTOCOL_FROZEN = PASS")
    print("EVALUATION_STARTED = NO")
    print("PROTOCOL =", PROTOCOL_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_OUT))
    print("QUERY_EMBEDDINGS =", QUERY_EMB_OUT)
    print("GOLD_EMBEDDINGS =", GOLD_EMB_OUT)
    print("PROTOCOL_STATUS =", protocol["status"])
    print("NO_TEST_PAYLOAD_READ = YES")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Encode Dev texts and freeze S8 protocol; no ranking evaluation")
    group.add_argument("--evaluate", action="store_true", help="Verify frozen protocol and run S8 ranking evaluation")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
