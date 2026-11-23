#!/usr/bin/env python3
"""S12C STEP 2F-5D-2: Dev full-library fold-vs-ensemble ranking audit.

DEV ONLY. Uses the existing S8 Dev query/Gold embeddings and frozen TRAIN
response library. No Test payload is opened. No training/checkpoint selection.
The per-query candidate set follows S8: remove the TRAIN response whose
normalized identity equals this query's Gold, then inject the separately
encoded Gold as the positive candidate.

Primary tie rule is the conservative external-evaluation rule (score >= Gold).
STRICT_GREATER ranks and exact tie counts are also saved as diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RFutureFullMatchedModel,
    RFutureNoProductMatchedModel,
    QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

DIM = 1024
QUERY_BATCH = 16
CANDIDATE_CHUNK = 2048
BOOTSTRAP_SEED = 20261027

DEV_ALL = ROOT / "data/splits/psydial_targets_dev_all.jsonl"
DEV_LONG20 = ROOT / "data/splits/psydial_targets_dev_long20.jsonl"
TRAIN_RESPONSES = ROOT / "data/canonical/fsar_s2/train_unique_responses_v1.jsonl"
TRAIN_INDEX = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"
TRAIN_EMB = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"

S8_DIR = ROOT / "results/phase_s8"
QUERY_EMB = S8_DIR / "dev_all_query_embeddings_bge_v1.npy"
GOLD_EMB = S8_DIR / "dev_all_gold_embeddings_bge_v1.npy"
EMB_MANIFEST = S8_DIR / "dev_all_embedding_manifest_v1.json"

MODEL_CODE = ROOT / "src/models/s12c_matched_ablation_models.py"
ARCH_PROTOCOL = ROOT / "protocols/fsar_s12c_matched_ablation_v1.json"

OUT_DIR = ROOT / "results/phase_s12c/step_2f5d_fold_vs_ensemble_audit_v1"
SUMMARY_OUT = OUT_DIR / "summary.json"
REPORT_OUT = OUT_DIR / "report.txt"
PRED_OUT = OUT_DIR / "predictions.npz"
MANIFEST_OUT = OUT_DIR / "run_manifest.json"

EXPECTED_HASHES = {
    str(DEV_ALL): "8909877897573c0d7fdf517ae53e739f4a68f96c89fb93819282521349562ca5",
    str(DEV_LONG20): "6721ffa84c55ab63942f0a173fc4d5747f0190ce4394a57677ba70f98138484d",
    str(TRAIN_RESPONSES): "6ed2a5807c7a3bdcb30b7b6b947265129b19925a2cf7e8872fa36231b4cab25e",
    str(TRAIN_INDEX): "fd8263c8fd00b4d8d23b915c7f241fb84d1d2ffc7b24c747bf34e4a9b8e47274",
    str(TRAIN_EMB): "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f",
    str(MODEL_CODE): "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c",
}

CLASSES = {
    "RANK_ONLY": RankOnlyFullMatchedModel,
    "R_FUTURE_FULL": RFutureFullMatchedModel,
    "R_FUTURE_NO_PRODUCT": RFutureNoProductMatchedModel,
    "QR_FUTURE_FULL": QRFutureFullMatchedModel,
    "QR_FUTURE_NO_PRODUCT": QRFutureNoProductMatchedModel,
}
CONDITIONS = list(CLASSES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.strip()
    return re.sub(r"\s+", " ", text)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    raise RuntimeError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
    return rows


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def checkpoint_paths(condition: str):
    if condition == "RANK_ONLY":
        base = ROOT / "results/phase_s4/checkpoints/rank_only"
        slug = "rank_only"
    else:
        base = ROOT / "results/phase_s12c/checkpoints" / condition.lower()
        slug = condition.lower()
    return [
        base / f"{slug}_fold{fold}_best_v1.pt"
        for fold in range(5)
    ]


def load_and_validate_inputs():
    for path in [
        DEV_ALL, DEV_LONG20, TRAIN_RESPONSES, TRAIN_INDEX, TRAIN_EMB,
        QUERY_EMB, GOLD_EMB, EMB_MANIFEST, MODEL_CODE, ARCH_PROTOCOL,
    ]:
        require(path.is_file(), f"Required input missing: {path}")

    input_hashes = {}
    for raw_path, expected in EXPECTED_HASHES.items():
        path = Path(raw_path)
        got = sha256_file(path)
        require(got == expected, f"Frozen input SHA256 mismatch: {path}: {got}")
        input_hashes[str(path.relative_to(ROOT))] = got
        print("HASH PASS:", path.relative_to(ROOT), flush=True)

    protocol = json.loads(ARCH_PROTOCOL.read_text(encoding="utf-8"))
    require(protocol.get("status") == "FROZEN_BEFORE_TRAINING",
            "S12C architecture protocol status mismatch")
    require(protocol.get("source_artifacts", {}).get("s12c_model_sha256")
            == EXPECTED_HASHES[str(MODEL_CODE)],
            "S12C model/protocol hash mismatch")

    dev = load_jsonl(DEV_ALL)
    long20 = load_jsonl(DEV_LONG20)
    responses = load_jsonl(TRAIN_RESPONSES)
    index = load_jsonl(TRAIN_INDEX)

    require(len(dev) == 9070, f"Unexpected Dev row count: {len(dev)}")
    require(len(long20) == 4375, f"Unexpected Dev Long20 row count: {len(long20)}")
    require(len(responses) == 61502, f"Unexpected response row count: {len(responses)}")
    require(len(index) == len(responses), "Response/index row count mismatch")

    required = {"sample_id", "dialogue_id", "query", "gold_answer"}
    for name, rows in (("Dev", dev), ("Long20", long20)):
        require(bool(rows), f"{name} is empty")
        require(required.issubset(rows[0]), f"{name} missing required fields")
        ids = [str(x["sample_id"]) for x in rows]
        require(len(ids) == len(set(ids)), f"{name} duplicate sample_id")

    dev_ids = [str(x["sample_id"]) for x in dev]
    long_ids = [str(x["sample_id"]) for x in long20]
    require(set(long_ids).issubset(set(dev_ids)), "Long20 is not a Dev subset")

    response_norms = []
    response_uids = []
    for i, (row, idx) in enumerate(zip(responses, index)):
        require("response_text" in row and "response_uid" in row,
                f"Response row {i} missing required fields")
        n = norm_text(row["response_text"])
        if "normalized_response_text" in row:
            require(n == norm_text(row["normalized_response_text"]),
                    f"Response normalization mismatch at row {i}")
        response_norms.append(n)
        response_uids.append(str(row["response_uid"]))
        require(int(idx.get("embedding_row", -1)) == i,
                f"Response embedding row mismatch at {i}")
        require(str(idx.get("response_uid", "")) == str(row["response_uid"]),
                f"Response UID mismatch at {i}")
        require(int(idx.get("occurrence_count", -1)) == int(row.get("occurrence_count", -2)),
                f"Response occurrence_count mismatch at {i}")

    require(len(set(response_norms)) == len(response_norms),
            "Train responses are not unique after frozen normalization")
    require(len(set(response_uids)) == len(response_uids),
            "Duplicate response_uid")

    norm_to_idx = {n: i for i, n in enumerate(response_norms)}
    match_idx = np.asarray(
        [norm_to_idx.get(norm_text(x["gold_answer"]), -1) for x in dev],
        dtype=np.int32,
    )

    q_np = np.load(QUERY_EMB, allow_pickle=False)
    g_np = np.load(GOLD_EMB, allow_pickle=False)
    r_np = np.load(TRAIN_EMB, mmap_mode="r", allow_pickle=False)

    require(q_np.shape == (len(dev), DIM) and q_np.dtype == np.float32,
            f"Dev query embedding shape/dtype mismatch: {q_np.shape}/{q_np.dtype}")
    require(g_np.shape == (len(dev), DIM) and g_np.dtype == np.float32,
            f"Dev Gold embedding shape/dtype mismatch: {g_np.shape}/{g_np.dtype}")
    require(r_np.shape == (len(responses), DIM) and r_np.dtype == np.float32,
            f"Train response embedding shape/dtype mismatch: {r_np.shape}/{r_np.dtype}")
    require(np.isfinite(q_np).all() and np.isfinite(g_np).all(),
            "Non-finite Dev embeddings")
    require(np.isfinite(r_np).all(), "Non-finite response embeddings")

    manifest = json.loads(EMB_MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("status") == "PASS", "S8 embedding manifest status mismatch")
    require(manifest.get("scope") == "DEV_ONLY", "S8 embedding scope mismatch")
    require(manifest.get("test_payload_used") is False, "S8 manifest indicates Test payload")
    require(manifest.get("query_source_sha256") == sha256_file(DEV_ALL),
            "Query embedding source hash mismatch")
    require(manifest.get("gold_source_sha256") == sha256_file(DEV_ALL),
            "Gold embedding source hash mismatch")
    require(manifest.get("query_embedding_sha256") == sha256_file(QUERY_EMB),
            "Query embedding hash mismatch")
    require(manifest.get("gold_embedding_sha256") == sha256_file(GOLD_EMB),
            "Gold embedding hash mismatch")

    q_norm_err = float(np.max(np.abs(np.linalg.norm(q_np, axis=1) - 1.0)))
    g_norm_err = float(np.max(np.abs(np.linalg.norm(g_np, axis=1) - 1.0)))
    r_norm_err = float(np.max(np.abs(np.linalg.norm(r_np, axis=1) - 1.0)))
    require(max(q_norm_err, g_norm_err, r_norm_err) <= 1e-4,
            "Embedding normalization check failed")

    long_set = set(long_ids)
    long_mask = np.asarray([str(x["sample_id"]) in long_set for x in dev], dtype=bool)
    require(int(long_mask.sum()) == len(long20), "Long20 row mapping mismatch")

    print("DEV_ROWS =", len(dev))
    print("DEV_DIALOGUES =", len(set(str(x["dialogue_id"]) for x in dev)))
    print("DEV_LONG20_ROWS =", int(long_mask.sum()))
    print("TRAIN_RESPONSE_ROWS =", len(responses))
    print("GOLD_MATCH_ROWS =", int((match_idx >= 0).sum()))
    print("GOLD_UNMATCHED_ROWS =", int((match_idx < 0).sum()))
    print("CANDIDATE_COUNTS =", {
        61502: int((match_idx >= 0).sum()),
        61503: int((match_idx < 0).sum()),
    })
    print("NORMALIZATION_MAX_ERRORS =", {
        "query": q_norm_err, "gold": g_norm_err, "response": r_norm_err
    })
    print("TEST_PAYLOAD_USED = NO")
    return dev, long_mask, match_idx, q_np, g_np, r_np, input_hashes


def fast_score(model, zq, zr):
    w = model.ranker.rank_head.weight[0]
    if model.ranker.rank_head.bias is not None:
        raise RuntimeError("Unexpected ranking-head bias")
    a = (zq * w[:DIM]).sum(-1, keepdim=True)
    b = (zr * w[DIM:2 * DIM]).sum(-1).unsqueeze(0)
    c = (torch.abs(zq[:, None, :] - zr[None, :, :]) * w[2 * DIM:3 * DIM]).sum(-1)
    if w.numel() == 4 * DIM:
        return a + b + c + torch.matmul(zq * w[3 * DIM:4 * DIM], zr.T)
    if w.numel() == 3 * DIM:
        return a + b + c
    raise RuntimeError(f"Unexpected rank-head dimension: {w.numel()}")


def validate_fast_scorer(model, device):
    with torch.inference_mode():
        q = torch.randn(2, DIM, device=device)
        r = torch.randn(3, DIM, device=device)
        zq = model.ranker.adapt_query(q)
        zr = model.ranker.adapt_response(r)
        fast = fast_score(model, zq, zr)
        zqe = zq[:, None, :].expand(-1, zr.shape[0], -1)
        zre = zr[None, :, :].expand(zq.shape[0], -1, -1)
        ref = model.ranker.score_from_adapted(zqe, zre)
        return float((fast - ref).abs().max().item())


def load_models(condition, device):
    paths = checkpoint_paths(condition)
    require(all(p.is_file() for p in paths),
            f"{condition}: one or more checkpoints missing")
    models, hashes = [], []
    for fold, path in enumerate(paths):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        require(ck.get("condition") == condition,
                f"{condition} fold {fold}: checkpoint condition metadata mismatch")
        require(int(ck.get("fold", -1)) == fold,
                f"{condition} fold {fold}: checkpoint fold metadata mismatch")
        model = CLASSES[condition]()
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device).eval()
        expected_dim = 3072 if "NO_PRODUCT" in condition else 4096
        require(int(model.ranker.rank_head.weight.shape[1]) == expected_dim,
                f"{condition} fold {fold}: rank-head dimension mismatch")
        diff = validate_fast_scorer(model, device)
        require(diff <= 1e-5,
                f"{condition} fold {fold}: fast scorer mismatch {diff}")
        print(f"  fold={fold} LOAD_PASS fast_diff={diff:.3e} sha256={sha256_file(path)}",
              flush=True)
        models.append(model)
        hashes.append({
            "fold": fold,
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "best_epoch": int(ck.get("best_epoch", -1)),
        })
    return models, hashes


def rank_dense_full(q_np, g_np, r_gpu, match_idx, device):
    q = torch.from_numpy(q_np).to(device)
    g = torch.from_numpy(g_np).to(device)
    n = len(q_np)
    ge_out = np.zeros(n, dtype=np.int32)
    gt_out = np.zeros(n, dtype=np.int32)
    tie_out = np.zeros(n, dtype=np.int32)
    with torch.inference_mode():
        gold = (q * g).sum(1)
        for qs in range(0, n, QUERY_BATCH):
            qe = min(qs + QUERY_BATCH, n)
            ge = torch.zeros(qe - qs, dtype=torch.int64, device=device)
            gt = torch.zeros_like(ge)
            ties = torch.zeros_like(ge)
            mi = match_idx[qs:qe]
            for cs in range(0, r_gpu.shape[0], CANDIDATE_CHUNK):
                ce = min(cs + CANDIDATE_CHUNK, r_gpu.shape[0])
                scores = q[qs:qe] @ r_gpu[cs:ce].T
                gs = gold[qs:qe, None]
                ge += (scores >= gs).sum(1)
                gt += (scores > gs).sum(1)
                ties += (scores == gs).sum(1)
                mask = (mi >= cs) & (mi < ce)
                if np.any(mask):
                    ri = torch.from_numpy(np.flatnonzero(mask)).to(device)
                    ci = torch.from_numpy((mi[mask] - cs).astype(np.int64)).to(device)
                    matched = scores[ri, ci]
                    ge[ri] -= (matched >= gold[qs:qe][ri]).to(torch.int64)
                    gt[ri] -= (matched > gold[qs:qe][ri]).to(torch.int64)
                    ties[ri] -= (matched == gold[qs:qe][ri]).to(torch.int64)
            ge_out[qs:qe] = (ge + 1).cpu().numpy()
            gt_out[qs:qe] = (gt + 1).cpu().numpy()
            tie_out[qs:qe] = ties.cpu().numpy()
            if qe % 512 == 0 or qe == n:
                print(f"  DENSE_BGE: queries {qe}/{n}", flush=True)
    return ge_out, gt_out, tie_out


def rank_condition_full(condition, models, q_np, g_np, r_gpu, match_idx, device):
    n = len(q_np)
    zq_folds, zr_folds, gold_scores = [], [], []
    with torch.inference_mode():
        q_gpu = torch.from_numpy(q_np).to(device)
        g_gpu = torch.from_numpy(g_np).to(device)
        for model in models:
            zq = model.ranker.adapt_query(q_gpu)
            zg = model.ranker.adapt_response(g_gpu)
            zr = model.ranker.adapt_response(r_gpu)
            gs = model.ranker.score_from_adapted(zq, zg)
            zq_folds.append(zq)
            zr_folds.append(zr)
            gold_scores.append(gs)

        gold_mean = torch.stack(gold_scores, dim=0).mean(0)

        fold_ge = [np.zeros(n, dtype=np.int32) for _ in models]
        fold_gt = [np.zeros(n, dtype=np.int32) for _ in models]
        fold_ties = [np.zeros(n, dtype=np.int32) for _ in models]
        ens_ge = np.zeros(n, dtype=np.int32)
        ens_gt = np.zeros(n, dtype=np.int32)
        ens_ties = np.zeros(n, dtype=np.int32)

        for qs in range(0, n, QUERY_BATCH):
            qe = min(qs + QUERY_BATCH, n)
            bsz = qe - qs
            local_ge = [torch.zeros(bsz, dtype=torch.int64, device=device) for _ in models]
            local_gt = [torch.zeros_like(local_ge[0]) for _ in models]
            local_ties = [torch.zeros_like(local_ge[0]) for _ in models]
            local_ens_ge = torch.zeros(bsz, dtype=torch.int64, device=device)
            local_ens_gt = torch.zeros_like(local_ens_ge)
            local_ens_ties = torch.zeros_like(local_ens_ge)
            local_gold_mean = gold_mean[qs:qe]

            for cs in range(0, r_gpu.shape[0], CANDIDATE_CHUNK):
                ce = min(cs + CANDIDATE_CHUNK, r_gpu.shape[0])
                score_sum = torch.zeros(
                    (bsz, ce - cs), dtype=torch.float32, device=device
                )
                mi = match_idx[qs:qe]
                mask = (mi >= cs) & (mi < ce)
                row_idx = col_idx = None
                if np.any(mask):
                    row_idx = torch.from_numpy(np.flatnonzero(mask)).to(device)
                    col_idx = torch.from_numpy((mi[mask] - cs).astype(np.int64)).to(device)

                for fi, model in enumerate(models):
                    scores = fast_score(
                        model,
                        zq_folds[fi][qs:qe],
                        zr_folds[fi][cs:ce],
                    )
                    gs = gold_scores[fi][qs:qe]
                    ge = scores >= gs[:, None]
                    gt = scores > gs[:, None]
                    eq = scores == gs[:, None]
                    local_ge[fi] += ge.sum(1)
                    local_gt[fi] += gt.sum(1)
                    local_ties[fi] += eq.sum(1)

                    if row_idx is not None:
                        matched = scores[row_idx, col_idx]
                        local_ge[fi][row_idx] -= (matched >= gs[row_idx]).to(torch.int64)
                        local_gt[fi][row_idx] -= (matched > gs[row_idx]).to(torch.int64)
                        local_ties[fi][row_idx] -= (matched == gs[row_idx]).to(torch.int64)

                    score_sum += scores
                    del scores, ge, gt, eq

                score_mean = score_sum / len(models)
                ens_ge_mask = score_mean >= local_gold_mean[:, None]
                ens_gt_mask = score_mean > local_gold_mean[:, None]
                ens_eq_mask = score_mean == local_gold_mean[:, None]
                local_ens_ge += ens_ge_mask.sum(1)
                local_ens_gt += ens_gt_mask.sum(1)
                local_ens_ties += ens_eq_mask.sum(1)
                if row_idx is not None:
                    matched_mean = score_mean[row_idx, col_idx]
                    local_ens_ge[row_idx] -= (matched_mean >= local_gold_mean[row_idx]).to(torch.int64)
                    local_ens_gt[row_idx] -= (matched_mean > local_gold_mean[row_idx]).to(torch.int64)
                    local_ens_ties[row_idx] -= (matched_mean == local_gold_mean[row_idx]).to(torch.int64)
                del score_sum, score_mean, ens_ge_mask, ens_gt_mask, ens_eq_mask

            for fi in range(len(models)):
                fold_ge[fi][qs:qe] = (local_ge[fi] + 1).cpu().numpy()
                fold_gt[fi][qs:qe] = (local_gt[fi] + 1).cpu().numpy()
                fold_ties[fi][qs:qe] = local_ties[fi].cpu().numpy()
            ens_ge[qs:qe] = (local_ens_ge + 1).cpu().numpy()
            ens_gt[qs:qe] = (local_ens_gt + 1).cpu().numpy()
            ens_ties[qs:qe] = local_ens_ties.cpu().numpy()

            if qe % 512 == 0 or qe == n:
                print(
                    f"  {condition}: queries {qe}/{n}",
                    flush=True,
                )

    result = {}
    for fi in range(len(models)):
        require(np.array_equal(fold_ge[fi] - fold_gt[fi], fold_ties[fi]),
                f"{condition} fold {fi}: tie identity check failed")
        result[f"fold{fi}"] = {
            "rank_ge": fold_ge[fi],
            "rank_gt": fold_gt[fi],
            "ties": fold_ties[fi],
        }
    require(np.array_equal(ens_ge - ens_gt, ens_ties),
            f"{condition} ensemble: tie identity check failed")
    result["ensemble"] = {
        "rank_ge": ens_ge,
        "rank_gt": ens_gt,
        "ties": ens_ties,
    }
    return result


def query_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    return np.column_stack([
        1.0 / ranks,
        (ranks <= 1).astype(np.float64),
        (ranks <= 5).astype(np.float64),
        (ranks <= 10).astype(np.float64),
        (ranks <= 20).astype(np.float64),
    ])


def dialogue_values(ranks, dialogue_ids):
    qm = query_metrics(ranks)
    uniq, inv = np.unique(dialogue_ids, return_inverse=True)
    sums = np.zeros((len(uniq), qm.shape[1]), dtype=np.float64)
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    np.add.at(sums, inv, qm)
    return sums / counts[:, None], uniq


def macro_metrics(ranks, dialogue_ids):
    gv, uniq = dialogue_values(ranks, dialogue_ids)
    return {
        "n_queries": int(len(ranks)),
        "n_dialogues": int(len(uniq)),
        "dialogue_macro": {
            name: float(gv[:, i].mean())
            for i, name in enumerate(["MRR", "R@1", "R@5", "R@10", "R@20"])
        },
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "rank_p90": float(np.quantile(ranks, 0.90)),
        "rank_p99": float(np.quantile(ranks, 0.99)),
    }


def pairwise_fold_rank_agreement(fold_ranks):
    names = list(fold_ranks)
    correlations, exact_rates, median_abs = [], [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = fold_ranks[names[i]].astype(np.float64)
            b = fold_ranks[names[j]].astype(np.float64)
            corr = float(np.corrcoef(a, b)[0, 1])
            correlations.append(corr)
            exact_rates.append(float(np.mean(a == b)))
            median_abs.append(float(np.median(np.abs(a - b))))
    return {
        "pairwise_rank_vector_pearson_mean": float(np.mean(correlations)),
        "pairwise_rank_vector_pearson_min": float(np.min(correlations)),
        "pairwise_exact_rank_agreement_mean": float(np.mean(exact_rates)),
        "pairwise_median_absolute_rank_difference_mean": float(np.mean(median_abs)),
        "pair_count": len(correlations),
    }


def summarize_condition_rankings(condition_result, dialogue_ids, long_mask):
    output = {}
    for ds_name, mask in (
        ("dev_all", np.ones(len(dialogue_ids), dtype=bool)),
        ("dev_long20", long_mask),
    ):
        folds = {
            key: value["rank_ge"][mask]
            for key, value in condition_result.items()
            if key.startswith("fold")
        }
        ens = condition_result["ensemble"]["rank_ge"][mask]
        dids = dialogue_ids[mask]
        fold_metrics = {k: macro_metrics(v, dids) for k, v in folds.items()}
        ensemble_metrics = macro_metrics(ens, dids)
        metric_names = ["MRR", "R@1", "R@5", "R@10", "R@20"]
        mean_fold = {
            m: float(np.mean([
                fold_metrics[f"fold{i}"]["dialogue_macro"][m]
                for i in range(5)
            ]))
            for m in metric_names
        }
        delta = {
            m: ensemble_metrics["dialogue_macro"][m] - mean_fold[m]
            for m in metric_names
        }
        output[ds_name] = {
            "folds": fold_metrics,
            "ensemble": ensemble_metrics,
            "mean_single_fold_dialogue_macro": mean_fold,
            "ensemble_minus_mean_single_fold": delta,
            "fold_rank_agreement": pairwise_fold_rank_agreement(folds),
        }
    return output


def evaluate():
    require(not OUT_DIR.exists(),
            f"Output collision; refusing to overwrite: {OUT_DIR}")
    for p in (SUMMARY_OUT, REPORT_OUT, PRED_OUT, MANIFEST_OUT):
        require(not p.exists(), f"Output collision: {p}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "This full-library diagnostic requires CUDA")
    print("GPU =", torch.cuda.get_device_name(0), flush=True)

    dev, long_mask, match_idx, q_np, g_np, r_np, input_hashes = load_and_validate_inputs()
    dialogue_ids = np.asarray([str(x["dialogue_id"]) for x in dev], dtype=str)

    print("\n===== DENSE BGE BASELINE =====", flush=True)
    r_gpu = torch.from_numpy(np.array(r_np, dtype=np.float32, copy=True)).to(device)
    dense_ge, dense_gt, dense_ties = rank_dense_full(
        q_np, g_np, r_gpu, match_idx, device
    )
    require(np.array_equal(dense_ge - dense_gt, dense_ties),
            "Dense tie identity check failed")
    dense_result = {
        "rank_ge": dense_ge,
        "rank_gt": dense_gt,
        "ties": dense_ties,
    }
    print("DENSE_BGE COMPLETE; median_rank =", float(np.median(dense_ge)), flush=True)

    all_results = {}
    checkpoint_manifest = {}
    scorer_max_diff = {}

    for condition in CONDITIONS:
        print(f"\n===== CONDITION {condition} =====", flush=True)
        models, ck_hashes = load_models(condition, device)
        checkpoint_manifest[condition] = ck_hashes
        result = rank_condition_full(
            condition, models, q_np, g_np, r_gpu, match_idx, device
        )
        all_results[condition] = result
        scorer_max_diff[condition] = max(
            validate_fast_scorer(m, device) for m in models
        )
        print(
            f"{condition} COMPLETE "
            f"ensemble_median_rank={np.median(result['ensemble']['rank_ge']):.1f} "
            f"fast_diff={scorer_max_diff[condition]:.3e}",
            flush=True,
        )
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dense_metrics = {}
    for ds_name, mask in (
        ("dev_all", np.ones(len(dev), dtype=bool)),
        ("dev_long20", long_mask),
    ):
        dense_metrics[ds_name] = macro_metrics(dense_ge[mask], dialogue_ids[mask])

    condition_summaries = {}
    for condition, result in all_results.items():
        condition_summaries[condition] = summarize_condition_rankings(
            result, dialogue_ids, long_mask
        )

    summary = {
        "artifact": "S12C STEP 2F-5D-2 Dev Full-Library Fold-vs-Ensemble Audit V1",
        "status": "PASS",
        "scope": "DEV_ONLY_FULL_LIBRARY_RANKING_DIAGNOSTIC",
        "candidate_construction": {
            "base_pool": "61,502 unique normalized TRAIN counselor responses",
            "per_query_rule": "Exclude the matching normalized TRAIN Gold identity, then inject the separately encoded Gold answer",
            "candidate_counts": {
                "61502": int((match_idx >= 0).sum()),
                "61503": int((match_idx < 0).sum()),
            },
            "interpretation": "Gold-injected candidate-set ranking generalization; not strict TRAIN-library retrieval coverage",
        },
        "dataset": {
            "dev_rows": len(dev),
            "dev_dialogues": int(len(np.unique(dialogue_ids))),
            "dev_long20_rows": int(long_mask.sum()),
            "dev_long20_dialogues": int(len(np.unique(dialogue_ids[long_mask]))),
        },
        "ranking": {
            "primary_tie_rule": "CONSERVATIVE_GREATER_OR_EQUAL",
            "sensitivity_tie_rule": "STRICT_GREATER",
            "rank_definition_ge": "1 + COUNT(negative_score >= Gold_score)",
            "rank_definition_gt": "1 + COUNT(negative_score > Gold_score)",
            "matched_gold_identity_excluded_from_negative_pool": True,
            "fold_aggregation": "Arithmetic mean of five folds' raw scores before rank computation",
            "folds": 5,
            "query_batch": QUERY_BATCH,
            "candidate_chunk": CANDIDATE_CHUNK,
        },
        "dense_bge": dense_metrics,
        "conditions": condition_summaries,
        "fast_scorer_max_abs_diff": scorer_max_diff,
        "checkpoint_manifest": checkpoint_manifest,
        "input_sha256": input_hashes,
        "query_embedding_sha256": sha256_file(QUERY_EMB),
        "gold_embedding_sha256": sha256_file(GOLD_EMB),
        "embedding_manifest_sha256": sha256_file(EMB_MANIFEST),
        "model_code_sha256": sha256_file(MODEL_CODE),
        "architecture_protocol_sha256": sha256_file(ARCH_PROTOCOL),
        "test_payload_used": False,
        "training": False,
        "checkpoint_reselection": False,
        "test_time_fitting": False,
        "checkpoints_modified": False,
    }

    arrays = {
        "sample_ids": np.asarray([str(x["sample_id"]) for x in dev], dtype=str),
        "dialogue_ids": dialogue_ids,
        "dev_long20_mask": long_mask,
        "match_idx": match_idx,
        "dense_rank_ge": dense_ge,
        "dense_rank_gt": dense_gt,
        "dense_ties": dense_ties,
    }
    for condition, result in all_results.items():
        for name, values in result.items():
            arrays[f"{condition}__{name}__rank_ge"] = values["rank_ge"]
            arrays[f"{condition}__{name}__rank_gt"] = values["rank_gt"]
            arrays[f"{condition}__{name}__ties"] = values["ties"]

    report = [
        "S12C STEP 2F-5D-2: DEV FULL-LIBRARY FOLD-VS-ENSEMBLE RANKING AUDIT",
        f"dev_rows={len(dev)}; dev_dialogues={len(np.unique(dialogue_ids))}",
        f"dev_long20_rows={int(long_mask.sum())}; dev_long20_dialogues={len(np.unique(dialogue_ids[long_mask]))}",
        f"train_response_pool={len(r_np)}",
        f"gold_match_rows={int((match_idx >= 0).sum())}; gold_unmatched_rows={int((match_idx < 0).sum())}",
        "primary_tie_rule=CONSERVATIVE_GREATER_OR_EQUAL",
        "sensitivity_tie_rule=STRICT_GREATER",
        "test_payload_used=NO; training=NO; checkpoint_reselection=NO; test_time_fitting=NO",
        "",
        "===== DENSE BGE BASELINE =====",
    ]
    for ds_name, m in dense_metrics.items():
        report.append(
            f"{ds_name}: "
            + " | ".join(f"{k}={v:.6f}" for k, v in m["dialogue_macro"].items())
            + f" | median_rank={m['median_rank']:.1f} | p90={m['rank_p90']:.1f}"
        )

    metric_names = ["MRR", "R@1", "R@5", "R@10", "R@20"]
    for condition, datasets in condition_summaries.items():
        report.extend(["", f"===== {condition} ====="])
        for ds_name, ds in datasets.items():
            report.append(f"[{ds_name}]")
            for fold_name, m in ds["folds"].items():
                vals = m["dialogue_macro"]
                report.append(
                    f"{fold_name}: "
                    + " | ".join(f"{k}={vals[k]:.6f}" for k in metric_names)
                    + f" | median_rank={m['median_rank']:.1f}"
                )
            vals = ds["ensemble"]["dialogue_macro"]
            report.append(
                "ENSEMBLE: "
                + " | ".join(f"{k}={vals[k]:.6f}" for k in metric_names)
                + f" | median_rank={ds['ensemble']['median_rank']:.1f}"
            )
            report.append(
                "MEAN_SINGLE_FOLD: "
                + " | ".join(f"{k}={ds['mean_single_fold_dialogue_macro'][k]:.6f}" for k in metric_names)
            )
            report.append(
                "ENSEMBLE_MINUS_MEAN_FOLD: "
                + " | ".join(f"{k}={ds['ensemble_minus_mean_single_fold'][k]:+.6f}" for k in metric_names)
            )
            agree = ds["fold_rank_agreement"]
            report.append(
                "FOLD_RANK_AGREEMENT: "
                f"pearson_mean={agree['pairwise_rank_vector_pearson_mean']:.6f}; "
                f"pearson_min={agree['pairwise_rank_vector_pearson_min']:.6f}; "
                f"exact_rank_mean={agree['pairwise_exact_rank_agreement_mean']:.6f}; "
                f"median_abs_rank_diff_mean={agree['pairwise_median_absolute_rank_difference_mean']:.1f}"
            )

    report.extend([
        "",
        "===== STRICT-GREATER SENSITIVITY =====",
        "Exact tie counts are saved in predictions.npz; rank_ge - rank_gt must equal ties.",
        "",
        "S12C_STEP_2F5D2_STATUS=PASS",
        "TEST_PAYLOAD_USED=NO",
        "TRAINING=NO",
        "CHECKPOINTS_MODIFIED=NO",
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    pred_tmp = PRED_OUT.with_suffix(".npz.tmp")
    with pred_tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    PRED_OUT_TMP = pred_tmp

    summary_tmp = SUMMARY_OUT.with_suffix(".json.tmp")
    summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    report_tmp = REPORT_OUT.with_suffix(".txt.tmp")
    report_tmp.write_text("\n".join(report) + "\n", encoding="utf-8")

    run_manifest = {
        "artifact": "S12C STEP 2F-5D-2 Run Manifest V1",
        "status": "PASS",
        "script_sha256": sha256_file(Path(__file__)),
        "input_sha256": input_hashes,
        "query_embedding_sha256": sha256_file(QUERY_EMB),
        "gold_embedding_sha256": sha256_file(GOLD_EMB),
        "embedding_manifest_sha256": sha256_file(EMB_MANIFEST),
        "model_code_sha256": sha256_file(MODEL_CODE),
        "architecture_protocol_sha256": sha256_file(ARCH_PROTOCOL),
        "checkpoint_manifest": checkpoint_manifest,
        "outputs": {
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "report": str(REPORT_OUT.relative_to(ROOT)),
            "predictions": str(PRED_OUT.relative_to(ROOT)),
        },
        "test_payload_used": False,
        "training": False,
        "checkpoints_modified": False,
    }
    manifest_tmp = MANIFEST_OUT.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")

    summary_tmp.replace(SUMMARY_OUT)
    report_tmp.replace(REPORT_OUT)
    PRED_OUT_TMP.replace(PRED_OUT)
    manifest_tmp.replace(MANIFEST_OUT)

    print("\n" + "=" * 76)
    print("S12C_STEP_2F5D2_STATUS = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("REPORT =", REPORT_OUT)
    print("PREDICTIONS =", PRED_OUT)
    print("MANIFEST =", MANIFEST_OUT)
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINTS_MODIFIED = NO")
    print("=" * 76)


def preflight():
    require(not OUT_DIR.exists(),
            f"Output collision; refusing to overwrite: {OUT_DIR}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "This full-library diagnostic requires CUDA")
    print("GPU =", torch.cuda.get_device_name(0), flush=True)
    dev, long_mask, match_idx, q_np, g_np, r_np, input_hashes = load_and_validate_inputs()
    print("INPUT_AUDIT = PASS")
    print("DEV_ROWS =", len(dev), "DEV_LONG20_ROWS =", int(long_mask.sum()))
    print("TRAIN_RESPONSE_ROWS =", int(r_np.shape[0]))
    print("GOLD_MATCH_ROWS =", int((match_idx >= 0).sum()))
    print("QUERY_EMBEDDINGS =", q_np.shape)
    print("GOLD_EMBEDDINGS =", g_np.shape)
    checkpoint_manifest = {}
    for condition in CONDITIONS:
        print(f"\nPREFLIGHT CONDITION = {condition}", flush=True)
        models, hashes = load_models(condition, device)
        checkpoint_manifest[condition] = hashes
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\nS12C_STEP_2F5D2_PREFLIGHT = PASS")
    print("CHECKPOINTS = 25/25")
    print("RANKING_INFERENCE = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINTS_MODIFIED = NO")
    print("OUTPUTS_WRITTEN = NO")


def main():
    parser = argparse.ArgumentParser(
        description="S12C STEP 2F-5D-2 Dev full-library fold-vs-ensemble audit"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        evaluate()


if __name__ == "__main__":
    main()
