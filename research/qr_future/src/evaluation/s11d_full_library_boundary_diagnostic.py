#!/usr/bin/env python3
"""S11-D: full-library score-boundary diagnostic (Dev only; read-only models/data).

This is an exploratory post-hoc diagnostic. It does not train models, modify
checkpoints, inspect Test payloads, or select a model/configuration.

Run from the repository root:
  python src/evaluation/s11d_full_library_boundary_diagnostic.py prepare
  python src/evaluation/s11d_full_library_boundary_diagnostic.py verify
  python src/evaluation/s11d_full_library_boundary_diagnostic.py evaluate
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import s8_dev_retrieval_evaluation as s8  # noqa: E402
import s9_dev_top100_reranking_diagnostic as s9  # noqa: E402

DIM = 1024
Q_BATCH = 32
C_CHUNK = 2048
CONDITIONS = ("RANK_ONLY", "R_FUTURE_BETA8", "QR_FUTURE_BETA8")

PROTOCOL = ROOT / "protocols/fsar_s11d_full_library_boundary_diagnostic_v1.json"
S8_PROTOCOL = ROOT / "protocols/fsar_s8_dev_retrieval_evaluation_v1.json"
S9_PROTOCOL = ROOT / "protocols/fsar_s9_dev_top100_reranking_v1.json"
S8_PRED = ROOT / "results/phase_s8/s8_dev_retrieval_predictions_v1.npz"
S9_PRED = ROOT / "results/phase_s9/s9_dev_top100_reranking_predictions_v1.npz"
OUT_DIR = ROOT / "results/phase_s11d"
SUMMARY_OUT = OUT_DIR / "s11d_full_library_boundary_summary_v1.json"
PRED_OUT = OUT_DIR / "s11d_full_library_boundary_predictions_v1.npz"
QUERY_CSV_OUT = OUT_DIR / "s11d_full_library_boundary_query_rows_v1.csv"
EXAMPLES_CSV_OUT = OUT_DIR / "s11d_outside_top100_gold_competitors_v1.csv"
REPORT_OUT = OUT_DIR / "s11d_full_library_boundary_report_v1.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def current_input_hashes() -> dict[str, Any]:
    required = [PROTOCOL.parent, S8_PROTOCOL, S9_PROTOCOL, S8_PRED, S9_PRED]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError("Required upstream artifact missing:\n" + "\n".join(missing))
    return {
        "upstream_s8_frozen_inputs": s8.input_paths_and_hashes(),
        "upstream_s8_protocol_sha256": sha256_file(S8_PROTOCOL),
        "upstream_s8_script_sha256": sha256_file(Path(s8.__file__).resolve()),
        "upstream_s8_predictions_sha256": sha256_file(S8_PRED),
        "upstream_s9_protocol_sha256": sha256_file(S9_PROTOCOL),
        "upstream_s9_script_sha256": sha256_file(Path(s9.__file__).resolve()),
        "upstream_s9_predictions_sha256": sha256_file(S9_PRED),
        "s11d_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def verify_upstream() -> None:
    # Reuse the upstream frozen protocol validators; these are read-only checks.
    s8.verify_protocol()
    s9.verify_s9_protocol()


def build_protocol() -> dict[str, Any]:
    verify_upstream()
    _, train_responses, dev, long20, response_norms, response_uids, match_idx, _ = s9.read_s8_data()
    q_np, gold_np, response_np = s9.load_embeddings()
    if len(dev) != 9070 or len(long20) != 4375:
        raise RuntimeError(f"Unexpected Dev sizes: dev={len(dev)}, long20={len(long20)}")
    if len(response_np) != 61502 or len(response_norms) != len(response_np) or len(response_uids) != len(response_np):
        raise RuntimeError("Frozen response library alignment/size mismatch")
    if q_np.shape != (len(dev), DIM) or gold_np.shape != q_np.shape:
        raise RuntimeError(f"Unexpected embedding shapes: q={q_np.shape}, gold={gold_np.shape}")
    if np.asarray(match_idx).shape != (len(dev),):
        raise RuntimeError("Gold-to-TRAIN match index shape mismatch")
    return {
        "project": "PsychResponse_FSAR",
        "phase": "S11-D",
        "artifact": "Full-Library Score-Boundary Diagnostic V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "conditions": list(CONDITIONS),
        "formula": "score(q,r)=wq·zq + wr·zr + wd·abs(zq-zr) + wp·(zq*zr)",
        "rank_relevant_formula": "wr·zr + wd·abs(zq-zr) + wp·(zq*zr); query-only term cancels within each query",
        "fold_aggregation": "Average raw full scores across the same five frozen fold checkpoints before rank computation; component scores are also averaged across folds",
        "candidate_universe": "All 61,502 unique normalized TRAIN counselor responses; use S8 Gold identity mapping and exclude the mapped row from its own rank count",
        "boundary_partition": "S9 BGE Top-100 negative candidate rows versus all remaining TRAIN response rows; S9 Top-100 excludes the mapped Gold identity",
        "tie_rule": "Conservative: every negative candidate with score >= Gold score counts ahead of Gold",
        "dataset": {"dev_rows": len(dev), "dev_long20_rows": len(long20), "train_response_rows": len(response_np), "embedding_dimension": DIM},
        "runtime": {"query_batch_size": Q_BATCH, "candidate_chunk_size": C_CHUNK, "tf32_allowed": False},
        "inputs_sha256": current_input_hashes(),
        "test_payload_used": False,
        "training_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
        "posthoc_success_gate": False,
        "output_paths": {
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "predictions": str(PRED_OUT.relative_to(ROOT)),
            "query_rows_csv": str(QUERY_CSV_OUT.relative_to(ROOT)),
            "outside_top100_examples_csv": str(EXAMPLES_CSV_OUT.relative_to(ROOT)),
            "report": str(REPORT_OUT.relative_to(ROOT)),
        },
    }


def prepare_protocol() -> None:
    if PROTOCOL.exists() or PROTOCOL.with_suffix(PROTOCOL.suffix + ".tmp").exists():
        raise RuntimeError(f"Protocol already exists; refusing overwrite: {PROTOCOL}")
    protocol = build_protocol()
    atomic_json(PROTOCOL, protocol)
    print("S11D_PROTOCOL_PREPARED = PASS")
    print("PROTOCOL =", PROTOCOL)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL))
    print("SCOPE = EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY")
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINT_CHANGES = NO")


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL.is_file():
        raise RuntimeError(f"S11-D protocol not found; run prepare first: {PROTOCOL}")
    frozen = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise RuntimeError("S11-D protocol status is not FROZEN_BEFORE_EVALUATION")
    current = build_protocol()
    if frozen.get("inputs_sha256") != current.get("inputs_sha256"):
        old = frozen.get("inputs_sha256", {})
        new = current.get("inputs_sha256", {})
        diffs = []
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                diffs.append(key)
        raise RuntimeError("Frozen S11-D input hash mismatch: " + ", ".join(diffs[:30]))
    if frozen.get("conditions") != list(CONDITIONS):
        raise RuntimeError("Frozen S11-D condition list mismatch")
    print("S11D_PROTOCOL_VERIFY = PASS")
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL))
    return frozen


def _components_adapted(model: torch.nn.Module, zq: torch.Tensor, zr: torch.Tensor):
    """Return full score and the three response-dependent components.

    The operation order for full scores matches s8.fast_score_adapted so ranks
    can be checked against the frozen S8 predictions. The query-only term is
    included in full scores but cancels in candidate-minus-Gold margins.
    """
    w = model.ranker.rank_head.weight[0]
    wq = w[:DIM]
    wr = w[DIM:2 * DIM]
    wd = w[2 * DIM:3 * DIM]
    wp = w[3 * DIM:4 * DIM]
    q_term = (zq * wq).sum(dim=-1, keepdim=True)
    response_term = (zr * wr).sum(dim=-1).unsqueeze(0)
    abs_term = (torch.abs(zq[:, None, :] - zr[None, :, :]) * wd).sum(dim=-1)
    product_term = torch.matmul(zq * wp, zr.transpose(0, 1))
    full = q_term + response_term + abs_term + product_term
    return full, response_term.expand(zq.shape[0], -1), abs_term, product_term


def _gold_components(model: torch.nn.Module, zq: torch.Tensor, zg: torch.Tensor):
    w = model.ranker.rank_head.weight[0]
    wr = w[DIM:2 * DIM]
    wd = w[2 * DIM:3 * DIM]
    wp = w[3 * DIM:4 * DIM]
    response_term = (zg * wr).sum(dim=-1)
    abs_term = (torch.abs(zq - zg) * wd).sum(dim=-1)
    product_term = (zq * wp * zg).sum(dim=-1)
    full = model.ranker.score_from_adapted(zq, zg).reshape(-1)
    return full, response_term, abs_term, product_term


def _safe_float(x: Any) -> float:
    return float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)


def _metrics(rank: np.ndarray) -> dict[str, float]:
    r = np.asarray(rank, dtype=np.int64)
    return {
        "n": int(len(r)),
        "median_rank": float(np.median(r)),
        "mean_rank": float(np.mean(r)),
        "MRR_query_micro": float(np.mean(1.0 / r)),
        "R@1": float(np.mean(r <= 1)),
        "R@5": float(np.mean(r <= 5)),
        "R@10": float(np.mean(r <= 10)),
        "R@20": float(np.mean(r <= 20)),
        "R@100": float(np.mean(r <= 100)),
    }


def _summary_slice(rows: list[dict[str, Any]], condition: str, mask: np.ndarray) -> dict[str, Any]:
    selected = [rows[i] for i in np.flatnonzero(mask)]
    rank = np.asarray([int(x["full_rank"]) for x in selected], dtype=np.int32)
    total = np.asarray([int(x["total_ge_gold"]) for x in selected], dtype=np.int64)
    top100 = np.asarray([int(x["top100_ge_gold"]) for x in selected], dtype=np.int64)
    outside = np.asarray([int(x["outside_top100_ge_gold"]) for x in selected], dtype=np.int64)
    outside_margin = np.asarray([float(x["outside_best_margin"]) for x in selected], dtype=np.float64)
    top100_margin = np.asarray([float(x["top100_best_margin"]) for x in selected], dtype=np.float64)
    parity = np.asarray([int(x["full_rank_s8_reference"]) == int(x["full_rank"]) for x in selected], dtype=bool)
    denom = int(total.sum())
    return {
        "condition": condition,
        "rank_metrics_query_micro": _metrics(rank),
        "s8_rank_parity": {
            "exact_match_n": int(parity.sum()),
            "mismatch_n": int((~parity).sum()),
            "max_abs_rank_delta": int(max((abs(int(x["full_rank"]) - int(x["full_rank_s8_reference"])) for x in selected), default=0)),
        },
        "competitors_ge_gold": {
            "mean_total": float(total.mean()),
            "median_total": float(np.median(total)),
            "mean_from_s9_dense_top100": float(top100.mean()),
            "mean_from_outside_s9_dense_top100": float(outside.mean()),
            "queries_with_any_outside_competitor_ge_gold_n": int(np.sum(outside > 0)),
            "queries_with_any_outside_competitor_ge_gold_rate": float(np.mean(outside > 0)),
            "outside_fraction_of_all_ge_gold_competitors": float(outside.sum() / denom) if denom else 0.0,
        },
        "best_competitor_margin_vs_gold": {
            "outside_top100_mean": float(outside_margin.mean()),
            "outside_top100_median": float(np.median(outside_margin)),
            "outside_top100_p90": float(np.quantile(outside_margin, 0.90)),
            "outside_top100_ge_gold_rate": float(np.mean(outside_margin >= 0.0)),
            "top100_mean": float(top100_margin.mean()),
            "top100_median": float(np.median(top100_margin)),
            "top100_ge_gold_rate": float(np.mean(top100_margin >= 0.0)),
        },
    }


def evaluate() -> None:
    frozen = verify_protocol()
    verify_upstream()
    outputs = [SUMMARY_OUT, PRED_OUT, QUERY_CSV_OUT, EXAMPLES_CSV_OUT, REPORT_OUT]
    collisions = [str(p) for p in outputs if p.exists() or p.with_suffix(p.suffix + ".tmp").exists()]
    if collisions:
        raise RuntimeError("S11-D output collision; refusing overwrite:\n" + "\n".join(collisions))

    _, train_responses, dev, long20, response_norms, response_uids, match_idx, _ = s9.read_s8_data()
    q_np, gold_np, response_np = s9.load_embeddings()
    match_idx = np.asarray(match_idx, dtype=np.int32)
    dev_sample_ids = np.asarray([str(row["sample_id"]) for row in dev], dtype=str)
    dialogue_ids = np.asarray([str(row["dialogue_id"]) for row in dev], dtype=str)
    long_ids = {str(row["sample_id"]) for row in long20}
    long_mask = np.asarray([sid in long_ids for sid in dev_sample_ids], dtype=bool)
    if len(dev_sample_ids) != 9070 or int(long_mask.sum()) != 4375:
        raise RuntimeError("Frozen Dev/Long20 identity alignment mismatch")

    with np.load(S9_PRED, allow_pickle=False) as p:
        s9_ids = p["sample_ids"].astype(str)
        s9_dialogue_ids = p["dialogue_ids"].astype(str)
        top100_rows = p["candidate_response_rows"].astype(np.int32)
        top100_dense_scores = p["candidate_dense_scores"].astype(np.float32)
    if not np.array_equal(s9_ids, dev_sample_ids) or not np.array_equal(s9_dialogue_ids, dialogue_ids):
        raise RuntimeError("S9 candidate predictions do not align exactly with frozen S8 Dev rows")
    if top100_rows.shape != (len(dev), 100) or top100_dense_scores.shape != top100_rows.shape:
        raise RuntimeError(f"Unexpected S9 Top-100 shapes: rows={top100_rows.shape}, scores={top100_dense_scores.shape}")
    if np.any(top100_rows < 0) or np.any(top100_rows >= len(response_np)):
        raise RuntimeError("S9 Top-100 candidate row out of response-library range")
    if np.any((match_idx[:, None] >= 0) & (top100_rows == match_idx[:, None])):
        raise RuntimeError("S9 Top-100 contains a mapped Gold identity row")
    if any(len(np.unique(x)) != 100 for x in top100_rows):
        raise RuntimeError("Duplicate response rows detected inside an S9 Top-100 candidate set")

    with np.load(S8_PRED, allow_pickle=False) as p:
        s8_rank_refs = {}
        for condition in CONDITIONS:
            key = f"ranks__{condition}"
            if key not in p.files:
                raise RuntimeError(f"Missing S8 rank reference array: {key}")
            s8_rank_refs[condition] = p[key].astype(np.int32)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device, flush=True)
    if device.type == "cuda":
        print("GPU =", torch.cuda.get_device_name(0), flush=True)

    # One response matrix on device. No full query-by-library score matrix is saved.
    response_gpu = torch.from_numpy(np.array(response_np, dtype=np.float32, copy=True)).to(device)
    all_rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {
        "sample_ids": dev_sample_ids,
        "dialogue_ids": dialogue_ids,
        "dev_long20_mask": long_mask,
        "gold_match_idx": match_idx,
        "s9_top100_response_rows": top100_rows,
    }
    summaries: dict[str, Any] = {}

    for condition in CONDITIONS:
        print(f"\n===== S11D CONDITION: {condition} =====", flush=True)
        models = s8.load_condition_models(condition, device)
        zq_folds, zg_folds, zr_folds, gold_full_folds = [], [], [], []
        gold_comp_folds = []
        fast_diffs = []
        q_gpu = torch.from_numpy(np.asarray(q_np, dtype=np.float32)).to(device)
        gold_gpu = torch.from_numpy(np.asarray(gold_np, dtype=np.float32)).to(device)
        with torch.inference_mode():
            for model in models:
                fast_diffs.append(float(s8.validate_fast_scorer(model, device)))
                zq = model.ranker.adapt_query(q_gpu)
                zg = model.ranker.adapt_response(gold_gpu)
                zr = model.ranker.adapt_response(response_gpu)
                gfull, gr, ga, gp = _gold_components(model, zq, zg)
                zq_folds.append(zq)
                zg_folds.append(zg)
                zr_folds.append(zr)
                gold_full_folds.append(gfull)
                gold_comp_folds.append((gr, ga, gp))

        gold_full_all = torch.stack(gold_full_folds, dim=0).mean(dim=0)
        gold_resp_all = torch.stack([x[0] for x in gold_comp_folds], dim=0).mean(dim=0)
        gold_abs_all = torch.stack([x[1] for x in gold_comp_folds], dim=0).mean(dim=0)
        gold_prod_all = torch.stack([x[2] for x in gold_comp_folds], dim=0).mean(dim=0)
        ranks = np.empty(len(dev), dtype=np.int32)
        condition_rows: list[dict[str, Any] | None] = [None] * len(dev)

        with torch.inference_mode():
            for qs in range(0, len(dev), Q_BATCH):
                qe = min(qs + Q_BATCH, len(dev))
                batch_n = qe - qs
                gold_score = gold_full_all[qs:qe]
                gold_resp = gold_resp_all[qs:qe]
                gold_abs = gold_abs_all[qs:qe]
                gold_prod = gold_prod_all[qs:qe]
                count_ge = torch.zeros(batch_n, dtype=torch.int64, device=device)
                top100_scores = torch.full((batch_n, 100), -torch.inf, dtype=torch.float32, device=device)
                top100_resp = torch.full_like(top100_scores, torch.nan)
                top100_abs = torch.full_like(top100_scores, torch.nan)
                top100_prod = torch.full_like(top100_scores, torch.nan)
                outside_best_score = torch.full((batch_n,), -torch.inf, dtype=torch.float32, device=device)
                outside_best_resp = torch.full((batch_n,), torch.nan, dtype=torch.float32, device=device)
                outside_best_abs = torch.full((batch_n,), torch.nan, dtype=torch.float32, device=device)
                outside_best_prod = torch.full((batch_n,), torch.nan, dtype=torch.float32, device=device)
                outside_best_row = torch.full((batch_n,), -1, dtype=torch.int64, device=device)

                local_top_rows = top100_rows[qs:qe]
                local_match = match_idx[qs:qe]
                for cs in range(0, response_gpu.shape[0], C_CHUNK):
                    ce = min(cs + C_CHUNK, response_gpu.shape[0])
                    score_sum = resp_sum = abs_sum = prod_sum = None
                    for fold_i, model in enumerate(models):
                        full_f, resp_f, abs_f, prod_f = _components_adapted(
                            model, zq_folds[fold_i][qs:qe], zr_folds[fold_i][cs:ce]
                        )
                        score_sum = full_f if score_sum is None else score_sum + full_f
                        resp_sum = resp_f if resp_sum is None else resp_sum + resp_f
                        abs_sum = abs_f if abs_sum is None else abs_sum + abs_f
                        prod_sum = prod_f if prod_sum is None else prod_sum + prod_f
                    scores = score_sum / len(models)
                    resp_scores = resp_sum / len(models)
                    abs_scores = abs_sum / len(models)
                    prod_scores = prod_sum / len(models)

                    ge = scores >= gold_score[:, None]
                    count_ge += ge.sum(dim=1)
                    in_match = (local_match >= cs) & (local_match < ce)
                    if np.any(in_match):
                        rr = torch.from_numpy(np.flatnonzero(in_match)).to(device)
                        cc = torch.from_numpy((local_match[in_match] - cs).astype(np.int64)).to(device)
                        counted = scores[rr, cc] >= gold_score[rr]
                        count_ge[rr] -= counted.to(torch.int64)

                    # Gather scores/components for the S9 BGE Top-100 candidates that fall in this chunk.
                    top_local = local_top_rows.astype(np.int64, copy=False) - cs
                    valid_top = (top_local >= 0) & (top_local < (ce - cs))
                    top_local_t = torch.from_numpy(top_local).to(device)
                    valid_top_t = torch.from_numpy(valid_top).to(device)
                    safe_top = top_local_t.clamp(0, (ce - cs) - 1)
                    chunk_top_scores = scores.gather(1, safe_top)
                    chunk_top_resp = resp_scores.gather(1, safe_top)
                    chunk_top_abs = abs_scores.gather(1, safe_top)
                    chunk_top_prod = prod_scores.gather(1, safe_top)
                    top100_scores = torch.where(valid_top_t, chunk_top_scores, top100_scores)
                    top100_resp = torch.where(valid_top_t, chunk_top_resp, top100_resp)
                    top100_abs = torch.where(valid_top_t, chunk_top_abs, top100_abs)
                    top100_prod = torch.where(valid_top_t, chunk_top_prod, top100_prod)

                    # Find the best candidate outside Top-100 and outside the mapped Gold identity.
                    outside_mask = torch.ones_like(scores, dtype=torch.bool)
                    rr_top, kk_top = torch.where(valid_top_t)
                    if rr_top.numel():
                        outside_mask[rr_top, top_local_t[rr_top, kk_top]] = False
                    match_local = local_match.astype(np.int64, copy=False) - cs
                    match_in = (local_match >= cs) & (local_match < ce)
                    if np.any(match_in):
                        rr_match = torch.from_numpy(np.flatnonzero(match_in)).to(device)
                        cc_match = torch.from_numpy(match_local[match_in]).to(device)
                        outside_mask[rr_match, cc_match] = False
                    masked = scores.masked_fill(~outside_mask, -torch.inf)
                    chunk_best, chunk_arg = masked.max(dim=1)
                    update = chunk_best > outside_best_score
                    if torch.any(update):
                        rows_update = torch.where(update)[0]
                        cols_update = chunk_arg[update]
                        outside_best_score[update] = chunk_best[update]
                        outside_best_resp[update] = resp_scores[rows_update, cols_update]
                        outside_best_abs[update] = abs_scores[rows_update, cols_update]
                        outside_best_prod[update] = prod_scores[rows_update, cols_update]
                        outside_best_row[update] = cols_update + cs

                    if cs == 0 or ce == response_gpu.shape[0] or (cs // C_CHUNK) % 10 == 0:
                        print(f"{condition} queries={qe}/{len(dev)} candidates={ce}/{response_gpu.shape[0]}", flush=True)

                if not torch.isfinite(top100_scores).all():
                    raise RuntimeError(f"Some Top-100 candidates were not scored for {condition}, rows {qs}:{qe}")
                rank_batch = count_ge.cpu().numpy().astype(np.int32) + 1
                ranks[qs:qe] = rank_batch
                top100_count = (top100_scores >= gold_score[:, None]).sum(dim=1)
                outside_count = count_ge - top100_count.to(torch.int64)
                if torch.any(outside_count < 0):
                    raise RuntimeError("Negative outside-Top-100 competitor count; candidate partition is inconsistent")
                top100_arg = top100_scores.argmax(dim=1)
                top100_best_score = top100_scores.gather(1, top100_arg[:, None]).squeeze(1)
                top100_best_resp = top100_resp.gather(1, top100_arg[:, None]).squeeze(1)
                top100_best_abs = top100_abs.gather(1, top100_arg[:, None]).squeeze(1)
                top100_best_prod = top100_prod.gather(1, top100_arg[:, None]).squeeze(1)
                top100_best_row = torch.from_numpy(local_top_rows).to(device).gather(1, top100_arg[:, None]).squeeze(1)

                outside_margin = outside_best_score - gold_score
                top100_margin = top100_best_score - gold_score
                full_best_is_top100 = top100_best_score >= outside_best_score
                full_best_score = torch.where(full_best_is_top100, top100_best_score, outside_best_score)
                full_best_row = torch.where(full_best_is_top100, top100_best_row, outside_best_row)
                full_best_resp = torch.where(full_best_is_top100, top100_best_resp, outside_best_resp)
                full_best_abs = torch.where(full_best_is_top100, top100_best_abs, outside_best_abs)
                full_best_prod = torch.where(full_best_is_top100, top100_best_prod, outside_best_prod)

                # Transfer this query batch's small diagnostic arrays to CPU.
                arrays = {
                    "gold_score": gold_score,
                    "gold_resp": gold_resp,
                    "gold_abs": gold_abs,
                    "gold_prod": gold_prod,
                    "count_ge": count_ge,
                    "top100_count": top100_count,
                    "outside_count": outside_count,
                    "top100_margin": top100_margin,
                    "outside_margin": outside_margin,
                    "top100_best_score": top100_best_score,
                    "outside_best_score": outside_best_score,
                    "top100_best_row": top100_best_row,
                    "outside_best_row": outside_best_row,
                    "full_best_score": full_best_score,
                    "full_best_row": full_best_row,
                    "top100_best_resp": top100_best_resp,
                    "top100_best_abs": top100_best_abs,
                    "top100_best_prod": top100_best_prod,
                    "outside_best_resp": outside_best_resp,
                    "outside_best_abs": outside_best_abs,
                    "outside_best_prod": outside_best_prod,
                    "full_best_resp": full_best_resp,
                    "full_best_abs": full_best_abs,
                    "full_best_prod": full_best_prod,
                }
                cpu = {k: v.detach().cpu().numpy() for k, v in arrays.items()}
                for local_i, i in enumerate(range(qs, qe)):
                    tr = int(cpu["top100_best_row"][local_i])
                    outr = int(cpu["outside_best_row"][local_i])
                    fr = int(cpu["full_best_row"][local_i])
                    # S9 stores Top-100 in descending BGE score order; report its position as BGE rank.
                    pos = int(np.flatnonzero(top100_rows[i] == tr)[0])
                    tr_text = str(response_norms[tr])
                    outr_text = str(response_norms[outr])
                    fr_text = str(response_norms[fr])
                    gold_score_i = float(cpu["gold_score"][local_i])
                    row = {
                        "condition": condition,
                        "sample_id": str(dev_sample_ids[i]),
                        "dialogue_id": str(dialogue_ids[i]),
                        "dev_long20": bool(long_mask[i]),
                        "full_rank": int(rank_batch[local_i]),
                        "full_rank_s8_reference": int(s8_rank_refs[condition][i]),
                        "rank_delta_vs_s8": int(rank_batch[local_i]) - int(s8_rank_refs[condition][i]),
                        "total_ge_gold": int(cpu["count_ge"][local_i]),
                        "top100_ge_gold": int(cpu["top100_count"][local_i]),
                        "outside_top100_ge_gold": int(cpu["outside_count"][local_i]),
                        "gold_score_full": gold_score_i,
                        "top100_best_margin": float(cpu["top100_margin"][local_i]),
                        "outside_best_margin": float(cpu["outside_margin"][local_i]),
                        "full_best_margin": float(cpu["full_best_score"][local_i] - gold_score_i),
                        "top100_best_row": tr,
                        "top100_best_uid": str(response_uids[tr]),
                        "top100_best_text": tr_text,
                        "top100_best_bge_rank": pos + 1,
                        "top100_best_bge_score": float(top100_dense_scores[i, pos]),
                        "outside_best_row": outr,
                        "outside_best_uid": str(response_uids[outr]),
                        "outside_best_text": outr_text,
                        "full_best_row": fr,
                        "full_best_uid": str(response_uids[fr]),
                        "full_best_text": fr_text,
                        "top100_best_response_component_margin": float(cpu["top100_best_resp"][local_i] - cpu["gold_resp"][local_i]),
                        "top100_best_abs_diff_component_margin": float(cpu["top100_best_abs"][local_i] - cpu["gold_abs"][local_i]),
                        "top100_best_product_component_margin": float(cpu["top100_best_prod"][local_i] - cpu["gold_prod"][local_i]),
                        "outside_best_response_component_margin": float(cpu["outside_best_resp"][local_i] - cpu["gold_resp"][local_i]),
                        "outside_best_abs_diff_component_margin": float(cpu["outside_best_abs"][local_i] - cpu["gold_abs"][local_i]),
                        "outside_best_product_component_margin": float(cpu["outside_best_prod"][local_i] - cpu["gold_prod"][local_i]),
                        "full_best_response_component_margin": float(cpu["full_best_resp"][local_i] - cpu["gold_resp"][local_i]),
                        "full_best_abs_diff_component_margin": float(cpu["full_best_abs"][local_i] - cpu["gold_abs"][local_i]),
                        "full_best_product_component_margin": float(cpu["full_best_prod"][local_i] - cpu["gold_prod"][local_i]),
                        "outside_best_ge_gold": bool(cpu["outside_margin"][local_i] >= 0.0),
                        "top100_best_ge_gold": bool(cpu["top100_margin"][local_i] >= 0.0),
                    }
                    condition_rows[i] = row

                print(f"{condition} batch={qe}/{len(dev)} rank_mismatch_so_far={int(np.sum(ranks[:qe] != s8_rank_refs[condition][:qe]))}", flush=True)

        if any(row is None for row in condition_rows):
            raise RuntimeError(f"Missing query diagnostics for {condition}")
        concrete_rows = [row for row in condition_rows if row is not None]
        all_rows.extend(concrete_rows)
        summaries[condition] = {
            "fast_scorer_max_abs_diff_by_fold": fast_diffs,
            "dev_all": _summary_slice(concrete_rows, condition, np.ones(len(dev), dtype=bool)),
            "dev_long20": _summary_slice(concrete_rows, condition, long_mask),
        }
        saved_arrays[f"ranks__{condition}"] = ranks
        for field, key in [
            ("total_ge_gold", "total_ge_gold"),
            ("top100_ge_gold", "top100_ge_gold"),
            ("outside_top100_ge_gold", "outside_top100_ge_gold"),
            ("top100_best_margin", "top100_best_margin"),
            ("outside_best_margin", "outside_best_margin"),
            ("full_best_margin", "full_best_margin"),
            ("top100_best_row", "top100_best_row"),
            ("outside_best_row", "outside_best_row"),
            ("full_best_row", "full_best_row"),
        ]:
            dtype = np.int32 if field.endswith("_row") or field.endswith("_ge_gold") else np.float32
            saved_arrays[f"{key}__{condition}"] = np.asarray([int(r[field]) if dtype == np.int32 else float(r[field]) for r in concrete_rows], dtype=dtype)

        del models, zq_folds, zg_folds, zr_folds, gold_full_folds, gold_comp_folds
        del gold_full_all, gold_resp_all, gold_abs_all, gold_prod_all
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Final reconstruction check and descriptive report.
    for condition in CONDITIONS:
        parity = summaries[condition]["dev_all"]["s8_rank_parity"]
        print(f"S8_RANK_PARITY {condition}: mismatch_n={parity['mismatch_n']} max_abs_delta={parity['max_abs_rank_delta']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_tmp = PRED_OUT.with_suffix(PRED_OUT.suffix + ".tmp")
    if pred_tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {pred_tmp}")
    with pred_tmp.open("wb") as f:
        np.savez_compressed(f, **saved_arrays)
    os.replace(pred_tmp, PRED_OUT)

    fieldnames = list(all_rows[0].keys())
    csv_tmp = QUERY_CSV_OUT.with_suffix(QUERY_CSV_OUT.suffix + ".tmp")
    if csv_tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {csv_tmp}")
    with csv_tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(csv_tmp, QUERY_CSV_OUT)

    examples = [r for r in all_rows if r["outside_best_ge_gold"]]
    ex_tmp = EXAMPLES_CSV_OUT.with_suffix(EXAMPLES_CSV_OUT.suffix + ".tmp")
    if ex_tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {ex_tmp}")
    with ex_tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)
    os.replace(ex_tmp, EXAMPLES_CSV_OUT)

    report_lines = [
        "FSAR S11-D FULL-LIBRARY SCORE-BOUNDARY DIAGNOSTIC",
        f"protocol_sha256 = {sha256_file(PROTOCOL)}",
        "scope = EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        f"dev_rows = {len(dev)}; dev_long20_rows = {int(long_mask.sum())}; train_responses = {len(response_np)}",
        "S9 boundary = BGE Top-100 negatives versus all remaining TRAIN responses",
        "Gold identity row is excluded from the full-library rank count; ties use score >= Gold",
        "All reported margins are within-query candidate-minus-Gold score differences; absolute scores are not comparable across queries.",
        "",
    ]
    for condition in CONDITIONS:
        report_lines.append(f"===== {condition} =====")
        for subset in ("dev_all", "dev_long20"):
            item = summaries[condition][subset]
            mm = item["rank_metrics_query_micro"]
            cc = item["competitors_ge_gold"]
            bm = item["best_competitor_margin_vs_gold"]
            pp = item["s8_rank_parity"]
            report_lines.extend([
                f"[{subset}] n={mm['n']} median_rank={mm['median_rank']:.4f} mean_rank={mm['mean_rank']:.4f} MRR={mm['MRR_query_micro']:.6f}",
                f"R@1={mm['R@1']:.6f} R@5={mm['R@5']:.6f} R@10={mm['R@10']:.6f} R@20={mm['R@20']:.6f} R@100={mm['R@100']:.6f}",
                f"S8_rank_mismatch_n={pp['mismatch_n']} max_abs_rank_delta={pp['max_abs_rank_delta']}",
                f"mean_ge_gold_total={cc['mean_total']:.4f} mean_top100={cc['mean_from_s9_dense_top100']:.4f} mean_outside_top100={cc['mean_from_outside_s9_dense_top100']:.4f}",
                f"queries_with_outside_ge_gold={cc['queries_with_any_outside_competitor_ge_gold_n']} ({cc['queries_with_any_outside_competitor_ge_gold_rate']:.4%}) outside_fraction_of_ge_gold={cc['outside_fraction_of_all_ge_gold_competitors']:.4%}",
                f"outside_best_margin median={bm['outside_top100_median']:.6f} p90={bm['outside_top100_p90']:.6f} ge_gold_rate={bm['outside_top100_ge_gold_rate']:.4%}",
                f"top100_best_margin median={bm['top100_median']:.6f} ge_gold_rate={bm['top100_ge_gold_rate']:.4%}",
            ])
        report_lines.append("")
    report_lines.extend([
        f"query_rows_csv = {QUERY_CSV_OUT}",
        f"outside_top100_examples_rows = {len(examples)} (condition-query rows; includes rows where the best outside-Top-100 candidate scores >= Gold)",
        "TEST_PAYLOAD_USED = NO",
        "TRAINING = NO",
        "TRAINING_ARTIFACTS_MODIFIED = NO",
        "CHECKPOINTS_MODIFIED = NO",
    ])
    report_tmp = REPORT_OUT.with_suffix(REPORT_OUT.suffix + ".tmp")
    if report_tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {report_tmp}")
    report_tmp.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(report_tmp, REPORT_OUT)

    summary = {
        "artifact": "FSAR S11-D Full-Library Score-Boundary Diagnostic Results V1",
        "status": "PASS",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "protocol_path": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs_sha256": frozen["inputs_sha256"],
        "dataset": {"dev_rows": len(dev), "dev_long20_rows": int(long_mask.sum()), "train_response_rows": len(response_np)},
        "candidate_partition": {
            "s9_dense_top100_per_query": 100,
            "outside_top100_response_rows_before_gold_identity_exclusion": int(len(response_np) - 100),
            "outside_top100_rankable_rows_if_gold_mapped": int(len(response_np) - 101),
            "outside_top100_rankable_rows_if_gold_not_mapped": int(len(response_np) - 100),
            "note": "For queries with a mapped Gold identity, that one row is excluded from the rank count; the Top-100 negatives already exclude it.",
        },
        "conditions": summaries,
        "outside_top100_examples_rows": len(examples),
        "output_sha256": {
            "predictions": sha256_file(PRED_OUT),
            "query_rows_csv": sha256_file(QUERY_CSV_OUT),
            "outside_top100_examples_csv": sha256_file(EXAMPLES_CSV_OUT),
            "report": sha256_file(REPORT_OUT),
        },
        "test_payload_used": False,
        "training_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
        "interpretation_note": "Descriptive Dev-only score-boundary diagnosis. It does not establish causal mechanism, future-state prediction validity, or end-to-end retrieval improvement. S9 Top-100 is selected by BGE, not by the learned scorer.",
    }
    atomic_json(SUMMARY_OUT, summary)
    print("\nS11D_EVALUATION = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("PREDICTIONS =", PRED_OUT)
    print("QUERY_ROWS_CSV =", QUERY_CSV_OUT)
    print("OUTSIDE_TOP100_EXAMPLES_CSV =", EXAMPLES_CSV_OUT)
    print("REPORT =", REPORT_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL))
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINTS_MODIFIED = NO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "evaluate"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_protocol()
    elif args.action == "verify":
        verify_protocol()
    else:
        evaluate()


if __name__ == "__main__":
    main()
