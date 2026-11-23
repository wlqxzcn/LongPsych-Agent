#!/usr/bin/env python3
"""FSAR S9: Dev Top-100 dense-candidate reranking diagnostic.

Two-step workflow:
  1) python src/evaluation/s9_dev_top100_reranking_diagnostic.py --prepare
  2) python src/evaluation/s9_dev_top100_reranking_diagnostic.py --evaluate

This is a Dev-only diagnostic. It does not read Test payloads and does not modify
training artifacts or checkpoints. It imports frozen validation/scoring helpers
from S8 to minimize divergence from the existing implementation.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import s8_dev_retrieval_evaluation as s8  # noqa: E402

DIM = 1024
TOP_K = 100
QUERY_BATCH = 64

DEV_FILE = ROOT / "data/splits/psydial_targets_dev_all.jsonl"
LONG20_FILE = ROOT / "data/splits/psydial_targets_dev_long20.jsonl"
TRAIN_RESPONSES_FILE = ROOT / "data/canonical/fsar_s2/train_unique_responses_v1.jsonl"
QUERY_EMB_FILE = ROOT / "results/phase_s8/dev_all_query_embeddings_bge_v1.npy"
GOLD_EMB_FILE = ROOT / "results/phase_s8/dev_all_gold_embeddings_bge_v1.npy"
RESPONSE_EMB_FILE = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
S8_PROTOCOL_FILE = ROOT / "protocols/fsar_s8_dev_retrieval_evaluation_v1.json"
S8_SCRIPT_FILE = ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py"
MODEL_CODE_FILE = ROOT / "src/models/s3c1_fsar_models.py"
S9_SCRIPT_FILE = ROOT / "src/evaluation/s9_dev_top100_reranking_diagnostic.py"

S9_PROTOCOL_FILE = ROOT / "protocols/fsar_s9_dev_top100_reranking_v1.json"
OUT_DIR = ROOT / "results/phase_s9"
SUMMARY_OUT = OUT_DIR / "s9_dev_top100_reranking_summary_v1.json"
PREDICTIONS_OUT = OUT_DIR / "s9_dev_top100_reranking_predictions_v1.npz"
REPORT_OUT = OUT_DIR / "s9_dev_top100_reranking_report_v1.txt"

CONDITIONS = [
    "RANK_ONLY",
    "R_FUTURE_BETA1", "R_FUTURE_BETA2", "R_FUTURE_BETA4", "R_FUTURE_BETA8",
    "QR_FUTURE_BETA1", "QR_FUTURE_BETA2", "QR_FUTURE_BETA4", "QR_FUTURE_BETA8",
]


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def checkpoint_paths() -> dict[str, list[Path]]:
    all_paths = s8.all_checkpoint_paths()
    missing = [c for c in CONDITIONS if c not in all_paths]
    if missing:
        raise RuntimeError(f"S8 checkpoint registry missing conditions: {missing}")
    result = {c: list(all_paths[c]) for c in CONDITIONS}
    for condition, paths in result.items():
        if len(paths) != 5:
            raise RuntimeError(f"Expected five fold checkpoints for {condition}, got {len(paths)}")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
    return result


def input_path_map() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "dev_all": DEV_FILE,
        "dev_long20": LONG20_FILE,
        "train_unique_responses": TRAIN_RESPONSES_FILE,
        "dev_query_embeddings": QUERY_EMB_FILE,
        "dev_gold_embeddings": GOLD_EMB_FILE,
        "train_response_embeddings": RESPONSE_EMB_FILE,
        "s8_protocol": S8_PROTOCOL_FILE,
        "s8_evaluation_script": S8_SCRIPT_FILE,
        "model_code": MODEL_CODE_FILE,
        "s9_evaluation_script": S9_SCRIPT_FILE,
    }
    for condition, files in checkpoint_paths().items():
        for fold, path in enumerate(files):
            paths[f"checkpoint:{condition}:fold{fold}"] = path
    return paths


def current_input_hashes() -> dict[str, str]:
    paths = input_path_map()
    missing = [f"{key}: {path}" for key, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing S9 input files:\n" + "\n".join(missing))
    return {key: sha256_file(path) for key, path in paths.items()}


def read_s8_data():
    # Reuse S8's frozen data validation and normalization/Gold-identity mapping.
    result = s8.load_and_validate_text_data()
    if len(result) != 8:
        raise RuntimeError(f"Unexpected S8 data-loader return signature: {len(result)} items")
    train_queries, train_responses, dev, long20, response_norms, response_uids, match_idx, coverage = result
    return train_queries, train_responses, dev, long20, response_norms, response_uids, np.asarray(match_idx), coverage


def load_embeddings(check_shapes: bool = True):
    q = np.load(QUERY_EMB_FILE, allow_pickle=False)
    gold = np.load(GOLD_EMB_FILE, allow_pickle=False)
    responses = np.load(RESPONSE_EMB_FILE, mmap_mode="r", allow_pickle=False)
    if check_shapes:
        if q.ndim != 2 or q.shape[1] != DIM:
            raise RuntimeError(f"Unexpected query embedding shape: {q.shape}")
        if gold.shape != q.shape:
            raise RuntimeError(f"Gold embedding shape mismatch: {gold.shape} vs {q.shape}")
        if responses.ndim != 2 or responses.shape[1] != DIM:
            raise RuntimeError(f"Unexpected response embedding shape: {responses.shape}")
        if responses.dtype != np.float32:
            raise RuntimeError(f"Expected float32 response embeddings, got {responses.dtype}")
        if q.dtype != np.float32 or gold.dtype != np.float32:
            raise RuntimeError(f"Expected float32 Dev embeddings, got query={q.dtype}, gold={gold.dtype}")
        if not np.isfinite(q).all() or not np.isfinite(gold).all():
            raise RuntimeError("Non-finite Dev embeddings")
    return q, gold, responses


def build_dense_topk(
    q_np: np.ndarray,
    r_gpu: torch.Tensor,
    match_idx: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select exactly 100 dense negatives per query, excluding normalized Gold identity."""
    n = len(q_np)
    if match_idx.shape != (n,):
        raise RuntimeError(f"Gold match index shape mismatch: {match_idx.shape}, expected {(n,)}")
    if r_gpu.shape[0] <= TOP_K:
        raise RuntimeError(f"Response library too small for Top-{TOP_K}: {r_gpu.shape[0]}")

    q_gpu = torch.from_numpy(np.asarray(q_np, dtype=np.float32)).to(device)
    candidate_rows = np.empty((n, TOP_K), dtype=np.int32)
    candidate_scores = np.empty((n, TOP_K), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, QUERY_BATCH):
            end = min(start + QUERY_BATCH, n)
            scores = torch.matmul(q_gpu[start:end], r_gpu.T)
            local_match = match_idx[start:end].astype(np.int64, copy=False)
            valid = local_match >= 0
            if np.any(valid):
                rows = torch.from_numpy(np.flatnonzero(valid)).to(device)
                cols = torch.from_numpy(local_match[valid]).to(device)
                scores[rows, cols] = -torch.inf
            values, indices = torch.topk(scores, k=TOP_K, dim=1, largest=True, sorted=True)
            if not torch.isfinite(values).all():
                raise RuntimeError(f"Non-finite Top-{TOP_K} scores in query batch {start}:{end}")
            candidate_rows[start:end] = indices.cpu().numpy().astype(np.int32, copy=False)
            candidate_scores[start:end] = values.cpu().numpy().astype(np.float32, copy=False)
            if start == 0 or end == n or (start // QUERY_BATCH) % 20 == 0:
                print(f"DENSE_TOPK_PROGRESS queries={end}/{n}", flush=True)

    # The matching Gold identity must not appear among negatives.
    for i, matched in enumerate(match_idx):
        if matched >= 0 and np.any(candidate_rows[i] == int(matched)):
            raise RuntimeError(f"Gold response identity leaked into negative candidates at query row {i}")
    return candidate_rows, candidate_scores, q_gpu


def ranks_from_scores(candidate_scores: np.ndarray, gold_scores: np.ndarray) -> np.ndarray:
    # Conservative deterministic tie rule: candidates with score >= Gold precede Gold.
    return (1 + np.sum(candidate_scores >= gold_scores[:, None], axis=1)).astype(np.int32)


def evaluate_learned_condition(
    condition: str,
    q_np: np.ndarray,
    gold_np: np.ndarray,
    r_gpu: torch.Tensor,
    candidate_rows: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    models = s8.load_condition_models(condition, device)
    q_gpu = torch.from_numpy(np.asarray(q_np, dtype=np.float32)).to(device)
    gold_gpu = torch.from_numpy(np.asarray(gold_np, dtype=np.float32)).to(device)
    ranks = np.empty(len(q_np), dtype=np.int32)
    max_fast_diff = 0.0

    for model in models:
        max_fast_diff = max(max_fast_diff, float(s8.validate_fast_scorer(model, device)))

    with torch.inference_mode():
        for start in range(0, len(q_np), QUERY_BATCH):
            end = min(start + QUERY_BATCH, len(q_np))
            qb = q_gpu[start:end]
            gb = gold_gpu[start:end]
            idx = torch.from_numpy(candidate_rows[start:end].astype(np.int64, copy=False)).to(device)
            candidates = r_gpu[idx]  # [batch, 100, 1024]

            candidate_score_folds = []
            gold_score_folds = []
            for model in models:
                candidate_score_folds.append(model.rank_scores(qb, candidates))
                gold_score_folds.append(model.rank_scores(qb, gb[:, None, :]).squeeze(1))

            # Match S8's ensemble convention: average raw scores across the five fold models.
            candidate_scores = torch.stack(candidate_score_folds, dim=0).mean(dim=0)
            gold_scores = torch.stack(gold_score_folds, dim=0).mean(dim=0)
            batch_ranks = 1 + (candidate_scores >= gold_scores[:, None]).sum(dim=1)
            ranks[start:end] = batch_ranks.cpu().numpy().astype(np.int32, copy=False)

            if start == 0 or end == len(q_np) or (start // QUERY_BATCH) % 20 == 0:
                print(f"{condition}_PROGRESS queries={end}/{len(q_np)}", flush=True)

    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ranks, max_fast_diff


def prepare() -> None:
    if S9_PROTOCOL_FILE.exists():
        raise RuntimeError(f"S9 protocol already exists; refusing overwrite: {S9_PROTOCOL_FILE}")
    for path in (SUMMARY_OUT, PREDICTIONS_OUT, REPORT_OUT):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise RuntimeError(f"S9 output collision; refusing overwrite: {path}")

    # Ensure the upstream frozen S8 inputs and protocol still pass their own audit.
    s8.verify_protocol()
    _, _, dev, long20, response_norms, response_uids, match_idx, coverage = read_s8_data()
    q, gold, responses = load_embeddings()
    if len(dev) != len(q) or len(gold) != len(dev):
        raise RuntimeError(f"Dev/embedding row mismatch: dev={len(dev)}, q={len(q)}, gold={len(gold)}")
    if len(response_norms) != len(responses) or len(response_uids) != len(responses):
        raise RuntimeError("TRAIN response text/identity and embedding row counts do not align")
    if match_idx.shape != (len(dev),):
        raise RuntimeError("S8 Gold match index is not aligned to Dev rows")
    if len(long20) > len(dev):
        raise RuntimeError("Long20 subset cannot exceed Dev row count")

    hashes = current_input_hashes()
    s8_protocol_sha = sha256_file(S8_PROTOCOL_FILE)
    protocol = {
        "artifact": "FSAR S9 Dev Top-100 Dense-Candidate Reranking Diagnostic V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "EXTERNAL_DEV_TOP100_RERANKING_DIAGNOSTIC_ONLY",
        "upstream_s8_protocol_sha256": s8_protocol_sha,
        "input_sha256": hashes,
        "dataset": {
            "dev_rows": int(len(dev)),
            "dev_long20_rows": int(len(long20)),
            "train_unique_response_rows": int(len(responses)),
            "gold_row_coverage_diagnostic": json_safe(coverage),
        },
        "candidate_construction": {
            "negative_source": "TRAIN unique normalized counselor responses",
            "negative_count_per_query": TOP_K,
            "selection": "Top-100 by frozen BGE query-response dot product, after excluding the TRAIN response row whose normalized identity equals this query Gold, if present",
            "positive_candidate": "Exactly one separately encoded Dev Gold answer is injected per query",
            "same_dialogue_candidates_excluded": False,
            "candidate_pool_is_query_specific": True,
            "selection_implementation": "torch.topk over float32 BGE dot products; sorted=True",
            "gold_identity_exclusion_uses_s8_match_idx": True,
        },
        "models": {
            "conditions": CONDITIONS,
            "fold_ensemble": "Use all five frozen fold checkpoints; average raw candidate scores across folds before ranking",
            "rank_only_and_future_aux_models_share_ranker_interface": True,
        },
        "metrics": {
            "primary": "Dialogue-macro MRR, R@1, R@5, R@10, R@20",
            "aggregation": "Reuse S8 metric implementation: average query metric within dialogue, then unweighted mean across dialogues",
            "uncertainty": "Paired dialogue-cluster bootstrap using S8 implementation and seed",
            "candidate_rank_tie_rule": "Conservative: every negative candidate with score >= Gold score is ranked before Gold",
            "interpretation": "Conditional reranking quality on a BGE-selected Top-100 negative set plus one injected Gold; not end-to-end retrieval coverage",
        },
        "runtime": {
            "query_batch_size": QUERY_BATCH,
            "device_selection": "CUDA if available, otherwise CPU",
            "tf32_allowed": False,
        },
        "test_payload_used": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
    }
    atomic_json(S9_PROTOCOL_FILE, protocol)
    print("S9_PROTOCOL_FROZEN = PASS")
    print("PROTOCOL =", S9_PROTOCOL_FILE)
    print("PROTOCOL_SHA256 =", sha256_file(S9_PROTOCOL_FILE))
    print("dev_rows =", len(dev))
    print("dev_long20_rows =", len(long20))
    print("train_response_rows =", len(responses))
    print("gold_row_coverage =", json.dumps(json_safe(coverage), ensure_ascii=False))
    print("TEST_PAYLOAD_READ = NO")
    print("EVALUATION_STARTED = NO")


def verify_s9_protocol() -> dict[str, Any]:
    if not S9_PROTOCOL_FILE.is_file():
        raise FileNotFoundError(f"S9 protocol missing; run --prepare first: {S9_PROTOCOL_FILE}")
    protocol = json.loads(S9_PROTOCOL_FILE.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise RuntimeError(f"Unexpected S9 protocol status: {protocol.get('status')}")
    if protocol.get("upstream_s8_protocol_sha256") != sha256_file(S8_PROTOCOL_FILE):
        raise RuntimeError("Upstream S8 protocol hash changed after S9 freeze")
    current_hashes = current_input_hashes()
    frozen_hashes = protocol.get("input_sha256", {})
    if current_hashes != frozen_hashes:
        changed = [
            key for key in sorted(set(current_hashes) | set(frozen_hashes))
            if current_hashes.get(key) != frozen_hashes.get(key)
        ]
        raise RuntimeError("Frozen S9 input hash mismatch: " + ", ".join(changed))
    return protocol


def evaluate() -> None:
    protocol = verify_s9_protocol()
    for path in (SUMMARY_OUT, PREDICTIONS_OUT, REPORT_OUT):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise RuntimeError(f"S9 output collision; refusing overwrite: {path}")

    s8.verify_protocol()
    _, _, dev, long20, response_norms, response_uids, match_idx, coverage = read_s8_data()
    q_np, gold_np, response_np = load_embeddings()
    if len(dev) != len(q_np) or len(dev) != len(gold_np):
        raise RuntimeError("Dev embeddings do not align with frozen Dev target rows")
    if len(response_np) != len(response_norms) or len(response_np) != len(response_uids):
        raise RuntimeError("TRAIN response embeddings do not align with response identities")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device, flush=True)
    if device.type == "cuda":
        print("GPU =", torch.cuda.get_device_name(0), flush=True)

    # One response matrix on device; query batches keep score/candidate tensors bounded.
    response_gpu = torch.from_numpy(np.array(response_np, dtype=np.float32, copy=True)).to(device)
    candidate_rows, candidate_dense_scores, q_gpu = build_dense_topk(q_np, response_gpu, match_idx, device)
    gold_gpu = torch.from_numpy(np.asarray(gold_np, dtype=np.float32)).to(device)
    with torch.inference_mode():
        dense_gold_scores = (q_gpu * gold_gpu).sum(dim=1).cpu().numpy().astype(np.float32)
    # Keep the baseline key "DENSE_BGE" because the frozen S8 metric helper uses it
    # to construct paired contrasts. In S9 it means Dense ranking restricted to this Top-100 set.
    ranks_by_model: dict[str, np.ndarray] = {
        "DENSE_BGE": ranks_from_scores(candidate_dense_scores, dense_gold_scores)
    }
    print("DENSE_BGE_TOP100 rank median =", float(np.median(ranks_by_model["DENSE_BGE"])), flush=True)

    fast_diffs: dict[str, float] = {}
    for condition in CONDITIONS:
        print(f"\n===== {condition} =====", flush=True)
        ranks, max_diff = evaluate_learned_condition(
            condition, q_np, gold_np, response_gpu, candidate_rows, device
        )
        ranks_by_model[condition] = ranks
        fast_diffs[condition] = max_diff
        print("fast_scorer_max_abs_diff =", max_diff, flush=True)
        print("rank median =", float(np.median(ranks)), "MRR micro =", float(np.mean(1.0 / ranks)), flush=True)

    dev_dialogue_ids = np.array([str(row["dialogue_id"]) for row in dev], dtype=str)
    dev_sample_ids = np.array([str(row["sample_id"]) for row in dev], dtype=str)
    long_ids = {str(row["sample_id"]) for row in long20}
    long_mask = np.array([sid in long_ids for sid in dev_sample_ids], dtype=bool)
    if int(long_mask.sum()) != len(long20):
        raise RuntimeError("Long20 subset alignment failed")

    rng = np.random.default_rng(s8.BOOTSTRAP_SEED)
    all_mask = np.ones(len(dev), dtype=bool)
    metrics = {
        "dev_all": s8.evaluate_metrics(
            ranks_by_model, dev_dialogue_ids, dev_sample_ids, all_mask, "dev_all", rng
        ),
        "dev_long20": s8.evaluate_metrics(
            ranks_by_model, dev_dialogue_ids, dev_sample_ids, long_mask, "dev_long20", rng
        ),
    }

    summary = {
        "artifact": "FSAR S9 Dev Top-100 Dense-Candidate Reranking Diagnostic Results V1",
        "status": "PASS",
        "scope": "EXTERNAL_DEV_TOP100_RERANKING_DIAGNOSTIC_ONLY",
        "protocol_path": rel(S9_PROTOCOL_FILE),
        "protocol_sha256": sha256_file(S9_PROTOCOL_FILE),
        "upstream_s8_protocol_sha256": sha256_file(S8_PROTOCOL_FILE),
        "dataset": {
            "dev_rows": int(len(dev)),
            "dev_dialogues": int(len(set(dev_dialogue_ids))),
            "dev_long20_rows": int(long_mask.sum()),
            "dev_long20_dialogues": int(len(set(dev_dialogue_ids[long_mask]))),
            "gold_row_coverage_diagnostic": json_safe(coverage),
        },
        "candidate_set": {
            "negative_count_per_query": TOP_K,
            "positive_count_per_query": 1,
            "negative_source": "BGE Top-100 TRAIN responses after normalized Gold identity exclusion",
            "gold_injected": True,
            "candidate_rows_saved": True,
        },
        "fast_scorer_max_abs_diff_by_condition": fast_diffs,
        "datasets": json_safe(metrics),
        "test_payload_used": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
        "interpretation_note": (
            "This is conditional reranking on a BGE-selected Top-100 negative set plus one injected Gold. "
            "It diagnoses local ranking behavior and does not establish end-to-end retrieval coverage."
        ),
    }

    report_lines = [
        "FSAR S9 DEV TOP-100 DENSE-CANDIDATE RERANKING DIAGNOSTIC",
        f"protocol_sha256 = {summary['protocol_sha256']}",
        f"dev_all_rows = {len(dev)}; dev_all_dialogues = {len(set(dev_dialogue_ids))}",
        f"dev_long20_rows = {int(long_mask.sum())}; dev_long20_dialogues = {len(set(dev_dialogue_ids[long_mask]))}",
        f"top_k_dense_negatives = {TOP_K}; one injected Gold per query",
        f"gold_row_coverage_train_library = {coverage['gold_row_coverage_n']}/{coverage['dev_rows']} ({coverage['gold_row_coverage_rate']:.4%})",
        f"gold_unique_coverage_train_library = {coverage['unique_gold_covered_n']}/{coverage['unique_gold_n']} ({coverage['unique_gold_coverage_rate']:.4%})",
        "metrics = dialogue-macro MRR, R@1, R@5, R@10, R@20; paired dialogue-cluster bootstrap CI",
        "tie_rule = conservative; all negatives with score >= Gold are counted ahead of Gold",
        "test_payload_used = NO",
        "",
    ]
    for dataset_name, dataset_result in metrics.items():
        report_lines.append(
            f"===== {dataset_name} (n={dataset_result['n_queries']}, dialogues={dataset_result['n_dialogues']}) ====="
        )
        for model_name, model_result in dataset_result["models"].items():
            bits = [model_name]
            for metric_name in s8.METRIC_NAMES:
                m = model_result["dialogue_macro"][metric_name]
                bits.append(
                    f"{metric_name}={m['estimate']:.6f} [{m['ci95_lower']:.6f}, {m['ci95_upper']:.6f}]"
                )
            report_lines.append(" | ".join(bits))
        report_lines.append("----- Paired contrasts vs Dense Top-100 and Rank-only -----")
        for model_name, contrast_dict in dataset_result["contrasts"].items():
            for base, vals in contrast_dict.items():
                bits = [f"{model_name} {base}"]
                for metric_name in s8.METRIC_NAMES:
                    m = vals[metric_name]
                    bits.append(
                        f"{metric_name}={m['delta']:+.6f} [{m['ci95_lower']:+.6f}, {m['ci95_upper']:+.6f}]"
                    )
                report_lines.append(" | ".join(bits))
        report_lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_tmp = SUMMARY_OUT.with_suffix(SUMMARY_OUT.suffix + ".tmp")
    predictions_tmp = PREDICTIONS_OUT.with_suffix(PREDICTIONS_OUT.suffix + ".tmp")
    report_tmp = REPORT_OUT.with_suffix(REPORT_OUT.suffix + ".tmp")
    atomic_json(summary_tmp, json_safe(summary))
    with predictions_tmp.open("wb") as f:
        np.savez_compressed(
            f,
            sample_ids=dev_sample_ids,
            dialogue_ids=dev_dialogue_ids,
            dev_long20_mask=long_mask,
            gold_match_idx=match_idx.astype(np.int32),
            candidate_response_rows=candidate_rows,
            candidate_dense_scores=candidate_dense_scores,
            gold_dense_scores=dense_gold_scores,
            **{f"ranks__{name}": ranks for name, ranks in ranks_by_model.items()},
        )
    report_tmp.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    os.replace(summary_tmp, SUMMARY_OUT)
    os.replace(predictions_tmp, PREDICTIONS_OUT)
    os.replace(report_tmp, REPORT_OUT)

    print("\n===== S9 OUTPUTS =====")
    print("S9_EVALUATION = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("PREDICTIONS =", PREDICTIONS_OUT)
    print("REPORT =", REPORT_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(S9_PROTOCOL_FILE))
    for dataset_name, dataset_result in metrics.items():
        print(f"\n[{dataset_name}]")
        for model_name, model_result in dataset_result["models"].items():
            vals = model_result["dialogue_macro"]
            print(model_name, json.dumps({m: round(vals[m]["estimate"], 6) for m in s8.METRIC_NAMES}))
    print("TEST_PAYLOAD_READ = NO")
    print("TRAINING_ARTIFACTS_MODIFIED = NO")
    print("CHECKPOINTS_MODIFIED = NO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Audit inputs and freeze S9 protocol only")
    group.add_argument("--evaluate", action="store_true", help="Run S9 after verifying frozen protocol and hashes")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
