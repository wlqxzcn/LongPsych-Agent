#!/usr/bin/env python3
"""S11-B: response-only score-component diagnostic for FSAR.

Exploratory, post-hoc, Dev-only diagnostic. It does not train models or modify
checkpoints. Run --prepare first to freeze the protocol, then --evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.evaluation import s8_dev_retrieval_evaluation as s8  # noqa: E402

CONDITIONS = ["RANK_ONLY", "R_FUTURE_BETA8", "QR_FUTURE_BETA8"]

PROTOCOL_OUT = ROOT / "protocols/fsar_s11_response_only_component_diagnostic_v1.json"
RESULT_DIR = ROOT / "results/phase_s11"
SUMMARY_OUT = RESULT_DIR / "s11_response_only_component_summary_v1.json"
PREDICTIONS_OUT = RESULT_DIR / "s11_response_only_component_predictions_v1.npz"
REPORT_OUT = RESULT_DIR / "s11_response_only_component_report_v1.txt"
S8_PREDICTIONS = ROOT / "results/phase_s8/s8_dev_retrieval_predictions_v1.npz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Temporary output already exists; inspect before cleanup: {tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def input_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "dev_all": s8.DEV_ALL,
        "dev_long20": s8.DEV_LONG20,
        "train_responses": s8.TRAIN_RESPONSES,
        "train_response_index": s8.TRAIN_RESPONSE_INDEX,
        "query_embeddings": s8.QUERY_EMB_OUT,
        "gold_embeddings": s8.GOLD_EMB_OUT,
        "train_response_embeddings": s8.TRAIN_RESPONSE_EMB,
        "s8_predictions": S8_PREDICTIONS,
        "s8_evaluation_script": ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py",
        "model_code": ROOT / "src/models/s3c1_fsar_models.py",
        "s11_script": Path(__file__).resolve(),
    }
    checkpoint_registry = s8.all_checkpoint_paths()
    for condition in CONDITIONS:
        if condition not in checkpoint_registry:
            raise RuntimeError(f"Condition missing from S8 checkpoint registry: {condition}")
        for fold, path in enumerate(checkpoint_registry[condition]):
            paths[f"checkpoint_{condition}_fold{fold}"] = path
    return paths


def current_input_hashes() -> dict[str, str]:
    paths = input_paths()
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing S11 inputs:\n" + "\n".join(missing))
    return {name: sha256_file(path) for name, path in paths.items()}


def load_data():
    # Reuse S8's frozen validation, text normalization, and Gold-identity mapping.
    result = s8.load_and_validate_text_data()
    if len(result) != 8:
        raise RuntimeError(f"Unexpected S8 data-loader return signature: {len(result)}")
    responses, response_index, dev, long20, response_norms, train_norm_to_idx, match_idx, coverage = result

    q = np.load(s8.QUERY_EMB_OUT, mmap_mode="r", allow_pickle=False)
    gold = np.load(s8.GOLD_EMB_OUT, mmap_mode="r", allow_pickle=False)
    response_emb = np.load(s8.TRAIN_RESPONSE_EMB, mmap_mode="r", allow_pickle=False)

    if q.shape != (len(dev), s8.DIM):
        raise RuntimeError(f"Unexpected query embedding shape: {q.shape}")
    if gold.shape != (len(dev), s8.DIM):
        raise RuntimeError(f"Unexpected Gold embedding shape: {gold.shape}")
    if response_emb.shape != (len(responses), s8.DIM):
        raise RuntimeError(
            f"Response identity/embedding mismatch: {len(responses)} rows vs {response_emb.shape}"
        )
    if len(match_idx) != len(dev):
        raise RuntimeError("Gold match index is not aligned with Dev rows")

    return responses, dev, long20, np.asarray(match_idx, dtype=np.int64), gold, response_emb


def check_output_collisions() -> None:
    for path in (SUMMARY_OUT, PREDICTIONS_OUT, REPORT_OUT):
        for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
            if candidate.exists():
                raise RuntimeError(f"S11 output collision; refusing overwrite: {candidate}")


def prepare() -> None:
    if PROTOCOL_OUT.exists():
        raise RuntimeError(f"Protocol already exists; refusing overwrite: {PROTOCOL_OUT}")
    check_output_collisions()

    responses, dev, long20, match_idx, gold, response_emb = load_data()
    q = np.load(s8.QUERY_EMB_OUT, mmap_mode="r", allow_pickle=False)
    hashes = current_input_hashes()

    protocol = {
        "project": "PsychResponse_FSAR",
        "phase": "S11-B",
        "artifact": "FSAR S11-B Response-Only Score-Component Diagnostic V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "conditions": CONDITIONS,
        "response_component_definition": (
            "For each fold f, z_r_f = normalize(e_r + r_adapter_f(e_r)); "
            "component_f(r) = dot(z_r_f, rank_head_weight_f[DIM:2*DIM]); "
            "the reported component is the arithmetic mean over the five folds."
        ),
        "ranking_rule": "Rank = 1 + count(response_component >= Gold_component); ties precede Gold.",
        "gold_identity_rule": (
            "For Dev rows whose normalized Gold response maps to a TRAIN response_uid, "
            "exclude that exact mapped response row from the component-rank count."
        ),
        "candidate_universe": "All unique normalized TRAIN response identities",
        "expected_dev_rows": int(len(dev)),
        "expected_dev_long20_rows": int(len(long20)),
        "expected_unique_train_responses": int(len(responses)),
        "embedding_dimension": int(s8.DIM),
        "query_embeddings_used": False,
        "full_score_recomputed": False,
        "future_state_metrics_computed": False,
        "test_payload_used": False,
        "training_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
        "input_sha256": hashes,
    }
    PROTOCOL_OUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_json(PROTOCOL_OUT, protocol)

    print("===== S11-B PREPARE: READ-ONLY INPUT AUDIT =====")
    print("dev_rows =", len(dev))
    print("dev_long20_rows =", len(long20))
    print("unique_train_responses =", len(responses))
    print("query_embedding_shape =", tuple(q.shape))
    print("gold_embedding_shape =", tuple(gold.shape))
    print("response_embedding_shape =", tuple(response_emb.shape))
    print("test_payload_used = NO")
    print("model_inference_performed = NO")
    print("PROTOCOL =", PROTOCOL_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_OUT))
    print("PROTOCOL_STATUS = FROZEN_BEFORE_EVALUATION")


def component_ranks(
    library_scores: np.ndarray,
    gold_scores: np.ndarray,
    match_idx: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    """Rank Gold using only the candidate-response additive score component."""
    n = len(gold_scores)
    ranks = np.empty(n, dtype=np.int32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        gs = gold_scores[start:end]
        counts = (library_scores[None, :] >= gs[:, None]).sum(axis=1, dtype=np.int64)

        matched = match_idx[start:end]
        valid = matched >= 0
        if np.any(valid):
            local_rows = np.flatnonzero(valid)
            counted = library_scores[matched[valid]] >= gs[valid]
            counts[local_rows] -= counted.astype(np.int64)

        ranks[start:end] = (counts + 1).astype(np.int32)
        if start == 0 or end == n:
            print(f"COMPONENT_RANK_PROGRESS rows={end}/{n}", flush=True)
    return ranks


def metric_summary(ranks: np.ndarray) -> dict[str, Any]:
    ranks = np.asarray(ranks, dtype=np.int64)
    return {
        "n": int(len(ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "MRR_query_weighted": float(np.mean(1.0 / ranks)),
        "R@1": float(np.mean(ranks <= 1)),
        "R@5": float(np.mean(ranks <= 5)),
        "R@10": float(np.mean(ranks <= 10)),
        "R@100": float(np.mean(ranks <= 100)),
    }


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_OUT.is_file():
        raise FileNotFoundError(f"Frozen protocol missing; run --prepare first: {PROTOCOL_OUT}")
    protocol = json.loads(PROTOCOL_OUT.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise RuntimeError("S11 protocol status is not FROZEN_BEFORE_EVALUATION")
    if protocol.get("scope") != "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY":
        raise RuntimeError("Unexpected S11 scope")
    current = current_input_hashes()
    if current != protocol.get("input_sha256"):
        changed = [
            key for key in sorted(set(current) | set(protocol.get("input_sha256", {})))
            if current.get(key) != protocol.get("input_sha256", {}).get(key)
        ]
        raise RuntimeError("Frozen S11 input hash mismatch: " + ", ".join(changed))
    return protocol


def evaluate() -> None:
    protocol = verify_protocol()
    check_output_collisions()

    responses, dev, long20, match_idx, gold_np, response_np = load_data()
    s8_pred = np.load(S8_PREDICTIONS, allow_pickle=False)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("DEVICE =", device, flush=True)
    if device.type == "cuda":
        print("GPU =", torch.cuda.get_device_name(0), flush=True)

    response_gpu = torch.from_numpy(np.asarray(response_np, dtype=np.float32)).to(device)
    gold_gpu = torch.from_numpy(np.asarray(gold_np, dtype=np.float32)).to(device)

    sample_ids = np.asarray([str(row["sample_id"]) for row in dev], dtype=str)
    dialogue_ids = np.asarray([str(row["dialogue_id"]) for row in dev], dtype=str)
    long_ids = {str(row["sample_id"]) for row in long20}
    long_mask = np.asarray([sid in long_ids for sid in sample_ids], dtype=bool)

    summary: dict[str, Any] = {
        "project": "PsychResponse_FSAR",
        "phase": "S11-B",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "protocol_sha256": sha256_file(PROTOCOL_OUT),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "conditions": {},
        "test_payload_used": False,
        "training_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
        "interpretation_note": (
            "This ranks Gold using only the additive response-side score component. "
            "It is diagnostic, not a standalone retrieval model or a causal attribution."
        ),
    }

    pred_out: dict[str, np.ndarray] = {
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "dev_long20_mask": long_mask.astype(np.uint8),
        "match_idx": match_idx.astype(np.int32),
    }

    report_lines = [
        "FSAR S11-B RESPONSE-ONLY SCORE-COMPONENT DIAGNOSTIC",
        f"protocol_sha256 = {summary['protocol_sha256']}",
        "scope = EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        f"device = {device}",
        f"dev_rows = {len(dev)}",
        f"dev_long20_rows = {int(long_mask.sum())}",
        f"unique_train_responses = {len(responses)}",
        "test_payload_used = NO",
        "training_performed = NO",
        "",
    ]

    for condition in CONDITIONS:
        print(f"\n===== {condition}: RESPONSE-ONLY COMPONENT =====", flush=True)
        models = s8.load_condition_models(condition, device)

        library_sum = torch.zeros(response_gpu.shape[0], dtype=torch.float32, device=device)
        gold_sum = torch.zeros(gold_gpu.shape[0], dtype=torch.float32, device=device)

        with torch.inference_mode():
            for fold_i, model in enumerate(models):
                if model.ranker.rank_head.weight.shape != (1, 4 * s8.DIM):
                    raise RuntimeError(f"Unexpected rank head shape in {condition}, fold {fold_i}")
                if model.ranker.rank_head.bias is not None:
                    raise RuntimeError(f"Expected bias-free rank head in {condition}, fold {fold_i}")

                # Interaction order is [z_q, z_r, |z_q-z_r|, z_q*z_r].
                # This is the additive candidate-response-only block.
                w_r = model.ranker.rank_head.weight[0, s8.DIM:2 * s8.DIM]
                z_r = model.ranker.adapt_response(response_gpu)
                z_gold = model.ranker.adapt_response(gold_gpu)

                library_sum += torch.mv(z_r, w_r)
                gold_sum += (z_gold * w_r).sum(dim=-1)

                del z_r, z_gold
                print(f"FOLD_COMPONENT_COMPLETE condition={condition} fold={fold_i}", flush=True)

        library_scores = (library_sum / len(models)).cpu().numpy().astype(np.float32, copy=False)
        gold_scores = (gold_sum / len(models)).cpu().numpy().astype(np.float32, copy=False)
        del models, library_sum, gold_sum
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ranks = component_ranks(library_scores, gold_scores, match_idx)
        full_key = f"ranks__{condition}"
        if full_key not in s8_pred:
            raise RuntimeError(f"Missing S8 full-pool ranks: {full_key}")
        full_ranks = np.asarray(s8_pred[full_key], dtype=np.int32)
        if full_ranks.shape != ranks.shape:
            raise RuntimeError(f"Full-rank shape mismatch for {condition}")

        condition_summary: dict[str, Any] = {
            "response_only_component_rank": {},
            "full_score_rank_reference": {},
            "log_rank_pearson_r_with_full_score": {},
            "library_component_score_distribution": {
                "mean": float(np.mean(library_scores)),
                "std": float(np.std(library_scores)),
                "p01": float(np.quantile(library_scores, 0.01)),
                "p50": float(np.quantile(library_scores, 0.50)),
                "p99": float(np.quantile(library_scores, 0.99)),
                "max": float(np.max(library_scores)),
                "min": float(np.min(library_scores)),
            },
        }

        for subset_name, mask in (
            ("dev_all", np.ones(len(dev), dtype=bool)),
            ("dev_long20", long_mask),
        ):
            component_metric = metric_summary(ranks[mask])
            full_metric = metric_summary(full_ranks[mask])
            x = np.log1p(ranks[mask].astype(np.float64))
            y = np.log1p(full_ranks[mask].astype(np.float64))
            corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")

            condition_summary["response_only_component_rank"][subset_name] = component_metric
            condition_summary["full_score_rank_reference"][subset_name] = full_metric
            condition_summary["log_rank_pearson_r_with_full_score"][subset_name] = corr

            report_lines.append(f"[{condition} / {subset_name}]")
            report_lines.append("RESPONSE_ONLY " + json.dumps(component_metric, ensure_ascii=False))
            report_lines.append("FULL_SCORE_REF " + json.dumps(full_metric, ensure_ascii=False))
            report_lines.append(f"log_rank_pearson_r = {corr:.6f}")
            report_lines.append("")

            print(subset_name, "RESPONSE_ONLY", json.dumps(component_metric), flush=True)
            print(subset_name, "FULL_SCORE_REF", json.dumps(full_metric), flush=True)
            print("log_rank_pearson_r =", round(corr, 6), flush=True)

        summary["conditions"][condition] = condition_summary
        pred_out[f"response_only_ranks__{condition}"] = ranks
        pred_out[f"response_only_gold_component__{condition}"] = gold_scores.astype(np.float32)
        pred_out[f"response_only_library_component__{condition}"] = library_scores.astype(np.float32)
        pred_out[f"full_ranks__{condition}"] = full_ranks

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(SUMMARY_OUT, summary)
    np.savez_compressed(PREDICTIONS_OUT, **pred_out)
    REPORT_OUT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n===== S11-B OUTPUTS =====")
    print("S11_B_EVALUATION = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("PREDICTIONS =", PREDICTIONS_OUT)
    print("REPORT =", REPORT_OUT)
    print("PROTOCOL_SHA256 =", summary["protocol_sha256"])
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING_ARTIFACTS_MODIFIED = NO")
    print("CHECKPOINTS_MODIFIED = NO")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Audit inputs and freeze protocol; no model inference")
    group.add_argument("--evaluate", action="store_true", help="Run response-only component diagnostic")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
