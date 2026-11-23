#!/usr/bin/env python3
"""STEP_2F18T: ORCA-FSAR external Test Top-100 matched evaluation.

Run from LongPsych-Agent repository root:
  python src/evaluation/s12c_step_2f18t_orca_fsar_test_evaluation.py --prepare
  python src/evaluation/s12c_step_2f18t_orca_fsar_test_evaluation.py --preflight
  python src/evaluation/s12c_step_2f18t_orca_fsar_test_evaluation.py --evaluate

Prepare freezes the protocol without ranking Test queries. Preflight checks
all inputs, 30 checkpoints, ORCA folds and scorer parity without Test metrics.
Evaluate performs BGE Top-100 candidate construction and six-condition scoring.
No training, checkpoint reselection, Test-time fitting or overwrite is allowed.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
DEV_SCRIPT = ROOT / "src/evaluation/ORCA_FSAR_DEV_Evaluation_v2.py"
TEST_BASE_SCRIPT = ROOT / "src/evaluation/s12c_external_test_evaluation.py"

OUT = ROOT / "results/phase_s12c/step_2f18t_orca_fsar_test_evaluation_v1"
PROTOCOL = OUT / "test_evaluation_protocol_v1.json"
PRED = OUT / "test_predictions_v1.npz"
SUMMARY = OUT / "test_summary_v1.json"
REPORT = OUT / "test_report_v1.txt"
MANIFEST = OUT / "run_manifest.json"

TOP_K, N_CANDIDATES, N_FOLDS = 100, 101, 5
N_TEST, N_LONG20, DIM = 18073, 8727, 1024
N_BOOT, BOOT_SEED, BATCH_SIZE = 5000, 20261108, 32
METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def load_module(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    dev = load_module("step_2f18s_dev_for_2f18t", DEV_SCRIPT)
    test_base = load_module("s12c_test_data_loader_for_2f18t", TEST_BASE_SCRIPT)
    return dev, test_base


def load_test_data(dev, test_base):
    """Load frozen Test inputs without invoking the legacy 25-checkpoint audit.

    The legacy external-test protocol remains a source/data manifest only here.
    Its model-code hash and 25-checkpoint manifest belong to the old five-condition
    experiment and are deliberately not used by the six-condition STEP_2F18T run.
    Current model/checkpoint integrity is validated by the frozen DEV/training
    protocols through dev.verify_protocol() and dev.load_and_validate_training_run().
    """
    # Preserve all legacy frozen data-file hashes, excluding only the obsolete
    # model-code hash (the current model code is checked by the 2F18S protocol).
    legacy_model_code = Path(test_base.MODEL_CODE).resolve()
    for path_text, digest in test_base.EXPECTED.items():
        path = Path(path_text)
        if path.resolve() == legacy_model_code:
            continue
        test_base.check_hash(path, digest)

    ext_protocol = json.loads(test_base.P.read_text(encoding="utf-8"))
    if ext_protocol.get("status") != "FROZEN" or ext_protocol.get("training") is not False:
        raise RuntimeError("External Test data protocol is not frozen/no-training")
    if ext_protocol.get("inference") != "not_yet_run":
        raise RuntimeError("Legacy External Test protocol inference state mismatch")

    # Verify legacy data/source manifest entries, but intentionally do not
    # inspect or load its 25 obsolete model checkpoints.
    legacy_ckpt_paths = {
        str((ROOT / item["path"]).resolve())
        for item in ext_protocol.get("checkpoint_manifest", [])
        if isinstance(item, dict) and item.get("path")
    }
    for name, item in ext_protocol.get("file_manifest", {}).items():
        rel = str(item.get("path", ""))
        path = (ROOT / rel).resolve()
        if path == legacy_model_code or str(path) in legacy_ckpt_paths:
            continue
        if not rel or not item.get("sha256"):
            raise RuntimeError(f"Malformed legacy Test source manifest entry: {name}")
        test_base.check_hash(ROOT / rel, item["sha256"])

    em = json.loads(test_base.EMB_MANIFEST.read_text(encoding="utf-8"))
    if em.get("status") != "PASS" or em.get("scope") != "EXTERNAL_TEST":
        raise RuntimeError("Test embedding manifest status/scope mismatch")

    raw, raw_long = test_base.read_jsonl(test_base.TEST), test_base.read_jsonl(test_base.LONG)
    rows = test_base.read_jsonl(test_base.ALL_ROWS)
    long_rows = test_base.read_jsonl(test_base.LONG_ROWS)
    idx = np.load(test_base.LONG_IDX, allow_pickle=False)
    if (len(raw), len(rows), len(raw_long), len(long_rows), len(idx)) != (18074, 18073, 8728, 8727, 8727):
        raise RuntimeError("Test row counts do not match frozen protocol")

    excluded = {"psydial_1012_44"}
    raw_clean = [x for x in raw if str(x["sample_id"]) not in excluded]
    raw_long_clean = [x for x in raw_long if str(x["sample_id"]) not in excluded]
    raw_clean_ids = [str(x["sample_id"]) for x in raw_clean]
    all_ids = [str(x["sample_id"]) for x in rows]
    raw_long_ids = [str(x["sample_id"]) for x in raw_long_clean]
    long_ids = [str(x["sample_id"]) for x in long_rows]
    if all_ids != raw_clean_ids or long_ids != raw_long_ids:
        raise RuntimeError("Clean-row identity/order differs from source Test data")

    identity_fields = ("sample_id", "dialogue_id", "query", "gold_answer")
    for i, (clean_row, source_row) in enumerate(zip(rows, raw_clean)):
        for field in identity_fields:
            if clean_row.get(field) != source_row.get(field):
                raise RuntimeError(f"Test All source-content mismatch at row {i}, field={field}")
    for i, (clean_row, source_row) in enumerate(zip(long_rows, raw_long_clean)):
        for field in identity_fields:
            if clean_row.get(field) != source_row.get(field):
                raise RuntimeError(f"Long20 source-content mismatch at row {i}, field={field}")

    if idx.dtype.kind not in "iu" or len(np.unique(idx)) != len(idx):
        raise RuntimeError("Long20 index dtype/uniqueness failure")
    if np.any(idx < 0) or np.any(idx >= len(rows)):
        raise RuntimeError("Long20 index out of bounds")
    if [all_ids[int(i)] for i in idx] != long_ids:
        raise RuntimeError("Long20 indices do not reproduce clean Long20 row order")
    for j, source_index in enumerate(idx):
        for field in identity_fields:
            if long_rows[j].get(field) != rows[int(source_index)].get(field):
                raise RuntimeError(
                    f"Long20/Test All subset content mismatch at row {j}, field={field}"
                )
    for i, row in enumerate(long_rows):
        if int(row.get("history_pairs", -1)) < 20 or int(row.get("target_pair_index", -1)) < 20:
            raise RuntimeError(f"Long20 definition failure at row {i}")

    q = np.load(test_base.Q_EMB, allow_pickle=False)
    gold = np.load(test_base.G_EMB, allow_pickle=False)
    responses = np.load(test_base.RESP_EMB, allow_pickle=False)
    for name, arr, n in (("query", q, len(rows)), ("gold", gold, len(rows)), ("response", responses, 61502)):
        if arr.shape != (n, DIM) or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise RuntimeError(f"{name} embedding shape/dtype/finite check failed: {arr.shape}, {arr.dtype}")
        err = float(np.max(np.abs(np.linalg.norm(arr, axis=1) - 1.0)))
        if err > 1e-4:
            raise RuntimeError(f"{name} embeddings are not normalized: {err}")
        print(f"EMBEDDING_PASS {name} shape={arr.shape} max_norm_error={err:.3g}")

    response_rows = test_base.read_jsonl(test_base.RESP)
    response_index = test_base.read_jsonl(test_base.INDEX)
    if len(response_rows) != 61502 or len(response_index) != 61502:
        raise RuntimeError("Response/index row count mismatch")
    normalized_texts = []
    for i, row in enumerate(response_rows):
        normalized = test_base.norm(row["response_text"])
        if "normalized_response_text" in row and test_base.norm(row["normalized_response_text"]) != normalized:
            raise RuntimeError(f"Response normalization mismatch at row {i}")
        ix = response_index[i]
        if int(ix["embedding_row"]) != i or str(ix["response_uid"]) != str(row["response_uid"]):
            raise RuntimeError(f"Response/index alignment mismatch at row {i}")
        if int(ix["occurrence_count"]) != int(row["occurrence_count"]):
            raise RuntimeError(f"Response occurrence-count mismatch at row {i}")
        normalized_texts.append(normalized)
    if len(set(normalized_texts)) != 61502 or sum(x == "" for x in normalized_texts) != 1:
        raise RuntimeError("Response identity uniqueness/empty-row policy mismatch")

    lookup = {value: i for i, value in enumerate(normalized_texts)}
    match = np.asarray([lookup.get(test_base.norm(row["gold_answer"]), -1) for row in rows], dtype=np.int32)
    counts = 61503 - (match >= 0).astype(np.int32)
    unique_counts, count_values = np.unique(counts, return_counts=True)
    distribution = {int(a): int(b) for a, b in zip(unique_counts, count_values)}
    if distribution != {61502: 510, 61503: 17563}:
        raise RuntimeError(f"Test All candidate-count distribution mismatch: {distribution}")
    long_unique, long_values = np.unique(counts[idx], return_counts=True)
    long_distribution = {int(a): int(b) for a, b in zip(long_unique, long_values)}
    if long_distribution != {61502: 296, 61503: 8431}:
        raise RuntimeError(f"Long20 candidate-count distribution mismatch: {long_distribution}")

    print("LEGACY_25_CHECKPOINT_MANIFEST = IGNORED_FOR_DATA_ONLY_LOADING")
    print("LEGACY_CHECKPOINTS_LOADED = 0")
    print("TEST_ROWS =", len(rows), "LONG20_ROWS =", len(long_rows))
    print("GOLD_MATCH_ROWS =", int((match >= 0).sum()))
    print("CANDIDATE_COUNTS =", distribution)
    print("LONG20_CANDIDATE_COUNTS =", long_distribution)

    # Return a named mapping: all downstream STEP_2F18T stages consume
    # explicit fields (Prepare, Preflight and Evaluate share this contract).
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    dialogue_ids = np.asarray([str(row["dialogue_id"]) for row in rows], dtype=str)
    if sample_ids.shape != (N_TEST,) or dialogue_ids.shape != (N_TEST,):
        raise RuntimeError("Test sample/dialogue ID shape mismatch")

    return {
        "external_protocol": ext_protocol,
        "rows": rows,
        "long_rows": long_rows,
        "long_idx": idx.astype(np.int64, copy=False),
        "q": q,
        "gold": gold,
        "responses": responses,
        "match": match,
        "counts": counts,
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "response_pool_alignment": {
            "status": "PASS",
            "checks": [
                "response embedding file SHA256",
                "response JSONL/index row count",
                "embedding_row equals response row index",
                "response_uid alignment",
                "occurrence_count alignment",
                "normalized response identity uniqueness",
            ],
            "max_abs_diff": None,
            "max_abs_diff_note": (
                "Not applicable: the frozen response embedding matrix is the "
                "single source array; row alignment is verified against its index."
            ),
        },
    }


def source_paths(dev, test_base):
    return {
        "dev_evaluator": Path(dev.__file__).resolve(),
        "dev_protocol": dev.PROTOCOL_FILE,
        "training_protocol": dev.TRAIN_PROTOCOL,
        "training_run_manifest": dev.TRAIN_RUN_MANIFEST,
        "model_code": dev.MODEL_FILE,
        "training_script": dev.TRAIN_SCRIPT,
        "orca_basis_summary": dev.ORCA_BASIS_SUMMARY,
        "orca_projection_summary": dev.ORCA_PROJECTION_SUMMARY,
        "orca_integrity_audit": dev.ORCA_INTEGRITY_AUDIT,
        "external_test_loader": Path(test_base.__file__).resolve(),
        "external_test_protocol": test_base.P,
        "test_all": test_base.TEST,
        "test_long20": test_base.LONG,
        "test_all_clean_rows": test_base.ALL_ROWS,
        "test_long20_clean_rows": test_base.LONG_ROWS,
        "test_long20_indices": test_base.LONG_IDX,
        "test_query_embeddings": test_base.Q_EMB,
        "test_gold_embeddings": test_base.G_EMB,
        "test_embedding_manifest": test_base.EMB_MANIFEST,
        "train_response_texts": test_base.RESP,
        "train_response_index": test_base.INDEX,
        "train_response_embeddings": test_base.RESP_EMB,
    }


def prepare():
    if OUT.exists():
        raise RuntimeError(f"Output directory exists; refusing overwrite: {OUT}")
    dev, test_base = modules()
    dev_protocol = dev.verify_protocol()
    train_protocol, _, ckpts = dev.load_and_validate_training_run()
    orca = dev.load_and_validate_orca_artifacts(train_protocol)
    parity = dev.validate_projection_parity(orca)
    data = load_test_data(dev, test_base)

    hashes = {}
    for name, path in source_paths(dev, test_base).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing source {name}: {path}")
        hashes[name] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}

    protocol = {
        "artifact": "ORCA-FSAR S12C External Test Top100 Matched Evaluation Protocol V1",
        "step": "STEP_2F18T",
        "status": "FROZEN_BEFORE_TEST_RANKING_INFERENCE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_sha256": sha(Path(__file__).resolve()),
        "dev_protocol_sha256": sha(dev.PROTOCOL_FILE),
        "training_protocol_sha256": sha(dev.TRAIN_PROTOCOL),
        "training_run_manifest_sha256": sha(dev.TRAIN_RUN_MANIFEST),
        "external_test_protocol_sha256": sha(test_base.P),
        "test_embedding_manifest_sha256": sha(test_base.EMB_MANIFEST),
        "external_test_protocol_id": data["external_protocol"]["protocol_id"],
        "test": {
            "queries": N_TEST, "dialogues": int(len(np.unique(data["dialogue_ids"]))),
            "long20_queries": N_LONG20,
            "gold_matched": int((data["match"] >= 0).sum()),
            "gold_unmatched": int((data["match"] < 0).sum()),
            "query_shape": list(data["q"].shape), "gold_shape": list(data["gold"].shape),
            "response_shape": list(data["responses"].shape),
            "response_pool_alignment": data["response_pool_alignment"],
            "excluded_invalid_sample_id": "psydial_1012_44",
        },
        "candidate_set": {
            "negative_candidates": TOP_K, "injected_gold": 1, "total_candidates": N_CANDIDATES,
            "negative_source": "BGE Top-100 TRAIN responses",
            "selection": "torch.topk over float32 raw BGE dot products; sorted=True",
            "matched_gold_identity_excluded_before_topk": True,
            "interpretation": "conditional reranking; not end-to-end retrieval coverage",
        },
        "orca": {
            "method": "FOLD_SPECIFIC_ORCA_PROJECTION", "k": 2,
            "formula": "z = normalize(x - (xB_f)B_f.T)",
            "fold_artifacts": dev_protocol["orca"]["fold_artifacts"],
            "projection_parity": parity,
        },
        "models": {
            "conditions": ["DENSE_BGE"] + dev.CONDITIONS,
            "checkpoint_count": len(ckpts), "checkpoints": ckpts,
            "fold_pairing": "MODEL_FOLD_F_USES_ORCA_BASIS_F",
            "score_aggregation": "ARITHMETIC_MEAN_OF_FIVE_RAW_FOLD_SCORES_BEFORE_RANKING",
            "retraining": False, "checkpoint_reselection": False,
        },
        "ranking": {"tie_rule": "negative_score >= Gold ranks before Gold",
                    "rank_formula": "1 + count(negative_score >= gold_score)", "metrics": METRICS},
        "statistics": {"method": "paired_dialogue_cluster_bootstrap", "replicates": N_BOOT,
                       "seed": BOOT_SEED, "confidence_level": 0.95, "resampling_unit": "dialogue"},
        "source_sha256": hashes,
        "training": False, "checkpoint_reselection": False, "test_time_fitting": False,
        "test_ranking_inference_performed": False, "future_metrics_computed": False,
        "overwrite_existing": False,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    write_json(PROTOCOL, protocol)
    print("STEP_2F18T_PREPARE_COMPLETE")
    print("PROTOCOL_FROZEN = PASS")
    print("CHECKPOINTS =", len(ckpts), "/ 30")
    print("TEST_RANKING_INFERENCE = NO")
    print("TRAINING = NO")
    print("PROTOCOL =", PROTOCOL)
    print("PROTOCOL_SHA256 =", sha(PROTOCOL))


def verify(dev, test_base):
    if not PROTOCOL.is_file():
        raise FileNotFoundError("Run --prepare first")
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_BEFORE_TEST_RANKING_INFERENCE":
        raise RuntimeError("STEP_2F18T protocol status mismatch")
    if p.get("evaluator_sha256") != sha(Path(__file__).resolve()):
        raise RuntimeError("STEP_2F18T evaluator changed after freeze")
    dev_protocol = dev.verify_protocol()
    if sha(dev.PROTOCOL_FILE) != p["dev_protocol_sha256"]:
        raise RuntimeError("STEP_2F18S Dev protocol SHA changed")
    if sha(dev.TRAIN_PROTOCOL) != p["training_protocol_sha256"]:
        raise RuntimeError("Training protocol SHA changed")
    if sha(dev.TRAIN_RUN_MANIFEST) != p["training_run_manifest_sha256"]:
        raise RuntimeError("Training run manifest SHA changed")
    if sha(test_base.P) != p["external_test_protocol_sha256"]:
        raise RuntimeError("External Test protocol SHA changed")
    if sha(test_base.EMB_MANIFEST) != p["test_embedding_manifest_sha256"]:
        raise RuntimeError("External Test embedding manifest SHA changed")
    for name, item in p["source_sha256"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source SHA mismatch: {name}")
    if len(p["models"]["checkpoints"]) != 30:
        raise RuntimeError("Expected 30 frozen ORCA-FSAR checkpoints")
    for item in p["models"]["checkpoints"]:
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch: {path}")
    return p, dev_protocol


def preflight():
    dev, test_base = modules()
    protocol, dev_protocol = verify(dev, test_base)
    data = load_test_data(dev, test_base)
    train_protocol, _, ckpts = dev.load_and_validate_training_run()
    orca = dev.load_and_validate_orca_artifacts(train_protocol)
    dev.validate_projection_parity(orca)

    for fold in range(N_FOLDS):
        b = torch.from_numpy(orca["bases"][fold].copy())
        for name, raw in (("query", data["q"][:64]),
                          ("gold", data["gold"][:64]),
                          ("response", data["responses"][:64])):
            z = dev.project_torch(torch.from_numpy(raw.copy()), b)
            err = float(torch.max(torch.abs(torch.linalg.vector_norm(z, dim=-1) - 1.0)))
            if not torch.isfinite(z).all() or err > 2e-5:
                raise RuntimeError(f"Test {name} projection failed: fold={fold}, err={err}")
        print(f"TEST_ORCA_INPUT_FOLD_{fold}_PASS")

    for condition in dev.CONDITIONS:
        models = dev.load_condition_models(condition, dev_protocol, torch.device("cpu"))
        del models

    if len(ckpts) != 30:
        raise RuntimeError("Checkpoint audit is not 30/30")
    print("STEP_2F18T_PREFLIGHT = PASS")
    print("TEST_INPUT_ALIGNMENT = PASS")
    print("ORCA_PROJECTION_PARITY = PASS")
    print("CHECKPOINTS = 30/30")
    print("MODEL_SCORER_PARITY = PASS")
    print("TEST_RANKING_INFERENCE = NO")
    print("TRAINING = NO")


def build_top100(q, responses, match, device):
    n = len(q)
    rows = np.empty((n, TOP_K), dtype=np.int32)
    scores_out = np.empty((n, TOP_K), dtype=np.float32)
    q_gpu = torch.from_numpy(q.copy()).to(device)
    r_gpu = torch.from_numpy(responses.copy()).to(device)
    with torch.inference_mode():
        for s in range(0, n, BATCH_SIZE):
            e = min(s + BATCH_SIZE, n)
            scores = q_gpu[s:e] @ r_gpu.T
            local_match = match[s:e]
            valid = np.flatnonzero(local_match >= 0)
            if len(valid):
                rr = torch.as_tensor(valid, dtype=torch.long, device=device)
                cc = torch.as_tensor(local_match[valid], dtype=torch.long, device=device)
                scores[rr, cc] = float("-inf")
            vals, idx = torch.topk(scores, k=TOP_K, dim=1, largest=True, sorted=True)
            rows[s:e] = idx.cpu().numpy().astype(np.int32, copy=False)
            scores_out[s:e] = vals.cpu().numpy().astype(np.float32, copy=False)
            if s == 0 or e == n or e % 1024 == 0:
                print(f"BGE_TOP100: {e}/{n}", flush=True)
    if not np.isfinite(scores_out).all():
        raise RuntimeError("Non-finite Top-100 scores")
    for i, row in enumerate(rows):
        if len(np.unique(row)) != TOP_K:
            raise RuntimeError(f"Duplicate Top-100 candidates: row={i}")
        if match[i] >= 0 and np.any(row == match[i]):
            raise RuntimeError(f"Gold identity leaked into negatives: row={i}")
    return rows, scores_out, r_gpu


def evaluate():
    dev, test_base = modules()
    protocol, dev_protocol = verify(dev, test_base)
    for path in (PRED, SUMMARY, REPORT, MANIFEST):
        if path.exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --evaluate")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    print("GPU =", torch.cuda.get_device_name(0), "TF32=OFF")

    data = load_test_data(dev, test_base)
    candidates, dense_candidate_scores, responses_gpu = build_top100(
        data["q"], data["responses"], data["match"], device
    )
    q_gpu = torch.from_numpy(data["q"].copy()).to(device)
    gold_gpu = torch.from_numpy(data["gold"].copy()).to(device)
    dense_gold_scores = (q_gpu * gold_gpu).sum(dim=1).cpu().numpy().astype(np.float32)
    dense_ranks = (1 + (dense_candidate_scores >= dense_gold_scores[:, None]).sum(1)).astype(np.int32)
    dense_ce = (np.logaddexp(dense_gold_scores,
                 np.logaddexp.reduce(dense_candidate_scores, axis=1)) - dense_gold_scores).astype(np.float32)

    ranks = {"DENSE_BGE": dense_ranks}
    arrays = {
        "sample_ids": data["sample_ids"], "dialogue_ids": data["dialogue_ids"],
        "test_long20_indices": data["long_idx"], "gold_match_idx": data["match"],
        "candidate_response_rows": candidates, "candidate_dense_scores": dense_candidate_scores,
        "gold_dense_scores": dense_gold_scores, "ranks__DENSE_BGE": dense_ranks,
        "ranking_ce__DENSE_BGE": dense_ce,
    }
    scorer_diffs = {}
    for condition in dev.CONDITIONS:
        print("\nRUNNING", condition, flush=True)
        models = dev.load_condition_models(condition, dev_protocol, device)
        cr, cs, gs, ce = dev.score_condition(
            condition, models, q_gpu, gold_gpu, responses_gpu, candidates, device
        )
        ranks[condition] = cr
        arrays[f"ranks__{condition}"] = cr
        arrays[f"candidate_scores__{condition}"] = cs
        arrays[f"gold_scores__{condition}"] = gs
        arrays[f"ranking_ce__{condition}"] = ce
        scorer_diffs[condition] = {
            str(fold): float(dev.validate_model_scorer(model))
            for fold, model, _ in models
        }
        print(condition, "COMPLETE", "MRR =", float(np.mean(1.0 / cr)),
              "R@5 =", float(np.mean(cr <= 5)), "rank_min =", int(cr.min()),
              "rank_max =", int(cr.max()))
        del models
        torch.cuda.empty_cache()

    idx = {
        "TEST_ALL": np.arange(N_TEST, dtype=np.int64),
        "TEST_LONG20": data["long_idx"],
        "TEST_GOLD_MATCHED": np.flatnonzero(data["match"] >= 0),
        "TEST_GOLD_UNMATCHED": np.flatnonzero(data["match"] < 0),
    }
    datasets = {
        name: dev.evaluate_subset(ranks, data["dialogue_ids"], ids, BOOT_SEED + j)
        for j, (name, ids) in enumerate(idx.items())
    }
    summary = {
        "artifact": "ORCA-FSAR S12C External Test Top100 Matched Evaluation Summary V1",
        "status": "COMPLETED", "protocol_sha256": sha(PROTOCOL),
        "dev_protocol_sha256": sha(dev.PROTOCOL_FILE),
        "training_protocol_sha256": sha(dev.TRAIN_PROTOCOL),
        "external_test_protocol_sha256": sha(test_base.P),
        "conditions": ["DENSE_BGE"] + dev.CONDITIONS,
        "test_queries": N_TEST, "test_dialogues": int(len(np.unique(data["dialogue_ids"]))),
        "test_long20_queries": N_LONG20,
        "candidate_set": {"negative_candidates": TOP_K, "gold_candidates": 1,
                          "total_candidates": N_CANDIDATES,
                          "source": "BGE Top-100 TRAIN response negatives + injected Test Gold",
                          "matched_gold_excluded_from_negatives": True},
        "orca": {"method": "FOLD_SPECIFIC_ORCA_PROJECTION", "k": 2,
                 "formula": "z = normalize(x - (xB_f)B_f.T)",
                 "query_gold_and_candidates_projected_per_fold": True,
                 "fold_basis_hashes": {k: v["basis_sha256"] for k, v in dev_protocol["orca"]["fold_artifacts"].items()}},
        "ensemble": {"folds": N_FOLDS,
                     "aggregation": "arithmetic_mean_of_five_raw_fold_scores_before_ranking",
                     "fold_model_basis_pairing": "MODEL_FOLD_F_USES_ORCA_BASIS_F",
                     "retraining": False, "checkpoint_reselection": False},
        "ranking": {"tie_rule": "negative score >= Gold ranks before Gold", "metrics": METRICS},
        "statistics": {"method": "paired_dialogue_cluster_bootstrap", "replicates": N_BOOT,
                       "seed": BOOT_SEED, "confidence_level": 0.95, "resampling_unit": "dialogue"},
        "model_scorer_max_abs_diff": scorer_diffs, "datasets": datasets,
        "interpretation_limit": "Conditional reranking on BGE Top-100 negatives plus injected Gold; not full-library retrieval coverage.",
        "training": False, "checkpoint_reselection": False, "test_time_fitting": False,
        "future_metrics_computed": False,
    }
    tmp = PRED.with_suffix(PRED.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, PRED)
    write_json(SUMMARY, summary)

    lines = [
        "ORCA-FSAR S12C EXTERNAL TEST TOP-100 MATCHED EVALUATION",
        "Candidate set: BGE Top-100 TRAIN negatives + one injected Test Gold",
        "Gold identity excluded before Top-100 selection",
        "ORCA: fold-specific k=2; model fold-f paired with basis fold-f",
        "Ensemble: arithmetic mean of five raw fold scores before ranking",
        "Tie rule: negative score >= Gold ranks before Gold",
        f"Bootstrap: {N_BOOT} paired dialogue-cluster replicates",
        "Training=NO; checkpoint_reselection=NO; test_time_fitting=NO",
        "Interpretation: conditional Top-100 reranking, not full-library coverage",
    ]
    for name, result in datasets.items():
        lines += ["", "=" * 80, name, "=" * 80]
        for model, metrics in result["models"].items():
            fields = [f"{m}={v['estimate']:.6f}[{v['ci95_lower']:.6f},{v['ci95_upper']:.6f}]"
                      for m, v in metrics.items()]
            lines.append(model + ": " + " | ".join(fields))
        lines.append("PAIRED CONTRASTS")
        for contrast, metrics in result["paired_contrasts"].items():
            fields = [f"{m}={v['delta']:+.6f}[{v['ci95_lower']:+.6f},{v['ci95_upper']:+.6f}]"
                      for m, v in metrics.items()]
            lines.append(contrast + ": " + " | ".join(fields))
    write_text(REPORT, "\n".join(lines) + "\n")
    write_json(MANIFEST, {
        "artifact": "ORCA-FSAR S12C External Test Top100 Run Manifest V1",
        "status": "COMPLETED", "protocol_sha256": sha(PROTOCOL),
        "predictions_sha256": sha(PRED), "summary_sha256": sha(SUMMARY),
        "report_sha256": sha(REPORT), "test_queries": N_TEST,
        "test_dialogues": int(len(np.unique(data["dialogue_ids"]))),
        "test_long20_queries": N_LONG20, "conditions": ["DENSE_BGE"] + dev.CONDITIONS,
        "checkpoint_count": 30, "training": False, "checkpoint_reselection": False,
        "test_time_fitting": False, "ranking_inference": True,
    })
    print("STEP_2F18T_TEST_EVALUATION_COMPLETE")
    print("SUMMARY =", SUMMARY)
    print("PREDICTIONS =", PRED)
    print("REPORT =", REPORT)
    print("RUN_MANIFEST =", MANIFEST)
    print("TRAINING = NO; CHECKPOINT_RESELECTION = NO; TEST_TIME_FITTING = NO")


def main():
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
    else:
        evaluate()


if __name__ == "__main__":
    main()
