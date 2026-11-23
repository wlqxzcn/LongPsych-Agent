#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import src.evaluation.s9_dev_top100_reranking_diagnostic as s9

from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel,
    RankOnlyNoProductMatchedModel,
    RFutureFullMatchedModel,
    RFutureNoProductMatchedModel,
    QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

DIM = 1024
BATCH = 32
N_BOOT = 5000
BOOT_SEED = 20261108

OUT = ROOT / "results/phase_s12c/dev_top100_matched_v1"
PROTOCOL = OUT / "protocol.json"
PRED = OUT / "predictions.npz"
SUMMARY = OUT / "summary.json"
REPORT = OUT / "report.txt"
RUN = OUT / "run_manifest.json"

S9_PRED = ROOT / "results/phase_s9/s9_dev_top100_reranking_predictions_v1.npz"
S11D_PRED = ROOT / "results/phase_s11d/s11d_full_library_boundary_predictions_v2.npz"

RANK_MANIFEST = ROOT / "results/phase_s12c/rank_only_matched_v1/run_manifest_v1.json"
FUTURE_MANIFEST = ROOT / "results/phase_s12c/step_2f5d_fold_vs_ensemble_audit_v1/run_manifest.json"

MODEL_FILE = ROOT / "src/models/s12c_matched_ablation_models.py"

CLASSES = {
    "RANK_ONLY_FULL_MATCHED": RankOnlyFullMatchedModel,
    "RANK_ONLY_NO_PRODUCT_MATCHED": RankOnlyNoProductMatchedModel,
    "R_FUTURE_FULL": RFutureFullMatchedModel,
    "R_FUTURE_NO_PRODUCT": RFutureNoProductMatchedModel,
    "QR_FUTURE_FULL": QRFutureFullMatchedModel,
    "QR_FUTURE_NO_PRODUCT": QRFutureNoProductMatchedModel,
}

RANK_CONDITIONS = [
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
]

FUTURE_CONDITIONS = [
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

CONDITIONS = list(CLASSES)

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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def source_files():
    return [
        S9_PRED,
        S11D_PRED,
        ROOT / "src/evaluation/s9_dev_top100_reranking_diagnostic.py",
        ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py",
        MODEL_FILE,
        RANK_MANIFEST,
        FUTURE_MANIFEST,
    ]


def load_data():
    print("LOADING FROZEN DEV DATA")

    data = s9.read_s8_data()
    if len(data) != 8:
        raise RuntimeError("Unexpected S8 data-loader signature")

    _, _, dev, _, _, _, match_idx, _ = data

    q, gold, responses = s9.load_embeddings()

    with np.load(S9_PRED, allow_pickle=False) as z:
        p = {k: z[k] for k in z.files}

    with np.load(S11D_PRED, allow_pickle=False) as z:
        p11 = {k: z[k] for k in z.files}

    sample_ids = np.asarray(
        [str(x["sample_id"]) for x in dev], dtype=str
    )
    dialogue_ids = np.asarray(
        [str(x["dialogue_id"]) for x in dev], dtype=str
    )

    match_idx = np.asarray(match_idx, dtype=np.int32)

    assert len(dev) == 9070
    assert q.shape == gold.shape == (9070, DIM)
    assert responses.shape == (61502, DIM)

    assert np.array_equal(sample_ids, p["sample_ids"])
    assert np.array_equal(dialogue_ids, p["dialogue_ids"])
    assert np.array_equal(match_idx, p["gold_match_idx"])

    assert np.array_equal(sample_ids, p11["sample_ids"])
    assert np.array_equal(dialogue_ids, p11["dialogue_ids"])
    assert np.array_equal(match_idx, p11["gold_match_idx"])

    candidates = p["candidate_response_rows"].astype(np.int32)

    assert candidates.shape == (9070, 100)
    assert np.array_equal(
        candidates, p11["s9_top100_response_rows"]
    )

    assert np.all(candidates >= 0)
    assert np.all(candidates < len(responses))

    for i, row in enumerate(candidates):
        if len(np.unique(row)) != 100:
            raise RuntimeError(f"Duplicate candidates at query {i}")

        if match_idx[i] >= 0 and match_idx[i] in row:
            raise RuntimeError(f"Gold identity leakage at query {i}")

    assert np.isfinite(q).all()
    assert np.isfinite(gold).all()
    assert np.isfinite(responses).all()

    assert p["dev_long20_mask"].shape == (9070,)
    assert p["ranks__DENSE_BGE"].shape == (9070,)

    print("DEV_QUERIES =", len(dev))
    print("DEV_DIALOGUES =", len(np.unique(dialogue_ids)))
    print("LONG20_QUERIES =", int(p["dev_long20_mask"].sum()))
    print("GOLD_MATCHED =", int((match_idx >= 0).sum()))
    print("GOLD_UNMATCHED =", int((match_idx < 0).sum()))
    print("TOP100_CANDIDATES = PASS")
    print("COHORT_IDENTITY = PASS")

    return (
        q.astype(np.float32, copy=False),
        gold.astype(np.float32, copy=False),
        responses,
        candidates,
        sample_ids,
        dialogue_ids,
        p["dev_long20_mask"],
        match_idx,
        p["ranks__DENSE_BGE"].astype(np.int32),
        p["candidate_dense_scores"],
        p["gold_dense_scores"],
    )


def checkpoint_entries():
    rm = json.loads(RANK_MANIFEST.read_text())
    fm = json.loads(FUTURE_MANIFEST.read_text())

    if rm.get("status") != "PASS":
        raise RuntimeError("Matched Rank-Only manifest is not PASS")

    entries = []

    for condition in RANK_CONDITIONS:
        items = rm["outputs"][condition]["checkpoints"]
        for item in items:
            entries.append({
                "condition": condition,
                "path": item["path"],
                "sha256": item["sha256"],
            })

    for condition in FUTURE_CONDITIONS:
        items = fm["checkpoint_manifest"][condition]
        for item in items:
            entries.append({
                "condition": condition,
                "path": item["path"],
                "sha256": item["sha256"],
            })

    if len(entries) != 30:
        raise RuntimeError(f"Expected 30 checkpoints, got {len(entries)}")

    seen = set()

    for item in entries:
        path = ROOT / item["path"]
        if not path.is_file():
            raise RuntimeError(f"Missing checkpoint: {path}")

        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch: {path}")

        ck = torch.load(
            path, map_location="cpu", weights_only=False
        )

        condition = item["condition"]
        fold = int(ck["fold"])

        if ck.get("condition") != condition:
            raise RuntimeError(f"Condition mismatch: {path}")

        if fold not in range(5):
            raise RuntimeError(f"Invalid fold: {path}")

        key = (condition, fold)
        if key in seen:
            raise RuntimeError(f"Duplicate checkpoint: {key}")
        seen.add(key)

        item["fold"] = fold
        item["best_epoch"] = int(ck["best_epoch"])

    expected = {
        (c, f) for c in CONDITIONS for f in range(5)
    }

    if seen != expected:
        raise RuntimeError("Checkpoint condition/fold coverage mismatch")

    return sorted(entries, key=lambda x: (x["condition"], x["fold"]))


def freeze():
    if OUT.exists():
        raise RuntimeError(
            f"Output directory exists; refusing overwrite: {OUT}"
        )

    data = load_data()
    entries = checkpoint_entries()

    hashes = {}
    for p in source_files():
        if not p.is_file():
            raise RuntimeError(f"Missing source: {p}")
        hashes[str(p.relative_to(ROOT))] = sha(p)

    protocol = {
        "artifact": "FSAR S12C Dev Top100 Matched Evaluation V1",
        "status": "FROZEN",
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "dev_queries": 9070,
        "dev_dialogues": int(len(np.unique(data[5]))),
        "candidate_set": {
            "source": "Frozen S9 BGE Top-100",
            "negative_candidates": 100,
            "gold_injected": True,
            "total_candidates": 101,
        },
        "ranking": {
            "tie_rule": "negative_score >= gold_score ranks before Gold",
            "rank_formula": "1 + count(negative_score >= gold_score)",
            "aggregation": "mean of five raw fold scores before ranking",
            "metrics": METRICS,
        },
        "statistics": {
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "confidence_level": 0.95,
            "resampling_unit": "dialogue",
        },
        "checkpoint_manifest": entries,
        "source_sha256": hashes,
        "training": False,
        "test_payload_used": False,
        "checkpoint_reselection": False,
    }

    OUT.mkdir(parents=True)
    write_json(PROTOCOL, protocol)

    print("PROTOCOL_FROZEN = PASS")
    print("CHECKPOINTS =", len(entries))
    print("PROTOCOL =", PROTOCOL)


def verify_protocol():
    if not PROTOCOL.is_file():
        raise RuntimeError("Run --prepare first")

    p = json.loads(PROTOCOL.read_text())

    if p.get("status") != "FROZEN":
        raise RuntimeError("Protocol is not frozen")

    for source, expected in p["source_sha256"].items():
        path = ROOT / source
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Source SHA mismatch: {source}")

    for item in p["checkpoint_manifest"]:
        path = ROOT / item["path"]
        if sha(path) != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch: {path}")

    return p


def validate_model(model, device):
    model.eval()

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


def load_models(condition, protocol, device):
    items = [
        x for x in protocol["checkpoint_manifest"]
        if x["condition"] == condition
    ]

    items = sorted(items, key=lambda x: x["fold"])

    if len(items) != 5:
        raise RuntimeError(f"{condition}: expected five folds")

    models = []

    for item in items:
        path = ROOT / item["path"]
        ck = torch.load(
            path, map_location="cpu", weights_only=False
        )

        model = CLASSES[condition]()
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device).eval()

        expected_dim = (
            3072 if "NO_PRODUCT" in condition else 4096
        )

        if model.ranker.rank_head.weight.shape[1] != expected_dim:
            raise RuntimeError(f"Rank dimension mismatch: {condition}")

        diff = validate_model(model, device)

        if diff > 1e-5:
            raise RuntimeError(
                f"Scorer parity failure: {condition}, fold={item['fold']}, diff={diff}"
            )

        print(
            f"LOAD_PASS {condition} fold={item['fold']} "
            f"score_diff={diff:.3e}"
        )

        models.append(model)

    return models


def preflight():
    protocol = verify_protocol()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")

    print("GPU =", torch.cuda.get_device_name(0))

    for condition in CONDITIONS:
        models = load_models(condition, protocol, device)
        del models
        torch.cuda.empty_cache()

    print("STEP_2F14A_PREFLIGHT = PASS")
    print("DEV_RANKING_INFERENCE = NO")
    print("TRAINING = NO")
    print("TEST_USED = NO")


def score_condition(
    condition, models, q_gpu, gold_gpu, r_gpu, candidates, device
):
    n = len(candidates)

    ranks = np.empty(n, dtype=np.int32)
    candidate_scores_out = np.empty((n, 100), dtype=np.float32)
    gold_scores_out = np.empty(n, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)

            qb = q_gpu[start:end]
            gb = gold_gpu[start:end]

            idx = torch.from_numpy(
                candidates[start:end].astype(np.int64, copy=False)
            ).to(device)

            rb = r_gpu[idx]

            fold_candidate_scores = []
            fold_gold_scores = []

            for model in models:
                fold_candidate_scores.append(
                    model.rank_scores(qb, rb)
                )
                fold_gold_scores.append(
                    model.rank_scores(qb, gb[:, None, :]).squeeze(1)
                )

            candidate_scores = torch.stack(
                fold_candidate_scores
            ).mean(0)

            gold_scores = torch.stack(
                fold_gold_scores
            ).mean(0)

            batch_ranks = 1 + (
                candidate_scores >= gold_scores[:, None]
            ).sum(1)

            ranks[start:end] = batch_ranks.cpu().numpy()
            candidate_scores_out[start:end] = (
                candidate_scores.cpu().numpy()
            )
            gold_scores_out[start:end] = gold_scores.cpu().numpy()

            if start == 0 or end == n or end % 512 == 0:
                print(
                    f"{condition}: {end}/{n}",
                    flush=True
                )

    return ranks, candidate_scores_out, gold_scores_out


def metric_matrix(ranks):
    ranks = np.asarray(ranks)
    return np.column_stack([
        1.0 / ranks,
        ranks <= 1,
        ranks <= 5,
        ranks <= 10,
        ranks <= 20,
    ]).astype(np.float64)


def evaluate_subset(ranks, dialogue_ids, indices, seed):
    dids = dialogue_ids[indices].astype(str)
    unique, inverse = np.unique(dids, return_inverse=True)

    grouped = {}

    for name, arr in ranks.items():
        values = metric_matrix(arr[indices])
        sums = np.zeros((len(unique), 5), dtype=np.float64)
        counts = np.bincount(inverse, minlength=len(unique))
        np.add.at(sums, inverse, values)
        grouped[name] = sums / counts[:, None]

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(
        0, len(unique), size=(N_BOOT, len(unique))
    )

    result = {
        "n_queries": int(len(indices)),
        "n_dialogues": int(len(unique)),
        "models": {},
        "paired_contrasts": {},
    }

    for name, values in grouped.items():
        boot = values[boot_idx].mean(axis=1)
        estimate = values.mean(axis=0)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)

        result["models"][name] = {
            metric: {
                "estimate": float(estimate[j]),
                "ci95_lower": float(lo[j]),
                "ci95_upper": float(hi[j]),
            }
            for j, metric in enumerate(METRICS)
        }

    for a, b in CONTRASTS:
        delta = grouped[a] - grouped[b]
        boot = delta[boot_idx].mean(axis=1)
        estimate = delta.mean(axis=0)
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)

        result["paired_contrasts"][f"{a}_MINUS_{b}"] = {
            metric: {
                "delta": float(estimate[j]),
                "ci95_lower": float(lo[j]),
                "ci95_upper": float(hi[j]),
            }
            for j, metric in enumerate(METRICS)
        }

    return result


def evaluate():
    protocol = verify_protocol()

    for path in (PRED, SUMMARY, REPORT, RUN):
        if path.exists():
            raise RuntimeError(f"Output collision: {path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    (
        q, gold, responses, candidates, sample_ids,
        dialogue_ids, long_mask, match_idx, dense_ranks,
        dense_candidate_scores, dense_gold_scores,
    ) = load_data()

    q_gpu = torch.from_numpy(q.copy()).to(device)
    gold_gpu = torch.from_numpy(gold.copy()).to(device)
    r_gpu = torch.from_numpy(
        np.array(responses, dtype=np.float32, copy=True)
    ).to(device)

    ranks = {"DENSE_BGE": dense_ranks.copy()}

    prediction_arrays = {
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "dev_long20_mask": long_mask,
        "gold_match_idx": match_idx,
        "candidate_response_rows": candidates,
        "candidate_dense_scores": dense_candidate_scores,
        "gold_dense_scores": dense_gold_scores,
        "ranks__DENSE_BGE": dense_ranks,
    }

    scorer_diffs = {}

    for condition in CONDITIONS:
        print("\nRUNNING", condition)

        models = load_models(condition, protocol, device)

        rr, cs, gs = score_condition(
            condition, models, q_gpu, gold_gpu,
            r_gpu, candidates, device
        )

        ranks[condition] = rr

        prediction_arrays[f"ranks__{condition}"] = rr
        prediction_arrays[f"candidate_scores__{condition}"] = cs
        prediction_arrays[f"gold_scores__{condition}"] = gs

        scorer_diffs[condition] = max(
            validate_model(m, device) for m in models
        )

        print(
            condition,
            "COMPLETE",
            "MRR =", float(np.mean(1.0 / rr)),
            "rank_min =", int(rr.min()),
            "rank_max =", int(rr.max()),
        )

        del models
        torch.cuda.empty_cache()

    all_idx = np.arange(len(q), dtype=np.int64)

    subsets = {
        "DEV_ALL": all_idx,
        "DEV_LONG20": np.flatnonzero(long_mask),
        "DEV_GOLD_MATCHED": np.flatnonzero(match_idx >= 0),
        "DEV_GOLD_UNMATCHED": np.flatnonzero(match_idx < 0),
    }

    datasets = {}

    for i, (name, indices) in enumerate(subsets.items()):
        if len(indices) == 0:
            raise RuntimeError(f"Empty evaluation subset: {name}")

        datasets[name] = evaluate_subset(
            ranks, dialogue_ids, indices, BOOT_SEED + i
        )

    summary = {
        "artifact": "FSAR S12C Dev Top100 Matched Evaluation V1",
        "status": "COMPLETED",
        "protocol_sha256": sha(PROTOCOL),
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "candidate_set": {
            "negative_candidates": 100,
            "gold_candidates": 1,
            "source": "Frozen S9 BGE Top-100",
        },
        "ensemble": {
            "folds": 5,
            "aggregation": "arithmetic_mean_of_five_raw_scores",
            "retraining": False,
            "checkpoint_reselection": False,
        },
        "ranking": {
            "tie_rule": "negative score >= Gold score ranks before Gold",
            "metrics": METRICS,
        },
        "statistics": {
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": BOOT_SEED,
            "confidence_level": 0.95,
        },
        "fast_scorer_max_abs_diff": scorer_diffs,
        "datasets": datasets,
        "training": False,
        "test_payload_used": False,
    }

    with PRED.with_suffix(".npz.tmp").open("wb") as f:
        np.savez_compressed(f, **prediction_arrays)

    os.replace(
        PRED.with_suffix(".npz.tmp"), PRED
    )

    write_json(SUMMARY, summary)

    lines = [
        "S12C DEV TOP100 MATCHED RERANKING EVALUATION",
        "=" * 76,
        "Candidate set: frozen S9 BGE Top-100 negatives + one injected Gold",
        "Tie rule: negative score >= Gold ranks before Gold",
        "Ensemble: arithmetic mean of five raw fold scores",
        f"Bootstrap: {N_BOOT} paired dialogue-cluster replicates",
        "Training = NO",
        "Test payload = NO",
    ]

    for ds, result in datasets.items():
        lines.extend(["", "=" * 76, ds, "=" * 76])

        for model, metrics in result["models"].items():
            fields = [
                f"{m}={v['estimate']:.6f}"
                f"[{v['ci95_lower']:.6f},{v['ci95_upper']:.6f}]"
                for m, v in metrics.items()
            ]
            lines.append(model + ": " + " | ".join(fields))

        lines.append("PAIRED CONTRASTS")

        for contrast, metrics in result["paired_contrasts"].items():
            fields = [
                f"{m}={v['delta']:+.6f}"
                f"[{v['ci95_lower']:+.6f},{v['ci95_upper']:+.6f}]"
                for m, v in metrics.items()
            ]
            lines.append(contrast + ": " + " | ".join(fields))

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    run_manifest = {
        "artifact": "S12C Dev Top100 Matched Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": sha(PROTOCOL),
        "evaluator_sha256": sha(Path(__file__)),
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "dev_queries": len(q),
        "dev_long20_queries": int(long_mask.sum()),
        "checkpoints": protocol["checkpoint_manifest"],
        "training": False,
        "test_used": False,
        "ranking_inference": True,
    }

    write_json(RUN, run_manifest)

    print("\nSTEP_2F14_DEV_TOP100_EVALUATION_COMPLETE")
    print("SUMMARY =", SUMMARY)
    print("PREDICTIONS =", PRED)
    print("REPORT =", REPORT)
    print("RUN_MANIFEST =", RUN)
    print("TEST_USED = NO")
    print("TRAINING = NO")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--evaluate", action="store_true")

    args = parser.parse_args()

    if args.prepare:
        freeze()
    elif args.preflight:
        preflight()
    elif args.evaluate:
        evaluate()


if __name__ == "__main__":
    main()
