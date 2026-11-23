#!/usr/bin/env python3
"""S12C frozen external Test evaluator.
Run:
  python src/evaluation/s12c_external_test_evaluation.py --preflight
  python src/evaluation/s12c_external_test_evaluation.py --evaluate
Preflight does not rank Test queries. Evaluation does not train or select checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, unicodedata
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.models.s12c_matched_ablation_models import (
    RankOnlyFullMatchedModel, RFutureFullMatchedModel,
    RFutureNoProductMatchedModel, QRFutureFullMatchedModel,
    QRFutureNoProductMatchedModel,
)

DIM, QB, CC = 1024, 16, 2048
N_BOOT, BOOT_SEED = 2000, 20261027
METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]
P = ROOT / "protocols/fsar_s12c_external_test_evaluation_v1.json"
MODEL_CODE = ROOT / "src/models/s12c_matched_ablation_models.py"
RESP = ROOT / "data/canonical/fsar_s2/train_unique_responses_v1.jsonl"
INDEX = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_index_v1.jsonl"
RESP_EMB = ROOT / "cache/embeddings/fsar_train_unique_responses_bge_v1.npy"
TEST = ROOT / "data/splits/psydial_targets_test_all.jsonl"
LONG = ROOT / "data/splits/psydial_targets_test_long20.jsonl"
D = ROOT / "results/phase_s12c/external_test_v1"
ALL_ROWS = D / "test_all_clean_rows_v1.jsonl"
LONG_ROWS = D / "test_long20_clean_rows_v1.jsonl"
LONG_IDX = D / "test_long20_clean_indices_v1.npy"
Q_EMB = D / "test_all_clean_query_embeddings_bge_v1.npy"
G_EMB = D / "test_all_clean_gold_embeddings_bge_v1.npy"
EMB_MANIFEST = D / "test_embedding_manifest_v1.json"

OUT = {
    "predictions": D / "s12c_external_test_ranking_predictions_v1.npz",
    "summary": D / "s12c_external_test_ranking_summary_v1.json",
    "report": D / "s12c_external_test_ranking_report_v1.txt",
    "manifest": D / "s12c_external_test_ranking_run_manifest_v1.json",
}
EXPECTED = {
    str(P): "52d14bce309a854dd9a2722ecc118dea68ab5e61ff7c4052c91c25c8110894cd",
    str(MODEL_CODE): "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c",
    str(RESP): "6ed2a5807c7a3bdcb30b7b6b947265129b19925a2cf7e8872fa36231b4cab25e",
    str(INDEX): "fd8263c8fd00b4d8d23b915c7f241fb84d1d2ffc7b24c747bf34e4a9b8e47274",
    str(RESP_EMB): "7a4ee8d462adce73174c853bf21b758132f5243020fa6253b1c01a99973bc96f",
    str(TEST): "f3b7190c735d51666b1035b5724dfb71528d80eb2e6036d244817f783f4788a3",
    str(LONG): "7100328fb3ec9f64735f07b0594be4a899559cf71a1075bc990c2846ab783f53",
    str(ALL_ROWS): "64479b1c8c07d0c8530a4b9a0c94e30e540ca149d9505cd77b7919f0e3624aca",
    str(LONG_ROWS): "408b0d768d835a9f6caef455d8afaa3b874af17703d4e8514dc829d706f5cb6c",
    str(LONG_IDX): "1495cbf2d8e8e9d872561826744172eceb5832a6ef1b124c8da4aa58a4adb273",
    str(Q_EMB): "41ab1778e1576380c4c053ff806d7178807a264d145d59579a1603a0e4341a11",
    str(G_EMB): "9bc63307efe88cea9e4f4c93c88832978a92d76df547f8b649d635b8f1ef59cf",
    str(EMB_MANIFEST): "ee1a572c7c397b9a0b334b801e67547776539c66ffdaf4979f683079bb11cf10",
}
CLASSES = {
    "RANK_ONLY": RankOnlyFullMatchedModel,
    "R_FUTURE_FULL": RFutureFullMatchedModel,
    "R_FUTURE_NO_PRODUCT": RFutureNoProductMatchedModel,
    "QR_FUTURE_FULL": QRFutureFullMatchedModel,
    "QR_FUTURE_NO_PRODUCT": QRFutureNoProductMatchedModel,
}
CONDITIONS = list(CLASSES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def check_hash(path, expected):
    if not Path(path).is_file():
        raise RuntimeError(f"Missing file: {path}")
    got = sha(path)
    if got != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}\nexpected={expected}\nactual={got}")
    print("HASH PASS:", Path(path).name)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def norm(text):
    text = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", text)


def load_inputs():
    for path, digest in EXPECTED.items():
        check_hash(path, digest)

    protocol = json.loads(P.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN" or protocol.get("training") is not False:
        raise RuntimeError("External Test protocol is not frozen/no-training")
    if protocol.get("inference") != "not_yet_run":
        raise RuntimeError("Protocol inference state is not not_yet_run")

    # Verify every source and all 25 checkpoint hashes from the frozen protocol.
    for name, item in protocol["file_manifest"].items():
        check_hash(ROOT / item["path"], item["sha256"])
    ckpts = protocol["checkpoint_manifest"]
    if len(ckpts) != 25:
        raise RuntimeError(f"Expected 25 checkpoints, got {len(ckpts)}")
    seen = set()
    for item in ckpts:
        key = (item["condition"], int(item["fold"]))
        if key in seen:
            raise RuntimeError(f"Duplicate checkpoint entry: {key}")
        seen.add(key)
        check_hash(ROOT / item["path"], item["sha256"])
    if seen != {(c, f) for c in CONDITIONS for f in range(5)}:
        raise RuntimeError("Checkpoint condition/fold manifest mismatch")

    em = json.loads(EMB_MANIFEST.read_text(encoding="utf-8"))
    if em.get("status") != "PASS" or em.get("scope") != "EXTERNAL_TEST":
        raise RuntimeError("Test embedding manifest status/scope mismatch")

    raw, raw_long = read_jsonl(TEST), read_jsonl(LONG)
    rows, long_rows = read_jsonl(ALL_ROWS), read_jsonl(LONG_ROWS)
    idx = np.load(LONG_IDX, allow_pickle=False)
    if (len(raw), len(rows), len(raw_long), len(long_rows), len(idx)) != (18074, 18073, 8728, 8727, 8727):
        raise RuntimeError("Test row counts do not match frozen protocol")

    excluded = {"psydial_1012_44"}
    raw_clean_ids = [str(x["sample_id"]) for x in raw if str(x["sample_id"]) not in excluded]
    all_ids = [str(x["sample_id"]) for x in rows]
    raw_long_ids = [str(x["sample_id"]) for x in raw_long if str(x["sample_id"]) not in excluded]
    long_ids = [str(x["sample_id"]) for x in long_rows]
    if all_ids != raw_clean_ids or long_ids != raw_long_ids:
        raise RuntimeError("Clean-row identity/order differs from source Test data")

    # Verify that the cleaned artifacts preserve source text and grouping fields.
    identity_fields = ("sample_id", "dialogue_id", "query", "gold_answer")
    for i, (clean_row, source_row) in enumerate(zip(rows, [x for x in raw if str(x["sample_id"]) not in excluded])):
        for field in identity_fields:
            if clean_row.get(field) != source_row.get(field):
                raise RuntimeError(f"Test All source-content mismatch at row {i}, field={field}")
    for i, (clean_row, source_row) in enumerate(zip(long_rows, [x for x in raw_long if str(x["sample_id"]) not in excluded])):
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
                    f"Long20/Test All subset content mismatch at long20 row {j}, field={field}"
                )
    for i, row in enumerate(long_rows):
        if int(row.get("history_pairs", -1)) < 20 or int(row.get("target_pair_index", -1)) < 20:
            raise RuntimeError(f"Long20 definition failure at row {i}")

    q, g = np.load(Q_EMB, allow_pickle=False), np.load(G_EMB, allow_pickle=False)
    r = np.load(RESP_EMB, allow_pickle=False)
    for name, a, n in [("query", q, len(rows)), ("gold", g, len(rows)), ("response", r, 61502)]:
        if a.shape != (n, DIM) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise RuntimeError(f"{name} embedding shape/dtype/finite check failed: {a.shape}, {a.dtype}")
        err = float(np.max(np.abs(np.linalg.norm(a, axis=1) - 1)))
        if err > 1e-4:
            raise RuntimeError(f"{name} embeddings are not normalized: {err}")
        print(f"EMBEDDING PASS: {name}, shape={a.shape}, max_norm_error={err:.3g}")

    responses, index = read_jsonl(RESP), read_jsonl(INDEX)
    if len(responses) != 61502 or len(index) != 61502:
        raise RuntimeError("Response/index row count mismatch")
    norms = []
    for i, row in enumerate(responses):
        n = norm(row["response_text"])
        if "normalized_response_text" in row and norm(row["normalized_response_text"]) != n:
            raise RuntimeError(f"Normalization mismatch at response row {i}")
        ix = index[i]
        if int(ix["embedding_row"]) != i or str(ix["response_uid"]) != str(row["response_uid"]):
            raise RuntimeError(f"Response/index alignment mismatch at row {i}")
        if int(ix["occurrence_count"]) != int(row["occurrence_count"]):
            raise RuntimeError(f"Occurrence-count mismatch at row {i}")
        norms.append(n)
    if len(set(norms)) != 61502 or sum(x == "" for x in norms) != 1:
        raise RuntimeError("Response identity uniqueness/empty-row policy mismatch")

    lookup = {x: i for i, x in enumerate(norms)}
    match = np.asarray([lookup.get(norm(x["gold_answer"]), -1) for x in rows], dtype=np.int32)
    counts = 61503 - (match >= 0).astype(np.int32)
    u, c = np.unique(counts, return_counts=True)
    dist = {int(a): int(b) for a, b in zip(u, c)}
    if dist != {61502: 510, 61503: 17563}:
        raise RuntimeError(f"Test All candidate-count distribution mismatch: {dist}")
    lu, lc = np.unique(counts[idx], return_counts=True)
    ldist = {int(a): int(b) for a, b in zip(lu, lc)}
    if ldist != {61502: 296, 61503: 8431}:
        raise RuntimeError(f"Long20 candidate-count distribution mismatch: {ldist}")

    print("ROW ALIGNMENT PASS")
    print("GOLD_MATCH_ROWS =", int((match >= 0).sum()))
    print("CANDIDATE_COUNTS =", dist)
    print("LONG20_CANDIDATE_COUNTS =", ldist)
    return protocol, rows, idx.astype(np.int64), q, g, r, match, counts


def fast_score(model, zq, zr):
    w = model.ranker.rank_head.weight[0]
    if model.ranker.rank_head.bias is not None:
        raise RuntimeError("Ranking head unexpectedly has bias")
    a = (zq * w[:DIM]).sum(-1, keepdim=True)
    b = (zr * w[DIM:2*DIM]).sum(-1).unsqueeze(0)
    c = (torch.abs(zq[:, None, :] - zr[None, :, :]) * w[2*DIM:3*DIM]).sum(-1)
    if w.numel() == 4 * DIM:
        return a + b + c + torch.matmul(zq * w[3*DIM:4*DIM], zr.T)
    if w.numel() == 3 * DIM:
        return a + b + c
    raise RuntimeError(f"Unexpected rank-head dimension: {w.numel()}")


def validate_scorer(model, device):
    q, r = torch.randn(2, DIM, device=device), torch.randn(3, DIM, device=device)
    with torch.inference_mode():
        zq, zr = model.ranker.adapt_query(q), model.ranker.adapt_response(r)
        fast = fast_score(model, zq, zr)
        ref = model.rank_scores(q, r.unsqueeze(0).expand(2, -1, -1))
        diff = float((fast - ref).abs().max().item())
    if diff > 1e-5:
        raise RuntimeError(f"Fast/reference scorer mismatch: {diff}")
    return diff


def load_models(condition, protocol, device):
    items = sorted([x for x in protocol["checkpoint_manifest"] if x["condition"] == condition],
                   key=lambda x: int(x["fold"]))
    if len(items) != 5 or [int(x["fold"]) for x in items] != list(range(5)):
        raise RuntimeError(f"{condition}: checkpoint folds are incomplete")
    models = []
    for item in items:
        ck = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
        if ck.get("condition") != condition or int(ck.get("fold", -1)) != int(item["fold"]):
            raise RuntimeError(f"Checkpoint metadata mismatch: {item['path']}")
        model = CLASSES[condition]()
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.to(device).eval()
        dim = int(model.ranker.rank_head.weight.shape[1])
        expected = 3072 if "NO_PRODUCT" in condition else 4096
        if dim != expected:
            raise RuntimeError(f"{condition}: rank-head dimension {dim}, expected {expected}")
        models.append(model)
    return models


def rank_dense(q_np, g_np, r_gpu, match, device):
    q, g = torch.from_numpy(q_np).to(device), torch.from_numpy(g_np).to(device)
    out = np.zeros(len(q_np), dtype=np.int32)
    with torch.inference_mode():
        gs = (q * g).sum(1)
        for qs in range(0, len(q_np), QB):
            qe = min(qs + QB, len(q_np))
            cnt = torch.zeros(qe - qs, dtype=torch.int64, device=device)
            for cs in range(0, r_gpu.shape[0], CC):
                ce = min(cs + CC, r_gpu.shape[0])
                s = q[qs:qe] @ r_gpu[cs:ce].T
                cnt += (s >= gs[qs:qe, None]).sum(1)
                mi = match[qs:qe]
                mask = (mi >= cs) & (mi < ce)
                if np.any(mask):
                    ri = torch.from_numpy(np.flatnonzero(mask)).to(device)
                    ci = torch.from_numpy((mi[mask] - cs).astype(np.int64)).to(device)
                    cnt[ri] -= (s[ri, ci] >= gs[qs:qe][ri]).to(torch.int64)
            out[qs:qe] = (cnt + 1).cpu().numpy()
    return out


def rank_learned(q_np, g_np, r_gpu, match, models, device):
    q, g = torch.from_numpy(q_np).to(device), torch.from_numpy(g_np).to(device)
    zqs, zrs, gss, maxdiff = [], [], [], 0.0
    with torch.inference_mode():
        for m in models:
            maxdiff = max(maxdiff, validate_scorer(m, device))
            zq, zg = m.ranker.adapt_query(q), m.ranker.adapt_response(g)
            zqs.append(zq)
            gss.append(m.ranker.score_from_adapted(zq, zg))
            zrs.append(m.ranker.adapt_response(r_gpu))
        gold_score = torch.stack(gss).mean(0)
        out = np.zeros(len(q_np), dtype=np.int32)
        for qs in range(0, len(q_np), QB):
            qe = min(qs + QB, len(q_np))
            cnt = torch.zeros(qe - qs, dtype=torch.int64, device=device)
            gs = gold_score[qs:qe]
            for cs in range(0, r_gpu.shape[0], CC):
                ce = min(cs + CC, r_gpu.shape[0])
                score_sum = None
                for fi, m in enumerate(models):
                    s = fast_score(m, zqs[fi][qs:qe], zrs[fi][cs:ce])
                    score_sum = s if score_sum is None else score_sum + s
                scores = score_sum / len(models)
                cnt += (scores >= gs[:, None]).sum(1)
                mi = match[qs:qe]
                mask = (mi >= cs) & (mi < ce)
                if np.any(mask):
                    ri = torch.from_numpy(np.flatnonzero(mask)).to(device)
                    ci = torch.from_numpy((mi[mask] - cs).astype(np.int64)).to(device)
                    cnt[ri] -= (scores[ri, ci] >= gs[ri]).to(torch.int64)
            out[qs:qe] = (cnt + 1).cpu().numpy()
    return out, maxdiff


def group_values(ranks, dids):
    v = np.column_stack([1/ranks, ranks <= 1, ranks <= 5, ranks <= 10, ranks <= 20]).astype(float)
    unique, inv = np.unique(dids.astype(str), return_inverse=True)
    sums = np.zeros((len(unique), 5), dtype=float)
    cnt = np.bincount(inv, minlength=len(unique)).astype(float)
    np.add.at(sums, inv, v)
    return unique, sums / cnt[:, None]


def evaluate_subset(ranks, dids, indices, rng):
    did = dids[indices].astype(str)
    unique = np.unique(did)
    grouped = {}
    for name, arr in ranks.items():
        ids, vals = group_values(arr[indices], did)
        if not np.array_equal(ids, unique):
            raise RuntimeError("Dialogue group ordering mismatch")
        grouped[name] = vals
    boot_idx = rng.integers(0, len(unique), size=(N_BOOT, len(unique)))

    def ci(v):
        est = v.mean(0)
        boot = v[boot_idx].mean(1)
        lo, hi = np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0)
        return {
            m: {"estimate": float(est[j]), "ci95_lower": float(lo[j]), "ci95_upper": float(hi[j])}
            for j, m in enumerate(METRICS)
        }

    result = {"n_queries": int(len(indices)), "n_dialogues": int(len(unique)), "models": {}, "contrasts": {}}
    for name, vals in grouped.items():
        result["models"][name] = {"dialogue_macro": ci(vals)}

    def contrast(category, name, base):
        v = grouped[name] - grouped[base]
        est = v.mean(0)
        boot = v[boot_idx].mean(1)
        lo, hi = np.quantile(boot, .025, axis=0), np.quantile(boot, .975, axis=0)
        result["contrasts"].setdefault(category, {})[f"{name}_MINUS_{base}"] = {
            m: {"delta": float(est[j]), "ci95_lower": float(lo[j]), "ci95_upper": float(hi[j])}
            for j, m in enumerate(METRICS)
        }

    for c in CONDITIONS:
        contrast("learned_vs_dense_bge", c, "DENSE_BGE")
    for c in CONDITIONS:
        if c != "RANK_ONLY":
            contrast("future_vs_rank_only", c, "RANK_ONLY")
    contrast("product_ablation", "R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL")
    contrast("product_ablation", "QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL")
    return result


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def preflight():
    protocol, rows, li, q, g, r, match, counts = load_inputs()
    if any(p.exists() for p in OUT.values()):
        raise RuntimeError("Evaluation output collision; refusing overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print("GPU =", torch.cuda.get_device_name(device))
    for c in CONDITIONS:
        models = load_models(c, protocol, device)
        diffs = [validate_scorer(m, device) for m in models]
        print(f"MODEL PASS: {c}, folds=5, rank_dim={models[0].ranker.rank_head.weight.shape[1]}, max_diff={max(diffs):.9g}")
        del models
        torch.cuda.empty_cache()
    print("TEST_ALL =", len(rows), "TEST_LONG20 =", len(li))
    print("TRAINING = NO")
    print("TEST_RANKING_INFERENCE = NO")
    print("S12C_STEP_2F4B6_PREFLIGHT = PASS")


def evaluate():
    protocol, rows, li, q, g, r, match, counts = load_inputs()
    if any(p.exists() for p in OUT.values()):
        raise RuntimeError("Evaluation output collision; refusing overwrite")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    print("GPU =", torch.cuda.get_device_name(device))
    r_gpu = torch.from_numpy(r).to(device)

    ranks = {"DENSE_BGE": rank_dense(q, g, r_gpu, match, device)}
    print("DENSE_BGE COMPLETE")
    maxdiff = {}
    for c in CONDITIONS:
        print("RUNNING", c)
        models = load_models(c, protocol, device)
        ranks[c], maxdiff[c] = rank_learned(q, g, r_gpu, match, models, device)
        print(c, "COMPLETE", "max_fast_diff=", maxdiff[c],
              "rank_min=", int(ranks[c].min()), "rank_max=", int(ranks[c].max()))
        del models
        torch.cuda.empty_cache()

    for name, arr in ranks.items():
        if arr.shape != (len(rows),) or np.any(arr < 1) or np.any(arr > counts):
            raise RuntimeError(f"Invalid rank array for {name}")

    dids = np.asarray([str(x["dialogue_id"]) for x in rows], dtype=str)
    sample_ids = np.asarray([str(x["sample_id"]) for x in rows], dtype=str)
    rng = np.random.default_rng(BOOT_SEED)
    all_idx = np.arange(len(rows), dtype=np.int64)
    result = {
        "artifact": "FSAR S12C External Test Candidate-Set Ranking Evaluation V1",
        "status": "COMPLETED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": EXPECTED[str(P)],
        "evaluation_type": protocol["evaluation_type"],
        "test_all_candidate_count_distribution": {
            str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))
        },
        "gold_matches_train": int((match >= 0).sum()),
        "gold_not_in_train": int((match < 0).sum()),
        "ensemble": {"folds": 5, "aggregation": "arithmetic_mean_of_five_raw_ranking_scores",
                     "retraining": False, "checkpoint_reselection": False},
        "ranking": {"tie_rule": "negative score >= Gold score ranks before Gold",
                    "rank_formula": "1 + count(negative_score >= gold_score)",
                    "metrics": METRICS},
        "statistics": {"method": "paired_dialogue_cluster_bootstrap", "replicates": N_BOOT,
                       "seed": BOOT_SEED, "confidence_level": .95,
                       "resampling_unit": "dialogue",
                       "shared_resampling_indices_across_conditions": True},
        "fast_scorer_max_abs_diff": maxdiff,
        "datasets": {
            "TEST_ALL": evaluate_subset(ranks, dids, all_idx, rng),
            "TEST_LONG20": evaluate_subset(ranks, dids, li, rng),
        },
        "interpretation_limit": "Gold-injected candidate-set ranking generalization, not retrieval coverage over the TRAIN response library alone.",
        "training": False, "test_time_fitting": False, "checkpoint_reselection": False,
    }

    pred_tmp = OUT["predictions"].with_suffix(".npz.tmp")
    with pred_tmp.open("wb") as f:
        np.savez_compressed(f, sample_ids=sample_ids, dialogue_ids=dids,
                            long20_indices=li, gold_match_idx=match,
                            candidate_counts=counts,
                            **{f"rank_{k}": v.astype(np.int32) for k, v in ranks.items()})

    report = [
        "S12C EXTERNAL TEST CANDIDATE-SET RANKING EVALUATION",
        "=" * 76,
        "Gold-injected candidate-set ranking generalization; not TRAIN-library coverage.",
        "Tie rule: negative score >= Gold ranks before Gold.",
        "Ensemble: arithmetic mean of five folds' raw scores.",
    ]
    for ds, d in result["datasets"].items():
        report += ["", "=" * 76, ds, "=" * 76]
        for model, md in d["models"].items():
            fields = []
            for metric, x in md["dialogue_macro"].items():
                fields.append(f"{metric}={x['estimate']:.6f} [{x['ci95_lower']:.6f},{x['ci95_upper']:.6f}]")
            report.append(model + ": " + " | ".join(fields))
        report.append("PAIRED CONTRASTS")
        for cat, contrasts in d["contrasts"].items():
            report.append("[" + cat + "]")
            for cname, vals in contrasts.items():
                fields = [f"{m}={x['delta']:.6f} [{x['ci95_lower']:.6f},{x['ci95_upper']:.6f}]"
                          for m, x in vals.items()]
                report.append(cname + ": " + " | ".join(fields))
    report += ["", "TRAINING = NO", "CHECKPOINT_RESELECTION = NO",
               "TEST_TIME_FITTING = NO", "RANKING_INFERENCE = COMPLETED"]

    run_manifest = {
        "artifact": "FSAR S12C External Test Evaluation Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": EXPECTED[str(P)],
        "evaluator_script_sha256": sha(Path(__file__)),
        "conditions": ["DENSE_BGE"] + CONDITIONS,
        "test_all_queries": len(rows), "test_long20_queries": len(li),
        "training": False, "test_time_fitting": False,
        "checkpoint_reselection": False, "ranking_inference": True,
    }

    # Refuse overwrite and promote all four artifacts only after generation.
    for p in OUT.values():
        if p.exists():
            raise RuntimeError(f"Output appeared during run; refusing overwrite: {p}")
    write_json(OUT["summary"].with_suffix(".json.tmp"), result)
    OUT["report"].with_suffix(".txt.tmp").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT["manifest"].with_suffix(".json.tmp"), run_manifest)
    for key in ("predictions", "summary", "report", "manifest"):
        tmp = OUT[key].with_suffix(OUT[key].suffix + ".tmp")
        os.replace(tmp, OUT[key])

    print("=" * 76)
    print("S12C_STEP_2F4B7_EXTERNAL_TEST_EVALUATION_COMPLETE")
    for k, p in OUT.items():
        print(k.upper(), "=", p)
    print("TRAINING = NO")
    print("CHECKPOINT_RESELECTION = NO")
    print("TEST_TIME_FITTING = NO")
    print("RANKING_INFERENCE = YES")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    preflight() if a.preflight else evaluate()


if __name__ == "__main__":
    main()
