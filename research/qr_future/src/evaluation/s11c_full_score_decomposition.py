#!/usr/bin/env python3
"""S11-C: Dev-only full-pool score decomposition. No training/checkpoint changes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.evaluation import s8_dev_retrieval_evaluation as s8

CONDITIONS = ["RANK_ONLY", "R_FUTURE_BETA8", "QR_FUTURE_BETA8"]
D = int(s8.DIM)
PROTOCOL = ROOT / "protocols/fsar_s11c_full_score_decomposition_v1.json"
OUT = ROOT / "results/phase_s11"
S8_PRED = OUT.parent / "phase_s8/s8_dev_retrieval_predictions_v1.npz"
S11B_PRED = OUT / "s11_response_only_component_predictions_v1.npz"
SUMMARY = OUT / "s11c_full_score_decomposition_summary_v1.json"
PRED = OUT / "s11c_full_score_decomposition_predictions_v1.npz"
REPORT = OUT / "s11c_full_score_decomposition_report_v1.txt"
QB, CB = 32, 2048
TERMS = ["RESPONSE_ONLY", "ABS_DIFF_ONLY", "PRODUCT_ONLY", "INTERACTIONS_ONLY",
         "RESPONSE_PLUS_ABS_DIFF", "RESPONSE_PLUS_PRODUCT", "FULL_RECONSTRUCTED"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"Temp output exists; refusing overwrite: {tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_data():
    result = s8.load_and_validate_text_data()
    if len(result) != 8:
        raise RuntimeError(f"Unexpected S8 loader signature: {len(result)}")
    responses, _, dev, long20, _, _, match_idx, _ = result
    q = np.load(s8.QUERY_EMB_OUT, mmap_mode="r", allow_pickle=False)
    g = np.load(s8.GOLD_EMB_OUT, mmap_mode="r", allow_pickle=False)
    r = np.load(s8.TRAIN_RESPONSE_EMB, mmap_mode="r", allow_pickle=False)
    if q.shape != (len(dev), D) or g.shape != (len(dev), D) or r.shape != (len(responses), D):
        raise RuntimeError(f"Embedding shape mismatch: q={q.shape}, gold={g.shape}, r={r.shape}")
    return responses, dev, long20, np.asarray(match_idx, dtype=np.int64), q, g, r


def paths_for_hash():
    p = {
        "dev_all": s8.DEV_ALL, "dev_long20": s8.DEV_LONG20,
        "train_responses": s8.TRAIN_RESPONSES,
        "train_response_index": s8.TRAIN_RESPONSE_INDEX,
        "query_embeddings": s8.QUERY_EMB_OUT, "gold_embeddings": s8.GOLD_EMB_OUT,
        "response_embeddings": s8.TRAIN_RESPONSE_EMB,
        "s8_predictions": S8_PRED, "s11b_predictions": S11B_PRED,
        "s8_script": ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py",
        "model_code": ROOT / "src/models/s3c1_fsar_models.py",
        "s11c_script": Path(__file__).resolve(),
    }
    reg = s8.all_checkpoint_paths()
    for c in CONDITIONS:
        if c not in reg:
            raise RuntimeError(f"Missing checkpoint condition {c}")
        for i, f in enumerate(reg[c]):
            p[f"checkpoint_{c}_fold{i}"] = f
    return p


def input_hashes():
    p = paths_for_hash()
    missing = [f"{k}: {v}" for k, v in p.items() if not Path(v).is_file()]
    if missing:
        raise FileNotFoundError("Missing S11-C inputs:\n" + "\n".join(missing))
    return {k: sha(v) for k, v in p.items()}


def prepare():
    if PROTOCOL.exists():
        raise RuntimeError(f"Protocol already exists; refusing overwrite: {PROTOCOL}")
    for p in (SUMMARY, PRED, REPORT):
        if p.exists() or p.with_suffix(p.suffix + ".tmp").exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {p}")
    responses, dev, long20, _, q, g, r = load_data()
    protocol = {
        "project": "PsychResponse_FSAR", "phase": "S11-C",
        "artifact": "Full Score Component Decomposition V1",
        "status": "FROZEN_BEFORE_EVALUATION",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "conditions": CONDITIONS,
        "formula": "score(q,r)=wq·zq + wr·zr + wd·abs(zq-zr) + wp·(zq*zr)",
        "rank_relevant_formula": "wr·zr + wd·abs(zq-zr) + wp·(zq*zr); wq·zq is query-constant",
        "feature_order": ["zq", "zr", "abs(zq-zr)", "zq*zr"],
        "fold_aggregation": "Average each score component over the same five folds before ranking",
        "candidate_universe": "All unique normalized TRAIN response identities",
        "gold_rule": "S8 normalized Gold-to-TRAIN mapping; exclude exact mapped Gold response from its own rank count",
        "tie_rule": "candidate score >= Gold score counts ahead of Gold",
        "terms": TERMS, "dev_rows": len(dev), "dev_long20_rows": len(long20),
        "train_response_rows": len(responses), "embedding_dimension": D,
        "query_batch_size": QB, "candidate_chunk_size": CB,
        "chunking_is_computational_only": True,
        "test_payload_used": False, "training_performed": False,
        "training_artifacts_modified": False, "checkpoints_modified": False,
        "input_sha256": input_hashes(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(PROTOCOL, protocol)
    print("S11C_PREPARE_PASS")
    print("dev_rows =", len(dev), "dev_long20_rows =", len(long20), "train_response_rows =", len(responses))
    print("embedding_shapes =", q.shape, g.shape, r.shape)
    print("PROTOCOL =", PROTOCOL)
    print("PROTOCOL_SHA256 =", sha(PROTOCOL))
    print("PROTOCOL_STATUS = FROZEN_BEFORE_EVALUATION")
    print("TEST_PAYLOAD_USED = NO")


def verify():
    if not PROTOCOL.is_file():
        raise FileNotFoundError("Run --prepare first; frozen protocol missing")
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN_BEFORE_EVALUATION" or p.get("scope") != "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY":
        raise RuntimeError("Unexpected S11-C protocol state/scope")
    now = input_hashes()
    old = p["input_sha256"]
    changed = [k for k in sorted(set(now) | set(old)) if now.get(k) != old.get(k)]
    if changed:
        raise RuntimeError("Frozen input hash mismatch: " + ", ".join(changed))
    return p


def metrics(rank):
    rank = np.asarray(rank, dtype=np.int64)
    return {"n": int(len(rank)), "median_rank": float(np.median(rank)),
            "mean_rank": float(np.mean(rank)), "MRR": float(np.mean(1.0 / rank)),
            "R@1": float(np.mean(rank <= 1)), "R@5": float(np.mean(rank <= 5)),
            "R@10": float(np.mean(rank <= 10)), "R@20": float(np.mean(rank <= 20)),
            "R@100": float(np.mean(rank <= 100))}


def run_condition(condition, q_np, g_np, r_np, match, dev, long_mask, s8pred, s11bpred, device):
    print(f"\n===== {condition} =====", flush=True)
    models = s8.load_condition_models(condition, device)
    if len(models) != 5:
        raise RuntimeError(f"Expected five folds, got {len(models)}")
    r_gpu = torch.from_numpy(np.asarray(r_np, dtype=np.float32)).to(device)
    zr_list, wr_list, wd_list, wp_list = [], [], [], []
    with torch.inference_mode():
        for fi, m in enumerate(models):
            head = m.ranker.rank_head
            if tuple(head.weight.shape) != (1, 4 * D) or head.bias is not None:
                raise RuntimeError(f"Unexpected rank head structure: {condition} fold {fi}")
            w = head.weight[0]
            wr_list.append(w[D:2*D].detach())
            wd_list.append(w[2*D:3*D].detach())
            wp_list.append(w[3*D:4*D].detach())
            zr_list.append(m.ranker.adapt_response(r_gpu).detach())
            print(f"RESPONSE_ADAPTER_READY fold={fi}", flush=True)

        # Candidate-only term, averaged across folds.
        r_lib = torch.zeros(len(r_np), dtype=torch.float32, device=device)
        for zr, wr in zip(zr_list, wr_list):
            r_lib += (zr * wr).sum(-1) / len(models)

    n = len(dev)
    ranks = {t: np.empty(n, dtype=np.int32) for t in TERMS}
    gold_parts = {t: np.empty(n, dtype=np.float32) for t in ["response", "abs_diff", "product"]}

    for qs in range(0, n, QB):
        qe = min(qs + QB, n)
        qb = torch.from_numpy(np.asarray(q_np[qs:qe], dtype=np.float32)).to(device)
        gb = torch.from_numpy(np.asarray(g_np[qs:qe], dtype=np.float32)).to(device)
        mb = torch.from_numpy(match[qs:qe].astype(np.int64, copy=False)).to(device)
        b = qe - qs
        gr = torch.zeros(b, device=device)
        gd = torch.zeros(b, device=device)
        gp = torch.zeros(b, device=device)
        zq_list = []
        with torch.inference_mode():
            for m, wr, wd, wp in zip(models, wr_list, wd_list, wp_list):
                zq = m.ranker.adapt_query(qb)
                zg = m.ranker.adapt_response(gb)
                zq_list.append(zq.detach())
                gr += (zg * wr).sum(-1) / len(models)
                gd += (torch.abs(zq - zg) * wd).sum(-1) / len(models)
                gp += (zq * zg * wp).sum(-1) / len(models)

            counts = {t: torch.zeros(b, dtype=torch.int64, device=device) for t in TERMS}
            matched_scores = {t: torch.full((b,), float("nan"), device=device) for t in TERMS}
            for cs in range(0, len(r_np), CB):
                ce = min(cs + CB, len(r_np))
                diff = torch.zeros((b, ce-cs), device=device)
                prod = torch.zeros_like(diff)
                for zq, zr, wd, wp in zip(zq_list, zr_list, wd_list, wp_list):
                    part = zr[cs:ce]
                    diff += (torch.abs(zq[:, None, :] - part[None, :, :]) * wd).sum(-1) / len(models)
                    prod += torch.matmul(zq * wp, part.T) / len(models)
                rscore = r_lib[cs:ce].unsqueeze(0).expand(b, -1)
                score = {
                    "RESPONSE_ONLY": rscore,
                    "ABS_DIFF_ONLY": diff,
                    "PRODUCT_ONLY": prod,
                    "INTERACTIONS_ONLY": diff + prod,
                    "RESPONSE_PLUS_ABS_DIFF": rscore + diff,
                    "RESPONSE_PLUS_PRODUCT": rscore + prod,
                    "FULL_RECONSTRUCTED": rscore + diff + prod,
                }
                goldscore = {
                    "RESPONSE_ONLY": gr, "ABS_DIFF_ONLY": gd, "PRODUCT_ONLY": gp,
                    "INTERACTIONS_ONLY": gd + gp, "RESPONSE_PLUS_ABS_DIFF": gr + gd,
                    "RESPONSE_PLUS_PRODUCT": gr + gp, "FULL_RECONSTRUCTED": gr + gd + gp,
                }
                for t in TERMS:
                    counts[t] += (score[t] >= goldscore[t][:, None]).sum(1)
                in_chunk = (mb >= cs) & (mb < ce)
                rows = torch.nonzero(in_chunk, as_tuple=False).flatten()
                if rows.numel():
                    cols = mb[rows] - cs
                    for t in TERMS:
                        matched_scores[t][rows] = score[t][rows, cols]
                if cs == 0 or ce == len(r_np):
                    print(f"PROGRESS q={qe}/{n} candidates={ce}/{len(r_np)}", flush=True)

            valid = mb >= 0
            for t in TERMS:
                subtract = valid & (matched_scores[t] >= goldscore[t])
                ranks[t][qs:qe] = (counts[t] + 1 - subtract.long()).cpu().numpy().astype(np.int32)
            gold_parts["response"][qs:qe] = gr.cpu().numpy()
            gold_parts["abs_diff"][qs:qe] = gd.cpu().numpy()
            gold_parts["product"][qs:qe] = gp.cpu().numpy()
        print(f"QUERY_BATCH_COMPLETE {qe}/{n}", flush=True)

    full_ref = np.asarray(s8pred[f"ranks__{condition}"], dtype=np.int32)
    s11b_ref = np.asarray(s11bpred[f"response_only_ranks__{condition}"], dtype=np.int32)
    recon_bad = ranks["FULL_RECONSTRUCTED"] != full_ref
    resp_bad = ranks["RESPONSE_ONLY"] != s11b_ref
    result = {
        "full_score_reconstruction": {
            "exact_match_rate": float(np.mean(~recon_bad)), "mismatch_count": int(recon_bad.sum()),
            "max_abs_rank_difference": int(np.max(np.abs(ranks["FULL_RECONSTRUCTED"].astype(np.int64)-full_ref.astype(np.int64)))),
            "status": "PASS" if not recon_bad.any() else "CHECK_FLOAT_TIE_SENSITIVITY"},
        "s11b_response_only_crosscheck": {
            "exact_match_rate": float(np.mean(~resp_bad)), "mismatch_count": int(resp_bad.sum()),
            "status": "PASS" if not resp_bad.any() else "CHECK_IMPLEMENTATION"},
        "terms": {}}
    arrays = {}
    for t in TERMS:
        arrays[f"ranks__{condition}__{t}"] = ranks[t]
        info = {}
        for sub, mask in [("dev_all", np.ones(n, dtype=bool)), ("dev_long20", long_mask)]:
            m = metrics(ranks[t][mask])
            m["log_rank_pearson_r_with_full"] = float(np.corrcoef(
                np.log1p(ranks[t][mask].astype(np.float64)),
                np.log1p(full_ref[mask].astype(np.float64)))[0, 1])
            info[sub] = m
        result["terms"][t] = info
    for k, v in gold_parts.items():
        arrays[f"gold_component__{condition}__{k}"] = v
    arrays[f"ranks__{condition}__S8_FULL_REFERENCE"] = full_ref
    arrays[f"ranks__{condition}__S11B_RESPONSE_REFERENCE"] = s11b_ref
    del models, zr_list, r_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, arrays


def evaluate():
    protocol = verify()
    for p in (SUMMARY, PRED, REPORT):
        if p.exists() or p.with_suffix(p.suffix + ".tmp").exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {p}")
    responses, dev, long20, match, q, g, r = load_data()
    s8pred = np.load(S8_PRED, allow_pickle=False)
    s11bpred = np.load(S11B_PRED, allow_pickle=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE =", device, flush=True)
    if device.type == "cuda":
        print("GPU =", torch.cuda.get_device_name(0), flush=True)
    sample_ids = np.asarray([str(x["sample_id"]) for x in dev], dtype=str)
    dialogue_ids = np.asarray([str(x["dialogue_id"]) for x in dev], dtype=str)
    long_ids = {str(x["sample_id"]) for x in long20}
    long_mask = np.asarray([x in long_ids for x in sample_ids], dtype=bool)
    summary = {"project": "PsychResponse_FSAR", "phase": "S11-C",
               "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
               "protocol_sha256": sha(PROTOCOL), "conditions": {},
               "test_payload_used": False, "training_performed": False,
               "training_artifacts_modified": False, "checkpoints_modified": False}
    out = {"sample_ids": sample_ids, "dialogue_ids": dialogue_ids,
           "dev_long20_mask": long_mask.astype(np.uint8), "match_idx": match.astype(np.int32)}
    report = [f"protocol_sha256 = {summary['protocol_sha256']}",
              "scope = EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY", "test_payload_used = NO", ""]
    for c in CONDITIONS:
        res, arr = run_condition(c, q, g, r, match, dev, long_mask, s8pred, s11bpred, device)
        summary["conditions"][c] = res
        out.update(arr)
        report.append(f"========== {c} ==========")
        report.append("FULL_RECONSTRUCTION " + json.dumps(res["full_score_reconstruction"]))
        report.append("S11B_CROSSCHECK " + json.dumps(res["s11b_response_only_crosscheck"]))
        for t in TERMS:
            report.append(f"\n[{t}]")
            report.append("dev_all " + json.dumps(res["terms"][t]["dev_all"]))
            report.append("dev_long20 " + json.dumps(res["terms"][t]["dev_long20"]))
            print(c, t, "dev_all", json.dumps(res["terms"][t]["dev_all"]), flush=True)
            print(c, t, "dev_long20", json.dumps(res["terms"][t]["dev_long20"]), flush=True)
        report.append("")
        print("FULL_RECONSTRUCTION", c, json.dumps(res["full_score_reconstruction"]), flush=True)
        print("S11B_CROSSCHECK", c, json.dumps(res["s11b_response_only_crosscheck"]), flush=True)
    summary["score_reconstruction_gate"] = (
        "PASS" if all(v["full_score_reconstruction"]["status"] == "PASS" for v in summary["conditions"].values())
        else "CHECK_FLOAT_TIE_SENSITIVITY")
    summary["s11b_crosscheck_gate"] = (
        "PASS" if all(v["s11b_response_only_crosscheck"]["status"] == "PASS" for v in summary["conditions"].values())
        else "CHECK_IMPLEMENTATION")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(SUMMARY, summary)
    np.savez_compressed(PRED, **out)
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("S11C_EVALUATION = PASS")
    print("SCORE_RECONSTRUCTION_GATE =", summary["score_reconstruction_gate"])
    print("S11B_CROSSCHECK_GATE =", summary["s11b_crosscheck_gate"])
    print("SUMMARY =", SUMMARY)
    print("PREDICTIONS =", PRED)
    print("REPORT =", REPORT)
    print("PROTOCOL_SHA256 =", summary["protocol_sha256"])
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING_ARTIFACTS_MODIFIED = NO")
    print("CHECKPOINTS_MODIFIED = NO")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--evaluate", action="store_true")
    a = p.parse_args()
    prepare() if a.prepare else evaluate()
