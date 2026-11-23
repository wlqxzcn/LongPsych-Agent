#!/usr/bin/env python3
"""Isolated S7 beta=4 OOF evaluation. Reads frozen inputs; refuses output overwrite."""
import hashlib, json, os, shutil, tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, mean_absolute_error, precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
PROTO = ROOT / "protocols/fsar_s7_beta4_oof_evaluation_v1.json"
OUT = ROOT / "results/phase_s7/beta_4/evaluation_v1"
FILES = ["s7_beta4_oof_evaluation_summary_v1.json", "s7_beta4_oof_evaluation_arrays_v1.npz", "s7_beta4_oof_evaluation_report_v1.txt"]
RC = ["DENSE_101", "RANK_ONLY", "R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"]
FC = ["R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"]
METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]
KS = [1, 5, 10, 20]
CLASSES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
THRESHOLDS = np.array([-2, -1, 0, 1], dtype=np.int64)
NBOOT = 5000

def req(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def npz(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k].copy() for k in z.files}

def sig(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -80, 80)
    return 1.0 / (1.0 + np.exp(-x))

def boot_means(dlg, seed):
    nd, nc, nm = dlg.shape
    rng = np.random.default_rng(seed)
    out = np.empty((NBOOT, nc, nm), dtype=np.float64)
    for b in range(NBOOT): out[b] = dlg[rng.integers(0, nd, size=nd)].mean(axis=0)
    return out

def contrasts(boots, points, definitions, metric_names, directions):
    out = []
    for name, a, b in definitions:
        for m, metric in enumerate(metric_names):
            ds = boots[:, a, m] - boots[:, b, m]
            lo, hi = np.quantile(ds, [0.025, 0.975])
            out.append({"contrast": name, "metric": metric,
                "delta_a_minus_b": float(points[a, m] - points[b, m]),
                "ci95_lower": float(lo), "ci95_upper": float(hi),
                "direction_of_improvement": directions[metric],
                "status": "EXPLORATORY_NO_CONFIRMATORY_GATE"})
    return out

def future_metrics(name, logits, y, avail, folds, dialogues):
    z = logits[avail].astype(np.float64); yy = y[avail]; ff = folds[avail]; dd = dialogues[avail]
    pred = (-2 + (z >= 0).sum(axis=1)).astype(np.int64)
    p, r, f, support = precision_recall_fscore_support(yy, pred, labels=CLASSES, zero_division=0)
    per_class = {str(int(c)): {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])} for i,c in enumerate(CLASSES)}
    per_thr = {}
    targets = (yy[:, None] > THRESHOLDS[None, :]).astype(np.int64)
    for j, t in enumerate(THRESHOLDS):
        tar = targets[:, j]; bp = (z[:, j] >= 0).astype(np.int64)
        auc = float(roc_auc_score(tar, z[:, j])) if np.unique(tar).size == 2 else None
        ap = float(average_precision_score(tar, z[:, j])) if np.unique(tar).size == 2 else None
        per_thr[str(int(t))] = {"positive_support": int(tar.sum()), "negative_support": int(len(tar)-tar.sum()),
            "accuracy": float(accuracy_score(tar,bp)), "balanced_accuracy": float(balanced_accuracy_score(tar,bp)),
            "f1": float(f1_score(tar,bp,zero_division=0)), "roc_auc": auc, "average_precision": ap}
    probs = sig(z); tb = targets.astype(np.float64)
    row = np.column_stack([
        ((probs-tb)**2).mean(axis=1),
        (np.logaddexp(0,z)-tb*z).mean(axis=1),
        np.abs((-2.0+probs.sum(axis=1))-yy),
        ((z>=0)==tb.astype(bool)).mean(axis=1),
    ])
    metrics = {"accuracy": float(accuracy_score(yy,pred)),
        "macro_f1": float(f1_score(yy,pred,labels=CLASSES,average="macro",zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(yy,pred)),
        "weighted_f1": float(f1_score(yy,pred,labels=CLASSES,average="weighted",zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(yy,pred))}
    unique_d, inv = np.unique(dd, return_inverse=True); counts = np.bincount(inv, minlength=len(unique_d))
    dlg = np.empty((len(unique_d), 4), dtype=np.float64)
    for m in range(4): dlg[:,m] = np.bincount(inv, weights=row[:,m], minlength=len(unique_d))/counts
    report = {"n": int(len(yy)), "labeled_dialogues": int(len(unique_d)), "metrics": metrics,
        "per_class": per_class, "confusion_matrix_labels": CLASSES.tolist(),
        "confusion_matrix": confusion_matrix(yy,pred,labels=CLASSES).tolist(), "per_threshold": per_thr,
        "ordinal_logit_monotonicity": {"violating_rows": int(np.any(z[:,:-1] < z[:,1:],axis=1).sum()),
            "eligible_rows": int(len(z)), "violation_rate": float(np.any(z[:,:-1] < z[:,1:],axis=1).mean())}}
    return report, pred, dlg, unique_d, row

def main():
    req(PROTO.is_file(), f"Missing frozen protocol: {PROTO}")
    protocol = json.loads(PROTO.read_text(encoding="utf-8"))
    req(protocol.get("status") == "FROZEN_BEFORE_S7_BETA4_METRIC_REVEAL", "Protocol not frozen")
    req(protocol.get("evaluation_script_sha256") == sha(Path(__file__)), "Evaluation script SHA mismatch")
    req(protocol["retrieval"]["bootstrap_replicates"] == NBOOT and protocol["retrieval"]["bootstrap_seed"] == 42, "Retrieval bootstrap protocol mismatch")
    req(protocol["future_state"]["bootstrap_replicates"] == NBOOT and protocol["future_state"]["bootstrap_seed"] == 43, "Future bootstrap protocol mismatch")
    req(not OUT.exists(), "Formal evaluation output directory collision; inspect before rerun")
    actual = {}
    for name, info in protocol["inputs"].items():
        pth = ROOT / info["path"]; req(pth.is_file(), f"Missing input {name}: {pth}")
        got = sha(pth); req(got == info["sha256"], f"SHA mismatch {name}: {got}")
        actual[name] = got; print("SHA_PASS", name, got)
    print("FROZEN_INPUT_SHA_GATE = PASS")
    master_path = ROOT / protocol["inputs"]["master"]["path"]
    with master_path.open(encoding="utf-8") as f: master = [json.loads(x) for x in f]
    n = len(master); req(n == 62889, f"master rows={n}")
    qids = np.asarray([str(x["query_id"]) for x in master], dtype=str)
    dids = np.asarray([int(x["dialogue_id"]) for x in master], dtype=np.int64)
    gold_rows = np.asarray([int(x["gold_response_row"]) for x in master], dtype=np.int32)
    avail = np.asarray([bool(x["future_value_available"]) for x in master], dtype=bool)
    y = np.asarray([int(x["future_value_label"]) if avail[i] else -2 for i,x in enumerate(master)], dtype=np.int64)
    req(avail.sum() == 17458, "future eligible row count mismatch")
    req(np.unique(dids[avail]).size == 1653, "future eligible dialogue count mismatch")
    fold = np.load(ROOT / protocol["inputs"]["fold"]["path"], allow_pickle=False)
    req(fold.shape == (n,), "fold shape mismatch")
    res = npz(ROOT / protocol["inputs"]["reservoir"]["path"])
    negscore = np.asarray(res["candidate_similarity"], dtype=np.float32)
    negrows = np.asarray(res["candidate_response_row"], dtype=np.int32)
    goldsim = np.asarray(res["gold_similarity"], dtype=np.float32)
    rgold = np.asarray(res["gold_response_row"], dtype=np.int32)
    req(negscore.shape == (n,100) and negrows.shape == (n,100), "reservoir shape mismatch")
    req(np.array_equal(gold_rows,rgold), "gold response row mismatch")
    req(not np.any(negrows == rgold[:,None]), "gold response occurs in negative reservoir")
    dense = np.concatenate([goldsim[:,None],negscore],axis=1)
    loaded = {}
    specs = [("RANK_ONLY","s4_oof","s4_summary",False), ("R_FUTURE_BETA1","s5_r_oof","s5_r_summary",True),
        ("QR_FUTURE_BETA1","s5_qr_oof","s5_qr_summary",True), ("R_FUTURE_BETA2","s7_r_oof","s7_r_summary",True),
        ("QR_FUTURE_BETA2","s7_qr_oof","s7_qr_summary",True),
        ("R_FUTURE_BETA4","s7_beta4_r_oof","s7_beta4_r_summary",True),
        ("QR_FUTURE_BETA4","s7_beta4_qr_oof","s7_beta4_qr_summary",True)]
    for name, ok, sk, needs_future in specs:
        d = npz(ROOT / protocol["inputs"][ok]["path"])
        req(d["candidate_scores"].shape == (n,101), f"{name} score shape mismatch")
        req(np.isfinite(d["candidate_scores"]).all(), f"{name} nonfinite ranking scores")
        req(np.array_equal(d["fold_id"],fold), f"{name} fold mismatch")
        req(np.all(d["gold_position"]==0), f"{name} gold position mismatch")
        if needs_future:
            req(d["future_logits"].shape==(n,4) and np.isfinite(d["future_logits"]).all(), f"{name} future logits invalid")
            req(np.array_equal(d["future_available"],avail), f"{name} availability mismatch")
            req(np.array_equal(d["future_value_label"][avail],y[avail]), f"{name} label mismatch")
        loaded[name] = d
        s = json.loads((ROOT / protocol["inputs"][sk]["path"]).read_text(encoding="utf-8"))
        req(s.get("status")=="PASS" and s.get("dev_payload_used", False) is False and s.get("test_payload_used", False) is False, f"{name} training summary invalid")
        if "BETA2" in name:
            req(float(s.get("beta"))==2.0 and s.get("formal_outputs_promoted") is True, f"{name} beta/outputs invalid")
        if "BETA4" in name:
            req(float(s.get("beta"))==4.0 and s.get("formal_outputs_promoted") is True, f"{name} beta/outputs invalid")
    scores = [dense, loaded["RANK_ONLY"]["candidate_scores"], loaded["R_FUTURE_BETA1"]["candidate_scores"],
        loaded["QR_FUTURE_BETA1"]["candidate_scores"], loaded["R_FUTURE_BETA2"]["candidate_scores"], loaded["QR_FUTURE_BETA2"]["candidate_scores"],
        loaded["R_FUTURE_BETA4"]["candidate_scores"], loaded["QR_FUTURE_BETA4"]["candidate_scores"]]
    ranks = np.empty((8,n),dtype=np.int16); ties=np.empty_like(ranks); qm=np.empty((n,8,5),dtype=np.float64)
    for c,s in enumerate(scores):
        ranks[c]=1+(s[:,1:]>s[:,0:1]).sum(axis=1); ties[c]=(s[:,1:]==s[:,0:1]).sum(axis=1)
        qm[:,c,0]=1.0/ranks[c]
        for j,k in enumerate(KS,1): qm[:,c,j]=ranks[c]<=k
    u_d, inv=np.unique(dids,return_inverse=True); counts=np.bincount(inv,minlength=len(u_d))
    dlg=np.empty((len(u_d),8,5),dtype=np.float64)
    for c in range(8):
        for m in range(5): dlg[:,c,m]=np.bincount(inv,weights=qm[:,c,m],minlength=len(u_d))/counts
    macro=dlg.mean(axis=0); micro=qm.mean(axis=0); rb=boot_means(dlg,42)
    rc=[("S7_BETA4_QR_MINUS_S5_QR",7,3),("S7_BETA4_R_MINUS_S5_R",6,2),("S7_BETA4_QR_MINUS_S7_BETA4_R",7,6),
        ("S7_BETA4_QR_MINUS_S7_BETA2_QR",7,5),("S7_BETA4_R_MINUS_S7_BETA2_R",6,4),
        ("S7_BETA2_QR_MINUS_S5_QR",5,3),("S7_BETA2_R_MINUS_S5_R",4,2),("S5_QR_MINUS_S5_R",3,2),
        ("S7_BETA4_QR_MINUS_RANK_ONLY",7,1),("S7_BETA4_R_MINUS_RANK_ONLY",6,1),("RANK_ONLY_MINUS_DENSE_101",1,0)]
    rdirs={m:"higher_is_better" for m in METRICS}
    rcontr=contrasts(rb,macro,rc,METRICS,rdirs)
    futlog=np.stack([loaded[x]["future_logits"] for x in FC],axis=0)
    freports={}; fpreds=[]; fdmetrics=[]; eligible_idx=np.flatnonzero(avail)
    fmetric_names=["brier","ordinal_logloss","expected_state_mae","threshold_accuracy"]
    fdirs={"brier":"lower_is_better","ordinal_logloss":"lower_is_better","expected_state_mae":"lower_is_better","threshold_accuracy":"higher_is_better"}
    fdlg_ids=None
    for i,name in enumerate(FC):
        rep,pred,dlgmet,ud,rows=future_metrics(name,futlog[i],y,avail,fold,dids)
        freports[name]=rep; fpreds.append(pred); fdmetrics.append(dlgmet); fdlg_ids=ud
    fmacro=np.stack([x.mean(axis=0) for x in fdmetrics],axis=0)
    # Same eligible dialogue universe across all four models; paired bootstrap.
    fb=np.empty((NBOOT,6,4),dtype=np.float64); rng=np.random.default_rng(43); nd=len(fdlg_ids)
    fd_all=np.stack(fdmetrics,axis=1)
    for b in range(NBOOT): fb[b]=fd_all[rng.integers(0,nd,size=nd)].mean(axis=0)
    fc=[("S7_BETA4_R_MINUS_S5_R",4,0),("S7_BETA4_QR_MINUS_S5_QR",5,1),("S7_BETA4_QR_MINUS_S7_BETA4_R",5,4),
        ("S7_BETA4_QR_MINUS_S7_BETA2_QR",5,3),("S7_BETA4_R_MINUS_S7_BETA2_R",4,2),
        ("S7_BETA2_QR_MINUS_S5_QR",3,1),("S7_BETA2_R_MINUS_S5_R",2,0),("S5_QR_MINUS_S5_R",1,0)]
    fcontr=contrasts(fb,fmacro,fc,fmetric_names,fdirs)
    for i,name in enumerate(FC):
        freports[name]["dialogue_macro_probability_metrics"]={m:float(fmacro[i,j]) for j,m in enumerate(fmetric_names)}
    retrieval_summary={}; tie_summary={}
    for i,name in enumerate(RC):
        retrieval_summary[name]={"dialogue_macro":{m:float(macro[i,j]) for j,m in enumerate(METRICS)},"query_micro":{m:float(micro[i,j]) for j,m in enumerate(METRICS)}}
        tie_summary[name]={"queries_with_gold_ties":int((ties[i]>0).sum()),"total_tied_non_gold_candidates":int(ties[i].sum()),"max_tied_candidates_per_query":int(ties[i].max())}
    summary={"artifact":"FSAR S7 beta=4 Unified OOF Evaluation V1","status":"PASS","scope":"TRAIN_INTERNAL_OOF_ONLY",
        "interpretation":"EXPLORATORY_BETA_SENSITIVITY_NO_CONFIRMATORY_GATE","protocol_sha256":sha(PROTO),"evaluation_script_sha256":sha(Path(__file__)),"input_sha256":actual,
        "query_rows":int(n),"dialogues":int(len(u_d)),"candidate_count":101,"future_eligible_rows":int(avail.sum()),"future_eligible_dialogues":int(len(fdlg_ids)),
        "class_counts":{str(int(c)):int((y[avail]==c).sum()) for c in CLASSES},
        "retrieval":{"primary_exploratory_contrast":"S7_BETA4_QR_MINUS_S5_QR","primary_metric":"DIALOGUE_MACRO_MRR","rank_definition":"1 + COUNT(candidate_score > gold_score)","tie_policy":"STRICT_GREATER",
            "condition_metrics":retrieval_summary,"tie_diagnostics":tie_summary,"bootstrap":{"unit":"dialogue_id","paired":True,"replicates":NBOOT,"seed":42,"ci":0.95,"method":"PERCENTILE","contrasts":rcontr}},
        "future_state":{"conditions":FC,"classification_metrics":freports,"probability_metric_names":fmetric_names,
            "probability_metric_dialogue_macro":{name:{m:float(fmacro[i,j]) for j,m in enumerate(fmetric_names)} for i,name in enumerate(FC)},
            "bootstrap":{"unit":"dialogue_id","paired":True,"replicates":NBOOT,"seed":43,"ci":0.95,"method":"PERCENTILE","contrasts":fcontr}},
        "dev_payload_used":False,"test_payload_used":False,"no_posthoc_success_gate":True}
    OUT.mkdir(parents=True,exist_ok=True); stage=Path(tempfile.mkdtemp(prefix=".stage_",dir=OUT))
    try:
        (stage/FILES[0]).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        np.savez_compressed(stage/FILES[1],retrieval_conditions=np.asarray(RC,dtype="U32"),query_ids=qids,dialogue_ids=dids,fold_id=fold,ranks=ranks,gold_tie_counts=ties,
            dialogue_query_counts=counts,dialogue_metrics=dlg,dialogue_macro=macro,query_micro=micro,retrieval_bootstrap_means=rb,
            future_conditions=np.asarray(FC,dtype="U32"),future_eligible_row_indices=eligible_idx,future_available=avail,future_value_label=y,future_logits=futlog,
            future_predictions=np.asarray(fpreds,dtype=np.int64),future_bootstrap_means=fb)
        lines=["FSAR S7 beta=4 Unified OOF Evaluation V1","Status: PASS","Scope: TRAIN_INTERNAL_OOF_ONLY",
            f"Queries={n}; dialogues={len(u_d)}; candidates/query=101",f"Future eligible rows={int(avail.sum())}; dialogues={len(fdlg_ids)}","","RETRIEVAL DIALOGUE-MACRO"]
        for name in RC: lines.append(name+": "+", ".join(f"{k}={v:.8f}" for k,v in retrieval_summary[name]["dialogue_macro"].items()))
        lines += ["","RETRIEVAL CONTRASTS: MRR, paired dialogue bootstrap"]
        for x in rcontr:
            if x["metric"]=="MRR": lines.append(f"{x['contrast']}: delta={x['delta_a_minus_b']:+.8f}, CI95=[{x['ci95_lower']:+.8f},{x['ci95_upper']:+.8f}]")
        lines += ["","FUTURE-STATE CLASSIFICATION"]
        for name in FC: lines.append(name+": "+", ".join(f"{k}={v:.8f}" for k,v in freports[name]["metrics"].items()))
        lines += ["","FUTURE-STATE PROBABILITY METRICS: dialogue macro"]
        for name in FC: lines.append(name+": "+", ".join(f"{k}={v:.8f}" for k,v in freports[name]["dialogue_macro_probability_metrics"].items()))
        lines += ["","FUTURE-STATE CONTRASTS: paired dialogue bootstrap"]
        for x in fcontr: lines.append(f"{x['contrast']} / {x['metric']}: delta={x['delta_a_minus_b']:+.8f}, CI95=[{x['ci95_lower']:+.8f},{x['ci95_upper']:+.8f}] ({x['direction_of_improvement']})")
        lines += ["","Exploratory analysis only; no confirmatory success gate applied.","No dev/test payload used."]
        (stage/FILES[2]).write_text("\n".join(lines)+"\n",encoding="utf-8")
        for name in FILES: os.replace(stage/name,OUT/name)
    finally: shutil.rmtree(stage,ignore_errors=True)
    print("\n===== RETRIEVAL DIALOGUE-MACRO =====")
    for name in RC: print(name,json.dumps(retrieval_summary[name]["dialogue_macro"]))
    print("\n===== KEY MRR CONTRASTS =====")
    for x in rcontr:
        if x["metric"]=="MRR" and x["contrast"] in {"S7_BETA4_QR_MINUS_S5_QR","S7_BETA4_R_MINUS_S5_R","S7_BETA4_QR_MINUS_S7_BETA2_QR"}: print(json.dumps(x))
    print("\n===== FUTURE-STATE METRICS =====")
    for name in FC: print(name,json.dumps(freports[name]["metrics"]),json.dumps(freports[name]["dialogue_macro_probability_metrics"]))
    print("EVALUATION_OUTPUTS_PASS")
    for name in FILES: print(OUT/name)

if __name__ == "__main__": main()
