#!/usr/bin/env python3
"""Freeze S7 beta=2 evaluation inputs; does not calculate performance metrics."""
import hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path.cwd().resolve()
PROTO=ROOT/"protocols/fsar_s7_beta2_oof_evaluation_v1.json"
EVAL=ROOT/"src/evaluation/s7_beta2_oof_evaluation.py"
OUT=ROOT/"results/phase_s7/beta_2/evaluation_v1"
EXPECTED_EVAL_SHA="f03c69865880dce8028829e02f5ab6cb12ac0dfa95a023989fbc337f720409e1"
def req(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
paths={
"master":"data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl",
"fold":"data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy",
"future_mask":"data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz",
"reservoir":"cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz",
"s4_oof":"results/phase_s4/s4_rank_only_oof_scores_v1.npz",
"s4_summary":"results/phase_s4/s4_rank_only_training_summary_v1.json",
"s5_r_oof":"results/phase_s5/s5_r_future_oof_scores_v1.npz",
"s5_r_summary":"results/phase_s5/s5_r_future_training_summary_v1.json",
"s5_qr_oof":"results/phase_s5/s5_qr_future_oof_scores_v1.npz",
"s5_qr_summary":"results/phase_s5/s5_qr_future_training_summary_v1.json",
"s7_protocol":"protocols/fsar_s7_future_aux_beta_sensitivity_v1.json",
"s7_training_script":"src/training/s7_train_future_aux_beta_sensitivity.py",
"s7_r_oof":"results/phase_s7/beta_2/s7_beta2_r_future_oof_scores_v1.npz",
"s7_r_summary":"results/phase_s7/beta_2/s7_beta2_r_future_training_summary_v1.json",
"s7_qr_oof":"results/phase_s7/beta_2/s7_beta2_qr_future_oof_scores_v1.npz",
"s7_qr_summary":"results/phase_s7/beta_2/s7_beta2_qr_future_training_summary_v1.json",
"s6a_protocol":"protocols/fsar_s6a_oof_retrieval_reveal_protocol_v1.json",
"s6a_script":"src/retrieval/s6a_oof_retrieval_reveal.py",
"s6b_protocol":"protocols/fsar_s6b_oof_future_prediction_evaluation_v1.json",
"s6b_script":"src/retrieval/s6b_oof_future_prediction_evaluation.py"}
fixed={
"master":"d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",
"fold":"a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",
"future_mask":"84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d",
"reservoir":"e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",
"s4_oof":"fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4",
"s4_summary":"be30581e5804c9b7368ed3f23037f79dc78b53659e3ca9abde63a2e4387de6d0",
"s5_r_oof":"12deca452373075b19b3699d037578c049d74cb7f49bdc88aebb36e42e9d4ecc",
"s5_r_summary":"ea73d0b84fba2d9415aa23c8f98c800e47b41be1c00e430f0806ad93000edf4c",
"s5_qr_oof":"8928ebb960bc6eb821f7419e188b83bdd306be1dfcbf96764b35381f633ad376",
"s5_qr_summary":"95564e659e1f53694e7bfb72d39f6b5575420ef5bced6995bbaf10cc52f9fa0a",
"s7_protocol":"6f3924f74a1b3168ea6b75f7f2383ce41b0b2715c236b66ec53f38341edcb23d",
"s7_training_script":"afbebfa650d5ada14b3e5ad23b7935c9a91c0d7fe561f50e3ede31f02dd9e9a7",
"s6a_protocol":"d0cd870354e2c4285bf9bbc0a5c26fea016853293d9364db935ab5883b552e7c",
"s6b_protocol":"72928cc629b7f32034628f26022af61903c1ee00a7e58e73624a38c7f73d9d58"}
req(ROOT.name=="PsychResponse_FSAR",f"Wrong project root: {ROOT}")
req(not PROTO.exists(),f"Protocol exists; inspect before proceeding: {PROTO}")
req(not OUT.exists(),f"Evaluation output directory exists; inspect before proceeding: {OUT}")
req(EVAL.is_file(),f"Missing evaluator: {EVAL}")
actual_eval=sha(EVAL); req(actual_eval==EXPECTED_EVAL_SHA,f"Evaluator SHA mismatch: {actual_eval}")
inputs={}
for name,rel in paths.items():
    p=ROOT/rel; req(p.is_file(),f"Missing input {name}: {rel}"); got=sha(p)
    if name in fixed: req(got==fixed[name],f"Frozen SHA mismatch {name}: expected={fixed[name]} actual={got}")
    inputs[name]={"path":rel,"sha256":got}; print("FROZEN_SHA_PASS",name,got)
with (ROOT/inputs["master"]["path"]).open(encoding="utf-8") as f: master=[json.loads(x) for x in f]
req(len(master)==62889,f"Unexpected master rows: {len(master)}")
fold=np.load(ROOT/inputs["fold"]["path"],allow_pickle=False); req(fold.shape==(62889,),"Fold shape mismatch")
avail=np.asarray([bool(x["future_value_available"]) for x in master],dtype=bool); req(int(avail.sum())==17458,"Eligible row count mismatch")
dids=np.asarray([int(x["dialogue_id"]) for x in master],dtype=np.int64); req(np.unique(dids[avail]).size==1653,"Eligible dialogue count mismatch")
for key in ["s4_summary","s5_r_summary","s5_qr_summary","s7_r_summary","s7_qr_summary"]:
    s=json.loads((ROOT/inputs[key]["path"]).read_text(encoding="utf-8"))
    req(s.get("status")=="PASS" and s.get("dev_payload_used", False) is False and s.get("test_payload_used", False) is False,f"Summary gate failed: {key}")
    if key in {"s7_r_summary","s7_qr_summary"}:
        req(float(s.get("beta"))==2.0 and s.get("formal_outputs_promoted") is True,f"S7 summary gate failed: {key}")
        req(s.get("retrieval_metrics_computed") is False and s.get("future_metrics_computed") is False,f"S7 metric reveal occurred during training: {key}")
for key in ["s4_oof","s5_r_oof","s5_qr_oof","s7_r_oof","s7_qr_oof"]:
    with np.load(ROOT/inputs[key]["path"],allow_pickle=False) as z:
        req(z["candidate_scores"].shape==(62889,101),f"Candidate score shape mismatch: {key}")
        req(np.array_equal(z["fold_id"],fold),f"Fold alignment mismatch: {key}")
        req(np.all(z["gold_position"]==0),f"Gold position mismatch: {key}")
        if key!="s4_oof": req(z["future_logits"].shape==(62889,4) and np.array_equal(z["future_available"],avail),f"Future logits/alignment mismatch: {key}")
print("OOF_STRUCTURE_GATE = PASS")
protocol={"project":"PsychResponse_FSAR","phase":"S7-BETA2-EVALUATION","artifact":"FSAR S7 beta=2 Unified OOF Evaluation Protocol V1",
"status":"FROZEN_BEFORE_S7_BETA2_METRIC_REVEAL","scope":"TRAIN_INTERNAL_OOF_ONLY","interpretation":"EXPLORATORY_BETA_SENSITIVITY_NO_CONFIRMATORY_GATE",
"evaluation_script":"src/evaluation/s7_beta2_oof_evaluation.py","evaluation_script_sha256":actual_eval,"inputs":inputs,
"retrieval":{"query_rows":62889,"dialogues":1666,"candidate_count":101,"candidate_set":"1_GOLD_PLUS_FROZEN_DENSE_TOP100_NON_GOLD",
"conditions":["DENSE_101","RANK_ONLY","R_FUTURE_BETA1","QR_FUTURE_BETA1","R_FUTURE_BETA2","QR_FUTURE_BETA2"],
"rank_definition":"1 + COUNT(candidate_score > gold_score)","tie_policy":"STRICT_GREATER","primary_metric":"DIALOGUE_MACRO_MRR","secondary_metrics":["R@1","R@5","R@10","R@20"],
"bootstrap_unit":"dialogue_id","bootstrap_paired":True,"bootstrap_replicates":5000,"bootstrap_seed":42,"ci":0.95,"method":"PERCENTILE",
"primary_exploratory_contrast":"S7_BETA2_QR_MINUS_S5_QR","other_contrasts_are_exploratory":True},
"future_state":{"eligible_rows":17458,"eligible_dialogues":1653,"classes":[-2,-1,0,1,2],"thresholds":[-2,-1,0,1],
"classification_metrics":["accuracy","macro_f1","balanced_accuracy","weighted_f1","ordinal_mae","per_class_precision_recall_f1","per_threshold_accuracy_balanced_accuracy_f1_roc_auc_average_precision"],
"probability_metrics":["brier","ordinal_logloss","expected_state_mae","threshold_accuracy"],"bootstrap_unit":"dialogue_id","bootstrap_paired":True,"bootstrap_replicates":5000,"bootstrap_seed":43,"ci":0.95,"method":"PERCENTILE","all_contrasts_exploratory":True},
"outputs":["results/phase_s7/beta_2/evaluation_v1/s7_beta2_oof_evaluation_summary_v1.json","results/phase_s7/beta_2/evaluation_v1/s7_beta2_oof_evaluation_arrays_v1.npz","results/phase_s7/beta_2/evaluation_v1/s7_beta2_oof_evaluation_report_v1.txt"],
"dev_payload_used":False,"test_payload_used":False,"no_posthoc_success_gate":True,"overwrite_policy":"REFUSE_IF_PROTOCOL_OR_OUTPUT_DIRECTORY_EXISTS"}
PROTO.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("PROTOCOL_CREATED =",PROTO); print("PROTOCOL_SHA256 =",sha(PROTO)); print("EVALUATION_SCRIPT_SHA256 =",actual_eval)
print("METRICS_COMPUTED = NO"); print("S7_BETA2_EVALUATION_PROTOCOL_FREEZE_PASS")
