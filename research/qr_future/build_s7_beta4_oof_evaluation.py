#!/usr/bin/env python3
"""Create a frozen, isolated S7 beta=4 OOF evaluator/protocol. Does not evaluate."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np

ROOT = Path.cwd()
OLD_SCRIPT = ROOT / "src/evaluation/s7_beta2_oof_evaluation.py"
OLD_PROTO = ROOT / "protocols/fsar_s7_beta2_oof_evaluation_v1.json"
NEW_SCRIPT = ROOT / "src/evaluation/s7_beta4_oof_evaluation.py"
NEW_PROTO = ROOT / "protocols/fsar_s7_beta4_oof_evaluation_v1.json"
OUT = ROOT / "results/phase_s7/beta_4/evaluation_v1"
EXPECTED_OLD_SCRIPT_SHA = "f03c69865880dce8028829e02f5ab6cb12ac0dfa95a023989fbc337f720409e1"
EXPECTED_OLD_PROTO_SHA = "61f135228851d9964599ae4b0aaa4b466e886a7d88fa634d0831477a78a87f83"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def replace_once(source, old, new, label, expected=1):
    count = source.count(old)
    require(count == expected, f"{label}: expected {expected} exact match(es), got {count}. No files written.")
    return source.replace(old, new)


# ---------- Immutable-source gates ----------
require(OLD_SCRIPT.is_file(), f"Missing original script: {OLD_SCRIPT}")
require(OLD_PROTO.is_file(), f"Missing original protocol: {OLD_PROTO}")
require(sha(OLD_SCRIPT) == EXPECTED_OLD_SCRIPT_SHA, "Original beta=2 evaluator SHA changed; stop.")
require(sha(OLD_PROTO) == EXPECTED_OLD_PROTO_SHA, "Original beta=2 protocol SHA changed; stop.")
require(not NEW_SCRIPT.exists(), f"Refusing to overwrite: {NEW_SCRIPT}")
require(not NEW_PROTO.exists(), f"Refusing to overwrite: {NEW_PROTO}")
require(not OUT.exists(), f"Refusing to overwrite existing beta=4 evaluation directory: {OUT}")

old_protocol = json.loads(OLD_PROTO.read_text(encoding="utf-8"))
require(old_protocol.get("status") == "FROZEN_BEFORE_S7_BETA2_METRIC_REVEAL", "Unexpected beta=2 protocol status")
require(old_protocol.get("evaluation_script_sha256") == EXPECTED_OLD_SCRIPT_SHA, "Original protocol/script SHA linkage mismatch")

# Verify every pre-existing dependency against the frozen beta=2 protocol before reusing it.
for name, info in old_protocol["inputs"].items():
    p = ROOT / info["path"]
    require(p.is_file(), f"Missing frozen input {name}: {p}")
    got = sha(p)
    require(got == info["sha256"], f"Frozen input SHA mismatch {name}: {got}")
print(f"ORIGINAL_BETA2_SCRIPT_SHA_PASS {EXPECTED_OLD_SCRIPT_SHA}")
print(f"ORIGINAL_BETA2_PROTOCOL_SHA_PASS {EXPECTED_OLD_PROTO_SHA}")
print(f"FROZEN_EXISTING_INPUTS_PASS count={len(old_protocol['inputs'])}")

# ---------- Generate beta=4 evaluator from the exact frozen beta=2 implementation ----------
s = OLD_SCRIPT.read_text(encoding="utf-8")
s = replace_once(s,
    '"""Isolated S7 beta=2 OOF evaluation. Reads frozen inputs; refuses output overwrite."""',
    '"""Isolated S7 beta=4 OOF evaluation. Reads frozen inputs; refuses output overwrite."""', 'docstring')
s = replace_once(s, 'PROTO = ROOT / "protocols/fsar_s7_beta2_oof_evaluation_v1.json"',
    'PROTO = ROOT / "protocols/fsar_s7_beta4_oof_evaluation_v1.json"', 'protocol path')
s = replace_once(s, 'OUT = ROOT / "results/phase_s7/beta_2/evaluation_v1"',
    'OUT = ROOT / "results/phase_s7/beta_4/evaluation_v1"', 'output directory')
s = replace_once(s,
    'FILES = ["s7_beta2_oof_evaluation_summary_v1.json", "s7_beta2_oof_evaluation_arrays_v1.npz", "s7_beta2_oof_evaluation_report_v1.txt"]',
    'FILES = ["s7_beta4_oof_evaluation_summary_v1.json", "s7_beta4_oof_evaluation_arrays_v1.npz", "s7_beta4_oof_evaluation_report_v1.txt"]', 'output filenames')
s = replace_once(s,
    'RC = ["DENSE_101", "RANK_ONLY", "R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2"]',
    'RC = ["DENSE_101", "RANK_ONLY", "R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"]', 'retrieval conditions')
s = replace_once(s,
    'FC = ["R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2"]',
    'FC = ["R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2", "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"]', 'future-state conditions')
s = replace_once(s,
    'protocol.get("status") == "FROZEN_BEFORE_S7_BETA2_METRIC_REVEAL"',
    'protocol.get("status") == "FROZEN_BEFORE_S7_BETA4_METRIC_REVEAL"', 'protocol status gate')
s = replace_once(s,
    'req(not any((OUT / n).exists() for n in FILES), "Formal evaluation output collision; inspect before rerun")',
    'req(not OUT.exists(), "Formal evaluation output directory collision; inspect before rerun")', 'output overwrite guard')
s = replace_once(s,
'''    specs = [("RANK_ONLY","s4_oof","s4_summary",False), ("R_FUTURE_BETA1","s5_r_oof","s5_r_summary",True),
        ("QR_FUTURE_BETA1","s5_qr_oof","s5_qr_summary",True), ("R_FUTURE_BETA2","s7_r_oof","s7_r_summary",True),
        ("QR_FUTURE_BETA2","s7_qr_oof","s7_qr_summary",True)]''',
'''    specs = [("RANK_ONLY","s4_oof","s4_summary",False), ("R_FUTURE_BETA1","s5_r_oof","s5_r_summary",True),
        ("QR_FUTURE_BETA1","s5_qr_oof","s5_qr_summary",True), ("R_FUTURE_BETA2","s7_r_oof","s7_r_summary",True),
        ("QR_FUTURE_BETA2","s7_qr_oof","s7_qr_summary",True),
        ("R_FUTURE_BETA4","s7_beta4_r_oof","s7_beta4_r_summary",True),
        ("QR_FUTURE_BETA4","s7_beta4_qr_oof","s7_beta4_qr_summary",True)]''', 'input condition specs')
s = replace_once(s,
'''        if "BETA2" in name:
            req(float(s.get("beta"))==2.0 and s.get("formal_outputs_promoted") is True, f"{name} beta/outputs invalid")''',
'''        if "BETA2" in name:
            req(float(s.get("beta"))==2.0 and s.get("formal_outputs_promoted") is True, f"{name} beta/outputs invalid")
        if "BETA4" in name:
            req(float(s.get("beta"))==4.0 and s.get("formal_outputs_promoted") is True, f"{name} beta/outputs invalid")''', 'beta validation')
s = replace_once(s,
'''    scores = [dense, loaded["RANK_ONLY"]["candidate_scores"], loaded["R_FUTURE_BETA1"]["candidate_scores"],
        loaded["QR_FUTURE_BETA1"]["candidate_scores"], loaded["R_FUTURE_BETA2"]["candidate_scores"], loaded["QR_FUTURE_BETA2"]["candidate_scores"]]
    ranks = np.empty((6,n),dtype=np.int16); ties=np.empty_like(ranks); qm=np.empty((n,6,5),dtype=np.float64)''',
'''    scores = [dense, loaded["RANK_ONLY"]["candidate_scores"], loaded["R_FUTURE_BETA1"]["candidate_scores"],
        loaded["QR_FUTURE_BETA1"]["candidate_scores"], loaded["R_FUTURE_BETA2"]["candidate_scores"], loaded["QR_FUTURE_BETA2"]["candidate_scores"],
        loaded["R_FUTURE_BETA4"]["candidate_scores"], loaded["QR_FUTURE_BETA4"]["candidate_scores"]]
    ranks = np.empty((8,n),dtype=np.int16); ties=np.empty_like(ranks); qm=np.empty((n,8,5),dtype=np.float64)''', 'retrieval arrays')
s = replace_once(s,
'''    dlg=np.empty((len(u_d),6,5),dtype=np.float64)
    for c in range(6):''',
'''    dlg=np.empty((len(u_d),8,5),dtype=np.float64)
    for c in range(8):''', 'dialogue metric dimensions')
s = replace_once(s,
'''    rc=[("S7_BETA2_QR_MINUS_S5_QR",5,3),("S7_BETA2_R_MINUS_S5_R",4,2),("S7_BETA2_QR_MINUS_S7_BETA2_R",5,4),
        ("S5_QR_MINUS_S5_R",3,2),("S7_BETA2_QR_MINUS_RANK_ONLY",5,1),("S7_BETA2_R_MINUS_RANK_ONLY",4,1),("RANK_ONLY_MINUS_DENSE_101",1,0)]''',
'''    rc=[("S7_BETA4_QR_MINUS_S5_QR",7,3),("S7_BETA4_R_MINUS_S5_R",6,2),("S7_BETA4_QR_MINUS_S7_BETA4_R",7,6),
        ("S7_BETA4_QR_MINUS_S7_BETA2_QR",7,5),("S7_BETA4_R_MINUS_S7_BETA2_R",6,4),
        ("S7_BETA2_QR_MINUS_S5_QR",5,3),("S7_BETA2_R_MINUS_S5_R",4,2),("S5_QR_MINUS_S5_R",3,2),
        ("S7_BETA4_QR_MINUS_RANK_ONLY",7,1),("S7_BETA4_R_MINUS_RANK_ONLY",6,1),("RANK_ONLY_MINUS_DENSE_101",1,0)]''', 'retrieval contrasts')
s = replace_once(s, 'fb=np.empty((NBOOT,4,4),dtype=np.float64)',
    'fb=np.empty((NBOOT,6,4),dtype=np.float64)', 'future bootstrap dimensions')
s = replace_once(s,
'''    fc=[("S7_BETA2_R_MINUS_S5_R",2,0),("S7_BETA2_QR_MINUS_S5_QR",3,1),("S7_BETA2_QR_MINUS_S7_BETA2_R",3,2),("S5_QR_MINUS_S5_R",1,0)]''',
'''    fc=[("S7_BETA4_R_MINUS_S5_R",4,0),("S7_BETA4_QR_MINUS_S5_QR",5,1),("S7_BETA4_QR_MINUS_S7_BETA4_R",5,4),
        ("S7_BETA4_QR_MINUS_S7_BETA2_QR",5,3),("S7_BETA4_R_MINUS_S7_BETA2_R",4,2),
        ("S7_BETA2_QR_MINUS_S5_QR",3,1),("S7_BETA2_R_MINUS_S5_R",2,0),("S5_QR_MINUS_S5_R",1,0)]''', 'future-state contrasts')
s = replace_once(s, '"FSAR S7 beta=2 Unified OOF Evaluation V1"',
    '"FSAR S7 beta=4 Unified OOF Evaluation V1"', 'artifact label', expected=2)
s = replace_once(s,
    '"retrieval":{"primary_exploratory_contrast":"S7_BETA2_QR_MINUS_S5_QR"',
    '"retrieval":{"primary_exploratory_contrast":"S7_BETA4_QR_MINUS_S5_QR"', 'primary contrast')
s = replace_once(s,
'''if x["metric"]=="MRR" and x["contrast"] in {"S7_BETA2_QR_MINUS_S5_QR","S7_BETA2_R_MINUS_S5_R"}: print(json.dumps(x))''',
'''if x["metric"]=="MRR" and x["contrast"] in {"S7_BETA4_QR_MINUS_S5_QR","S7_BETA4_R_MINUS_S5_R","S7_BETA4_QR_MINUS_S7_BETA2_QR"}: print(json.dumps(x))''', 'printed primary contrasts')

compile(s, str(NEW_SCRIPT), "exec")
require('results/phase_s7/beta_2/evaluation_v1' not in s, "Beta=2 output path leaked into new evaluator")
require('s7_beta2_oof_evaluation' not in s, "Beta=2 output filename leaked into new evaluator")
require('np.empty((8,n)' in s and 'np.empty((NBOOT,6,4)' in s, "Expanded dimensions missing")
print(f"BETA4_EVALUATOR_STATIC_CHECK_PASS bytes={len(s.encode())}")

# ---------- Validate beta=4 training artifacts before freezing their hashes ----------
beta4_items = [
    ("s7_beta4_r_oof", "results/phase_s7/beta_4/s7_beta4_r_future_oof_scores_v1.npz", "s7_beta4_r_summary", "results/phase_s7/beta_4/s7_beta4_r_future_training_summary_v1.json", "R_FUTURE"),
    ("s7_beta4_qr_oof", "results/phase_s7/beta_4/s7_beta4_qr_future_oof_scores_v1.npz", "s7_beta4_qr_summary", "results/phase_s7/beta_4/s7_beta4_qr_future_training_summary_v1.json", "QR_FUTURE"),
]
new_input_records = {}
for oof_key, oof_rel, summary_key, summary_rel, condition in beta4_items:
    oof_path = ROOT / oof_rel
    summary_path = ROOT / summary_rel
    require(oof_path.is_file(), f"Missing beta=4 OOF file: {oof_path}")
    require(summary_path.is_file(), f"Missing beta=4 summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(summary.get("status") == "PASS", f"{summary_key} status is not PASS")
    require(summary.get("condition") == condition, f"{summary_key} condition mismatch")
    require(float(summary.get("beta", -1)) == 4.0, f"{summary_key} beta is not 4")
    require(summary.get("formal_outputs_promoted") is True, f"{summary_key} outputs not promoted")
    require(summary.get("retrieval_metrics_computed") is False, f"{summary_key} already has retrieval metrics")
    require(summary.get("future_metrics_computed") is False, f"{summary_key} already has future metrics")
    require(summary.get("dev_payload_used") is False and summary.get("test_payload_used") is False,
            f"{summary_key} used dev/test payload")
    with np.load(oof_path, allow_pickle=False) as z:
        require(z["candidate_scores"].shape == (62889, 101), f"{oof_key} candidate_scores shape mismatch")
        require(z["future_logits"].shape == (62889, 4), f"{oof_key} future_logits shape mismatch")
        require(z["fold_id"].shape == (62889,), f"{oof_key} fold_id shape mismatch")
        require(np.isfinite(z["candidate_scores"]).all(), f"{oof_key} ranking scores non-finite")
        require(np.isfinite(z["future_logits"]).all(), f"{oof_key} future logits non-finite")
    new_input_records[oof_key] = {"path": oof_rel, "sha256": sha(oof_path)}
    new_input_records[summary_key] = {"path": summary_rel, "sha256": sha(summary_path)}
    print(f"BETA4_ARTIFACT_PASS {condition} oof_sha={new_input_records[oof_key]['sha256']} summary_sha={new_input_records[summary_key]['sha256']}")

# ---------- Freeze a distinct beta=4 protocol before evaluation ----------
protocol = json.loads(json.dumps(old_protocol))
protocol["phase"] = "S7_BETA4_OOF_EVALUATION"
protocol["artifact"] = "FSAR S7 beta=4 Unified OOF Evaluation V1"
protocol["status"] = "FROZEN_BEFORE_S7_BETA4_METRIC_REVEAL"
protocol["scope"] = "TRAIN_INTERNAL_OOF_ONLY"
protocol["interpretation"] = "EXPLORATORY_BETA_SENSITIVITY_NO_CONFIRMATORY_GATE"
protocol["evaluation_script"] = "src/evaluation/s7_beta4_oof_evaluation.py"
protocol["evaluation_script_sha256"] = hashlib.sha256(s.encode("utf-8")).hexdigest()
protocol["inputs"] = dict(old_protocol["inputs"])
protocol["inputs"].update(new_input_records)
protocol["retrieval"]["conditions"] = [
    "DENSE_101", "RANK_ONLY", "R_FUTURE_BETA1", "QR_FUTURE_BETA1",
    "R_FUTURE_BETA2", "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"
]
protocol["retrieval"]["primary_exploratory_contrast"] = "S7_BETA4_QR_MINUS_S5_QR"
protocol["retrieval"]["other_contrasts_are_exploratory"] = True
protocol["future_state"]["conditions"] = [
    "R_FUTURE_BETA1", "QR_FUTURE_BETA1", "R_FUTURE_BETA2",
    "QR_FUTURE_BETA2", "R_FUTURE_BETA4", "QR_FUTURE_BETA4"
]
protocol["future_state"]["all_contrasts_exploratory"] = True
protocol["outputs"] = [
    "results/phase_s7/beta_4/evaluation_v1/s7_beta4_oof_evaluation_summary_v1.json",
    "results/phase_s7/beta_4/evaluation_v1/s7_beta4_oof_evaluation_arrays_v1.npz",
    "results/phase_s7/beta_4/evaluation_v1/s7_beta4_oof_evaluation_report_v1.txt",
]
protocol["dev_payload_used"] = False
protocol["test_payload_used"] = False
protocol["no_posthoc_success_gate"] = True
protocol["overwrite_policy"] = "REFUSE_IF_PROTOCOL_OR_OUTPUT_DIRECTORY_EXISTS"

# Sanity-check the frozen protocol against the generated evaluator before writing either file.
require(protocol["evaluation_script_sha256"] == hashlib.sha256(s.encode("utf-8")).hexdigest(), "Generated script SHA mismatch")
require(protocol["status"] == "FROZEN_BEFORE_S7_BETA4_METRIC_REVEAL", "Wrong protocol freeze status")
require(len(protocol["retrieval"]["conditions"]) == 8, "Expected 8 retrieval conditions")
require(len(protocol["future_state"]["conditions"]) == 6, "Expected 6 future-state conditions")
protocol_text = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
json.loads(protocol_text)

# Exclusive creation protects both frozen artifacts from accidental overwrite.
require(not NEW_SCRIPT.exists() and not NEW_PROTO.exists() and not OUT.exists(), "A beta=4 target appeared during preflight; refusing write")
with NEW_SCRIPT.open("x", encoding="utf-8", newline="\n") as f:
    f.write(s)
try:
    with NEW_PROTO.open("x", encoding="utf-8", newline="\n") as f:
        f.write(protocol_text)
except Exception:
    # The beta=2 source is never touched. Remove only the just-created beta=4 script if protocol creation fails.
    NEW_SCRIPT.unlink(missing_ok=True)
    raise

require(sha(NEW_SCRIPT) == protocol["evaluation_script_sha256"], "Post-write evaluator SHA mismatch")
print("\nBETA4_EVALUATOR_CREATED", NEW_SCRIPT.relative_to(ROOT))
print("BETA4_PROTOCOL_CREATED", NEW_PROTO.relative_to(ROOT))
print("BETA4_EVALUATOR_SHA256", sha(NEW_SCRIPT))
print("BETA4_PROTOCOL_SHA256", sha(NEW_PROTO))
print("BETA4_INPUT_COUNT", len(protocol["inputs"]))
print("RETRIEVAL_CONDITIONS", len(protocol["retrieval"]["conditions"]))
print("FUTURE_STATE_CONDITIONS", len(protocol["future_state"]["conditions"]))
print("NO_EVALUATION_STARTED = YES")
