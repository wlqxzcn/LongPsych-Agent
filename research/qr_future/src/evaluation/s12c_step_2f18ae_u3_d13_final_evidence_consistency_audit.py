import json
import hashlib
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[4]
ROOT = BASE / "results/phase_s12c"
OUT = ROOT / "step_2f18ae_u3_d13_final_evidence_consistency_audit_v1"

if OUT.exists():
    raise RuntimeError(f"Output collision: {OUT}")

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    result = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m-rank) * pvalues[idx])
        result[idx] = min(running, 1.0)
    return result

checks = []
warnings = []

def check(name, condition, detail=""):
    checks.append({
        "name": name,
        "status": "PASS" if condition else "FAIL",
        "detail": str(detail)
    })
    print(("PASS" if condition else "FAIL"), name, detail)

def warn(name, detail):
    warnings.append({"name": name, "detail": str(detail)})
    print("WARNING", name, detail)

def verify_outputs(manifest, directory, mapping=None):
    outputs = manifest.get("outputs", {})
    for key, expected in outputs.items():
        filename = mapping.get(key, key) if mapping else key
        path = directory / filename
        check(
            f"OUTPUT_HASH::{directory.name}::{filename}",
            path.is_file() and sha(path) == expected,
            expected
        )

# --------------------------------------------------
# 1. Load frozen artifacts
# --------------------------------------------------

D = {
    "con": ROOT / "step_2f18aa_final_results_consolidation_v1",
    "d10": ROOT / "step_2f18ab_u3_d10_case_mechanism_audit_v1",
    "d11": ROOT / "step_2f18ac_u3_d11_paired_interaction_audit_v2",
    "d12": ROOT / "step_2f18ad_u3_d12_product_interaction_mechanism_v1",
    "holm": ROOT / "step_2f18z_final_holm_audit_v1",
    "perm": ROOT / "step_2f18x_independent_permutation_audit_v1",
    "v2": ROOT / "step_2f18y_holm_reanalysis_v2",
}

FILES = {
    "con_m": D["con"] / "run_manifest.json",
    "con_r": D["con"] / "final_results_v1.json",

    "d10_m": D["d10"] / "run_manifest.json",
    "d10_p": D["d10"] / "protocol_v1.json",
    "d10_s": D["d10"] / "case_statistics_v1.json",
    "d10_l": D["d10"] / "long_vs_nonlong_statistics_v1.json",

    "d11_m": D["d11"] / "run_manifest_v2.json",
    "d11_p": D["d11"] / "protocol_v1.json",
    "d11_r": D["d11"] / "interaction_results_v2.json",

    "d12_m": D["d12"] / "run_manifest_v1.json",
    "d12_p": D["d12"] / "protocol_v1.json",
    "d12_r": D["d12"] / "d12_results_v1.json",

    "holm_m": D["holm"] / "run_manifest.json",
    "holm_r": D["holm"] / "final_audit_results_v1.json",

    "perm_m": D["perm"] / "run_manifest.json",
    "perm_r": D["perm"] / "independent_permutation_results_v1.json",
}

for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"Missing source {name}: {path}")

A = {name: load(path) for name, path in FILES.items()}

# --------------------------------------------------
# 2. Freeze audit protocol
# --------------------------------------------------

OUT.mkdir(parents=True)

source_hashes = {
    name: sha(path) for name, path in FILES.items()
}

protocol = {
    "artifact": "S12C U3-D13 Final Evidence Consistency Audit V1",
    "status": "FROZEN_BEFORE_FINAL_AUDIT",
    "scope": [
        "source_integrity",
        "manifest_integrity",
        "cross_stage_sha256_consistency",
        "statistical_result_consistency",
        "multiple_testing_reproduction",
        "paper_claim_boundary"
    ],
    "training": False,
    "new_ranking_inference": False,
    "test_time_fitting": False,
    "historical_artifact_modification": False,
    "source_sha256": source_hashes
}

protocol_path = OUT / "protocol_v1.json"
protocol_path.write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

# --------------------------------------------------
# 3. Artifact status and output integrity
# --------------------------------------------------

for name in ["con_m", "d10_m", "d11_m", "d12_m", "perm_m"]:
    check(f"{name}_COMPLETED", A[name].get("status") == "COMPLETED")

check("FINAL_HOLM_PASS", A["holm_m"].get("status") == "PASS")
check("FINAL_HOLM_RESULT_PASS", A["holm_r"].get("status") == "PASS")

verify_outputs(
    A["con_m"], D["con"],
    {
        "json": "final_results_v1.json",
        "model_csv": "model_performance_v1.csv",
        "contrast_csv": "paired_contrasts_v1.csv",
        "report": "final_results_report_v1.txt"
    }
)

verify_outputs(A["d10_m"], D["d10"])
verify_outputs(A["d11_m"], D["d11"])
verify_outputs(A["d12_m"], D["d12"])

check(
    "FINAL_HOLM_RESULT_HASH",
    sha(D["holm"] / "final_audit_results_v1.json")
    == A["holm_m"]["result_sha256"]
)

check(
    "FINAL_HOLM_REPORT_HASH",
    sha(D["holm"] / "final_audit_report_v1.txt")
    == A["holm_m"]["report_sha256"]
)

check(
    "PERMUTATION_RESULT_HASH",
    sha(D["perm"] / "independent_permutation_results_v1.json")
    == A["perm_m"]["result_sha256"]
)

check(
    "PERMUTATION_REPORT_HASH",
    sha(D["perm"] / "independent_permutation_report_v1.txt")
    == A["perm_m"]["report_sha256"]
)

# --------------------------------------------------
# 4. Protocol and source-chain integrity
# --------------------------------------------------

check(
    "D10_PROTOCOL_HASH",
    sha(D["d10"] / "protocol_v1.json")
    == A["d10_m"].get("protocol_sha256")
)

check(
    "D10_INPUT_ALIGNMENT",
    A["d10_m"].get("input_sha256")
    == A["d10_p"].get("input_sha256")
)

check(
    "D10_PROTOCOL_FROZEN",
    A["d10_p"].get("status") == "FROZEN_BEFORE_CASE_EXTRACTION"
)

d11src = A["d11_m"]["source_sha256"]

check(
    "D11_D10_MANIFEST_LINK",
    d11src["d10_manifest"] == sha(FILES["d10_m"])
)

check(
    "D11_D10_PROTOCOL_LINK",
    d11src["d10_protocol"] == sha(FILES["d10_p"])
)

check(
    "D11_PROTOCOL_LINK",
    d11src["d11_protocol"] == sha(FILES["d11_p"])
)

check(
    "D11_PROTOCOL_FROZEN",
    A["d11_p"].get("status") == "FROZEN_BEFORE_INTERACTION_TEST"
)

check(
    "D11_FINALIZATION_ONLY",
    A["d11_m"].get("finalization_only") is True
)

check(
    "D11_STATISTICS_NOT_RECOMPUTED",
    A["d11_m"].get("statistics_recomputed") is False
)

d12src = A["d12_m"]["source_sha256"]

check(
    "D12_PROTOCOL_HASH",
    sha(FILES["d12_p"]) == A["d12_m"]["protocol_sha256"]
)

check(
    "D12_PROTOCOL_SOURCE_ALIGNMENT",
    d12src == A["d12_p"].get("source_sha256")
)

check(
    "D12_D11_MANIFEST_LINK",
    d12src["d11_manifest"] == sha(FILES["d11_m"])
)

check(
    "D12_D11_RESULT_LINK",
    d12src["d11_results"] == sha(FILES["d11_r"])
)

check(
    "D12_TEST_PREDICTION_LINK",
    d12src["test_predictions"]
    == A["perm_r"]["source_sha256"]["test_predictions"]
)

check(
    "D12_TEST_SUMMARY_LINK",
    d12src["test_summary"]
    == A["perm_r"]["source_sha256"]["test_summary"]
)

check(
    "D12_PROTOCOL_FROZEN",
    A["d12_p"].get("status") == "FROZEN_BEFORE_D12_INFERENCE"
)

# --------------------------------------------------
# 5. D6/D7 and final consolidation chain
# --------------------------------------------------

v2_result = D["v2"] / "holm_reanalysis_results_v2.json"
v2_report = D["v2"] / "holm_reanalysis_report_v2.txt"
v2_manifest = D["v2"] / "run_manifest.json"

for p in [v2_result, v2_report, v2_manifest]:
    if not p.is_file():
        raise FileNotFoundError(p)

consrc = A["con_m"]["source_sha256"]

check(
    "CONSOLIDATION_HOLM_V2_LINK",
    consrc["holm_v2"] == sha(v2_result)
)

check(
    "CONSOLIDATION_HOLM_AUDIT_LINK",
    consrc["holm_audit"] == sha(FILES["holm_r"])
)

check(
    "CONSOLIDATION_V2_MANIFEST_LINK",
    consrc["v2_manifest"] == sha(v2_manifest)
)

check(
    "CONSOLIDATION_AUDIT_MANIFEST_LINK",
    consrc["audit_manifest"] == sha(FILES["holm_m"])
)

check(
    "FINAL_HOLM_V2_RESULT_LINK",
    A["holm_m"]["v2_result_sha256"] == sha(v2_result)
)

check(
    "FINAL_HOLM_V2_REPORT_LINK",
    A["holm_m"]["v2_report_sha256"] == sha(v2_report)
)

check(
    "FINAL_AUDIT_V2_RESULT_LINK",
    A["holm_r"]["v2_result_sha256"] == sha(v2_result)
)

check(
    "FINAL_AUDIT_V2_REPORT_LINK",
    A["holm_r"]["v2_report_sha256"] == sha(v2_report)
)

# Locate dev/test manifest files by their frozen hashes.
json_hash_index = defaultdict(list)

for p in ROOT.rglob("*.json"):
    if p.is_file():
        json_hash_index[sha(p)].append(str(p))

for key in ["dev_manifest", "test_manifest"]:
    expected = consrc[key]
    check(
        f"CONSOLIDATION_{key.upper()}_SOURCE",
        expected in json_hash_index,
        json_hash_index.get(expected, [])
    )

perm_sources = A["perm_r"]["source_sha256"]

for key in ["dev_summary", "test_summary"]:
    check(
        f"CONSOLIDATION_{key.upper()}_SOURCE",
        consrc[key] == perm_sources[key]
    )

# --------------------------------------------------
# 6. Final consolidated result counts
# --------------------------------------------------

con_result = A["con_r"]

check(
    "MODEL_ROWS_280",
    con_result.get("model_row_count") == 280
    and A["con_m"].get("model_rows") == 280
)

check(
    "CONTRAST_ROWS_400",
    con_result.get("contrast_metric_row_count") == 400
    and A["con_m"].get("contrast_metric_rows") == 400
)

holm_result = A["holm_r"]

check(
    "HOLM_GLOBAL_FAMILY_400",
    holm_result.get("global_family") == 400
)

check(
    "HOLM_DATASET_FAMILIES_2",
    holm_result.get("dataset_families") == 2
)

check(
    "HOLM_SECONDARY_FAMILIES_40",
    holm_result.get("secondary_families") == 40
)

check(
    "HOLM_SECONDARY_SIZE_10",
    holm_result.get("secondary_family_size") == 10
)

check(
    "HOLM_UNIQUE_RECORDS_400",
    holm_result.get("unique_records") == 400
)

check(
    "HOLM_REPORT_ROWS_400",
    holm_result.get("report_rows_verified") == 400
)

check(
    "HOLM_INDEPENDENT_MAX_DIFF_ZERO",
    holm_result.get("independent_holm_max_abs_diff") == 0.0
)

# --------------------------------------------------
# 7. Independent permutation: reproduce actual Holm
# --------------------------------------------------

perm = A["perm_r"]
datasets = perm["datasets"]
metrics = perm["metrics"]

records = []

for subset, data in datasets.items():
    for contrast, metric_data in data["contrasts"].items():
        for metric in metrics:
            v = metric_data[metric]
            records.append({
                "subset": subset,
                "contrast": contrast,
                "metric": metric,
                "p_raw": float(v["p_raw"]),
                "global": float(v["p_holm_global_200"]),
                "local": float(v["p_holm_subset_metric_10"])
            })

check("PERMUTATION_TOTAL_RECORDS_400", len(records) == 400)
check("PERMUTATION_DATASETS_8", len(datasets) == 8)

# Reproduce global Holm across all 400 records.
global_values = holm([r["p_raw"] for r in records])

global_diff = max(
    abs(a["global"] - b)
    for a, b in zip(records, global_values)
)

check(
    "PERMUTATION_GLOBAL_HOLM_400_REPRODUCTION",
    global_diff <= 1e-12,
    f"max_diff={global_diff}"
)

# Reproduce 40 secondary families, each with 10 tests.
families = defaultdict(list)

for i, r in enumerate(records):
    families[(r["subset"], r["metric"])].append(i)

local_diff = 0.0
family_sizes = []

for key, indices in families.items():
    family_sizes.append(len(indices))
    adjusted = holm([records[i]["p_raw"] for i in indices])

    for i, value in zip(indices, adjusted):
        local_diff = max(
            local_diff,
            abs(records[i]["local"] - value)
        )

check("PERMUTATION_SECONDARY_FAMILIES_40", len(families) == 40)
check("PERMUTATION_FAMILY_SIZE_10", set(family_sizes) == {10})
check(
    "PERMUTATION_SECONDARY_HOLM_REPRODUCTION",
    local_diff <= 1e-12,
    f"max_diff={local_diff}"
)

# Historical labels are preserved but explicitly flagged.
mt_text = json.dumps(perm.get("multiple_testing", {}))
report_text = (
    D["perm"] / "independent_permutation_report_v1.txt"
).read_text(encoding="utf-8")

if "200 tests" in mt_text or "Primary Holm family: 200 tests" in report_text:
    warn(
        "PERMUTATION_V1_STALE_LABEL",
        "Actual global Holm family is 400; historical V1 metadata "
        "and report still say 200. Numerical values were independently "
        "reproduced using m=400. Preserve V1 unchanged."
    )

# --------------------------------------------------
# 8. D10/D11/D12 interpretation boundaries
# --------------------------------------------------

check(
    "D10_SAMPLE_ALIGNMENT",
    A["d10_m"].get("test_queries") == 18073
    and A["d10_m"].get("test_dialogues") == 477
    and A["d10_m"].get("long20_queries") == 8727
)

check(
    "D11_PAIRED_DIALOGUES_413",
    A["d11_m"].get("paired_dialogues") == 413
)

check(
    "D12_PAIRED_DIALOGUES_413",
    A["d12_r"].get("n_paired_dialogues") == 413
)

check(
    "D12_EXPLORATORY_POST_HOC",
    A["d12_r"].get("interpretation") == "EXPLORATORY_POST_HOC"
)

check(
    "D12_POINT_ESTIMATE_REPRODUCTION",
    A["d12_r"].get("d11_point_estimate_reproduction") is True
)

for name, obj in [
    ("D10_MANIFEST", A["d10_m"]),
    ("D12_MANIFEST", A["d12_m"]),
    ("D12_RESULT", A["d12_r"])
]:
    for field in ["training", "new_ranking_inference"]:
        if field in obj:
            check(
                f"{name}_{field.upper()}_FALSE",
                obj[field] is False
            )

for field in ["training", "new_ranking_inference", "test_time_fitting"]:
    if field in A["d12_r"]:
        check(
            f"D12_RESULT_{field.upper()}_FALSE",
            A["d12_r"][field] is False
        )

# --------------------------------------------------
# 9. Build final audit artifacts
# --------------------------------------------------

failed = [x for x in checks if x["status"] == "FAIL"]

if failed:
    overall = "FAIL"
elif warnings:
    overall = "PASS_WITH_DOCUMENTATION_WARNING"
else:
    overall = "PASS"

result = {
    "artifact": "S12C U3-D13 Final Evidence Consistency Audit V1",
    "status": overall,
    "total_checks": len(checks),
    "passed_checks": len(checks) - len(failed),
    "failed_checks": len(failed),
    "warning_count": len(warnings),
    "source_sha256": source_hashes,
    "permutation_reproduction": {
        "records": len(records),
        "global_family": len(records),
        "global_max_abs_diff": global_diff,
        "secondary_families": len(families),
        "secondary_family_size": 10,
        "secondary_max_abs_diff": local_diff
    },
    "checks": checks,
    "warnings": warnings,
    "training": False,
    "new_ranking_inference": False,
    "test_time_fitting": False,
    "historical_artifacts_modified": False
}

result_path = OUT / "audit_results_v1.json"
result_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

lines = [
    "S12C U3-D13 FINAL EVIDENCE CONSISTENCY AUDIT",
    "=" * 85,
    f"OVERALL: {overall}",
    f"TOTAL CHECKS: {len(checks)}",
    f"PASSED: {len(checks)-len(failed)}",
    f"FAILED: {len(failed)}",
    f"WARNINGS: {len(warnings)}",
    "",
    "STATISTICAL REPRODUCTION",
    f"Permutation records: {len(records)}",
    f"Global Holm family: {len(records)}",
    f"Global maximum difference: {global_diff:.12g}",
    f"Secondary families: {len(families)}",
    f"Secondary family size: 10",
    f"Secondary maximum difference: {local_diff:.12g}",
    "",
    "CHECK DETAILS"
]

for item in checks:
    lines.append(
        f"{item['status']} | {item['name']} | {item['detail']}"
    )

lines += ["", "WARNINGS"]

for item in warnings:
    lines.append(f"{item['name']}: {item['detail']}")

report_path = OUT / "audit_report_v1.txt"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "artifact": "S12C U3-D13 Final Evidence Audit Manifest V1",
    "status": overall,
    "protocol_sha256": sha(protocol_path),
    "result_sha256": sha(result_path),
    "report_sha256": sha(report_path),
    "script_sha256": sha(Path(__file__)),
    "source_sha256": source_hashes,
    "total_checks": len(checks),
    "failed_checks": len(failed),
    "warning_count": len(warnings),
    "training": False,
    "new_ranking_inference": False,
    "test_time_fitting": False
}

manifest_path = OUT / "run_manifest_v1.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("\n" + "=" * 85)
print("U3_D13_FINAL_EVIDENCE_AUDIT_COMPLETE")
print("STATUS:", overall)
print("TOTAL CHECKS:", len(checks))
print("FAILED:", len(failed))
print("WARNINGS:", len(warnings))
print("GLOBAL HOLM REPRODUCTION:", global_diff)
print("SECONDARY HOLM REPRODUCTION:", local_diff)
print("OUTPUT:", OUT)
print("MANIFEST:", manifest_path)
print("=" * 85)

if failed:
    raise SystemExit(1)
