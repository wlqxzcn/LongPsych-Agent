import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path("results/phase_s12c")

SRC = ROOT / "step_2f18x_independent_permutation_audit_v1"
OUT = ROOT / "step_2f18y_holm_reanalysis_v2"

SRC_JSON = SRC / "independent_permutation_results_v1.json"
SRC_REPORT = SRC / "independent_permutation_report_v1.txt"
SRC_MANIFEST = SRC / "run_manifest.json"

FILES = [
    SRC_JSON,
    SRC_REPORT,
    SRC_MANIFEST,
]

PROTOCOLS = [
    ROOT / "step_2f18s_orca_fsar_dev_evaluation_v1/dev_evaluation_protocol_v1.json",
    ROOT / "step_2f18t_orca_fsar_test_evaluation_v1/test_evaluation_protocol_v1.json",
    ROOT / "step_2f18m_orca_fsar_oof_evaluation_v1/evaluation_protocol_v1.json",
    ROOT / "step_2f18_orca_fsar_training_v1/training_protocol_v1.json",
]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def holm(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)

    if m == 0 or not np.isfinite(p).all():
        raise RuntimeError("Invalid p-value family")

    if np.any((p < 0) | (p > 1)):
        raise RuntimeError("P-values outside [0,1]")

    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]

    adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )

    result = np.empty(m, dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)

    return result

if any(p.exists() for p in [
    OUT / "holm_reanalysis_results_v2.json",
    OUT / "holm_reanalysis_report_v2.txt",
    OUT / "run_manifest.json",
]):
    raise RuntimeError("Output collision: refusing overwrite")

data = json.loads(SRC_JSON.read_text(encoding="utf-8"))

records = []

for subset, subset_data in data["datasets"].items():
    dataset = subset.split("_", 1)[0]

    for contrast, metrics in subset_data["contrasts"].items():
        for metric, value in metrics.items():

            records.append({
                "dataset": dataset,
                "subset": subset,
                "contrast": contrast,
                "metric": metric,
                "delta": float(value["delta"]),
                "p_raw": float(value["p_raw"]),
                "source": value,
            })

if len(records) != 400:
    raise RuntimeError(f"Expected 400 tests, got {len(records)}")

# 1. Global Holm-400
global_adj = holm([r["p_raw"] for r in records])

# 2. Dataset-level Holm-200
dataset_adj = {}

for dataset in ["DEV", "TEST"]:
    indices = [
        i for i, r in enumerate(records)
        if r["dataset"] == dataset
    ]

    if len(indices) != 200:
        raise RuntimeError(
            f"{dataset} expected 200 tests, got {len(indices)}"
        )

    values = holm([records[i]["p_raw"] for i in indices])

    for i, value in zip(indices, values):
        dataset_adj[i] = float(value)

# 3. Subset-metric Holm-10
local_adj = {}

families = {}

for i, r in enumerate(records):
    key = (r["subset"], r["metric"])
    families.setdefault(key, []).append(i)

if len(families) != 40:
    raise RuntimeError("Expected 40 secondary families")

for key, indices in families.items():
    if len(indices) != 10:
        raise RuntimeError(f"Invalid family size: {key}")

    values = holm([records[i]["p_raw"] for i in indices])

    for i, value in zip(indices, values):
        local_adj[i] = float(value)

# Independent reproduction against V1 fields
global_diff = []
local_diff = []

for i, r in enumerate(records):
    old = r["source"]

    if "p_holm_global_200" in old:
        global_diff.append(
            abs(global_adj[i] - old["p_holm_global_200"])
        )

    if "p_holm_subset_metric_10" in old:
        local_diff.append(
            abs(local_adj[i] - old["p_holm_subset_metric_10"])
        )

max_global_diff = max(global_diff, default=0)
max_local_diff = max(local_diff, default=0)

if max_global_diff > 1e-12:
    raise RuntimeError("Global Holm V1 reproduction failed")

if max_local_diff > 1e-12:
    raise RuntimeError("Local Holm V1 reproduction failed")

# Assemble V2
for i, r in enumerate(records):
    target = r["source"]

    target["p_holm_global_400"] = float(global_adj[i])
    target["p_holm_dataset_200_sensitivity"] = dataset_adj[i]
    target["p_holm_subset_metric_10_sensitivity"] = local_adj[i]

result = {
    "artifact": "S12C Holm Reanalysis V2",
    "status": "COMPLETED",
    "source_result_sha256": sha(SRC_JSON),
    "total_tests": 400,
    "multiple_testing_v2": {
        "primary_reported_correction": "Holm-400",
        "primary_family_size": 400,
        "method": "Holm-Bonferroni",
        "interpretation": "Conservative post-hoc global correction",
        "dev_sensitivity_family_size": 200,
        "test_sensitivity_family_size": 200,
        "subset_metric_sensitivity_family_size": 10,
        "secondary_family_count": 40,
        "prespecified_multiplicity_family": False,
    },
    "verification": {
        "global_v1_reproduction_max_abs_diff": max_global_diff,
        "local_v1_reproduction_max_abs_diff": max_local_diff,
        "global_records": len(global_adj),
        "dataset_records": len(dataset_adj),
        "secondary_families": len(families),
    },
    "datasets": data["datasets"],
}

OUT.mkdir(parents=True, exist_ok=True)

result_path = OUT / "holm_reanalysis_results_v2.json"
report_path = OUT / "holm_reanalysis_report_v2.txt"
manifest_path = OUT / "run_manifest.json"

result_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

lines = [
    "S12C HOLM REANALYSIS V2",
    "Global family: 400 tests",
    "DEV sensitivity family: 200 tests",
    "TEST sensitivity family: 200 tests",
    "Subset-metric sensitivity: 40 families x 10 tests",
    "Method: Holm-Bonferroni",
    "Interpretation: Conservative post-hoc correction",
    "",
]

for i, r in enumerate(records):
    lines.append(
        f"{r['subset']} | {r['contrast']} | {r['metric']} "
        f"delta={r['delta']:+.7f} "
        f"p={r['p_raw']:.8g} "
        f"Holm400={global_adj[i]:.8g} "
        f"Holm200={dataset_adj[i]:.8g} "
        f"Holm10={local_adj[i]:.8g}"
    )

report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

manifest = {
    "artifact": "S12C Holm Reanalysis V2 Manifest",
    "status": "COMPLETED",
    "source_sha256": {
        str(p): sha(p) for p in FILES + PROTOCOLS if p.exists()
    },
    "result_sha256": sha(result_path),
    "report_sha256": sha(report_path),
    "total_tests": 400,
    "global_holm_family": 400,
    "dataset_holm_families": 2,
    "secondary_families": 40,
    "global_reproduction_max_abs_diff": max_global_diff,
    "local_reproduction_max_abs_diff": max_local_diff,
}

manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("=" * 80)
print("U3_D6_HOLM_REANALYSIS_V2_COMPLETE")
print("TOTAL TESTS:", len(records))
print("GLOBAL HOLM:", len(global_adj))
print("DATASET FAMILIES:", 2)
print("SECONDARY FAMILIES:", len(families))
print("GLOBAL V1 REPRODUCTION DIFF:", max_global_diff)
print("LOCAL V1 REPRODUCTION DIFF:", max_local_diff)
print("RESULT:", result_path)
print("REPORT:", report_path)
print("MANIFEST:", manifest_path)
print("=" * 80)
