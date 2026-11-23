import json
import hashlib
from pathlib import Path

ROOT = Path("results/phase_s12c")

V1 = ROOT / "step_2f18x_independent_permutation_audit_v1"
V2 = ROOT / "step_2f18y_holm_reanalysis_v2"

V1_JSON = V1 / "independent_permutation_results_v1.json"
V1_REPORT = V1 / "independent_permutation_report_v1.txt"
V1_MANIFEST = V1 / "run_manifest.json"

V2_JSON = V2 / "holm_reanalysis_results_v2.json"
V2_REPORT = V2 / "holm_reanalysis_report_v2.txt"
V2_MANIFEST = V2 / "run_manifest.json"

OUT = ROOT / "step_2f18z_final_holm_audit_v1"

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

# Independent scalar Holm implementation
def independent_holm(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: (pvalues[i], i))
    result = [0.0] * m
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(
            running,
            (m - rank) * pvalues[idx]
        )
        result[idx] = min(1.0, running)

    return result

def check_close(a, b, name, tol=1e-12):
    if abs(float(a) - float(b)) > tol:
        raise RuntimeError(
            f"{name} mismatch: {a} vs {b}"
        )

for p in [V1_JSON, V1_REPORT, V1_MANIFEST,
          V2_JSON, V2_REPORT, V2_MANIFEST]:
    if not p.exists():
        raise FileNotFoundError(p)

if OUT.exists():
    raise RuntimeError("Output collision; refusing overwrite")

v1 = json.loads(V1_JSON.read_text(encoding="utf-8"))
v2 = json.loads(V2_JSON.read_text(encoding="utf-8"))
manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))

assert v1["status"] == "COMPLETED"
assert v2["status"] == "COMPLETED"
assert v2["total_tests"] == 400

check_close(
    v2["source_result_sha256"],
    sha(V1_JSON),
    "V1 source SHA"
) if False else None

if v2["source_result_sha256"] != sha(V1_JSON):
    raise RuntimeError("V1 source SHA mismatch")

records = []

for subset, sd in v1["datasets"].items():
    if len(sd["contrasts"]) != 10:
        raise RuntimeError(f"Contrast count error: {subset}")

    for contrast, metrics in sd["contrasts"].items():
        if len(metrics) != 5:
            raise RuntimeError(f"Metric count error: {subset}/{contrast}")

        for metric, value in metrics.items():
            records.append({
                "dataset": subset.split("_", 1)[0],
                "subset": subset,
                "contrast": contrast,
                "metric": metric,
                "delta": float(value["delta"]),
                "p": float(value["p_raw"]),
                "source": value,
            })

if len(records) != 400:
    raise RuntimeError(f"Record count error: {len(records)}")

keys = [
    (r["subset"], r["contrast"], r["metric"])
    for r in records
]

if len(set(keys)) != 400:
    raise RuntimeError("Duplicate test records")

# Recompute global Holm-400
global_adj = independent_holm([r["p"] for r in records])

# Recompute DEV/TEST Holm-200
dataset_adj = {}

for dataset in ("DEV", "TEST"):
    ids = [
        i for i, r in enumerate(records)
        if r["dataset"] == dataset
    ]

    if len(ids) != 200:
        raise RuntimeError(f"{dataset} family count error")

    values = independent_holm([records[i]["p"] for i in ids])

    for i, value in zip(ids, values):
        dataset_adj[i] = value

# Recompute 40 Holm-10 families
families = {}

for i, r in enumerate(records):
    key = (r["subset"], r["metric"])
    families.setdefault(key, []).append(i)

if len(families) != 40:
    raise RuntimeError("Secondary family count error")

local_adj = {}

for key, ids in families.items():
    if len(ids) != 10:
        raise RuntimeError(f"Family size error: {key}")

    values = independent_holm([records[i]["p"] for i in ids])

    for i, value in zip(ids, values):
        local_adj[i] = value

# Compare V1 and V2 source fields and all adjusted values
max_diff = 0.0

for i, r in enumerate(records):
    subset = r["subset"]
    contrast = r["contrast"]
    metric = r["metric"]

    old = r["source"]
    new = v2["datasets"][subset]["contrasts"][contrast][metric]

    for field in (
        "delta", "p_raw",
        "bootstrap_ci95_lower",
        "bootstrap_ci95_upper"
    ):
        check_close(old[field], new[field], field)

    checks = [
        (new["p_holm_global_400"], global_adj[i], "Holm400"),
        (new["p_holm_dataset_200_sensitivity"], dataset_adj[i], "Holm200"),
        (new["p_holm_subset_metric_10_sensitivity"], local_adj[i], "Holm10"),
    ]

    for actual, expected, name in checks:
        diff = abs(float(actual) - float(expected))
        max_diff = max(max_diff, diff)
        check_close(actual, expected, name)

    for raw, adjusted in [
        (r["p"], global_adj[i]),
        (r["p"], dataset_adj[i]),
        (r["p"], local_adj[i]),
    ]:
        if adjusted < raw - 1e-12 or adjusted > 1.0:
            raise RuntimeError("Holm monotonicity/bounds failure")

# Validate V2 report contains all 400 exact result rows
report_lines = V2_REPORT.read_text(encoding="utf-8").splitlines()
report_set = set(report_lines)

for i, r in enumerate(records):
    expected_line = (
        f"{r['subset']} | {r['contrast']} | {r['metric']} "
        f"delta={r['delta']:+.7f} "
        f"p={r['p']:.8g} "
        f"Holm400={global_adj[i]:.8g} "
        f"Holm200={dataset_adj[i]:.8g} "
        f"Holm10={local_adj[i]:.8g}"
    )

    if expected_line not in report_set:
        raise RuntimeError(f"Report mismatch: {r}")

# Validate V2 manifest
if manifest["result_sha256"] != sha(V2_JSON):
    raise RuntimeError("V2 result SHA mismatch")

if manifest["report_sha256"] != sha(V2_REPORT):
    raise RuntimeError("V2 report SHA mismatch")

for source, expected in manifest["source_sha256"].items():
    p = Path(source)
    if not p.exists() or sha(p) != expected:
        raise RuntimeError(f"Source SHA mismatch: {source}")

# Save independent audit
OUT.mkdir(parents=True)

result = {
    "artifact": "S12C Final Holm Independent Audit V1",
    "status": "PASS",
    "total_tests": 400,
    "global_family": 400,
    "dataset_families": 2,
    "secondary_families": 40,
    "secondary_family_size": 10,
    "independent_holm_max_abs_diff": max_diff,
    "unique_records": len(set(keys)),
    "report_rows_verified": 400,
    "v1_source_sha256": sha(V1_JSON),
    "v2_result_sha256": sha(V2_JSON),
    "v2_report_sha256": sha(V2_REPORT),
}

result_path = OUT / "final_audit_results_v1.json"
report_path = OUT / "final_audit_report_v1.txt"
manifest_path = OUT / "run_manifest.json"

result_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

report_path.write_text(
    "\n".join(f"{k}: {v}" for k, v in result.items()) + "\n",
    encoding="utf-8"
)

final_manifest = {
    "artifact": "S12C Final Holm Audit Manifest V1",
    "status": "PASS",
    "result_sha256": sha(result_path),
    "report_sha256": sha(report_path),
    "v1_source_sha256": sha(V1_JSON),
    "v2_result_sha256": sha(V2_JSON),
    "v2_report_sha256": sha(V2_REPORT),
}

manifest_path.write_text(
    json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("=" * 80)
print("U3_D7_FINAL_HOLM_AUDIT_PASS")
print("TOTAL TESTS:", len(records))
print("UNIQUE RECORDS:", len(set(keys)))
print("GLOBAL HOLM:", 400)
print("DATASET FAMILIES:", 2)
print("SECONDARY FAMILIES:", len(families))
print("REPORT ROWS VERIFIED:", 400)
print("INDEPENDENT HOLM MAX DIFF:", max_diff)
print("V1/V2 SOURCE INTEGRITY: PASS")
print("V2 MANIFEST INTEGRITY: PASS")
print("RESULT:", result_path)
print("REPORT:", report_path)
print("MANIFEST:", manifest_path)
print("=" * 80)
