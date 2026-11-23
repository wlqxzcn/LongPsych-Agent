import json
import csv
import hashlib
from pathlib import Path

ROOT = Path("results/phase_s12c")

DEV = ROOT / "step_2f18s_orca_fsar_dev_evaluation_v1"
TEST = ROOT / "step_2f18t_orca_fsar_test_evaluation_v1"
V2 = ROOT / "step_2f18y_holm_reanalysis_v2"
AUDIT = ROOT / "step_2f18z_final_holm_audit_v1"

OUT = ROOT / "step_2f18aa_final_results_consolidation_v1"

SOURCES = {
    "dev_summary": DEV / "dev_summary_v1.json",
    "test_summary": TEST / "test_summary_v1.json",
    "holm_v2": V2 / "holm_reanalysis_results_v2.json",
    "holm_audit": AUDIT / "final_audit_results_v1.json",
    "dev_manifest": DEV / "run_manifest.json",
    "test_manifest": TEST / "run_manifest.json",
    "v2_manifest": V2 / "run_manifest.json",
    "audit_manifest": AUDIT / "run_manifest.json",
}

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]

SUBSETS = [
    "DEV_ALL", "DEV_LONG20",
    "DEV_GOLD_MATCHED", "DEV_GOLD_UNMATCHED",
    "TEST_ALL", "TEST_LONG20",
    "TEST_GOLD_MATCHED", "TEST_GOLD_UNMATCHED",
]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))

if OUT.exists():
    raise RuntimeError("Output collision; refusing overwrite")

for name, path in SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")

dev = load(SOURCES["dev_summary"])
test = load(SOURCES["test_summary"])
v2 = load(SOURCES["holm_v2"])
audit = load(SOURCES["holm_audit"])

if audit["status"] != "PASS":
    raise RuntimeError("U3-D7 audit not PASS")

if audit["total_tests"] != 400:
    raise RuntimeError("Unexpected audit test count")

if v2["status"] != "COMPLETED":
    raise RuntimeError("Holm V2 incomplete")

if v2["source_result_sha256"] != sha(
    ROOT / "step_2f18x_independent_permutation_audit_v1"
    / "independent_permutation_results_v1.json"
):
    raise RuntimeError("V1 source hash mismatch")

if audit["v2_result_sha256"] != sha(SOURCES["holm_v2"]):
    raise RuntimeError("V2 audit hash mismatch")

summaries = {
    "DEV": dev,
    "TEST": test,
}

model_rows = []
contrast_rows = []

for subset in SUBSETS:
    dataset = subset.split("_", 1)[0]
    summary = summaries[dataset]

    if subset not in summary["datasets"]:
        raise RuntimeError(f"Missing summary subset: {subset}")

    if subset not in v2["datasets"]:
        raise RuntimeError(f"Missing V2 subset: {subset}")

    source_subset = summary["datasets"][subset]
    corrected_subset = v2["datasets"][subset]

    # Model performance and bootstrap confidence intervals
    for model, metrics in source_subset["models"].items():
        for metric in METRICS:
            if metric not in metrics:
                raise RuntimeError(f"Missing metric: {subset}/{model}/{metric}")

            value = metrics[metric]

            model_rows.append({
                "subset": subset,
                "model": model,
                "metric": metric,
                "estimate": value["estimate"],
                "ci95_lower": value["ci95_lower"],
                "ci95_upper": value["ci95_upper"],
            })

    # Paired contrasts and multiplicity corrections
    original_contrasts = source_subset["paired_contrasts"]
    corrected_contrasts = corrected_subset["contrasts"]

    if set(original_contrasts) != set(corrected_contrasts):
        raise RuntimeError(f"Contrast mismatch: {subset}")

    for contrast, metrics in original_contrasts.items():
        for metric in METRICS:
            original = metrics[metric]
            corrected = corrected_contrasts[contrast][metric]

            for field in ("delta", "ci95_lower", "ci95_upper"):
                if abs(float(original[field]) - float(
                    corrected["delta"] if field == "delta"
                    else corrected[
                        "bootstrap_ci95_lower" if field == "ci95_lower"
                        else "bootstrap_ci95_upper"
                    ]
                )) > 1e-10:
                    raise RuntimeError(
                        f"Contrast source mismatch: {subset}/{contrast}/{metric}"
                    )

            contrast_rows.append({
                "subset": subset,
                "contrast": contrast,
                "metric": metric,
                "delta": original["delta"],
                "bootstrap_ci95_lower": original["ci95_lower"],
                "bootstrap_ci95_upper": original["ci95_upper"],
                "p_raw": corrected["p_raw"],
                "p_holm_global_400": corrected["p_holm_global_400"],
                "p_holm_dataset_200_sensitivity":
                    corrected["p_holm_dataset_200_sensitivity"],
                "p_holm_subset_metric_10_sensitivity":
                    corrected["p_holm_subset_metric_10_sensitivity"],
            })

if len(model_rows) == 0:
    raise RuntimeError("Empty model results")

if len(contrast_rows) != 400:
    raise RuntimeError(
        f"Expected 400 contrast-metric rows, got {len(contrast_rows)}"
    )

OUT.mkdir(parents=True)

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

model_csv = OUT / "model_performance_v1.csv"
contrast_csv = OUT / "paired_contrasts_v1.csv"

write_csv(model_csv, model_rows)
write_csv(contrast_csv, contrast_rows)

result = {
    "artifact": "S12C Final Experimental Results Consolidation V1",
    "status": "COMPLETED",
    "sources": {
        name: {
            "path": str(path),
            "sha256": sha(path),
        }
        for name, path in SOURCES.items()
    },
    "subsets": SUBSETS,
    "metrics": METRICS,
    "model_row_count": len(model_rows),
    "contrast_metric_row_count": len(contrast_rows),
    "statistical_policy": {
        "primary_reported_correction": "Holm-400",
        "dev_holm_200": "sensitivity_analysis",
        "test_holm_200": "sensitivity_analysis",
        "subset_metric_holm_10": "sensitivity_analysis",
        "prespecified_multiplicity_family": False,
    },
    "model_results": model_rows,
    "contrast_results": contrast_rows,
}

json_path = OUT / "final_results_v1.json"
json_path.write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

report_path = OUT / "final_results_report_v1.txt"

lines = [
    "S12C FINAL EXPERIMENTAL RESULTS",
    "=" * 85,
    f"Model rows: {len(model_rows)}",
    f"Contrast metric rows: {len(contrast_rows)}",
    "Primary reported correction: Holm-400",
    "DEV/TEST Holm-200: sensitivity analyses",
    "Subset-Metric Holm-10: sensitivity analyses",
    "",
]

for subset in SUBSETS:
    lines.extend([
        "=" * 85,
        subset,
        "=" * 85,
        "MODEL PERFORMANCE",
    ])

    for row in model_rows:
        if row["subset"] != subset:
            continue

        if row["metric"] in ("MRR", "R@5"):
            lines.append(
                f"{row['model']} {row['metric']}="
                f"{row['estimate']:.6f} "
                f"[{row['ci95_lower']:.6f},"
                f"{row['ci95_upper']:.6f}]"
            )

    lines.append("KEY PAIRED CONTRASTS")

    for row in contrast_rows:
        if row["subset"] != subset:
            continue

        if row["metric"] in ("MRR", "R@5"):
            lines.append(
                f"{row['contrast']} {row['metric']} "
                f"delta={row['delta']:+.6f} "
                f"CI=[{row['bootstrap_ci95_lower']:+.6f},"
                f"{row['bootstrap_ci95_upper']:+.6f}] "
                f"p={row['p_raw']:.7g} "
                f"Holm400={row['p_holm_global_400']:.7g}"
            )

report_path.write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8"
)

manifest = {
    "artifact": "S12C Final Results Consolidation Manifest V1",
    "status": "COMPLETED",
    "model_rows": len(model_rows),
    "contrast_metric_rows": len(contrast_rows),
    "source_sha256": {
        name: sha(path) for name, path in SOURCES.items()
    },
    "outputs": {
        "json": sha(json_path),
        "model_csv": sha(model_csv),
        "contrast_csv": sha(contrast_csv),
        "report": sha(report_path),
    },
}

manifest_path = OUT / "run_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("=" * 80)
print("U3_D8_FINAL_RESULTS_CONSOLIDATION_COMPLETE")
print("SUBSETS:", len(SUBSETS))
print("MODEL ROWS:", len(model_rows))
print("CONTRAST METRIC ROWS:", len(contrast_rows))
print("SOURCE CONSISTENCY: PASS")
print("HOLM V2 INTEGRATION: PASS")
print("OUTPUT SHA256: PASS")
print("JSON:", json_path)
print("MODEL CSV:", model_csv)
print("CONTRAST CSV:", contrast_csv)
print("REPORT:", report_path)
print("MANIFEST:", manifest_path)
print("=" * 80)
