import json
import csv
import hashlib
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

OLD = PHASE / "step_2f18ac_u3_d11_paired_interaction_audit_v1"
D10 = PHASE / "step_2f18ab_u3_d10_case_mechanism_audit_v1"
PRED = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_predictions_v1.npz"

OUT = PHASE / "step_2f18ac_u3_d11_paired_interaction_audit_v2"

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * pvalues[order]
    )
    result = np.empty(m)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def check_close(a, b, label):
    if not np.isclose(a, b, atol=1e-12, rtol=1e-10):
        raise RuntimeError(
            f"HOLM_REPRODUCTION_FAILURE: {label}: {a} != {b}"
        )


def main():

    protocol_path = OLD / "protocol_v1.json"
    result_path = OLD / "interaction_results_v1.json"
    d10_manifest_path = D10 / "run_manifest.json"
    d10_protocol_path = D10 / "protocol_v1.json"

    for p in [
        protocol_path, result_path,
        d10_manifest_path, d10_protocol_path, PRED
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    if OUT.exists():
        raise RuntimeError("V2_OUTPUT_COLLISION: " + str(OUT))

    protocol = read_json(protocol_path)
    original = read_json(result_path)
    d10_manifest = read_json(d10_manifest_path)

    print("========== SOURCE INTEGRITY ==========")

    if d10_manifest["status"] != "COMPLETED":
        raise RuntimeError("D10_NOT_COMPLETED")

    if sha(d10_protocol_path) != d10_manifest["protocol_sha256"]:
        raise RuntimeError("D10_PROTOCOL_HASH_FAILURE")

    for name, expected in d10_manifest["outputs"].items():
        p = D10 / name
        if not p.exists() or sha(p) != expected:
            raise RuntimeError("D10_OUTPUT_HASH_FAILURE: " + name)

    inputs = protocol["inputs"]

    expected_sources = {
        "predictions_sha256": sha(PRED),
        "d10_manifest_sha256": sha(d10_manifest_path),
        "d10_protocol_sha256": sha(d10_protocol_path),
    }

    for key, actual in expected_sources.items():
        if inputs[key] != actual:
            raise RuntimeError("D11_SOURCE_HASH_FAILURE: " + key)

    print("D10_MANIFEST_INTEGRITY_PASS")
    print("D10_OUTPUT_HASH_PASS")
    print("D11_SOURCE_HASH_PASS")

    print("\n========== RESULT SCHEMA ==========")

    contrasts = original["contrasts"]

    if len(contrasts) != 8:
        raise RuntimeError("CONTRAST_COUNT_MISMATCH")

    if original["total_tests"] != 40:
        raise RuntimeError("TOTAL_TEST_COUNT_MISMATCH")

    records = []

    for contrast, data in contrasts.items():
        for metric in METRICS:
            if metric not in data["interaction_long_minus_nonlong"]:
                raise RuntimeError(
                    f"MISSING_METRIC: {contrast}/{metric}"
                )

            v = data["interaction_long_minus_nonlong"][metric]

            records.append({
                "contrast": contrast,
                "metric": metric,
                "n_paired_dialogues": data["n_paired_dialogues"],
                "long_delta": data["long_dialogue_macro_delta"][metric],
                "nonlong_delta": data["nonlong_dialogue_macro_delta"][metric],
                "interaction_delta": v["delta"],
                "ci95_lower": v["ci95_lower"],
                "ci95_upper": v["ci95_upper"],
                "p_raw": v["p_raw"],
                "stored": v,
            })

    if len(records) != 40:
        raise RuntimeError("RECORD_COUNT_MISMATCH")

    print("CONTRASTS:", len(contrasts))
    print("RECORDS:", len(records))

    print("\n========== INDEPENDENT HOLM REPRODUCTION ==========")

    global_adjusted = holm([
        r["p_raw"] for r in records
    ])

    mrr_indices = [
        i for i, r in enumerate(records)
        if r["metric"] == "MRR"
    ]

    r5_indices = [
        i for i, r in enumerate(records)
        if r["metric"] == "R@5"
    ]

    mrr_adjusted = holm([
        records[i]["p_raw"] for i in mrr_indices
    ])

    r5_adjusted = holm([
        records[i]["p_raw"] for i in r5_indices
    ])

    for i, r in enumerate(records):

        v = r["stored"]

        check_close(
            v["p_holm_global_40"],
            global_adjusted[i],
            r["contrast"] + "/" + r["metric"] + "/GLOBAL"
        )

        if r["metric"] == "MRR":
            check_close(
                v["p_holm_mrr_8"],
                mrr_adjusted[mrr_indices.index(i)],
                r["contrast"] + "/MRR"
            )

        if r["metric"] == "R@5":
            check_close(
                v["p_holm_r@5_8"],
                r5_adjusted[r5_indices.index(i)],
                r["contrast"] + "/R@5"
            )

    print("GLOBAL_HOLM_40_REPRODUCTION_PASS")
    print("MRR_HOLM_8_REPRODUCTION_PASS")
    print("R5_HOLM_8_REPRODUCTION_PASS")

    print("\n========== BUILD V2 ARTIFACTS ==========")

    OUT.mkdir(parents=True)

    shutil.copy2(
        protocol_path,
        OUT / "protocol_v1.json"
    )

    normalized = json.loads(json.dumps(original))

    for data in normalized["contrasts"].values():
        for metric, v in data["interaction_long_minus_nonlong"].items():
            if metric == "R@5":
                v["p_holm_r5_8"] = v.pop("p_holm_r@5_8")

    normalized["artifact"] = (
        "S12C U3-D11 Paired Interaction Results V2"
    )

    normalized["multiple_testing_audit"] = {
        "global_family_size": 40,
        "mrr_family_size": 8,
        "r5_family_size": 8,
        "independent_reproduction": "PASS",
    }

    result_out = OUT / "interaction_results_v2.json"
    write_json(result_out, normalized)

    fields = [
        "contrast",
        "metric",
        "n_paired_dialogues",
        "long_delta",
        "nonlong_delta",
        "interaction_delta",
        "ci95_lower",
        "ci95_upper",
        "p_raw",
        "p_holm_global_40",
        "p_holm_mrr_8",
        "p_holm_r5_8",
    ]

    csv_rows = []

    for r in records:

        v = r["stored"]

        row = {
            key: r[key]
            for key in fields[:9]
        }

        row["p_holm_global_40"] = v["p_holm_global_40"]
        row["p_holm_mrr_8"] = (
            v["p_holm_mrr_8"]
            if r["metric"] == "MRR" else ""
        )
        row["p_holm_r5_8"] = (
            v["p_holm_r@5_8"]
            if r["metric"] == "R@5" else ""
        )

        csv_rows.append(row)

    csv_path = OUT / "interaction_results_v2.csv"

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "S12C U3-D11 PAIRED INTERACTION AUDIT V2",
        "Statistical computation reused from verified V1 JSON",
        "No retraining or ranking inference",
        "Bootstrap: 5000",
        "Permutation: 20000",
        "Global Holm: 40",
        "MRR Holm: 8",
        "R@5 Holm: 8",
        "",
    ]

    for r in records:

        if r["metric"] not in ("MRR", "R@5"):
            continue

        v = r["stored"]

        secondary = (
            v["p_holm_mrr_8"]
            if r["metric"] == "MRR"
            else v["p_holm_r@5_8"]
        )

        lines.append(
            f"{r['contrast']} | {r['metric']} | "
            f"delta={r['interaction_delta']:+.7f} "
            f"CI=[{r['ci95_lower']:+.7f},"
            f"{r['ci95_upper']:+.7f}] "
            f"p={r['p_raw']:.7g} "
            f"Holm40={v['p_holm_global_40']:.7g} "
            f"Holm8={secondary:.7g}"
        )

    report_path = OUT / "interaction_report_v2.txt"

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "S12C U3-D11 Finalized Run Manifest V2",
        "status": "COMPLETED",
        "finalization_only": True,
        "statistics_recomputed": False,
        "source_sha256": {
            "predictions": sha(PRED),
            "d10_manifest": sha(d10_manifest_path),
            "d10_protocol": sha(d10_protocol_path),
            "d11_protocol": sha(protocol_path),
            "d11_original_results": sha(result_path),
        },
        "outputs": {
            name: sha(OUT / name)
            for name in [
                "protocol_v1.json",
                "interaction_results_v2.json",
                "interaction_results_v2.csv",
                "interaction_report_v2.txt",
            ]
        },
        "total_tests": 40,
        "global_holm_family": 40,
        "mrr_holm_family": 8,
        "r5_holm_family": 8,
        "paired_dialogues": 413,
    }

    manifest_path = OUT / "run_manifest_v2.json"
    write_json(manifest_path, manifest)

    # Final output integrity verification.
    check = read_json(manifest_path)

    for name, expected in check["outputs"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError("V2_OUTPUT_HASH_FAILURE: " + name)

    print("\n" + "=" * 75)
    print("U3_D11_FINALIZATION_V2_COMPLETE")
    print("SOURCE_INTEGRITY: PASS")
    print("TOTAL_TESTS:", len(records))
    print("GLOBAL_HOLM_40: PASS")
    print("MRR_HOLM_8: PASS")
    print("R5_HOLM_8: PASS")
    print("CSV_EXPORT: PASS")
    print("REPORT_EXPORT: PASS")
    print("MANIFEST_INTEGRITY: PASS")
    print("STATISTICS_RECOMPUTED: NO")
    print("OUTPUT:", OUT)
    print("=" * 75)


if __name__ == "__main__":
    main()
