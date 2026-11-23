import json
import hashlib
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

PHASE = ROOT / "results/phase_s12c"

PRED = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_predictions_v1.npz"
D10 = PHASE / "step_2f18ab_u3_d10_case_mechanism_audit_v1"
ROWS = PHASE / "external_test_v1/test_all_clean_rows_v1.jsonl"
SUMMARY = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_summary_v1.json"

OUT = PHASE / "step_2f18ac_u3_d11_paired_interaction_audit_v1"

N_BOOT = 5000
N_PERM = 20000
BATCH = 2048
SEED = 20261118

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]

CONTRASTS = [
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
    ("R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL"),
    ("QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL"),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def metric_matrix(ranks):
    ranks = np.asarray(ranks, dtype=np.int32)
    return np.column_stack([
        1.0 / ranks,
        ranks <= 1,
        ranks <= 5,
        ranks <= 10,
        ranks <= 20,
    ]).astype(np.float64)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    m = len(pvalues)

    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )

    result = np.empty(m, dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)

    return result


def validate_d10():

    manifest_path = D10 / "run_manifest.json"
    protocol_path = D10 / "protocol_v1.json"

    if not manifest_path.exists() or not protocol_path.exists():
        raise RuntimeError("D10_ARTIFACT_MISSING")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("status") != "COMPLETED":
        raise RuntimeError("D10_NOT_COMPLETED")

    if sha(protocol_path) != manifest["protocol_sha256"]:
        raise RuntimeError("D10_PROTOCOL_HASH_MISMATCH")

    for name, expected in manifest["outputs"].items():
        path = D10 / name

        if not path.exists() or sha(path) != expected:
            raise RuntimeError("D10_OUTPUT_HASH_FAILURE: " + name)

    expected_inputs = manifest["input_sha256"]

    for name, path in [
        ("predictions", PRED),
        ("clean_rows", ROWS),
        ("test_summary", SUMMARY),
    ]:
        if sha(path) != expected_inputs[name]:
            raise RuntimeError("D10_INPUT_HASH_FAILURE: " + name)

    print("D10_MANIFEST_INTEGRITY_PASS")
    print("D10_OUTPUT_HASH_PASS")


def paired_dialogue_differences(
    delta, dialogue_ids, long_mask
):
    unique, inverse = np.unique(
        dialogue_ids.astype(str),
        return_inverse=True
    )

    long_values = []
    nonlong_values = []
    paired_ids = []

    for d, did in enumerate(unique):

        selected = inverse == d

        lmask = selected & long_mask
        nmask = selected & (~long_mask)

        if not lmask.any() or not nmask.any():
            continue

        long_values.append(
            delta[lmask].mean(axis=0)
        )

        nonlong_values.append(
            delta[nmask].mean(axis=0)
        )

        paired_ids.append(str(did))

    if len(paired_ids) < 2:
        raise RuntimeError("INSUFFICIENT_PAIRED_DIALOGUES")

    long_values = np.asarray(long_values)
    nonlong_values = np.asarray(nonlong_values)

    interaction = long_values - nonlong_values

    return (
        np.asarray(paired_ids),
        long_values,
        nonlong_values,
        interaction
    )


def bootstrap_ci(values, seed):

    n = len(values)
    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, n, size=(N_BOOT, n)
    )

    boot = values[indices].mean(axis=1)

    lower, upper = np.quantile(
        boot, [0.025, 0.975], axis=0
    )

    return lower, upper


def sign_flip_permutation(values, seed):

    n = len(values)
    observed = values.mean(axis=0)

    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(METRICS), dtype=np.int64)

    for start in range(0, N_PERM, BATCH):

        size = min(BATCH, N_PERM - start)

        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(size, n)
        )

        permuted = signs @ values / n

        exceed += (
            np.abs(permuted) >= np.abs(observed)
        ).sum(axis=0)

    pvalues = (exceed + 1.0) / (N_PERM + 1.0)

    return observed, pvalues


def main():

    for path in (PRED, ROWS, SUMMARY):
        if not path.exists():
            raise FileNotFoundError(path)

    if OUT.exists():
        raise RuntimeError(
            "OUTPUT_COLLISION_REFUSING_OVERWRITE: " + str(OUT)
        )

    print("========== D10 INTEGRITY ==========")
    validate_d10()

    print("\n========== LOAD TEST PREDICTIONS ==========")

    with np.load(PRED, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    dialogue_ids = arrays["dialogue_ids"].astype(str)
    sample_ids = arrays["sample_ids"].astype(str)

    n = len(sample_ids)

    if len(np.unique(sample_ids)) != n:
        raise RuntimeError("DUPLICATE_SAMPLE_IDS")

    long_idx = arrays["test_long20_indices"].astype(np.int64)

    if len(np.unique(long_idx)) != len(long_idx):
        raise RuntimeError("DUPLICATE_LONG20_INDICES")

    if np.any(long_idx < 0) or np.any(long_idx >= n):
        raise RuntimeError("LONG20_INDEX_OUT_OF_RANGE")

    long_mask = np.zeros(n, dtype=bool)
    long_mask[long_idx] = True

    unique_dialogues = np.unique(dialogue_ids)

    print("TEST_QUERIES:", n)
    print("TEST_DIALOGUES:", len(unique_dialogues))
    print("LONG20_QUERIES:", int(long_mask.sum()))
    print("NONLONG20_QUERIES:", int((~long_mask).sum()))

    ranks = {}

    for a, b in CONTRASTS:
        for condition in (a, b):

            key = "ranks__" + condition

            if key not in arrays:
                raise RuntimeError("MISSING_RANK_ARRAY: " + key)

            r = arrays[key]

            if r.shape != (n,):
                raise RuntimeError("RANK_SHAPE_MISMATCH")

            if np.any(r < 1) or np.any(r > 101):
                raise RuntimeError("INVALID_RANK")

            ranks[condition] = r

    OUT.mkdir(parents=True)

    protocol = {
        "artifact": "S12C U3-D11 Paired Long Nonlong Interaction Audit V1",
        "status": "FROZEN_BEFORE_INTERACTION_TEST",
        "inputs": {
            "predictions_sha256": sha(PRED),
            "d10_manifest_sha256": sha(D10 / "run_manifest.json"),
            "d10_protocol_sha256": sha(D10 / "protocol_v1.json"),
        },
        "estimand": (
            "Mean within-dialogue difference between "
            "Long20 and NonLong20 treatment effects"
        ),
        "dialogue_pairing": (
            "Only dialogues containing both Long20 and NonLong20 queries"
        ),
        "contrasts": [
            {"A": a, "B": b, "name": a + "_MINUS_" + b}
            for a, b in CONTRASTS
        ],
        "metrics": METRICS,
        "statistics": {
            "bootstrap": N_BOOT,
            "permutations": N_PERM,
            "permutation_method": "paired_dialogue_sign_flip",
            "confidence_level": 0.95,
            "alternative": "two-sided",
            "global_holm_family": 40,
            "primary_mrr_holm_family": 8,
            "secondary_r5_holm_family": 8,
        },
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
    }

    write_json(OUT / "protocol_v1.json", protocol)

    print("\n========== PAIRED DIALOGUE AUDIT ==========")

    records = []
    results = {}

    for ci, (a, b) in enumerate(CONTRASTS):

        contrast = a + "_MINUS_" + b

        delta = (
            metric_matrix(ranks[a])
            - metric_matrix(ranks[b])
        )

        paired_ids, long_values, nonlong_values, interaction = (
            paired_dialogue_differences(
                delta, dialogue_ids, long_mask
            )
        )

        lower, upper = bootstrap_ci(
            interaction, SEED + 1000 + ci
        )

        observed, pvalues = sign_flip_permutation(
            interaction, SEED + 10000 + ci
        )

        long_mean = long_values.mean(axis=0)
        nonlong_mean = nonlong_values.mean(axis=0)

        contrast_result = {
            "n_paired_dialogues": len(paired_ids),
            "long_dialogue_macro_delta": {},
            "nonlong_dialogue_macro_delta": {},
            "interaction_long_minus_nonlong": {},
        }

        for j, metric in enumerate(METRICS):

            contrast_result["long_dialogue_macro_delta"][metric] = (
                float(long_mean[j])
            )

            contrast_result["nonlong_dialogue_macro_delta"][metric] = (
                float(nonlong_mean[j])
            )

            contrast_result["interaction_long_minus_nonlong"][metric] = {
                "delta": float(observed[j]),
                "ci95_lower": float(lower[j]),
                "ci95_upper": float(upper[j]),
                "p_raw": float(pvalues[j]),
            }

            records.append({
                "contrast": contrast,
                "metric": metric,
                "n_paired_dialogues": len(paired_ids),
                "long_delta": float(long_mean[j]),
                "nonlong_delta": float(nonlong_mean[j]),
                "interaction_delta": float(observed[j]),
                "ci95_lower": float(lower[j]),
                "ci95_upper": float(upper[j]),
                "p_raw": float(pvalues[j]),
            })

        results[contrast] = contrast_result

        print(
            contrast,
            "PAIRED_DIALOGUES=", len(paired_ids)
        )

        for metric in ("MRR", "R@5"):

            v = contrast_result[
                "interaction_long_minus_nonlong"
            ][metric]

            print(
                metric,
                "LONG=", round(
                    contrast_result["long_dialogue_macro_delta"][metric],
                    7
                ),
                "NONLONG=", round(
                    contrast_result["nonlong_dialogue_macro_delta"][metric],
                    7
                ),
                "INTERACTION=", round(v["delta"], 7),
                "CI=", (
                    round(v["ci95_lower"], 7),
                    round(v["ci95_upper"], 7)
                ),
                "P=", round(v["p_raw"], 7)
            )

    print("\n========== MULTIPLE TESTING ==========")

    global_adj = holm_adjust([
        x["p_raw"] for x in records
    ])

    for record, adjusted in zip(records, global_adj):
        record["p_holm_global_40"] = float(adjusted)

    for metric in ("MRR", "R@5"):

        selected = [
            x for x in records
            if x["metric"] == metric
        ]

        adjusted = holm_adjust([
            x["p_raw"] for x in selected
        ])

        field = "p_holm_" + metric.lower() + "_8"

        for item, value in zip(selected, adjusted):
            item[field] = float(value)

    for record in records:

        target = results[record["contrast"]][
            "interaction_long_minus_nonlong"
        ][record["metric"]]

        target["p_holm_global_40"] = record[
            "p_holm_global_40"
        ]

        if record["metric"] in ("MRR", "R@5"):
            field = "p_holm_" + record["metric"].lower() + "_8"
            target[field] = record[field]

    output = {
        "artifact": "S12C U3-D11 Paired Interaction Results V1",
        "status": "COMPLETED",
        "method": "paired_dialogue_sign_flip_permutation",
        "permutations": N_PERM,
        "bootstrap_replicates": N_BOOT,
        "total_tests": len(records),
        "paired_dialogue_definition": (
            "Dialogue must contain both Long20 and NonLong20 queries"
        ),
        "contrasts": results,
    }

    write_json(
        OUT / "interaction_results_v1.json",
        output
    )

    csv_path = OUT / "interaction_results_v1.csv"

    with csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "S12C U3-D11 PAIRED INTERACTION AUDIT",
        "Method: paired dialogue-level sign-flip permutation",
        f"Permutations: {N_PERM}",
        f"Bootstrap: {N_BOOT}",
        f"Total tests: {len(records)}",
        "",
    ]

    for record in records:

        if record["metric"] not in ("MRR", "R@5"):
            continue

        lines.append(
            f"{record['contrast']} | {record['metric']} | "
            f"delta={record['interaction_delta']:+.7f} "
            f"CI=[{record['ci95_lower']:+.7f},"
            f"{record['ci95_upper']:+.7f}] "
            f"p={record['p_raw']:.7g} "
            f"Holm40={record['p_holm_global_40']:.7g}"
        )

    report = OUT / "interaction_report_v1.txt"
    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "S12C U3-D11 Run Manifest V1",
        "status": "COMPLETED",
        "source_sha256": {
            "predictions": sha(PRED),
            "d10_manifest": sha(D10 / "run_manifest.json"),
            "d10_protocol": sha(D10 / "protocol_v1.json"),
        },
        "protocol_sha256": sha(OUT / "protocol_v1.json"),
        "outputs": {
            name: sha(OUT / name)
            for name in [
                "protocol_v1.json",
                "interaction_results_v1.json",
                "interaction_results_v1.csv",
                "interaction_report_v1.txt",
            ]
        },
        "test_queries": n,
        "test_dialogues": len(unique_dialogues),
        "paired_dialogues": int(
            results[
                CONTRASTS[0][0] + "_MINUS_" + CONTRASTS[0][1]
            ]["n_paired_dialogues"]
        ),
        "total_tests": len(records),
        "training": False,
        "new_ranking_inference": False,
    }

    write_json(OUT / "run_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("U3_D11_PAIRED_INTERACTION_AUDIT_COMPLETE")
    print("D10_INTEGRITY: PASS")
    print("PAIRED_DIALOGUE_TEST: PASS")
    print("TOTAL_TESTS:", len(records))
    print("GLOBAL_HOLM: PASS")
    print("MANIFEST: PASS")
    print("OUTPUT:", OUT)
    print("=" * 80)


if __name__ == "__main__":
    main()
