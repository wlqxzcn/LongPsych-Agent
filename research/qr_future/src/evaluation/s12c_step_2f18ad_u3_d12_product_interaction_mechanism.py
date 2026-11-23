import json
import csv
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

OUT = PHASE / "step_2f18ad_u3_d12_product_interaction_mechanism_v1"
PROTOCOL = OUT / "protocol_v1.json"

PRED = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_predictions_v1.npz"
SUMMARY = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_summary_v1.json"

D11 = PHASE / "step_2f18ac_u3_d11_paired_interaction_audit_v2"
D11_MANIFEST = D11 / "run_manifest_v2.json"
D11_RESULT = D11 / "interaction_results_v2.json"

EXPECTED_PROTOCOL_SHA = (
    "321b689c035ec69b13d63f0f875d94d67a221c7f50452f0317e1b4f5992a7097"
)

CONDITIONS = [
    "QR_FUTURE_FULL",
    "R_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
    "R_FUTURE_NO_PRODUCT",
]

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
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


def holm(pvalues):
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


def check_close(a, b, label, tolerance=1e-9):
    if not np.isclose(a, b, atol=tolerance, rtol=1e-9):
        raise RuntimeError(
            f"POINT_ESTIMATE_MISMATCH {label}: {a} vs {b}"
        )


def main():

    print("=" * 78)
    print("U3-D12 PRODUCT INTERACTION MECHANISM EVALUATION")
    print("=" * 78)

    # --------------------------------------------------
    # 1. Source integrity
    # --------------------------------------------------

    for p in [PROTOCOL, PRED, SUMMARY, D11_MANIFEST, D11_RESULT]:
        if not p.exists():
            raise FileNotFoundError(p)

    protocol_hash = sha(PROTOCOL)

    if protocol_hash != EXPECTED_PROTOCOL_SHA:
        raise RuntimeError("FROZEN_PROTOCOL_SHA_MISMATCH")

    protocol = read_json(PROTOCOL)
    d11_manifest = read_json(D11_MANIFEST)
    d11_result = read_json(D11_RESULT)

    if protocol["status"] != "FROZEN_BEFORE_D12_INFERENCE":
        raise RuntimeError("PROTOCOL_STATUS_FAILURE")

    if protocol["interpretation"] != "EXPLORATORY_POST_HOC":
        raise RuntimeError("INTERPRETATION_POLICY_MISMATCH")

    if d11_manifest["status"] != "COMPLETED":
        raise RuntimeError("D11_NOT_COMPLETED")

    for name, expected in d11_manifest["outputs"].items():
        p = D11 / name
        if not p.exists() or sha(p) != expected:
            raise RuntimeError("D11_OUTPUT_HASH_FAILURE: " + name)

    source_paths = {
        "test_predictions": PRED,
        "test_summary": SUMMARY,
        "d11_manifest": D11_MANIFEST,
        "d11_results": D11_RESULT,
    }

    for name, path in source_paths.items():
        if sha(path) != protocol["source_sha256"][name]:
            raise RuntimeError("SOURCE_SHA_FAILURE: " + name)

    print("FROZEN_PROTOCOL_HASH_PASS")
    print("D11_SOURCE_INTEGRITY_PASS")
    print("TEST_SOURCE_INTEGRITY_PASS")

    # --------------------------------------------------
    # 2. Load frozen ranking predictions
    # --------------------------------------------------

    with np.load(PRED, allow_pickle=False) as z:

        dialogue_ids = z["dialogue_ids"].astype(str)
        long_indices = z["test_long20_indices"].astype(np.int64)

        if len(dialogue_ids) != 18073:
            raise RuntimeError("TEST_QUERY_COUNT_MISMATCH")

        if len(np.unique(dialogue_ids)) != 477:
            raise RuntimeError("TEST_DIALOGUE_COUNT_MISMATCH")

        if len(long_indices) != 8727:
            raise RuntimeError("LONG20_COUNT_MISMATCH")

        if len(np.unique(long_indices)) != len(long_indices):
            raise RuntimeError("LONG20_DUPLICATE_INDICES")

        if np.any(long_indices < 0) or np.any(long_indices >= len(dialogue_ids)):
            raise RuntimeError("LONG20_INDEX_OUT_OF_RANGE")

        ranks = {}

        for condition in CONDITIONS:
            key = "ranks__" + condition

            if key not in z.files:
                raise RuntimeError("MISSING_RANK_ARRAY: " + key)

            value = np.asarray(z[key], dtype=np.int32)

            if value.shape != (len(dialogue_ids),):
                raise RuntimeError("RANK_SHAPE_MISMATCH: " + key)

            if np.any(value < 1) or np.any(value > 101):
                raise RuntimeError("RANK_RANGE_FAILURE: " + key)

            ranks[condition] = value.copy()

    long_mask = np.zeros(len(dialogue_ids), dtype=bool)
    long_mask[long_indices] = True

    if int(long_mask.sum()) != 8727:
        raise RuntimeError("LONG20_MASK_FAILURE")

    nonlong_count = int((~long_mask).sum())

    if nonlong_count != 9346:
        raise RuntimeError("NONLONG_COUNT_MISMATCH")

    print("TEST_PREDICTION_ALIGNMENT_PASS")
    print("TEST_QUERIES:", len(dialogue_ids))
    print("LONG20_QUERIES:", int(long_mask.sum()))
    print("NONLONG20_QUERIES:", nonlong_count)

    # --------------------------------------------------
    # 3. Query-level metrics
    # --------------------------------------------------

    query_metrics = {
        name: metric_matrix(value)
        for name, value in ranks.items()
    }

    # Each dialogue contributes one mean per condition
    # within Long20 and NonLong20.

    unique_dialogues = np.unique(dialogue_ids)

    long_means = {name: [] for name in CONDITIONS}
    nonlong_means = {name: [] for name in CONDITIONS}

    paired_dialogue_ids = []

    for did in unique_dialogues:

        dialogue_mask = dialogue_ids == did

        li = dialogue_mask & long_mask
        ni = dialogue_mask & (~long_mask)

        if not li.any() or not ni.any():
            continue

        paired_dialogue_ids.append(str(did))

        for condition in CONDITIONS:

            long_means[condition].append(
                query_metrics[condition][li].mean(axis=0)
            )

            nonlong_means[condition].append(
                query_metrics[condition][ni].mean(axis=0)
            )

    long_means = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in long_means.items()
    }

    nonlong_means = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in nonlong_means.items()
    }

    n_dialogues = len(paired_dialogue_ids)

    if n_dialogues != 413:
        raise RuntimeError(
            f"PAIRED_DIALOGUE_COUNT_MISMATCH: {n_dialogues}"
        )

    for condition in CONDITIONS:
        if long_means[condition].shape != (413, 5):
            raise RuntimeError("LONG_GROUP_SHAPE_FAILURE")

        if nonlong_means[condition].shape != (413, 5):
            raise RuntimeError("NONLONG_GROUP_SHAPE_FAILURE")

    print("PAIRED_DIALOGUE_AGGREGATION_PASS")
    print("PAIRED_DIALOGUES:", n_dialogues)

    # --------------------------------------------------
    # 4. Calculate three-way interaction
    # --------------------------------------------------

    # Full:
    # (QR_Full - R_Full)_Long
    # - (QR_Full - R_Full)_NonLong

    full_interaction = (
        long_means["QR_FUTURE_FULL"]
        - long_means["R_FUTURE_FULL"]
        - nonlong_means["QR_FUTURE_FULL"]
        + nonlong_means["R_FUTURE_FULL"]
    )

    # No Product:
    # (QR_NP - R_NP)_Long
    # - (QR_NP - R_NP)_NonLong

    no_product_interaction = (
        long_means["QR_FUTURE_NO_PRODUCT"]
        - long_means["R_FUTURE_NO_PRODUCT"]
        - nonlong_means["QR_FUTURE_NO_PRODUCT"]
        + nonlong_means["R_FUTURE_NO_PRODUCT"]
    )

    # Primary D12 estimand:
    # Full interaction - No-Product interaction.

    product_increment = (
        full_interaction - no_product_interaction
    )

    if not np.isfinite(product_increment).all():
        raise RuntimeError("NONFINITE_INTERACTION")

    # --------------------------------------------------
    # 5. Independently reproduce D11 point estimates
    # --------------------------------------------------

    d11_contrasts = d11_result["contrasts"]

    references = {
        "FULL": (
            "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
            full_interaction
        ),
        "NO_PRODUCT": (
            "QR_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_NO_PRODUCT",
            no_product_interaction
        ),
    }

    for label, (contrast, computed) in references.items():

        if contrast not in d11_contrasts:
            raise RuntimeError("D11_CONTRAST_MISSING: " + contrast)

        original = d11_contrasts[contrast][
            "interaction_long_minus_nonlong"
        ]

        for j, metric in enumerate(METRICS):
            check_close(
                computed.mean(axis=0)[j],
                original[metric]["delta"],
                f"{label}/{metric}"
            )

    print("D11_FULL_POINT_ESTIMATE_REPRODUCTION_PASS")
    print("D11_NO_PRODUCT_POINT_ESTIMATE_REPRODUCTION_PASS")

    # --------------------------------------------------
    # 6. Paired dialogue cluster bootstrap
    # --------------------------------------------------

    N_BOOT = protocol["statistics"]["bootstrap"]["replicates"]
    N_PERM = protocol["statistics"]["permutation"]["replicates"]
    SEED = protocol["statistics"]["seed"]

    if N_BOOT != 5000 or N_PERM != 20000:
        raise RuntimeError("STATISTICAL_PROTOCOL_MISMATCH")

    rng = np.random.default_rng(SEED)

    bootstrap_means = np.empty(
        (N_BOOT, len(METRICS)),
        dtype=np.float64
    )

    BATCH = 256

    for start in range(0, N_BOOT, BATCH):

        end = min(start + BATCH, N_BOOT)

        sample_indices = rng.integers(
            0,
            n_dialogues,
            size=(end - start, n_dialogues)
        )

        bootstrap_means[start:end] = (
            product_increment[sample_indices].mean(axis=1)
        )

        if end % 1024 == 0 or end == N_BOOT:
            print(f"BOOTSTRAP: {end}/{N_BOOT}", flush=True)

    point = product_increment.mean(axis=0)

    lower, upper = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
        axis=0
    )

    print("BOOTSTRAP_COMPLETE")

    # --------------------------------------------------
    # 7. Paired dialogue sign-flip permutation
    # --------------------------------------------------

    observed = np.abs(point)
    exceed = np.zeros(len(METRICS), dtype=np.int64)

    completed = 0
    PERM_BATCH = 512

    while completed < N_PERM:

        batch = min(PERM_BATCH, N_PERM - completed)

        signs = (
            rng.integers(
                0, 2,
                size=(batch, n_dialogues),
                dtype=np.int8
            ) * 2 - 1
        ).astype(np.float64)

        permuted = signs @ product_increment / n_dialogues

        exceed += np.sum(
            np.abs(permuted) >= observed[None, :] - 1e-15,
            axis=0
        )

        completed += batch

        if completed % 4096 == 0 or completed == N_PERM:
            print(f"PERMUTATION: {completed}/{N_PERM}", flush=True)

    p_raw = (exceed + 1.0) / (N_PERM + 1.0)

    print("PERMUTATION_COMPLETE")

    # --------------------------------------------------
    # 8. Holm correction across five primary metrics
    # --------------------------------------------------

    p_holm = holm(p_raw)

    if len(p_holm) != 5:
        raise RuntimeError("HOLM_FAMILY_SIZE_FAILURE")

    # --------------------------------------------------
    # 9. Build result JSON
    # --------------------------------------------------

    results = {}

    full_point = full_interaction.mean(axis=0)
    no_product_point = no_product_interaction.mean(axis=0)

    for j, metric in enumerate(METRICS):

        results[metric] = {
            "full_long_specific_effect": float(full_point[j]),
            "no_product_long_specific_effect": float(no_product_point[j]),
            "product_interaction_increment": float(point[j]),
            "ci95_lower": float(lower[j]),
            "ci95_upper": float(upper[j]),
            "p_raw": float(p_raw[j]),
            "p_holm_5": float(p_holm[j]),
            "ci_excludes_zero": bool(
                lower[j] > 0 or upper[j] < 0
            ),
            "significant_after_holm_005": bool(
                p_holm[j] < 0.05
            ),
        }

    output = {
        "artifact": "S12C U3-D12 Product Interaction Mechanism Results V1",
        "status": "COMPLETED",
        "interpretation": "EXPLORATORY_POST_HOC",
        "estimand": protocol["estimand"],
        "formula": protocol["formula"],
        "n_paired_dialogues": n_dialogues,
        "n_test_queries": len(dialogue_ids),
        "n_long20_queries": int(long_mask.sum()),
        "n_nonlong20_queries": nonlong_count,
        "aggregation": "unweighted_mean_of_dialogue_level_query_means",
        "statistics": {
            "bootstrap": N_BOOT,
            "permutation": N_PERM,
            "seed": SEED,
            "permutation_method": "paired_dialogue_sign_flip",
            "alternative": "two-sided",
            "multiple_testing": "Holm",
            "holm_family_size": 5,
            "confidence_level": 0.95,
        },
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "d11_point_estimate_reproduction": "PASS",
        "results": results,
    }

    # --------------------------------------------------
    # 10. Export artifacts
    # --------------------------------------------------

    result_path = OUT / "d12_results_v1.json"
    csv_path = OUT / "d12_results_v1.csv"
    report_path = OUT / "d12_report_v1.txt"
    manifest_path = OUT / "run_manifest_v1.json"

    for p in [result_path, csv_path, report_path, manifest_path]:
        if p.exists():
            raise RuntimeError("OUTPUT_COLLISION: " + str(p))

    write_json(result_path, output)

    fields = [
        "metric",
        "full_long_specific_effect",
        "no_product_long_specific_effect",
        "product_interaction_increment",
        "ci95_lower",
        "ci95_upper",
        "p_raw",
        "p_holm_5",
        "ci_excludes_zero",
        "significant_after_holm_005",
    ]

    with csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for metric in METRICS:
            writer.writerow({
                "metric": metric,
                **results[metric]
            })

    lines = [
        "S12C U3-D12 PRODUCT INTERACTION MECHANISM",
        "=" * 85,
        "Interpretation: EXPLORATORY_POST_HOC",
        "Paired dialogue-level three-way interaction",
        f"Paired dialogues: {n_dialogues}",
        f"Bootstrap: {N_BOOT}",
        f"Permutation: {N_PERM}",
        "Permutation: paired dialogue sign-flip, two-sided",
        "Multiple testing: Holm-5",
        "Training: NO",
        "New ranking inference: NO",
        "",
        "PRIMARY RESULTS",
        "-" * 85,
    ]

    for metric in METRICS:

        v = results[metric]

        lines.append(
            f"{metric}: "
            f"Full={v['full_long_specific_effect']:+.7f} | "
            f"NoProduct={v['no_product_long_specific_effect']:+.7f} | "
            f"D12={v['product_interaction_increment']:+.7f} | "
            f"CI95=[{v['ci95_lower']:+.7f},"
            f"{v['ci95_upper']:+.7f}] | "
            f"p={v['p_raw']:.7g} | "
            f"Holm5={v['p_holm_5']:.7g}"
        )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "S12C U3-D12 Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": protocol_hash,
        "evaluator_sha256": sha(Path(__file__)),
        "source_sha256": {
            name: sha(path)
            for name, path in source_paths.items()
        },
        "outputs": {
            name: sha(OUT / name)
            for name in [
                "d12_results_v1.json",
                "d12_results_v1.csv",
                "d12_report_v1.txt",
            ]
        },
        "paired_dialogues": n_dialogues,
        "total_metrics": len(METRICS),
        "bootstrap_replicates": N_BOOT,
        "permutations": N_PERM,
        "holm_family_size": 5,
        "training": False,
        "new_ranking_inference": False,
    }

    write_json(manifest_path, manifest)

    # Final hash verification
    for name, expected in manifest["outputs"].items():
        if sha(OUT / name) != expected:
            raise RuntimeError("FINAL_OUTPUT_HASH_FAILURE: " + name)

    print("\n" + "=" * 78)
    print("U3_D12_PRODUCT_INTERACTION_MECHANISM_COMPLETE")
    print("SOURCE_INTEGRITY: PASS")
    print("PAIRED_DIALOGUES:", n_dialogues)
    print("D11_POINT_ESTIMATE_REPRODUCTION: PASS")
    print("BOOTSTRAP:", N_BOOT)
    print("PERMUTATION:", N_PERM)
    print("HOLM_FAMILY: 5")
    print("RESULT_JSON:", result_path)
    print("RESULT_CSV:", csv_path)
    print("REPORT:", report_path)
    print("MANIFEST:", manifest_path)
    print("OUTPUT_HASH_VERIFICATION: PASS")
    print("=" * 78)


if __name__ == "__main__":
    main()
