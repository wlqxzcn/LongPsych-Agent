#!/usr/bin/env python3
from pathlib import Path
import csv
import json
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

NPZ = ROOT / "results/phase_s11/s11c_full_score_decomposition_predictions_v1.npz"

OUT = ROOT / "results/phase_s12"
CSV_OUT = OUT / "s12a_component_ablation_metrics_v1.csv"
JSON_OUT = OUT / "s12a_component_ablation_summary_v1.json"
REPORT_OUT = OUT / "s12a_component_ablation_report_v1.txt"

CONDITIONS = [
    "RANK_ONLY",
    "R_FUTURE_BETA8",
    "QR_FUTURE_BETA8",
]

VARIANTS = [
    "RESPONSE_ONLY",
    "ABS_DIFF_ONLY",
    "PRODUCT_ONLY",
    "INTERACTIONS_ONLY",
    "RESPONSE_PLUS_ABS_DIFF",
    "RESPONSE_PLUS_PRODUCT",
    "FULL_RECONSTRUCTED",
]

METRICS = ["MRR", "R@5", "R@100", "MEAN_RANK"]

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261102


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_values(ranks, metric):
    ranks = ranks.astype(np.float64)

    if metric == "MRR":
        return 1.0 / ranks

    if metric == "R@5":
        return (ranks <= 5).astype(np.float64)

    if metric == "R@100":
        return (ranks <= 100).astype(np.float64)

    if metric == "MEAN_RANK":
        return ranks

    raise ValueError(metric)


def bootstrap_group_mean(values, group_index, draws, n_groups):

    group_sum = np.bincount(
        group_index,
        weights=values,
        minlength=n_groups,
    )

    group_count = np.bincount(
        group_index,
        minlength=n_groups,
    ).astype(np.float64)

    numerator = group_sum[draws].sum(axis=1)
    denominator = group_count[draws].sum(axis=1)

    return numerator / denominator


def main():

    if not NPZ.exists():
        raise FileNotFoundError(NPZ)

    for path in [CSV_OUT, JSON_OUT, REPORT_OUT]:
        if path.exists():
            raise FileExistsError(
                f"Output already exists; inspect before rerunning: {path}"
            )

    print("S12A COMPONENT ABLATION REANALYSIS")
    print("INPUT =", NPZ)
    print("INPUT_SHA256 =", sha256_file(NPZ))

    z = np.load(NPZ, allow_pickle=False)

    required = [
        "sample_ids",
        "dialogue_ids",
        "dev_long20_mask",
    ]

    for key in required:
        if key not in z.files:
            raise KeyError(key)

    sample_ids = z["sample_ids"]
    dialogue_ids = z["dialogue_ids"].astype(str)
    long_mask = z["dev_long20_mask"].astype(bool)

    n = len(sample_ids)

    assert dialogue_ids.shape == (n,)
    assert long_mask.shape == (n,)
    assert len(np.unique(sample_ids)) == n

    print("QUERY_COUNT =", n)
    print("DIALOGUE_COUNT =", len(np.unique(dialogue_ids)))
    print("DEV_LONG20_COUNT =", int(long_mask.sum()))

    subsets = {
        "dev_all": np.ones(n, dtype=bool),
        "dev_long20": long_mask,
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []

    report = []

    for subset_name, mask in subsets.items():

        subset_ids = dialogue_ids[mask]

        unique_dialogues, group_index = np.unique(
            subset_ids,
            return_inverse=True,
        )

        n_groups = len(unique_dialogues)
        n_queries = int(mask.sum())

        draws = rng.integers(
            0,
            n_groups,
            size=(BOOTSTRAP_REPS, n_groups),
        )

        print("\n" + "=" * 78)
        print("SUBSET =", subset_name)
        print("QUERIES =", n_queries)
        print("DIALOGUES =", n_groups)

        report.append(
            f"\nSUBSET={subset_name} "
            f"N={n_queries} DIALOGUES={n_groups}"
        )

        for condition in CONDITIONS:

            ref_key = (
                f"ranks__{condition}__S8_FULL_REFERENCE"
            )

            if ref_key not in z.files:
                raise KeyError(ref_key)

            reference = z[ref_key][mask].astype(np.int64)

            assert np.all(reference >= 1)

            print("\nCONDITION =", condition)

            report.append("\nCONDITION=" + condition)

            for variant in VARIANTS:

                key = f"ranks__{condition}__{variant}"

                if key not in z.files:
                    raise KeyError(key)

                ranks = z[key][mask].astype(np.int64)

                assert ranks.shape == reference.shape
                assert np.all(ranks >= 1)

                ref_median = float(np.median(reference))
                var_median = float(np.median(ranks))

                metric_results = {}

                for metric in METRICS:

                    ref_values = metric_values(
                        reference,
                        metric,
                    )

                    var_values = metric_values(
                        ranks,
                        metric,
                    )

                    ref_point = float(ref_values.mean())
                    var_point = float(var_values.mean())

                    ref_boot = bootstrap_group_mean(
                        ref_values,
                        group_index,
                        draws,
                        n_groups,
                    )

                    var_boot = bootstrap_group_mean(
                        var_values,
                        group_index,
                        draws,
                        n_groups,
                    )

                    if metric == "MEAN_RANK":
                        improvement = ref_point - var_point
                        boot_diff = ref_boot - var_boot
                    else:
                        improvement = var_point - ref_point
                        boot_diff = var_boot - ref_boot

                    ci_low, ci_high = np.quantile(
                        boot_diff,
                        [0.025, 0.975],
                    )

                    result = {
                        "reference": ref_point,
                        "variant": var_point,
                        "improvement": float(improvement),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                    }

                    metric_results[metric] = result

                    rows.append({
                        "subset": subset_name,
                        "condition": condition,
                        "variant": variant,
                        "metric": metric,
                        "n_queries": n_queries,
                        "n_dialogues": n_groups,
                        "reference_value": ref_point,
                        "variant_value": var_point,
                        "improvement": float(improvement),
                        "ci_low": float(ci_low),
                        "ci_high": float(ci_high),
                        "reference_median_rank": ref_median,
                        "variant_median_rank": var_median,
                    })

                line = (
                    f"{variant:<25} "
                    f"MRR={metric_results['MRR']['improvement']:+.8f} "
                    f"CI=[{metric_results['MRR']['ci_low']:+.8f},"
                    f"{metric_results['MRR']['ci_high']:+.8f}] "
                    f"R5={metric_results['R@5']['improvement']:+.8f} "
                    f"R100={metric_results['R@100']['improvement']:+.8f} "
                    f"MeanRank={metric_results['MEAN_RANK']['improvement']:+.3f}"
                )

                print(line)
                report.append(line)

    # Save CSV
    with CSV_OUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "phase": "S12-A",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "baseline": "S8_FULL_REFERENCE",
        "conditions": CONDITIONS,
        "variants": VARIANTS,
        "metrics": METRICS,
        "bootstrap": {
            "method": "PAIRED_DIALOGUE_CLUSTER_BOOTSTRAP",
            "repetitions": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "input_sha256": sha256_file(NPZ),
        "query_count": n,
        "dev_long20_count": int(long_mask.sum()),
        "result_rows": len(rows),
        "training": False,
        "model_inference_performed": False,
        "test_payload_used": False,
        "checkpoint_changes": False,
        "results": rows,
    }

    JSON_OUT.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    REPORT_OUT.write_text(
        "\n".join(report)
        + "\n\nS12A_COMPONENT_ABLATION_COMPLETE\n",
        encoding="utf-8",
    )

    z.close()

    print("\n" + "=" * 78)
    print("S12A_COMPONENT_ABLATION_COMPLETE")
    print("RESULT_ROWS =", len(rows))
    print("BOOTSTRAP_REPS =", BOOTSTRAP_REPS)
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINT_CHANGES = NO")
    print("CSV =", CSV_OUT)
    print("JSON =", JSON_OUT)
    print("REPORT =", REPORT_OUT)


if __name__ == "__main__":
    main()
