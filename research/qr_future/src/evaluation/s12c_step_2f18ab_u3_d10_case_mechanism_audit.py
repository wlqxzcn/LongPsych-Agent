import json
import csv
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

PRED = ROOT / "results/phase_s12c/step_2f18t_orca_fsar_test_evaluation_v1/test_predictions_v1.npz"
ROWS = ROOT / "results/phase_s12c/external_test_v1/test_all_clean_rows_v1.jsonl"
SUMMARY = ROOT / "results/phase_s12c/step_2f18t_orca_fsar_test_evaluation_v1/test_summary_v1.json"

OUT = ROOT / "results/phase_s12c/step_2f18ab_u3_d10_case_mechanism_audit_v1"

N_BOOT = 5000
SEED = 20261118
MAX_CASES = 50

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


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def metric_matrix(ranks):
    ranks = np.asarray(ranks, dtype=np.int32)
    return np.column_stack([
        1.0 / ranks,
        ranks <= 1,
        ranks <= 5,
        ranks <= 10,
        ranks <= 20,
    ]).astype(np.float64)


def dialogue_bootstrap(delta, dialogue_ids, seed):
    unique, inverse = np.unique(
        dialogue_ids.astype(str), return_inverse=True
    )

    sums = np.zeros((len(unique), delta.shape[1]), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique))

    np.add.at(sums, inverse, delta)
    grouped = sums / counts[:, None]

    estimate = grouped.mean(axis=0)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(unique), size=(N_BOOT, len(unique))
    )

    boot = grouped[sampled].mean(axis=1)
    lower, upper = np.quantile(
        boot, [0.025, 0.975], axis=0
    )

    return {
        "n_dialogues": len(unique),
        "estimate": estimate,
        "lower": lower,
        "upper": upper,
    }


def long_nonlong_bootstrap(delta, dialogue_ids, long_mask, seed):
    unique, inverse = np.unique(
        dialogue_ids.astype(str), return_inverse=True
    )

    n_dialogues = len(unique)
    n_metrics = delta.shape[1]

    def grouped_subset(mask):
        grouped = np.full(
            (n_dialogues, n_metrics),
            np.nan,
            dtype=np.float64
        )

        for d in range(n_dialogues):
            selected = mask & (inverse == d)
            if selected.any():
                grouped[d] = delta[selected].mean(axis=0)

        return grouped

    long_group = grouped_subset(long_mask)
    nonlong_group = grouped_subset(~long_mask)

    long_estimate = np.nanmean(long_group, axis=0)
    nonlong_estimate = np.nanmean(nonlong_group, axis=0)

    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, n_dialogues, size=(N_BOOT, n_dialogues)
    )

    long_boot = np.nanmean(long_group[sampled], axis=1)
    nonlong_boot = np.nanmean(nonlong_group[sampled], axis=1)

    diff_boot = long_boot - nonlong_boot

    lower, upper = np.nanquantile(
        diff_boot, [0.025, 0.975], axis=0
    )

    return {
        "n_total_dialogues": n_dialogues,
        "n_long_dialogues": int(np.isfinite(long_group[:, 0]).sum()),
        "n_nonlong_dialogues": int(np.isfinite(nonlong_group[:, 0]).sum()),
        "long_dialogue_macro_delta": long_estimate.tolist(),
        "nonlong_dialogue_macro_delta": nonlong_estimate.tolist(),
        "long_minus_nonlong": (
            long_estimate - nonlong_estimate
        ).tolist(),
        "ci95_lower": lower.tolist(),
        "ci95_upper": upper.tolist(),
    }


def top_negatives(i, condition, arrays):
    scores = arrays["candidate_scores__" + condition][i]
    response_rows = arrays["candidate_response_rows"][i]

    order = np.argsort(-scores, kind="stable")[:5]

    return [
        {
            "response_row": int(response_rows[j]),
            "score": float(scores[j])
        }
        for j in order
    ]


def build_case(i, contrast, case_type, a, b,
               rank_a, rank_b, delta_mrr, row, arrays, is_long20):

    return {
        "case_id": f"{row['sample_id']}__{contrast}",
        "sample_id": str(row["sample_id"]),
        "dialogue_id": str(row["dialogue_id"]),
        "contrast": contrast,
        "case_type": case_type,

        "base_model": b,
        "future_model": a,

        "base_rank": int(rank_b),
        "future_rank": int(rank_a),
        "rank_shift_base_minus_future": int(rank_b - rank_a),
        "delta_mrr": float(delta_mrr),

        "is_long20": bool(is_long20),

        "query": row.get("query", ""),
        "gold_answer": row.get("gold_answer", ""),
        "history_pairs": row.get("history_pairs", []),

        "target_pair_index": row.get("target_pair_index"),
        "relative_position": row.get("relative_position"),
        "dialogue_pairs_total": row.get("dialogue_pairs_total"),
        "horizon_bin": row.get("horizon_bin"),

        "base_gold_score": float(
            arrays["gold_scores__" + b][i]
        ),
        "future_gold_score": float(
            arrays["gold_scores__" + a][i]
        ),

        "base_top5_negative_candidates": top_negatives(
            i, b, arrays
        ),
        "future_top5_negative_candidates": top_negatives(
            i, a, arrays
        ),
    }


def main():

    for p in (PRED, ROWS, SUMMARY):
        if not p.exists():
            raise FileNotFoundError(p)

    if OUT.exists():
        raise RuntimeError(
            f"Output collision, refusing overwrite: {OUT}"
        )

    print("========== LOAD INPUTS ==========")

    with np.load(PRED, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    with ROWS.open(encoding="utf-8") as f:
        rows = [
            json.loads(line)
            for line in f if line.strip()
        ]

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    sample_ids = arrays["sample_ids"].astype(str)
    dialogue_ids = arrays["dialogue_ids"].astype(str)

    n = len(sample_ids)

    if len(rows) != n:
        raise RuntimeError(
            f"Row count mismatch: {len(rows)} vs {n}"
        )

    row_ids = np.asarray(
        [str(x["sample_id"]) for x in rows]
    )

    row_dialogues = np.asarray(
        [str(x["dialogue_id"]) for x in rows]
    )

    if not np.array_equal(sample_ids, row_ids):
        raise RuntimeError("SAMPLE_ID_ALIGNMENT_FAILED")

    if not np.array_equal(dialogue_ids, row_dialogues):
        raise RuntimeError("DIALOGUE_ID_ALIGNMENT_FAILED")

    if len(np.unique(sample_ids)) != n:
        raise RuntimeError("DUPLICATE_SAMPLE_ID")

    long_idx = arrays["test_long20_indices"].astype(np.int64)

    if len(np.unique(long_idx)) != len(long_idx):
        raise RuntimeError("DUPLICATE_LONG20_INDEX")

    if np.any(long_idx < 0) or np.any(long_idx >= n):
        raise RuntimeError("LONG20_INDEX_OUT_OF_RANGE")

    long_mask = np.zeros(n, dtype=bool)
    long_mask[long_idx] = True

    if len(long_idx) != summary["test_long20_queries"]:
        raise RuntimeError("LONG20_COUNT_MISMATCH")

    if n != summary["test_queries"]:
        raise RuntimeError("TEST_QUERY_COUNT_MISMATCH")

    print("SAMPLE_ID_ALIGNMENT_PASS")
    print("DIALOGUE_ID_ALIGNMENT_PASS")
    print("LONG20_ALIGNMENT_PASS")
    print("TEST_QUERY_COUNT:", n)
    print("TEST_DIALOGUES:", len(np.unique(dialogue_ids)))
    print("LONG20_QUERIES:", int(long_mask.sum()))
    print("NONLONG20_QUERIES:", int((~long_mask).sum()))

    ranks = {}

    for condition in set(
        x for pair in CONTRASTS for x in pair
    ):
        rank_key = "ranks__" + condition
        score_key = "candidate_scores__" + condition
        gold_key = "gold_scores__" + condition

        for key in (rank_key, score_key, gold_key):
            if key not in arrays:
                raise RuntimeError("MISSING_ARRAY: " + key)

        r = arrays[rank_key]

        if r.shape != (n,):
            raise RuntimeError("RANK_SHAPE_ERROR: " + condition)

        if np.any(r < 1) or np.any(r > 101):
            raise RuntimeError("INVALID_RANK: " + condition)

        if not np.isfinite(arrays[score_key]).all():
            raise RuntimeError("NONFINITE_SCORES: " + condition)

        ranks[condition] = r

    if arrays["candidate_response_rows"].shape != (n, 100):
        raise RuntimeError("CANDIDATE_RESPONSE_SHAPE_ERROR")

    OUT.mkdir(parents=True)

    input_hashes = {
        "predictions": sha(PRED),
        "clean_rows": sha(ROWS),
        "test_summary": sha(SUMMARY),
    }

    protocol = {
        "artifact": "S12C U3-D10 Case Mechanism Audit Protocol V1",
        "status": "FROZEN_BEFORE_CASE_EXTRACTION",
        "input_sha256": input_hashes,
        "test_queries": n,
        "test_dialogues": int(len(np.unique(dialogue_ids))),
        "long20_queries": int(long_mask.sum()),
        "contrasts": [
            {"A": a, "B": b, "name": a + "_MINUS_" + b}
            for a, b in CONTRASTS
        ],
        "subsets": ["TEST_ALL", "TEST_LONG20", "TEST_NONLONG20"],
        "case_selection": {
            "top1_rescue": "base_rank>1 and future_rank==1",
            "top5_rescue": "base_rank>5 and future_rank<=5",
            "top1_regression": "base_rank==1 and future_rank>1",
            "top5_regression": "base_rank>1 and base_rank<=5 and future_rank>5",
            "general_improvement": "future_rank<base_rank and not rescue",
            "general_regression": "future_rank>base_rank and not regression",
            "max_examples_per_contrast_and_type": MAX_CASES,
        },
        "statistics": {
            "point_estimator": "unweighted_mean_of_dialogue_level_query_means",
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": SEED,
            "confidence_level": 0.95,
            "resampling_unit": "dialogue",
        },
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
    }

    write_json(OUT / "protocol_v1.json", protocol)

    print("\n========== CASE EXTRACTION ==========")


    success_buckets = {}
    regression_buckets = {}

    for a, b in CONTRASTS:

        contrast = a + "_MINUS_" + b
        ra = ranks[a]
        rb = ranks[b]

        for i in range(n):

            delta = 1.0 / float(ra[i]) - 1.0 / float(rb[i])

            if ra[i] < rb[i]:

                if rb[i] > 1 and ra[i] == 1:
                    kind = "TOP1_RESCUE"
                elif rb[i] > 5 and ra[i] <= 5:
                    kind = "TOP5_RESCUE"
                else:
                    kind = "RANK_IMPROVEMENT"

                success_buckets.setdefault(
                    (contrast, kind), []
                ).append((float(delta), i))

            elif ra[i] > rb[i]:

                if rb[i] == 1:
                    kind = "TOP1_REGRESSION"
                elif rb[i] <= 5 and ra[i] > 5:
                    kind = "TOP5_REGRESSION"
                else:
                    kind = "RANK_REGRESSION"

                regression_buckets.setdefault(
                    (contrast, kind), []
                ).append((float(delta), i))

    def materialize(buckets, descending):

        output = []

        for (contrast, kind), events in buckets.items():

            events.sort(
                key=lambda x: x[0],
                reverse=descending
            )

            a, b = contrast.split("_MINUS_", 1)

            for delta, i in events[:MAX_CASES]:

                output.append(
                    build_case(
                        i, contrast, kind, a, b,
                        ranks[a][i], ranks[b][i],
                        delta, rows[i], arrays,
                        bool(long_mask[i])
                    )
                )

        return output

    success_records = materialize(
        success_buckets, True
    )

    regression_records = materialize(
        regression_buckets, False
    )

    write_jsonl(
        OUT / "success_cases_v1.jsonl",
        success_records
    )

    write_jsonl(
        OUT / "regression_cases_v1.jsonl",
        regression_records
    )

    qr_records = [
        x for x in success_records + regression_records
        if x["contrast"] in (
            "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
            "QR_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_NO_PRODUCT",
        )
    ]

    write_jsonl(
        OUT / "success_cases_v1.jsonl",
        success_records
    )

    write_jsonl(
        OUT / "regression_cases_v1.jsonl",
        regression_records
    )

    write_jsonl(
        OUT / "qr_specific_cases_v1.jsonl",
        qr_records
    )

    print("SUCCESS_EXAMPLES:", len(success_records))
    print("REGRESSION_EXAMPLES:", len(regression_records))
    print("QR_SPECIFIC_EXAMPLES:", len(qr_records))

    print("\n========== SUBSET STATISTICS ==========")

    subset_masks = {
        "TEST_ALL": np.ones(n, dtype=bool),
        "TEST_LONG20": long_mask,
        "TEST_NONLONG20": ~long_mask,
    }

    statistics = {
        "artifact": "S12C U3-D10 Case Statistics V1",
        "status": "COMPLETED",
        "datasets": {},
    }

    csv_rows = []

    for si, (subset_name, mask) in enumerate(subset_masks.items()):

        idx = np.flatnonzero(mask)
        subset_result = {}

        for ci, (a, b) in enumerate(CONTRASTS):

            contrast = a + "_MINUS_" + b

            delta = (
                metric_matrix(ranks[a][idx])
                - metric_matrix(ranks[b][idx])
            )

            dids = dialogue_ids[idx]

            result = dialogue_bootstrap(
                delta, dids, SEED + ci * 100 + si
            )

            base = ranks[b][idx]
            future = ranks[a][idx]

            counts = {
                "rank_improved": int((future < base).sum()),
                "rank_tied": int((future == base).sum()),
                "rank_regressed": int((future > base).sum()),
                "top1_rescue": int(((base > 1) & (future == 1)).sum()),
                "top5_rescue_including_top1": int(
                    ((base > 5) & (future <= 5)).sum()
                ),
                "top5_rescue_excluding_top1": int(
                    ((base > 5) & (future > 1) & (future <= 5)).sum()
                ),
                "top1_regression": int(
                    ((base == 1) & (future > 1)).sum()
                ),
                "top5_regression_excluding_top1": int(
                    ((base > 1) & (base <= 5) & (future > 5)).sum()
                ),
            }

            metrics = {}

            for j, metric in enumerate(METRICS):

                metrics[metric] = {
                    "delta": float(result["estimate"][j]),
                    "ci95_lower": float(result["lower"][j]),
                    "ci95_upper": float(result["upper"][j]),
                }

                csv_rows.append({
                    "subset": subset_name,
                    "contrast": contrast,
                    "metric": metric,
                    "delta": float(result["estimate"][j]),
                    "ci95_lower": float(result["lower"][j]),
                    "ci95_upper": float(result["upper"][j]),
                    "n_queries": len(idx),
                    "n_dialogues": result["n_dialogues"],
                })

            subset_result[contrast] = {
                "n_queries": len(idx),
                "n_dialogues": result["n_dialogues"],
                "metrics": metrics,
                "query_micro_delta_mrr": float(delta[:, 0].mean()),
                "mean_rank_shift": float((base - future).mean()),
                "positive_mrr_fraction": float(
                    (delta[:, 0] > 0).mean()
                ),
                "counts": counts,
            }

        statistics["datasets"][subset_name] = subset_result

        print(
            subset_name,
            "queries=", len(idx),
            "dialogues=", len(np.unique(dialogue_ids[idx]))
        )

    write_json(
        OUT / "case_statistics_v1.json",
        statistics
    )

    with (OUT / "case_statistics_v1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset", "contrast", "metric", "delta",
                "ci95_lower", "ci95_upper",
                "n_queries", "n_dialogues"
            ]
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n========== LONG VS NONLONG ==========")

    long_results = {}

    for ci, (a, b) in enumerate(CONTRASTS):

        contrast = a + "_MINUS_" + b

        delta = (
            metric_matrix(ranks[a])
            - metric_matrix(ranks[b])
        )

        result = long_nonlong_bootstrap(
            delta,
            dialogue_ids,
            long_mask,
            SEED + 50000 + ci
        )

        long_results[contrast] = result

        print(
            contrast,
            "LONG_MRR=",
            round(result["long_dialogue_macro_delta"][0], 7),
            "NONLONG_MRR=",
            round(result["nonlong_dialogue_macro_delta"][0], 7),
            "DIFF=",
            round(result["long_minus_nonlong"][0], 7),
            "CI=",
            (
                round(result["ci95_lower"][0], 7),
                round(result["ci95_upper"][0], 7)
            )
        )

    write_json(
        OUT / "long_vs_nonlong_statistics_v1.json",
        {
            "artifact": "S12C U3-D10 Long vs Nonlong Statistics V1",
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": N_BOOT,
            "seed": SEED + 50000,
            "resampling_unit": "dialogue",
            "interpretation": "Long20 minus NonLong20 dialogue-macro metric difference",
            "contrasts": long_results,
        }
    )

    print("\n========== FINAL REPORT ==========")

    lines = [
        "S12C U3-D10 CASE MECHANISM AUDIT",
        "Training: NO",
        "New ranking inference: NO",
        f"Test queries: {n}",
        f"Test dialogues: {len(np.unique(dialogue_ids))}",
        f"Long20 queries: {int(long_mask.sum())}",
        f"Bootstrap replicates: {N_BOOT}",
        "",
    ]

    for subset, result in statistics["datasets"].items():

        lines.extend(["=" * 80, subset, "=" * 80])

        for contrast, item in result.items():

            m = item["metrics"]["MRR"]

            lines.append(
                f"{contrast}: "
                f"MRR={m['delta']:+.7f} "
                f"CI=[{m['ci95_lower']:+.7f},"
                f"{m['ci95_upper']:+.7f}] "
                f"improved={item['counts']['rank_improved']} "
                f"regressed={item['counts']['rank_regressed']} "
                f"tied={item['counts']['rank_tied']}"
            )

    (OUT / "final_report_v1.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    output_names = [
        "protocol_v1.json",
        "success_cases_v1.jsonl",
        "regression_cases_v1.jsonl",
        "qr_specific_cases_v1.jsonl",
        "case_statistics_v1.json",
        "case_statistics_v1.csv",
        "long_vs_nonlong_statistics_v1.json",
        "final_report_v1.txt",
    ]

    manifest = {
        "artifact": "S12C U3-D10 Run Manifest V1",
        "status": "COMPLETED",
        "input_sha256": input_hashes,
        "protocol_sha256": sha(OUT / "protocol_v1.json"),
        "outputs": {
            name: sha(OUT / name)
            for name in output_names
        },
        "test_queries": n,
        "test_dialogues": int(len(np.unique(dialogue_ids))),
        "long20_queries": int(long_mask.sum()),
        "contrasts": len(CONTRASTS),
        "bootstrap_replicates": N_BOOT,
        "training": False,
        "new_ranking_inference": False,
    }

    write_json(OUT / "run_manifest.json", manifest)

    print("\n" + "=" * 80)
    print("U3_D10_CASE_MECHANISM_AUDIT_COMPLETE")
    print("INPUT_ALIGNMENT: PASS")
    print("CONTRASTS:", len(CONTRASTS))
    print("BOOTSTRAP:", N_BOOT)
    print("SUCCESS_CASES:", len(success_records))
    print("REGRESSION_CASES:", len(regression_records))
    print("QR_SPECIFIC_CASES:", len(qr_records))
    print("OUTPUT:", OUT)
    print("MANIFEST: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()
