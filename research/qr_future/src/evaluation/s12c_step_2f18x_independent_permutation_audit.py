import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

DEV_PRED = PHASE / "step_2f18s_orca_fsar_dev_evaluation_v1/dev_predictions_v1.npz"
DEV_SUM = PHASE / "step_2f18s_orca_fsar_dev_evaluation_v1/dev_summary_v1.json"

TEST_PRED = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_predictions_v1.npz"
TEST_SUM = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1/test_summary_v1.json"

OUT = PHASE / "step_2f18x_independent_permutation_audit_v1"

CONDITIONS = [
    "DENSE_BGE",
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

CONTRASTS = [
    ("RANK_ONLY_FULL_MATCHED", "DENSE_BGE"),
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("RANK_ONLY_NO_PRODUCT_MATCHED", "DENSE_BGE"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
    ("R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL"),
    ("QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL"),
]

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]

N_PERM = 20000
SEED = 20261119
BATCH = 2000


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metric_matrix(ranks):
    ranks = np.asarray(ranks, dtype=np.int32)

    if np.any(ranks < 1) or np.any(ranks > 101):
        raise RuntimeError("Invalid rank values")

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


def grouped_metrics(ranks, dialogue_ids, indices):

    did = dialogue_ids[indices].astype(str)

    unique, inverse = np.unique(
        did, return_inverse=True
    )

    counts = np.bincount(
        inverse, minlength=len(unique)
    ).astype(np.float64)

    grouped = {}

    for name, values in ranks.items():

        matrix = metric_matrix(values[indices])

        sums = np.zeros(
            (len(unique), len(METRICS)),
            dtype=np.float64
        )

        np.add.at(sums, inverse, matrix)

        grouped[name] = sums / counts[:, None]

    return unique, grouped


def permutation_test(delta, rng):

    n_dialogues = delta.shape[0]

    observed = delta.mean(axis=0)

    exceed = np.zeros(
        len(METRICS), dtype=np.int64
    )

    completed = 0

    while completed < N_PERM:

        batch = min(BATCH, N_PERM - completed)

        signs = rng.integers(
            0, 2,
            size=(batch, n_dialogues),
            dtype=np.int8
        )

        signs = signs.astype(np.float64) * 2.0 - 1.0

        null = signs @ delta / n_dialogues

        exceed += np.sum(
            np.abs(null) >= np.abs(observed)[None, :],
            axis=0
        )

        completed += batch

    pvalues = (exceed + 1.0) / (N_PERM + 1.0)

    return observed, pvalues


def load_dataset(name, pred_path, summary_path):

    summary = json.loads(
        summary_path.read_text(encoding="utf-8")
    )

    with np.load(pred_path, allow_pickle=False) as z:

        dialogue_ids = np.asarray(
            z["dialogue_ids"]
        ).astype(str)

        match = np.asarray(
            z["gold_match_idx"], dtype=np.int32
        )

        ranks = {
            c: np.asarray(
                z[f"ranks__{c}"], dtype=np.int32
            )
            for c in CONDITIONS
        }

        n = len(dialogue_ids)

        if name == "DEV":

            long_mask = np.asarray(
                z["dev_long20_mask"], dtype=bool
            )

            subsets = {
                "DEV_ALL": np.arange(n),
                "DEV_LONG20": np.flatnonzero(long_mask),
                "DEV_GOLD_MATCHED": np.flatnonzero(match >= 0),
                "DEV_GOLD_UNMATCHED": np.flatnonzero(match < 0),
            }

        else:

            long_idx = np.asarray(
                z["test_long20_indices"], dtype=np.int64
            )

            subsets = {
                "TEST_ALL": np.arange(n),
                "TEST_LONG20": long_idx,
                "TEST_GOLD_MATCHED": np.flatnonzero(match >= 0),
                "TEST_GOLD_UNMATCHED": np.flatnonzero(match < 0),
            }

    if len(match) != n:
        raise RuntimeError("Prediction alignment error")

    for c, r in ranks.items():
        if r.shape != (n,):
            raise RuntimeError(f"Rank shape mismatch: {c}")

    return dialogue_ids, ranks, subsets, summary


def run():

    OUT.mkdir(parents=True, exist_ok=True)

    output_files = [
        OUT / "independent_permutation_results_v1.json",
        OUT / "independent_permutation_report_v1.txt",
        OUT / "run_manifest.json",
    ]

    if any(p.exists() for p in output_files):
        raise RuntimeError(
            "Output collision. Refusing to overwrite existing audit."
        )

    datasets = {}

    records = []

    sources = {
        "DEV": (DEV_PRED, DEV_SUM),
        "TEST": (TEST_PRED, TEST_SUM),
    }

    for dataset_id, (pred_path, summary_path) in sources.items():

        print("\n" + "=" * 75)
        print(dataset_id, "INDEPENDENT PERMUTATION AUDIT")
        print("=" * 75)

        dids, ranks, subsets, summary = load_dataset(
            dataset_id, pred_path, summary_path
        )

        for si, (subset_name, indices) in enumerate(subsets.items()):

            unique, grouped = grouped_metrics(
                ranks, dids, indices
            )

            if len(unique) == 0:
                raise RuntimeError("Empty dialogue subset")

            expected = summary["datasets"][subset_name][
                "paired_contrasts"
            ]

            subset_result = {
                "n_queries": int(len(indices)),
                "n_dialogues": int(len(unique)),
                "contrasts": {},
            }

            print(
                f"\n{subset_name}: "
                f"queries={len(indices)}, "
                f"dialogues={len(unique)}"
            )

            for ci, (a, b) in enumerate(CONTRASTS):

                key = f"{a}_MINUS_{b}"

                delta = grouped[a] - grouped[b]

                rng = np.random.default_rng(
                    SEED + si * 10000 + ci * 100
                    + (0 if dataset_id == "DEV" else 50000)
                )

                observed, pvalues = permutation_test(
                    delta, rng
                )

                original = expected[key]

                contrast_result = {}

                for j, metric in enumerate(METRICS):

                    old_delta = float(
                        original[metric]["delta"]
                    )

                    if abs(observed[j] - old_delta) > 1e-10:
                        raise RuntimeError(
                            f"Point estimate mismatch: "
                            f"{subset_name}/{key}/{metric}: "
                            f"{observed[j]} vs {old_delta}"
                        )

                    record = {
                        "dataset": dataset_id,
                        "subset": subset_name,
                        "contrast": key,
                        "metric": metric,
                        "delta": float(observed[j]),
                        "p_raw": float(pvalues[j]),
                    }

                    records.append(record)

                    contrast_result[metric] = {
                        "delta": float(observed[j]),
                        "p_raw": float(pvalues[j]),
                        "bootstrap_ci95_lower": float(
                            original[metric]["ci95_lower"]
                        ),
                        "bootstrap_ci95_upper": float(
                            original[metric]["ci95_upper"]
                        ),
                    }

                subset_result["contrasts"][key] = contrast_result

            datasets[subset_name] = subset_result

            print(
                "POINT_ESTIMATE_REPRODUCTION = PASS"
            )

    # Global Holm correction: all 200 tests.
    global_adjusted = holm_adjust([
        r["p_raw"] for r in records
    ])

    for r, adjusted in zip(records, global_adjusted):
        r["p_holm_global_200"] = float(adjusted)

    # Secondary correction: ten contrasts within
    # each subset-metric family.
    families = {}

    for i, r in enumerate(records):
        key = (r["subset"], r["metric"])
        families.setdefault(key, []).append(i)

    for family, indices in families.items():

        adjusted = holm_adjust([
            records[i]["p_raw"] for i in indices
        ])

        for i, value in zip(indices, adjusted):
            records[i]["p_holm_subset_metric_10"] = float(value)

    # Assemble corrected results.
    for r in records:

        target = datasets[r["subset"]]["contrasts"][
            r["contrast"]
        ][r["metric"]]

        target["p_holm_global_200"] = r[
            "p_holm_global_200"
        ]

        target["p_holm_subset_metric_10"] = r[
            "p_holm_subset_metric_10"
        ]

    result = {
        "artifact": "S12C Independent Dialogue-Level Permutation Audit V1",
        "status": "COMPLETED",
        "method": "paired_dialogue_level_sign_flip_permutation",
        "alternative": "two_sided",
        "permutations": N_PERM,
        "seed": SEED,
        "metrics": METRICS,
        "contrast_count": len(CONTRASTS),
        "total_tests": len(records),
        "multiple_testing": {
            "primary": "Holm-Bonferroni across all 200 tests",
            "secondary": "Holm-Bonferroni within each subset-metric family of 10 tests",
        },
        "source_sha256": {
            "dev_predictions": sha(DEV_PRED),
            "dev_summary": sha(DEV_SUM),
            "test_predictions": sha(TEST_PRED),
            "test_summary": sha(TEST_SUM),
        },
        "datasets": datasets,
    }

    json_path = output_files[0]

    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    lines = [
        "S12C INDEPENDENT PERMUTATION AUDIT",
        f"Permutations: {N_PERM}",
        f"Seed: {SEED}",
        "Primary Holm family: 200 tests",
        "Secondary Holm family: 10 tests per subset-metric",
        "",
    ]

    for subset, data in datasets.items():

        lines.append("=" * 85)
        lines.append(
            f"{subset}: "
            f"queries={data['n_queries']}, "
            f"dialogues={data['n_dialogues']}"
        )
        lines.append("=" * 85)

        for contrast, metrics in data["contrasts"].items():

            lines.append(contrast)

            for metric in ("MRR", "R@5"):

                v = metrics[metric]

                lines.append(
                    f"  {metric}: delta={v['delta']:+.7f} "
                    f"bootstrap_CI="
                    f"[{v['bootstrap_ci95_lower']:+.7f},"
                    f"{v['bootstrap_ci95_upper']:+.7f}] "
                    f"p={v['p_raw']:.7g} "
                    f"Holm200={v['p_holm_global_200']:.7g} "
                    f"Holm10={v['p_holm_subset_metric_10']:.7g}"
                )

    report_path = output_files[1]

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "S12C Independent Permutation Audit Manifest V1",
        "status": "COMPLETED",
        "result_sha256": sha(json_path),
        "report_sha256": sha(report_path),
        "source_sha256": result["source_sha256"],
        "total_tests": len(records),
        "permutations": N_PERM,
    }

    output_files[2].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("U3_D2_INDEPENDENT_PERMUTATION_COMPLETE")
    print("TOTAL TESTS:", len(records))
    print("GLOBAL HOLM FAMILY:", len(records))
    print("POINT ESTIMATE REPRODUCTION: PASS")
    print("RESULT:", json_path)
    print("REPORT:", report_path)
    print("MANIFEST:", output_files[2])
    print("=" * 75)


if __name__ == "__main__":
    run()
