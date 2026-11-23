import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

BASE = ROOT / "results/phase_s12c/step_2f18t_orca_fsar_test_evaluation_v1"
A0 = ROOT / "results/phase_s12c/step_2f18ag_u3_d14_a0_index_rank_audit_v1"

OUT = ROOT / "results/phase_s12c/step_2f18ah_u3_d14_a1_gold_negative_margin_v1"

PRED = BASE / "test_predictions_v1.npz"
SUMMARY = BASE / "test_summary_v1.json"
TEST_PROTOCOL = BASE / "test_evaluation_protocol_v1.json"
TEST_MANIFEST = BASE / "run_manifest.json"

A0_RESULT = A0 / "audit_results_v1.json"
A0_MANIFEST = A0 / "run_manifest_v1.json"
A0_PROTOCOL = A0 / "protocol_v1.json"

PROTOCOL = OUT / "protocol_v1.json"
RESULT = OUT / "margin_results_v1.json"
DESC_CSV = OUT / "margin_descriptive_v1.csv"
CONTRAST_CSV = OUT / "margin_contrasts_v1.csv"
REPORT = OUT / "margin_report_v1.txt"
MANIFEST = OUT / "run_manifest_v1.json"

N = 18073
N_NEG = 100

N_BOOT = 5000
N_PERM = 20000
SEED = 20261119
BATCH = 256

EPS = 1e-8

KS = [1, 5, 10, 20, 100]

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
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
    ("R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL"),
    ("QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL"),
]

INFER_SUBSETS = [
    "TEST_ALL",
    "TEST_LONG20",
    "TEST_NONLONG20",
]

DESC_SUBSETS = [
    "TEST_ALL",
    "TEST_LONG20",
    "TEST_NONLONG20",
    "TEST_GOLD_MATCHED",
    "TEST_GOLD_UNMATCHED",
]

INFER_METRICS = [
    "RAW_MARGIN@1",
    "NORMALIZED_MARGIN@1",
    "NORMALIZED_MARGIN@5",
]

SOURCE_FILES = {
    "predictions": PRED,
    "summary": SUMMARY,
    "test_protocol": TEST_PROTOCOL,
    "test_manifest": TEST_MANIFEST,
    "a0_result": A0_RESULT,
    "a0_manifest": A0_MANIFEST,
    "a0_protocol": A0_PROTOCOL,
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    tmp.replace(path)


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )
    result = np.empty(m, dtype=np.float64)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def prepare():
    if OUT.exists():
        raise RuntimeError(
            f"Output collision, refusing overwrite: {OUT}"
        )

    for name, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    a0 = json.loads(A0_RESULT.read_text(encoding="utf-8"))
    a0m = json.loads(A0_MANIFEST.read_text(encoding="utf-8"))

    if a0.get("status") != "PASS":
        raise RuntimeError("D14-A0 did not PASS")

    if a0.get("failed_checks") != 0:
        raise RuntimeError("D14-A0 contains failed checks")

    if a0m.get("result_sha256") != sha(A0_RESULT):
        raise RuntimeError("D14-A0 result hash mismatch")

    OUT.mkdir(parents=True)

    protocol = {
        "artifact": "S12C U3 D14-A1 Gold Negative Margin Analysis V1",
        "status": "FROZEN_BEFORE_MARGIN_ANALYSIS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            k: sha(v) for k, v in SOURCE_FILES.items()
        },
        "a0_status": "PASS",
        "queries": N,
        "negative_candidates": N_NEG,
        "conditions": CONDITIONS,
        "margin_thresholds": KS,
        "normalization": "candidate_score_sample_std_ddof_1",
        "primary_metric": "NORMALIZED_MARGIN@1",
        "primary_contrast": "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
        "primary_subset": "TEST_LONG20",
        "inference": {
            "method": "paired_dialogue_cluster_bootstrap",
            "permutation": "paired_dialogue_sign_flip_two_sided",
            "bootstrap_replicates": N_BOOT,
            "permutations": N_PERM,
            "seed": SEED,
            "resampling_unit": "dialogue",
            "point_estimator": "unweighted_mean_of_dialogue_level_query_means",
            "global_holm_tests": 72,
            "secondary_family_size": 8,
        },
        "inferential_subsets": INFER_SUBSETS,
        "descriptive_subsets": DESC_SUBSETS,
        "inferential_metrics": INFER_METRICS,
        "contrasts": [
            f"{a}_MINUS_{b}" for a, b in CONTRASTS
        ],
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "overwrite_existing": False,
    }

    write_json(PROTOCOL, protocol)

    print("U3_D14_A1_PROTOCOL_FREEZE_PASS")
    print("PROTOCOL_SHA256:", sha(PROTOCOL))


def grouped_means(values, dialogue_ids, indices):
    selected_ids = dialogue_ids[indices]
    unique, inverse = np.unique(
        selected_ids, return_inverse=True
    )

    selected = values[indices].astype(np.float64)

    sums = np.zeros(
        (len(unique), values.shape[1]),
        dtype=np.float64
    )

    np.add.at(sums, inverse, selected)

    counts = np.bincount(
        inverse, minlength=len(unique)
    ).astype(np.float64)

    return unique, sums / counts[:, None]


def bootstrap_ci(delta, seed):
    rng = np.random.default_rng(seed)
    n = len(delta)

    samples = np.empty(N_BOOT, dtype=np.float64)
    completed = 0

    while completed < N_BOOT:
        b = min(BATCH, N_BOOT - completed)

        idx = rng.integers(0, n, size=(b, n))
        samples[completed:completed+b] = delta[idx].mean(axis=1)

        completed += b

    return np.quantile(samples, [0.025, 0.975])


def permutation_p(delta, seed):
    rng = np.random.default_rng(seed)
    n = len(delta)

    observed = abs(float(np.mean(delta)))
    exceed = 0
    completed = 0

    while completed < N_PERM:
        b = min(BATCH, N_PERM - completed)

        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(b, n)
        )

        permuted = np.mean(
            signs * delta[None, :],
            axis=1
        )

        exceed += int(
            np.sum(np.abs(permuted) >= observed)
        )

        completed += b

    return (exceed + 1.0) / (N_PERM + 1.0)


def audit():
    if not PROTOCOL.is_file():
        raise RuntimeError("Protocol missing; run --prepare")

    for p in (
        RESULT, DESC_CSV, CONTRAST_CSV, REPORT, MANIFEST
    ):
        if p.exists():
            raise RuntimeError(f"Output collision: {p}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    if protocol.get("status") != "FROZEN_BEFORE_MARGIN_ANALYSIS":
        raise RuntimeError("Protocol status mismatch")

    for name, path in SOURCE_FILES.items():
        if sha(path) != protocol["source_sha256"][name]:
            raise RuntimeError(f"Source hash mismatch: {name}")

    if sha(A0_RESULT) != json.loads(
        A0_MANIFEST.read_text(encoding="utf-8")
    )["result_sha256"]:
        raise RuntimeError("A0 integrity failure")

    with np.load(PRED, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    sample_ids = arrays["sample_ids"]
    dialogue_ids = arrays["dialogue_ids"]
    long_idx = arrays["test_long20_indices"]
    match = arrays["gold_match_idx"]

    if len(sample_ids) != N:
        raise RuntimeError("Test query count mismatch")

    if len(dialogue_ids) != N:
        raise RuntimeError("Dialogue alignment mismatch")

    if len(np.unique(sample_ids)) != N:
        raise RuntimeError("Duplicate sample IDs")

    if len(long_idx) != 8727:
        raise RuntimeError("Long20 count mismatch")

    if len(np.unique(long_idx)) != len(long_idx):
        raise RuntimeError("Duplicate Long20 indices")

    if match.shape != (N,):
        raise RuntimeError("Gold index shape mismatch")

    if np.any(match < -1) or np.any(match >= 61503):
        raise RuntimeError("Gold global index invalid")

    long_mask = np.zeros(N, dtype=bool)
    long_mask[long_idx] = True

    subsets = {
        "TEST_ALL": np.arange(N),
        "TEST_LONG20": long_idx,
        "TEST_NONLONG20": np.flatnonzero(~long_mask),
        "TEST_GOLD_MATCHED": np.flatnonzero(match >= 0),
        "TEST_GOLD_UNMATCHED": np.flatnonzero(match < 0),
    }

    for name, indices in subsets.items():
        if len(indices) == 0:
            raise RuntimeError(f"Empty subset: {name}")

    # Each condition: five raw and five normalized margins.
    raw_margin = np.empty(
        (N, len(CONDITIONS), len(KS)),
        dtype=np.float64
    )

    normalized_margin = np.empty_like(raw_margin)

    rank_checks = {}

    print("\n========== MARGIN CONSTRUCTION ==========")

    for ci, condition in enumerate(CONDITIONS):

        if condition == "DENSE_BGE":
            candidate_key = "candidate_dense_scores"
            gold_key = "gold_dense_scores"
        else:
            candidate_key = "candidate_scores__" + condition
            gold_key = "gold_scores__" + condition

        candidates = arrays[candidate_key]
        gold = arrays[gold_key]

        if candidates.shape != (N, N_NEG):
            raise RuntimeError(
                f"Candidate shape mismatch: {condition}"
            )

        if gold.shape != (N,):
            raise RuntimeError(
                f"Gold shape mismatch: {condition}"
            )

        if not np.isfinite(candidates).all():
            raise RuntimeError(
                f"Non-finite candidate scores: {condition}"
            )

        if not np.isfinite(gold).all():
            raise RuntimeError(
                f"Non-finite gold scores: {condition}"
            )

        std = np.std(candidates, axis=1, ddof=1)
        std = np.maximum(std, EPS)

        for ki, k in enumerate(KS):

            # kth largest negative score.
            threshold = np.partition(
                candidates,
                N_NEG - k,
                axis=1
            )[:, N_NEG - k]

            margin = gold - threshold

            raw_margin[:, ci, ki] = margin
            normalized_margin[:, ci, ki] = margin / std

        # Independent consistency with stored ranking.
        ranks = arrays["ranks__" + condition]

        rank1_ok = np.array_equal(
            raw_margin[:, ci, 0] > 0,
            ranks == 1
        )

        rank5_ok = np.array_equal(
            raw_margin[:, ci, 1] > 0,
            ranks <= 5
        )

        rank_checks[condition] = {
            "rank1_margin_consistency": bool(rank1_ok),
            "rank5_margin_consistency": bool(rank5_ok),
        }

        if not rank1_ok or not rank5_ok:
            raise RuntimeError(
                f"Margin/rank consistency failure: {condition}"
            )

        print(
            "MARGIN_PASS",
            condition,
            "rank1=PASS",
            "rank5=PASS"
        )

    if not np.isfinite(raw_margin).all():
        raise RuntimeError("Non-finite raw margins")

    if not np.isfinite(normalized_margin).all():
        raise RuntimeError("Non-finite normalized margins")

    # Descriptive statistics: all thresholds and both scales.
    desc_names = (
        [f"RAW_MARGIN@{k}" for k in KS]
        + [f"NORMALIZED_MARGIN@{k}" for k in KS]
    )

    desc_values = np.concatenate(
        [raw_margin, normalized_margin],
        axis=2
    ).reshape(N, -1)

    desc_records = []

    print("\n========== DESCRIPTIVE ANALYSIS ==========")

    for subset_name in DESC_SUBSETS:
        indices = subsets[subset_name]

        unique, grouped = grouped_means(
            desc_values, dialogue_ids, indices
        )

        for ci, condition in enumerate(CONDITIONS):
            for mi, metric in enumerate(desc_names):

                values = grouped[:, ci * len(desc_names) + mi]

                desc_records.append({
                    "subset": subset_name,
                    "condition": condition,
                    "metric": metric,
                    "queries": int(len(indices)),
                    "dialogues": int(len(unique)),
                    "dialogue_macro_mean": float(np.mean(values)),
                    "dialogue_sd": float(np.std(values, ddof=1)),
                    "dialogue_median": float(np.median(values)),
                })

    # Inferential metrics: raw @1, normalized @1, normalized @5.
    selected = np.stack([
        raw_margin[:, :, 0],
        normalized_margin[:, :, 0],
        normalized_margin[:, :, 1],
    ], axis=2)

    infer_values = selected.reshape(N, -1)

    infer_records = []

    print("\n========== PAIRED DIALOGUE INFERENCE ==========")

    test_index = 0

    for subset_name in INFER_SUBSETS:
        indices = subsets[subset_name]

        unique, grouped = grouped_means(
            infer_values, dialogue_ids, indices
        )

        print(
            subset_name,
            "queries=", len(indices),
            "dialogues=", len(unique)
        )

        for ai, (a, b) in enumerate(CONTRASTS):

            ia = CONDITIONS.index(a)
            ib = CONDITIONS.index(b)

            contrast = f"{a}_MINUS_{b}"

            for mi, metric in enumerate(INFER_METRICS):

                col_a = ia * len(INFER_METRICS) + mi
                col_b = ib * len(INFER_METRICS) + mi

                delta = grouped[:, col_a] - grouped[:, col_b]

                seed = SEED + test_index * 100

                observed = float(np.mean(delta))

                ci = bootstrap_ci(delta, seed + 1)
                p = permutation_p(delta, seed + 2)

                infer_records.append({
                    "subset": subset_name,
                    "contrast": contrast,
                    "metric": metric,
                    "queries": int(len(indices)),
                    "dialogues": int(len(unique)),
                    "delta": observed,
                    "ci95_lower": float(ci[0]),
                    "ci95_upper": float(ci[1]),
                    "p_raw": float(p),
                })

                test_index += 1

    if len(infer_records) != 72:
        raise RuntimeError(
            f"Unexpected inferential record count: {len(infer_records)}"
        )

    # Primary global Holm family: all 72 tests.
    global_adj = holm_adjust([
        r["p_raw"] for r in infer_records
    ])

    for r, value in zip(infer_records, global_adj):
        r["p_holm_global_72"] = float(value)

    # Secondary families: 8 contrasts per subset-metric.
    families = {}

    for i, r in enumerate(infer_records):
        key = (r["subset"], r["metric"])
        families.setdefault(key, []).append(i)

    for family, indices in families.items():

        adjusted = holm_adjust([
            infer_records[i]["p_raw"] for i in indices
        ])

        for i, value in zip(indices, adjusted):
            infer_records[i]["p_holm_subset_metric_8"] = float(value)

    # CSV export.
    with DESC_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(desc_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(desc_records)

    with CONTRAST_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(infer_records[0].keys())
        )
        writer.writeheader()
        writer.writerows(infer_records)

    # Primary result.
    primary = next(
        r for r in infer_records
        if r["subset"] == "TEST_LONG20"
        and r["contrast"] ==
        "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL"
        and r["metric"] == "NORMALIZED_MARGIN@1"
    )

    result = {
        "artifact": "S12C U3 D14-A1 Gold Negative Margin Results V1",
        "status": "COMPLETED",
        "interpretation": "MECHANISM_ANALYSIS",
        "queries": N,
        "conditions": CONDITIONS,
        "thresholds": KS,
        "rank_consistency": rank_checks,
        "descriptive_records": len(desc_records),
        "inferential_records": len(infer_records),
        "global_holm_family": len(infer_records),
        "secondary_families": len(families),
        "secondary_family_size": 8,
        "primary_result": primary,
        "descriptive": desc_records,
        "contrasts": infer_records,
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
    }

    write_json(RESULT, result)

    # Human-readable report.
    lines = [
        "S12C U3 D14-A1 GOLD-NEGATIVE MARGIN ANALYSIS",
        "=" * 85,
        "STATUS: COMPLETED",
        f"QUERIES: {N}",
        f"BOOTSTRAP: {N_BOOT}",
        f"PERMUTATION: {N_PERM}",
        "PRIMARY SUBSET: TEST_LONG20",
        "PRIMARY METRIC: NORMALIZED_MARGIN@1",
        "PRIMARY CONTRAST: QR_FUTURE_FULL - R_FUTURE_FULL",
        "",
        "PRIMARY RESULT",
        "-" * 85,
    ]

    lines.append(
        f"Delta={primary['delta']:+.7f} "
        f"CI95=[{primary['ci95_lower']:+.7f},"
        f"{primary['ci95_upper']:+.7f}] "
        f"p={primary['p_raw']:.7g} "
        f"Holm72={primary['p_holm_global_72']:.7g} "
        f"Holm8={primary['p_holm_subset_metric_8']:.7g}"
    )

    lines += [
        "",
        "INFERENTIAL RESULTS",
        "-" * 85,
    ]

    for r in infer_records:
        lines.append(
            f"{r['subset']} | {r['contrast']} | {r['metric']} "
            f"| delta={r['delta']:+.7f} "
            f"| CI=[{r['ci95_lower']:+.7f},"
            f"{r['ci95_upper']:+.7f}] "
            f"| p={r['p_raw']:.7g} "
            f"| Holm72={r['p_holm_global_72']:.7g} "
            f"| Holm8={r['p_holm_subset_metric_8']:.7g}"
        )

    lines += [
        "",
        "INTEGRITY",
        "-" * 85,
        "RANK/MARGIN CONSISTENCY: PASS",
        "SOURCE HASHES: PASS",
        "TRAINING: NO",
        "NEW RANKING INFERENCE: NO",
        "TEST-TIME FITTING: NO",
    ]

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "U3 D14-A1 Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": sha(PROTOCOL),
        "result_sha256": sha(RESULT),
        "descriptive_csv_sha256": sha(DESC_CSV),
        "contrast_csv_sha256": sha(CONTRAST_CSV),
        "report_sha256": sha(REPORT),
        "source_sha256": protocol["source_sha256"],
        "descriptive_records": len(desc_records),
        "inferential_records": len(infer_records),
        "global_holm_family": len(infer_records),
        "secondary_families": len(families),
        "bootstrap": N_BOOT,
        "permutations": N_PERM,
    }

    write_json(MANIFEST, manifest)

    # Output hash verification.
    for name, path, key in [
        ("RESULT", RESULT, "result_sha256"),
        ("DESC_CSV", DESC_CSV, "descriptive_csv_sha256"),
        ("CONTRAST_CSV", CONTRAST_CSV, "contrast_csv_sha256"),
        ("REPORT", REPORT, "report_sha256"),
    ]:
        if sha(path) != manifest[key]:
            raise RuntimeError(f"Output hash failure: {name}")

    print("\n" + "=" * 75)
    print("U3_D14_A1_GOLD_NEGATIVE_MARGIN_COMPLETE")
    print("DESCRIPTIVE RECORDS:", len(desc_records))
    print("INFERENTIAL RECORDS:", len(infer_records))
    print("GLOBAL HOLM:", len(infer_records))
    print("SECONDARY FAMILIES:", len(families))
    print("PRIMARY DELTA:", primary["delta"])
    print("PRIMARY CI:", primary["ci95_lower"], primary["ci95_upper"])
    print("PRIMARY P:", primary["p_raw"])
    print("PRIMARY HOLM72:", primary["p_holm_global_72"])
    print("OUTPUT HASH VERIFICATION: PASS")
    print("OUTPUT:", OUT)
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--audit", action="store_true")

    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        audit()
