import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

TEST = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1"
A1 = PHASE / "step_2f18ah_u3_d14_a1_gold_negative_margin_v1"
OUT = PHASE / "step_2f18ai_u3_d14_a2_score_decomposition_v2"

PRED = TEST / "test_predictions_v1.npz"
SUMMARY = TEST / "test_summary_v1.json"
TEST_PROTOCOL = TEST / "test_evaluation_protocol_v1.json"
TEST_MANIFEST = TEST / "run_manifest.json"

A1_PROTOCOL = A1 / "protocol_v1.json"
A1_RESULT = A1 / "margin_results_v1.json"
A1_MANIFEST = A1 / "run_manifest_v1.json"

PROTOCOL = OUT / "protocol_v1.json"
RESULT = OUT / "decomposition_results_v1.json"
CSV_OUT = OUT / "decomposition_results_v1.csv"
REPORT = OUT / "decomposition_report_v1.txt"
MANIFEST = OUT / "run_manifest_v1.json"

N = 18073
N_LONG20 = 8727
N_BOOT = 5000
N_PERM = 20000
SEED = 20261120
EPS = 1e-8
BATCH = 256

CONDITIONS = [
    "R_FUTURE_FULL",
    "QR_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_NO_PRODUCT",
]

CONTRASTS = [
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
]

METRICS = ["Z_GOLD", "Z_HARDNEG", "Z_MARGIN"]

SUBSETS = ["TEST_ALL", "TEST_LONG20", "TEST_NONLONG20"]

SOURCES = {
    "predictions": PRED,
    "summary": SUMMARY,
    "test_protocol": TEST_PROTOCOL,
    "test_manifest": TEST_MANIFEST,
    "a1_protocol": A1_PROTOCOL,
    "a1_result": A1_RESULT,
    "a1_manifest": A1_MANIFEST,
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
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


def holm(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def prepare():
    if OUT.exists():
        raise RuntimeError(f"Output collision: {OUT}")

    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    a1m = json.loads(A1_MANIFEST.read_text(encoding="utf-8"))

    if a1m["result_sha256"] != sha(A1_RESULT):
        raise RuntimeError("D14-A1 result hash mismatch")

    if a1m["protocol_sha256"] != sha(A1_PROTOCOL):
        raise RuntimeError("D14-A1 protocol hash mismatch")

    a1 = json.loads(A1_RESULT.read_text(encoding="utf-8"))

    if a1.get("status") != "COMPLETED":
        raise RuntimeError("D14-A1 incomplete")

    OUT.mkdir(parents=True)

    protocol = {
        "artifact": "S12C U3 D14-A2 Score Component Decomposition V2",
        "status": "FROZEN_BEFORE_COMPONENT_ANALYSIS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_script_sha256": sha(Path(__file__).resolve()),
        "source_sha256": {
            k: sha(v) for k, v in SOURCES.items()
        },
        "queries": N,
        "long20_queries": N_LONG20,
        "conditions": CONDITIONS,
        "contrasts": [
            f"{a}_MINUS_{b}" for a, b in CONTRASTS
        ],
        "components": METRICS,
        "normalization": (
            "(score - negative_candidate_mean) / "
            "negative_candidate_sample_std"
        ),
        "primary_subset": "TEST_LONG20",
        "primary_contrast": "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
        "primary_tests": ["Z_GOLD", "Z_HARDNEG"],
        "primary_holm_family": 2,
        "bootstrap": N_BOOT,
        "permutation": N_PERM,
        "resampling_unit": "dialogue",
        "alternative": "two-sided",
        "seed": SEED,
        "point_estimator": (
            "unweighted_mean_of_dialogue_level_query_means"
        ),
        "secondary_inference": "bootstrap_CI_only",
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "overwrite_existing": False,
    }

    write_json(PROTOCOL, protocol)

    print("U3_D14_A2_PROTOCOL_FREEZE_PASS")
    print("PROTOCOL_SHA256:", sha(PROTOCOL))


def grouped(values, dialogue_ids, indices):
    ids = dialogue_ids[indices].astype(str)
    unique, inverse = np.unique(ids, return_inverse=True)

    x = values[indices].astype(np.float64)

    sums = np.zeros(
        (len(unique), x.shape[1]), dtype=np.float64
    )
    np.add.at(sums, inverse, x)

    counts = np.bincount(
        inverse, minlength=len(unique)
    ).astype(np.float64)

    return unique, sums / counts[:, None]


def bootstrap_ci(delta, seed):
    rng = np.random.default_rng(seed)
    n = len(delta)

    samples = np.empty(
        (N_BOOT, delta.shape[1]), dtype=np.float64
    )

    completed = 0

    while completed < N_BOOT:
        b = min(BATCH, N_BOOT - completed)
        idx = rng.integers(0, n, size=(b, n))
        samples[completed:completed+b] = delta[idx].mean(axis=1)
        completed += b

    return np.quantile(samples, [0.025, 0.975], axis=0)


def permutation_p(delta, seed):
    rng = np.random.default_rng(seed)
    n = len(delta)
    observed = abs(float(delta.mean()))

    exceed = 0
    completed = 0

    while completed < N_PERM:
        b = min(BATCH, N_PERM - completed)

        signs = rng.choice(
            [-1.0, 1.0], size=(b, n)
        )

        values = (signs * delta[None, :]).mean(axis=1)

        exceed += int(
            np.sum(np.abs(values) >= observed)
        )
        completed += b

    return (exceed + 1.0) / (N_PERM + 1.0)


def audit():
    if not PROTOCOL.is_file():
        raise RuntimeError("Run --prepare first")

    for p in (RESULT, CSV_OUT, REPORT, MANIFEST):
        if p.exists():
            raise RuntimeError(f"Output collision: {p}")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    if protocol.get("evaluator_script_sha256") != sha(Path(__file__).resolve()):
        raise RuntimeError("Evaluator SHA256 mismatch")

    if protocol["status"] != "FROZEN_BEFORE_COMPONENT_ANALYSIS":
        raise RuntimeError("Protocol status mismatch")

    for name, path in SOURCES.items():
        if sha(path) != protocol["source_sha256"][name]:
            raise RuntimeError(f"Source hash mismatch: {name}")

    a1m = json.loads(A1_MANIFEST.read_text(encoding="utf-8"))

    if a1m["result_sha256"] != sha(A1_RESULT):
        raise RuntimeError("A1 result integrity failure")

    a1 = json.loads(A1_RESULT.read_text(encoding="utf-8"))

    with np.load(PRED, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    dialogue_ids = arrays["dialogue_ids"]
    long_idx = arrays["test_long20_indices"]

    if len(dialogue_ids) != N:
        raise RuntimeError("Dialogue count mismatch")

    if len(long_idx) != N_LONG20:
        raise RuntimeError("Long20 alignment mismatch")

    long_mask = np.zeros(N, dtype=bool)
    long_mask[long_idx] = True

    subsets = {
        "TEST_ALL": np.arange(N),
        "TEST_LONG20": long_idx,
        "TEST_NONLONG20": np.flatnonzero(~long_mask),
    }

    # Components: [Z_GOLD, Z_HARDNEG, Z_MARGIN]
    components = np.empty(
        (N, len(CONDITIONS), len(METRICS)),
        dtype=np.float64
    )

    print("\n========== SCORE COMPONENT CONSTRUCTION ==========")

    for ci, condition in enumerate(CONDITIONS):

        candidate = np.asarray(arrays["candidate_scores__" + condition], dtype=np.float64)
        gold = np.asarray(arrays["gold_scores__" + condition], dtype=np.float64)

        if candidate.shape != (N, 100):
            raise RuntimeError(f"Candidate shape error: {condition}")

        if gold.shape != (N,):
            raise RuntimeError(f"Gold shape error: {condition}")

        if not (
            np.isfinite(candidate).all()
            and np.isfinite(gold).all()
        ):
            raise RuntimeError(f"Non-finite scores: {condition}")

        neg_mean = candidate.mean(axis=1)
        neg_std = candidate.std(axis=1, ddof=1)
        neg_std = np.maximum(neg_std, EPS)

        hardest = candidate.max(axis=1)

        z_gold = (gold - neg_mean) / neg_std
        z_hard = (hardest - neg_mean) / neg_std
        z_margin = z_gold - z_hard

        direct_margin = (gold - hardest) / neg_std

        if not np.allclose(
            z_margin, direct_margin, rtol=0, atol=1e-10
        ):
            raise RuntimeError(
                f"Component identity failure: {condition}"
            )

        components[:, ci, 0] = z_gold
        components[:, ci, 1] = z_hard
        components[:, ci, 2] = z_margin

        print(
            "COMPONENT_PASS", condition,
            "IDENTITY_PASS"
        )

    if not np.isfinite(components).all():
        raise RuntimeError("Non-finite components")

    records = []
    primary_tests = []

    print("\n========== PAIRED COMPONENT ANALYSIS ==========")

    test_number = 0

    for subset in SUBSETS:

        indices = subsets[subset]

        grouped_conditions = {}

        for ci, condition in enumerate(CONDITIONS):
            unique, values = grouped(
                components[:, ci, :],
                dialogue_ids,
                indices
            )
            grouped_conditions[condition] = values

        print(
            subset,
            "queries=", len(indices),
            "dialogues=", len(unique)
        )

        for a, b in CONTRASTS:

            delta = (
                grouped_conditions[a] -
                grouped_conditions[b]
            )

            ci = bootstrap_ci(delta, SEED + test_number * 10)

            contrast = f"{a}_MINUS_{b}"

            for mi, metric in enumerate(METRICS):

                record = {
                    "subset": subset,
                    "contrast": contrast,
                    "metric": metric,
                    "queries": int(len(indices)),
                    "dialogues": int(len(unique)),
                    "delta": float(delta[:, mi].mean()),
                    "ci95_lower": float(ci[0, mi]),
                    "ci95_upper": float(ci[1, mi]),
                    "p_raw": None,
                    "p_holm_primary_2": None,
                }

                # Only two pre-registered primary tests.
                if (
                    subset == "TEST_LONG20"
                    and a == "QR_FUTURE_FULL"
                    and b == "R_FUTURE_FULL"
                    and metric in ("Z_GOLD", "Z_HARDNEG")
                ):
                    p = permutation_p(
                        delta[:, mi],
                        SEED + 100000 + mi
                    )

                    record["p_raw"] = float(p)

                    primary_tests.append(record)

                records.append(record)

            test_number += 1

    if len(primary_tests) != 2:
        raise RuntimeError("Primary test count mismatch")

    adjusted = holm([
        r["p_raw"] for r in primary_tests
    ])

    for r, p in zip(primary_tests, adjusted):
        r["p_holm_primary_2"] = float(p)

    # Exact reproduction of D14-A1 primary margin point estimate.
    primary_margin = next(
        r for r in records
        if r["subset"] == "TEST_LONG20"
        and r["contrast"] ==
        "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL"
        and r["metric"] == "Z_MARGIN"
    )

    a1_primary = a1["primary_result"]["delta"]

    reproduction_diff = abs(
        primary_margin["delta"] - a1_primary
    )

    if reproduction_diff > 1e-10:
        raise RuntimeError(
            "D14-A1 margin reproduction failure: "
            f"{primary_margin['delta']} vs {a1_primary}"
        )

    print("D14_A1_MARGIN_POINT_REPRODUCTION_PASS")
    print("REPRODUCTION_DIFF:", reproduction_diff)

    # Export.
    with CSV_OUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    primary_gold = next(
        r for r in primary_tests if r["metric"] == "Z_GOLD"
    )

    primary_hard = next(
        r for r in primary_tests if r["metric"] == "Z_HARDNEG"
    )

    result = {
        "artifact": "S12C U3 D14-A2 Score Decomposition V1",
        "status": "COMPLETED",
        "interpretation": "POST_HOC_SCORE_MECHANISM_DECOMPOSITION",
        "components": METRICS,
        "contrasts": records,
        "primary_tests": primary_tests,
        "primary_holm_family": 2,
        "primary_result": {
            "gold_component": primary_gold,
            "hard_negative_component": primary_hard,
            "margin_component": primary_margin,
        },
        "a1_margin_reproduction": {
            "status": "PASS",
            "a1_point_estimate": float(a1_primary),
            "a2_point_estimate": primary_margin["delta"],
            "absolute_difference": reproduction_diff,
        },
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
    }

    write_json(RESULT, result)

    lines = [
        "S12C U3 D14-A2 SCORE COMPONENT DECOMPOSITION",
        "=" * 85,
        "PRIMARY SUBSET: TEST_LONG20",
        "PRIMARY CONTRAST: QR_FUTURE_FULL - R_FUTURE_FULL",
        "BOOTSTRAP: 5000",
        "PERMUTATION: 20000",
        "PRIMARY HOLM FAMILY: 2",
        "",
        "PRIMARY COMPONENT RESULTS",
        "-" * 85,
    ]

    for r in primary_tests + [primary_margin]:
        lines.append(
            f"{r['metric']}: "
            f"delta={r['delta']:+.7f} "
            f"CI95=[{r['ci95_lower']:+.7f},"
            f"{r['ci95_upper']:+.7f}] "
            f"p={r['p_raw']} "
            f"Holm2={r['p_holm_primary_2']}"
        )

    lines += [
        "",
        "INTERPRETATION GUIDE",
        "Positive Z_GOLD delta: Gold strengthened relative to negative pool.",
        "Negative Z_HARDNEG delta: hardest negative weakened relative to pool.",
        "Positive Z_MARGIN delta: Gold-hardest-negative separation increased.",
        "",
        "A1 POINT ESTIMATE REPRODUCTION: PASS",
        "TRAINING: NO",
        "NEW RANKING INFERENCE: NO",
        "TEST-TIME FITTING: NO",
        "",
        "ALL CONTRASTS",
        "-" * 85,
    ]

    for r in records:
        lines.append(
            f"{r['subset']} | {r['contrast']} | {r['metric']} "
            f"| delta={r['delta']:+.7f} "
            f"| CI=[{r['ci95_lower']:+.7f},"
            f"{r['ci95_upper']:+.7f}]"
        )

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    manifest = {
        "artifact": "U3 D14-A2 Manifest V2",
        "status": "COMPLETED",
        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_script_sha256": sha(Path(__file__).resolve()),
        "protocol_sha256": sha(PROTOCOL),
        "result_sha256": sha(RESULT),
        "csv_sha256": sha(CSV_OUT),
        "report_sha256": sha(REPORT),
        "source_sha256": protocol["source_sha256"],
        "contrast_metric_rows": len(records),
        "primary_tests": len(primary_tests),
        "primary_holm_family": 2,
        "bootstrap": N_BOOT,
        "permutations": N_PERM,
        "a1_reproduction_diff": reproduction_diff,
    }

    write_json(MANIFEST, manifest)

    for path, key in [
        (RESULT, "result_sha256"),
        (CSV_OUT, "csv_sha256"),
        (REPORT, "report_sha256"),
    ]:
        if sha(path) != manifest[key]:
            raise RuntimeError(f"Output hash failure: {path.name}")

    print("\n" + "=" * 75)
    print("U3_D14_A2_SCORE_DECOMPOSITION_COMPLETE")
    print("CONTRAST METRIC ROWS:", len(records))
    print("PRIMARY TESTS:", len(primary_tests))
    print("PRIMARY HOLM: 2")
    print("A1 REPRODUCTION: PASS")
    print("OUTPUT HASH: PASS")
    print("RESULT:", RESULT)
    print("REPORT:", REPORT)
    print("MANIFEST:", MANIFEST)
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
