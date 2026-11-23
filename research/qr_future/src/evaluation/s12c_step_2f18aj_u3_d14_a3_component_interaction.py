import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

TEST = PHASE / "step_2f18t_orca_fsar_test_evaluation_v1"
A2 = PHASE / "step_2f18ai_u3_d14_a2_score_decomposition_v2"
D11 = PHASE / "step_2f18ac_u3_d11_paired_interaction_audit_v2"

OUT = PHASE / "step_2f18aj_u3_d14_a3_component_interaction_v1"

PRED = TEST / "test_predictions_v1.npz"
SUMMARY = TEST / "test_summary_v1.json"
TEST_PROTOCOL = TEST / "test_evaluation_protocol_v1.json"
TEST_MANIFEST = TEST / "run_manifest.json"

A2_SCRIPT = ROOT / "src/evaluation/s12c_step_2f18ai_u3_d14_a2_score_decomposition_v2.py"
A2_PROTOCOL = A2 / "protocol_v1.json"
A2_RESULT = A2 / "decomposition_results_v1.json"
A2_MANIFEST = A2 / "run_manifest_v1.json"

D11_SCRIPT = ROOT / "src/evaluation/s12c_step_2f18ac_u3_d11_paired_interaction_audit.py"
D11_PROTOCOL = D11 / "protocol_v1.json"
D11_RESULT = D11 / "interaction_results_v2.json"
D11_MANIFEST = D11 / "run_manifest_v2.json"

PROTOCOL = OUT / "protocol_v1.json"
RESULT = OUT / "a3_results_v1.json"
CSV_OUT = OUT / "a3_results_v1.csv"
REPORT = OUT / "a3_report_v1.txt"
MANIFEST = OUT / "run_manifest_v1.json"

N = 18073
N_LONG20 = 8727
N_DIALOGUES = 477
N_PAIRED = 413

N_BOOT = 5000
N_PERM = 20000
SEED = 20261121
BATCH = 256
EPS = 1e-8

CONDITIONS = [
    "R_FUTURE_FULL",
    "QR_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_NO_PRODUCT",
]

METRICS = ["Z_GOLD", "Z_HARDNEG", "Z_MARGIN"]

SOURCE_PATHS = {
    "predictions": PRED,
    "summary": SUMMARY,
    "test_protocol": TEST_PROTOCOL,
    "test_manifest": TEST_MANIFEST,
    "a2_script": A2_SCRIPT,
    "a2_protocol": A2_PROTOCOL,
    "a2_result": A2_RESULT,
    "a2_manifest": A2_MANIFEST,
    "d11_script": D11_SCRIPT,
    "d11_protocol": D11_PROTOCOL,
    "d11_result": D11_RESULT,
    "d11_manifest": D11_MANIFEST,
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


def holm(pvalues):
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


def load_arrays():
    with np.load(PRED, allow_pickle=False) as z:
        required = [
            "dialogue_ids",
            "sample_ids",
            "test_long20_indices",
        ]

        for condition in CONDITIONS:
            required.extend([
                "candidate_scores__" + condition,
                "gold_scores__" + condition,
            ])

        missing = [k for k in required if k not in z.files]
        if missing:
            raise RuntimeError("MISSING_ARRAYS: " + str(missing))

        arrays = {k: z[k] for k in required}

    return arrays


def validate_alignment(arrays):
    dids = arrays["dialogue_ids"].astype(str)
    sample_ids = arrays["sample_ids"].astype(str)
    long_idx = arrays["test_long20_indices"].astype(np.int64)

    if len(dids) != N or len(sample_ids) != N:
        raise RuntimeError("QUERY_COUNT_MISMATCH")

    if len(np.unique(sample_ids)) != N:
        raise RuntimeError("DUPLICATE_SAMPLE_IDS")

    if len(np.unique(dids)) != N_DIALOGUES:
        raise RuntimeError("DIALOGUE_COUNT_MISMATCH")

    if len(long_idx) != N_LONG20:
        raise RuntimeError("LONG20_COUNT_MISMATCH")

    if len(np.unique(long_idx)) != N_LONG20:
        raise RuntimeError("DUPLICATE_LONG20_INDICES")

    if np.any(long_idx < 0) or np.any(long_idx >= N):
        raise RuntimeError("LONG20_INDEX_OUT_OF_RANGE")

    long_mask = np.zeros(N, dtype=bool)
    long_mask[long_idx] = True

    paired_ids = []

    for did in np.unique(dids):
        selected = dids == did
        if np.any(selected & long_mask) and np.any(selected & ~long_mask):
            paired_ids.append(did)

    long_dialogues = len(np.unique(dids[long_mask]))

    if long_dialogues != N_PAIRED:
        raise RuntimeError(
            f"LONG_DIALOGUE_COUNT_MISMATCH: {long_dialogues}"
        )

    if len(paired_ids) != N_PAIRED:
        raise RuntimeError(
            f"PAIRED_DIALOGUE_COUNT_MISMATCH: {len(paired_ids)}"
        )

    for condition in CONDITIONS:
        candidate = arrays["candidate_scores__" + condition]
        gold = arrays["gold_scores__" + condition]

        if candidate.shape != (N, 100):
            raise RuntimeError("CANDIDATE_SHAPE_ERROR: " + condition)

        if gold.shape != (N,):
            raise RuntimeError("GOLD_SHAPE_ERROR: " + condition)

        if not np.isfinite(candidate).all():
            raise RuntimeError("NONFINITE_CANDIDATE: " + condition)

        if not np.isfinite(gold).all():
            raise RuntimeError("NONFINITE_GOLD: " + condition)

    print("SAMPLE_ALIGNMENT_PASS")
    print("DIALOGUE_ALIGNMENT_PASS")
    print("LONG20_ALIGNMENT_PASS")
    print("PAIRED_DIALOGUE_ALIGNMENT_PASS")
    print("TEST_QUERIES:", N)
    print("TEST_DIALOGUES:", N_DIALOGUES)
    print("LONG20_QUERIES:", N_LONG20)
    print("NONLONG20_QUERIES:", N - N_LONG20)
    print("PAIRED_DIALOGUES:", len(paired_ids))

    return dids, long_mask, np.asarray(paired_ids)


def validate_sources():
    for name, path in SOURCE_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    a2p = json.loads(A2_PROTOCOL.read_text(encoding="utf-8"))
    a2 = json.loads(A2_RESULT.read_text(encoding="utf-8"))

    if a2p.get("status") != "FROZEN_BEFORE_COMPONENT_ANALYSIS":
        raise RuntimeError("A2_PROTOCOL_STATUS_ERROR")

    if a2p.get("evaluator_script_sha256") != sha(A2_SCRIPT):
        raise RuntimeError("A2_SCRIPT_HASH_MISMATCH")

    if a2.get("status") != "COMPLETED":
        raise RuntimeError("A2_RESULT_NOT_COMPLETED")

    for name, path in [
        ("predictions", PRED),
        ("summary", SUMMARY),
        ("test_protocol", TEST_PROTOCOL),
        ("test_manifest", TEST_MANIFEST),
    ]:
        if a2p["source_sha256"][name] != sha(path):
            raise RuntimeError("A2_SOURCE_MISMATCH: " + name)

    test_manifest = json.loads(
        TEST_MANIFEST.read_text(encoding="utf-8")
    )

    if test_manifest.get("status") != "COMPLETED":
        raise RuntimeError("TEST_MANIFEST_NOT_COMPLETED")

    test_protocol = json.loads(
        TEST_PROTOCOL.read_text(encoding="utf-8")
    )

    if test_protocol.get("status") != "FROZEN_BEFORE_TEST_RANKING_INFERENCE":
        raise RuntimeError("TEST_PROTOCOL_STATUS_ERROR")

    print("A2_SOURCE_INTEGRITY_PASS")
    print("D11_SOURCE_FILES_PRESENT_PASS")
    print("TEST_SOURCE_INTEGRITY_PASS")

    return a2


def prepare():
    if OUT.exists():
        raise RuntimeError("OUTPUT_COLLISION: " + str(OUT))

    validate_sources()

    arrays = load_arrays()
    dids, long_mask, paired_ids = validate_alignment(arrays)

    protocol = {
        "artifact": "S12C U3 D14-A3 Component Interaction Audit V1",
        "status": "FROZEN_BEFORE_COMPONENT_INTERACTION_ANALYSIS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "evaluator_script": str(Path(__file__).resolve()),
        "evaluator_script_sha256": sha(Path(__file__).resolve()),

        "source_sha256": {
            name: sha(path)
            for name, path in SOURCE_PATHS.items()
        },

        "data": {
            "test_queries": N,
            "test_dialogues": N_DIALOGUES,
            "long20_queries": N_LONG20,
            "nonlong20_queries": N - N_LONG20,
            "paired_dialogues": N_PAIRED,
            "paired_dialogue_definition": (
                "Dialogue must contain at least one Long20 query "
                "and at least one NonLong20 query."
            ),
        },

        "components": METRICS,

        "normalization": (
            "(score - negative_candidate_mean) "
            "/ negative_candidate_sample_std"
        ),

        "component_definitions": {
            "Z_GOLD": "(gold - negative_mean) / negative_std",
            "Z_HARDNEG": "(hardest_negative - negative_mean) / negative_std",
            "Z_MARGIN": "(gold - hardest_negative) / negative_std",
        },

        "estimands": [
            "FULL_LONG_SPECIFIC_INTERACTION",
            "NO_PRODUCT_LONG_SPECIFIC_INTERACTION",
            "PRODUCT_INTERACTION_INCREMENT",
        ],

        "product_increment_formula": (
            "[(QR_FULL-R_FULL)_LONG-(QR_FULL-R_FULL)_NONLONG] "
            "- [(QR_NO_PRODUCT-R_NO_PRODUCT)_LONG "
            "- (QR_NO_PRODUCT-R_NO_PRODUCT)_NONLONG]"
        ),

        "primary_tests": 9,
        "multiple_testing": {
            "method": "Holm-Bonferroni",
            "family_size": 9,
            "family": "All nine component interaction tests",
        },

        "bootstrap": N_BOOT,
        "permutation": N_PERM,
        "permutation_method": "paired_dialogue_sign_flip",
        "alternative": "two-sided",
        "resampling_unit": "paired_dialogue",
        "point_estimator": (
            "Unweighted mean of paired dialogue-level interactions"
        ),

        "seed": SEED,
        "batch_size": BATCH,
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "checkpoint_reselection": False,

        "interpretation": "EXPLORATORY_POST_HOC",
        "overwrite_existing": False,
    }

    OUT.mkdir(parents=True, exist_ok=False)
    write_json(PROTOCOL, protocol)

    print("\n" + "=" * 75)
    print("U3_D14_A3_PROTOCOL_FREEZE_PASS")
    print("PAIRED_DIALOGUES:", len(paired_ids))
    print("TESTS:", 9)
    print("HOLM_FAMILY:", 9)
    print("PROTOCOL_SHA256:", sha(PROTOCOL))
    print("PROTOCOL:", PROTOCOL)
    print("=" * 75)


def build_components(arrays):
    components = {}

    for condition in CONDITIONS:
        candidate = np.asarray(
            arrays["candidate_scores__" + condition],
            dtype=np.float64
        )
        gold = np.asarray(
            arrays["gold_scores__" + condition],
            dtype=np.float64
        )

        neg_mean = candidate.mean(axis=1)
        neg_std = np.maximum(
            candidate.std(axis=1, ddof=1),
            EPS
        )

        hardest = candidate.max(axis=1)

        z_gold = (gold - neg_mean) / neg_std
        z_hard = (hardest - neg_mean) / neg_std
        z_margin = (gold - hardest) / neg_std

        if not np.allclose(
            z_gold - z_hard,
            z_margin,
            rtol=0,
            atol=1e-10
        ):
            raise RuntimeError(
                "COMPONENT_IDENTITY_FAILURE: " + condition
            )

        components[condition] = np.column_stack([
            z_gold,
            z_hard,
            z_margin,
        ])

        print("COMPONENT_PASS", condition)

    return components


def paired_interaction(delta, dids, long_mask, expected_ids):
    long_values = []
    nonlong_values = []
    paired_ids = []

    for did in np.unique(dids):
        selected = dids == did

        lm = selected & long_mask
        nm = selected & (~long_mask)

        if not lm.any() or not nm.any():
            continue

        long_values.append(delta[lm].mean(axis=0))
        nonlong_values.append(delta[nm].mean(axis=0))
        paired_ids.append(did)

    paired_ids = np.asarray(paired_ids)
    long_values = np.asarray(long_values, dtype=np.float64)
    nonlong_values = np.asarray(nonlong_values, dtype=np.float64)

    if not np.array_equal(paired_ids, expected_ids):
        raise RuntimeError("PAIRED_ID_ORDER_MISMATCH")

    interaction = long_values - nonlong_values

    return paired_ids, long_values, nonlong_values, interaction


def bootstrap_ci(matrix, seed):
    rng = np.random.default_rng(seed)

    n, k = matrix.shape
    samples = np.empty((N_BOOT, k), dtype=np.float64)

    completed = 0

    while completed < N_BOOT:
        b = min(BATCH, N_BOOT - completed)

        idx = rng.integers(0, n, size=(b, n))
        samples[completed:completed+b] = matrix[idx].mean(axis=1)

        completed += b

        if completed % 1024 == 0 or completed == N_BOOT:
            print("BOOTSTRAP:", completed, "/", N_BOOT)

    return np.quantile(samples, [0.025, 0.975], axis=0)


def permutation_p(matrix, seed):
    rng = np.random.default_rng(seed)

    n, k = matrix.shape
    observed = np.abs(matrix.mean(axis=0))

    exceed = np.zeros(k, dtype=np.int64)
    completed = 0

    while completed < N_PERM:
        b = min(BATCH, N_PERM - completed)

        signs = rng.choice(
            [-1.0, 1.0],
            size=(b, n)
        )

        values = signs @ matrix / n

        exceed += np.sum(
            np.abs(values) >= observed[None, :],
            axis=0
        )

        completed += b

        if completed % 4096 == 0 or completed == N_PERM:
            print("PERMUTATION:", completed, "/", N_PERM)

    return (exceed + 1.0) / (N_PERM + 1.0)


def reproduce_a2_long20(a2, long_values_full, long_values_no):
    lookup = {}

    for r in a2["contrasts"]:
        key = (
            r["subset"],
            r["contrast"],
            r["metric"],
        )
        lookup[key] = float(r["delta"])

    checks = [
        (
            "QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
            long_values_full,
        ),
        (
            "QR_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_NO_PRODUCT",
            long_values_no,
        ),
    ]

    diffs = []

    for contrast, values in checks:
        for mi, metric in enumerate(METRICS):
            key = ("TEST_LONG20", contrast, metric)

            if key not in lookup:
                raise RuntimeError("A2_RESULT_FIELD_MISSING: " + str(key))

            diff = abs(float(values[:, mi].mean()) - lookup[key])
            diffs.append(diff)

    maximum = max(diffs)

    if maximum > 1e-10:
        raise RuntimeError(
            f"A2_LONG20_REPRODUCTION_FAILURE: {maximum}"
        )

    print("A2_LONG20_POINT_ESTIMATE_REPRODUCTION_PASS")
    print("A2_REPRODUCTION_MAX_DIFF:", maximum)

    return maximum


def audit():
    if not PROTOCOL.is_file():
        raise RuntimeError("RUN_PREPARE_FIRST")

    for p in (RESULT, CSV_OUT, REPORT, MANIFEST):
        if p.exists():
            raise RuntimeError("OUTPUT_COLLISION: " + str(p))

    protocol = json.loads(
        PROTOCOL.read_text(encoding="utf-8")
    )

    if protocol["status"] != "FROZEN_BEFORE_COMPONENT_INTERACTION_ANALYSIS":
        raise RuntimeError("PROTOCOL_STATUS_ERROR")

    if protocol["evaluator_script_sha256"] != sha(Path(__file__).resolve()):
        raise RuntimeError("EVALUATOR_HASH_MISMATCH")

    for name, path in SOURCE_PATHS.items():
        if sha(path) != protocol["source_sha256"][name]:
            raise RuntimeError("SOURCE_HASH_MISMATCH: " + name)

    a2 = validate_sources()

    arrays = load_arrays()
    dids, long_mask, expected_ids = validate_alignment(arrays)

    components = build_components(arrays)

    full_delta = (
        components["QR_FUTURE_FULL"]
        - components["R_FUTURE_FULL"]
    )

    no_delta = (
        components["QR_FUTURE_NO_PRODUCT"]
        - components["R_FUTURE_NO_PRODUCT"]
    )

    ids, full_long, full_nonlong, full_interaction = (
        paired_interaction(
            full_delta, dids, long_mask, expected_ids
        )
    )

    ids2, no_long, no_nonlong, no_interaction = (
        paired_interaction(
            no_delta, dids, long_mask, expected_ids
        )
    )

    if not np.array_equal(ids, ids2):
        raise RuntimeError("CONDITION_PAIRING_MISMATCH")

    product_increment = full_interaction - no_interaction

    a2_diff = reproduce_a2_long20(
        a2, full_long, no_long
    )

    # Nine tests in one frozen family.
    columns = []
    specs = []

    groups = [
        (
            "FULL_LONG_SPECIFIC",
            full_interaction,
            "Long20 interaction: QR_FULL - R_FULL",
        ),
        (
            "NO_PRODUCT_LONG_SPECIFIC",
            no_interaction,
            "Long20 interaction: QR_NO_PRODUCT - R_NO_PRODUCT",
        ),
        (
            "PRODUCT_INCREMENT",
            product_increment,
            "Full interaction minus No-Product interaction",
        ),
    ]

    for group, values, formula in groups:
        for mi, metric in enumerate(METRICS):
            columns.append(values[:, mi])

            specs.append({
                "test": group + "_" + metric,
                "group": group,
                "metric": metric,
                "formula": formula,
            })

    matrix = np.column_stack(columns)

    if matrix.shape != (N_PAIRED, 9):
        raise RuntimeError("FINAL_TEST_MATRIX_SHAPE_ERROR")

    print("\n========== PAIRED INTERACTION ANALYSIS ==========")
    print("PAIRED_DIALOGUES:", len(ids))
    print("TEST_MATRIX:", matrix.shape)

    point = matrix.mean(axis=0)

    ci = bootstrap_ci(matrix, SEED + 100)
    pvalues = permutation_p(matrix, SEED + 200)
    adjusted = holm(pvalues)

    records = []

    for i, spec in enumerate(specs):
        records.append({
            **spec,
            "paired_dialogues": N_PAIRED,
            "delta": float(point[i]),
            "ci95_lower": float(ci[0, i]),
            "ci95_upper": float(ci[1, i]),
            "p_raw": float(pvalues[i]),
            "p_holm_9": float(adjusted[i]),
            "ci_excludes_zero": bool(
                ci[0, i] > 0 or ci[1, i] < 0
            ),
            "holm_significant": bool(adjusted[i] < 0.05),
        })

    print("\n========== FINAL RESULTS ==========")

    for r in records:
        print(
            r["test"],
            "DELTA=", f"{r['delta']:+.7f}",
            "CI=", (
                f"[{r['ci95_lower']:+.7f},"
                f"{r['ci95_upper']:+.7f}]"
            ),
            "P=", f"{r['p_raw']:.7g}",
            "HOLM9=", f"{r['p_holm_9']:.7g}",
        )

    result = {
        "artifact": "S12C U3 D14-A3 Component Interaction Results V1",
        "status": "COMPLETED",
        "interpretation": "EXPLORATORY_POST_HOC",
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "test_queries": N,
        "paired_dialogues": N_PAIRED,
        "long20_queries": N_LONG20,
        "nonlong20_queries": N - N_LONG20,
        "components": METRICS,
        "tests": records,
        "multiple_testing": {
            "method": "Holm-Bonferroni",
            "family_size": 9,
            "pvalues": "Two-sided paired dialogue sign-flip permutation",
        },
        "bootstrap_replicates": N_BOOT,
        "permutations": N_PERM,
        "seed": SEED,
        "a2_long20_reproduction_max_abs_diff": a2_diff,
        "paired_dialogue_id_sha256": hashlib.sha256(
            "\n".join(ids.tolist()).encode("utf-8")
        ).hexdigest(),
        "protocol_sha256": sha(PROTOCOL),
    }

    write_json(RESULT, result)

    with CSV_OUT.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "S12C U3 D14-A3 COMPONENT INTERACTION AUDIT",
        "=" * 85,
        "Interpretation: EXPLORATORY_POST_HOC",
        "Paired dialogue-level Long20 versus NonLong20 interaction",
        f"Paired dialogues: {N_PAIRED}",
        f"Bootstrap: {N_BOOT}",
        f"Permutation: {N_PERM}",
        "Permutation: paired dialogue sign-flip, two-sided",
        "Multiple testing: Holm-9",
        "Training: NO",
        "New ranking inference: NO",
        "Test-time fitting: NO",
        "",
        f"A2 Long20 reproduction max diff: {a2_diff:.12g}",
        "",
        "FINAL RESULTS",
        "-" * 85,
    ]

    for r in records:
        lines.append(
            f"{r['test']}: "
            f"delta={r['delta']:+.7f} "
            f"CI95=[{r['ci95_lower']:+.7f},"
            f"{r['ci95_upper']:+.7f}] "
            f"p={r['p_raw']:.7g} "
            f"Holm9={r['p_holm_9']:.7g}"
        )

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    outputs = [RESULT, CSV_OUT, REPORT]

    manifest = {
        "artifact": "S12C U3 D14-A3 Run Manifest V1",
        "status": "COMPLETED",
        "protocol_sha256": sha(PROTOCOL),
        "evaluator_script_sha256": sha(Path(__file__).resolve()),
        "source_sha256": protocol["source_sha256"],
        "outputs": {
            p.name: sha(p) for p in outputs
        },
        "test_count": 9,
        "holm_family_size": 9,
        "paired_dialogues": N_PAIRED,
        "bootstrap": N_BOOT,
        "permutations": N_PERM,
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "a2_reproduction_max_abs_diff": a2_diff,
    }

    write_json(MANIFEST, manifest)

    for p in outputs:
        if sha(p) != manifest["outputs"][p.name]:
            raise RuntimeError("OUTPUT_HASH_FAILURE: " + p.name)

    print("\n" + "=" * 75)
    print("U3_D14_A3_COMPONENT_INTERACTION_COMPLETE")
    print("SOURCE_INTEGRITY: PASS")
    print("PAIRED_DIALOGUES:", N_PAIRED)
    print("TESTS:", len(records))
    print("HOLM_FAMILY:", len(records))
    print("A2_REPRODUCTION: PASS")
    print("BOOTSTRAP:", N_BOOT)
    print("PERMUTATION:", N_PERM)
    print("OUTPUT_HASH_VERIFICATION: PASS")
    print("RESULT:", RESULT)
    print("REPORT:", REPORT)
    print("MANIFEST:", MANIFEST)
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare",
        action="store_true"
    )
    parser.add_argument(
        "--audit",
        action="store_true"
    )

    args = parser.parse_args()

    if args.prepare == args.audit:
        parser.error("Specify exactly one of --prepare or --audit")

    if args.prepare:
        prepare()
    else:
        audit()
