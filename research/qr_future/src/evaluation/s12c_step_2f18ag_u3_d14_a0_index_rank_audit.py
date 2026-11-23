import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

BASE = ROOT / "results/phase_s12c/step_2f18t_orca_fsar_test_evaluation_v1"

OUT = ROOT / "results/phase_s12c/step_2f18ag_u3_d14_a0_index_rank_audit_v1"

PRED = BASE / "test_predictions_v1.npz"
SUMMARY = BASE / "test_summary_v1.json"
PROTOCOL = BASE / "test_evaluation_protocol_v1.json"
REPORT_SOURCE = BASE / "test_report_v1.txt"
SOURCE_MANIFEST = BASE / "run_manifest.json"

EVALUATOR = ROOT / "src/evaluation/s12c_step_2f18t_orca_fsar_test_evaluation.py"

AUDIT_PROTOCOL = OUT / "protocol_v1.json"
RESULT = OUT / "audit_results_v1.json"
REPORT = OUT / "audit_report_v1.txt"
MANIFEST = OUT / "run_manifest_v1.json"

N = 18073
N_LONG = 8727
N_DIALOGUES = 477
N_LONG_DIALOGUES = 413

LIB_ROWS = 61503
TOP_K = 100

CONDITIONS = [
    "DENSE_BGE",
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

SOURCES = {
    "predictions": PRED,
    "summary": SUMMARY,
    "protocol": PROTOCOL,
    "report": REPORT_SOURCE,
    "source_manifest": SOURCE_MANIFEST,
    "evaluator": EVALUATOR,
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


def prepare():
    if OUT.exists():
        raise RuntimeError(
            f"Output collision, refusing overwrite: {OUT}"
        )

    for name, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    OUT.mkdir(parents=True)

    protocol = {
        "artifact": "S12C U3 D14-A0 Index and Rank Audit V1",
        "status": "FROZEN_BEFORE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            name: sha(path)
            for name, path in SOURCES.items()
        },
        "audit_script_sha256": sha(Path(__file__)),
        "test": {
            "queries": N,
            "dialogues": N_DIALOGUES,
            "long20_queries": N_LONG,
            "long20_dialogues": N_LONG_DIALOGUES,
        },
        "response_library": {
            "rows": LIB_ROWS,
            "valid_global_index_min": 0,
            "valid_global_index_max": LIB_ROWS - 1,
        },
        "candidate_set": {
            "negative_candidates": TOP_K,
            "total_candidates": TOP_K + 1,
            "candidate_indices": "global_response_library_rows",
        },
        "gold_match": {
            "matched_condition": "0 <= gold_match_idx < 61503",
            "unmatched_sentinel": -1,
        },
        "ranking": {
            "formula": "1 + sum(candidate_score >= gold_score)",
            "tie_rule": "negative score >= Gold ranks before Gold",
        },
        "conditions": CONDITIONS,
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
        "overwrite_existing": False,
    }

    write_json(AUDIT_PROTOCOL, protocol)

    print("U3_D14_A0_PROTOCOL_FREEZE_PASS")
    print("PROTOCOL:", AUDIT_PROTOCOL)
    print("PROTOCOL_SHA256:", sha(AUDIT_PROTOCOL))


def audit():
    if not AUDIT_PROTOCOL.is_file():
        raise RuntimeError("Frozen protocol missing; run --prepare first")

    for p in (RESULT, REPORT, MANIFEST):
        if p.exists():
            raise RuntimeError(f"Output collision: {p}")

    protocol = json.loads(
        AUDIT_PROTOCOL.read_text(encoding="utf-8")
    )

    if protocol["status"] != "FROZEN_BEFORE_AUDIT":
        raise RuntimeError("Protocol status mismatch")

    if sha(Path(__file__)) != protocol["audit_script_sha256"]:
        raise RuntimeError("Audit script changed after protocol freeze")

    checks = []

    def record(name, passed, detail=""):
        item = {
            "name": name,
            "status": "PASS" if passed else "FAIL",
            "detail": str(detail),
        }
        checks.append(item)
        print(item["status"], name, detail)

    # 1. Source integrity
    for name, path in SOURCES.items():
        record(
            "SOURCE_HASH_" + name.upper(),
            sha(path) == protocol["source_sha256"][name],
        )

    source_manifest = json.loads(
        SOURCE_MANIFEST.read_text(encoding="utf-8")
    )

    record(
        "ORIGINAL_RUN_MANIFEST_COMPLETED",
        source_manifest.get("status") == "COMPLETED",
    )

    record(
        "ORIGINAL_PREDICTION_HASH",
        source_manifest.get("predictions_sha256") == sha(PRED),
    )

    record(
        "ORIGINAL_PROTOCOL_HASH",
        source_manifest.get("protocol_sha256") == sha(PROTOCOL),
    )

    # 2. Load immutable predictions
    with np.load(PRED, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    required = [
        "sample_ids",
        "dialogue_ids",
        "test_long20_indices",
        "gold_match_idx",
        "candidate_response_rows",
        "candidate_dense_scores",
        "gold_dense_scores",
    ]

    for condition in CONDITIONS:
        required.append("ranks__" + condition)

        if condition != "DENSE_BGE":
            required.extend([
                "candidate_scores__" + condition,
                "gold_scores__" + condition,
            ])

    missing = [k for k in required if k not in arrays]

    if missing:
        raise RuntimeError(f"Missing NPZ fields: {missing}")

    sample_ids = arrays["sample_ids"]
    dialogue_ids = arrays["dialogue_ids"]
    long_idx = arrays["test_long20_indices"]
    match = arrays["gold_match_idx"]
    candidates = arrays["candidate_response_rows"]

    record("QUERY_COUNT", len(sample_ids) == N, len(sample_ids))
    record("DIALOGUE_COUNT", len(np.unique(dialogue_ids)) == N_DIALOGUES)
    record("SAMPLE_ID_UNIQUENESS", len(np.unique(sample_ids)) == N)

    # 3. Gold global-index audit
    valid = match >= 0
    matched = int(valid.sum())
    unmatched = int((match == -1).sum())

    invalid = (
        np.any(match < -1)
        or np.any(match >= LIB_ROWS)
    )

    record("GOLD_GLOBAL_INDEX_RANGE", not invalid)
    record("GOLD_MATCHED_COUNT", matched == 510, matched)
    record("GOLD_UNMATCHED_COUNT", unmatched == 17563, unmatched)

    # 4. Candidate global-index audit
    record(
        "CANDIDATE_SHAPE",
        candidates.shape == (N, TOP_K),
        candidates.shape,
    )

    candidate_range_ok = (
        candidates.min() >= 0
        and candidates.max() < LIB_ROWS
    )

    record(
        "CANDIDATE_GLOBAL_INDEX_RANGE",
        candidate_range_ok,
        f"min={candidates.min()}, max={candidates.max()}",
    )

    sorted_rows = np.sort(candidates, axis=1)

    unique_ok = np.all(
        np.diff(sorted_rows, axis=1) > 0
    )

    record("CANDIDATE_UNIQUENESS", unique_ok)

    leaked = int(np.any(
        candidates[valid] == match[valid, None],
        axis=1
    ).sum())

    record(
        "MATCHED_GOLD_EXCLUDED_FROM_NEGATIVES",
        leaked == 0,
        f"leaked_queries={leaked}",
    )

    # 5. Long20 alignment
    long_ok = (
        len(long_idx) == N_LONG
        and long_idx.min() >= 0
        and long_idx.max() < N
        and len(np.unique(long_idx)) == N_LONG
    )

    record("LONG20_INDEX_ALIGNMENT", long_ok)

    long_match = int((match[long_idx] >= 0).sum())

    record("LONG20_MATCHED_COUNT", long_match == 296, long_match)
    record(
        "LONG20_UNMATCHED_COUNT",
        int(len(long_idx) - long_match) == 8431,
    )

    record(
        "LONG20_DIALOGUE_COUNT",
        len(np.unique(dialogue_ids[long_idx])) == N_LONG_DIALOGUES,
    )

    # 6. Independent ranking reconstruction
    score_pairs = {
        "DENSE_BGE": (
            "candidate_dense_scores",
            "gold_dense_scores",
        )
    }

    for condition in CONDITIONS:
        if condition != "DENSE_BGE":
            score_pairs[condition] = (
                "candidate_scores__" + condition,
                "gold_scores__" + condition,
            )

    rank_results = {}

    for condition in CONDITIONS:
        candidate_key, gold_key = score_pairs[condition]

        candidate_scores = arrays[candidate_key]
        gold_scores = arrays[gold_key]
        stored_ranks = arrays["ranks__" + condition]

        shape_ok = (
            candidate_scores.shape == (N, TOP_K)
            and gold_scores.shape == (N,)
            and stored_ranks.shape == (N,)
        )

        record(condition + "_SCORE_SHAPES", shape_ok)

        finite_ok = (
            np.isfinite(candidate_scores).all()
            and np.isfinite(gold_scores).all()
        )

        record(condition + "_FINITE_SCORES", finite_ok)

        rank_range_ok = (
            stored_ranks.min() >= 1
            and stored_ranks.max() <= TOP_K + 1
        )

        record(condition + "_RANK_RANGE", rank_range_ok)

        reconstructed = (
            1 + np.sum(
                candidate_scores >= gold_scores[:, None],
                axis=1,
            )
        ).astype(np.int32)

        mismatch = int(
            np.sum(reconstructed != stored_ranks)
        )

        record(
            condition + "_RANK_RECONSTRUCTION",
            mismatch == 0,
            f"mismatches={mismatch}",
        )

        rank_results[condition] = {
            "rank_min": int(stored_ranks.min()),
            "rank_max": int(stored_ranks.max()),
            "reconstruction_mismatches": mismatch,
        }

    # 7. Dense Top-100 ordering
    dense_scores = arrays["candidate_dense_scores"]

    dense_sorted = np.all(
        dense_scores[:, :-1] >= dense_scores[:, 1:]
    )

    record("DENSE_TOP100_SCORE_ORDER", dense_sorted)

    # Final result
    failed = [
        x for x in checks
        if x["status"] != "PASS"
    ]

    status = "PASS" if not failed else "FAIL"

    result = {
        "artifact": "U3 D14-A0 Index and Rank Audit V1",
        "status": status,
        "queries": N,
        "response_library_rows": LIB_ROWS,
        "matched_gold": matched,
        "unmatched_gold": unmatched,
        "candidate_count_per_query": TOP_K,
        "long20_queries": N_LONG,
        "conditions": rank_results,
        "total_checks": len(checks),
        "failed_checks": len(failed),
        "checks": checks,
        "training": False,
        "new_ranking_inference": False,
        "test_time_fitting": False,
    }

    write_json(RESULT, result)

    lines = [
        "U3 D14-A0 INDEX AND RANK AUDIT",
        "=" * 75,
        "STATUS: " + status,
        f"QUERIES: {N}",
        f"LIBRARY_ROWS: {LIB_ROWS}",
        f"MATCHED_GOLD: {matched}",
        f"UNMATCHED_GOLD: {unmatched}",
        f"TOP100_NEGATIVES: {TOP_K}",
        "",
        "RANK RECONSTRUCTION",
    ]

    for condition, values in rank_results.items():
        lines.append(
            f"{condition}: "
            f"mismatches={values['reconstruction_mismatches']}, "
            f"rank_range=[{values['rank_min']},{values['rank_max']}]"
        )

    lines += [
        "",
        f"TOTAL_CHECKS: {len(checks)}",
        f"FAILED: {len(failed)}",
        "TRAINING: NO",
        "NEW_RANKING_INFERENCE: NO",
        "TEST_TIME_FITTING: NO",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "artifact": "U3 D14-A0 Audit Manifest V1",
        "status": status,
        "protocol_sha256": sha(AUDIT_PROTOCOL),
        "result_sha256": sha(RESULT),
        "report_sha256": sha(REPORT),
        "source_sha256": protocol["source_sha256"],
        "total_checks": len(checks),
        "failed_checks": len(failed),
    }

    write_json(MANIFEST, manifest)

    print("\n" + "=" * 75)
    print("U3_D14_A0_INDEX_RANK_AUDIT_" + status)
    print("TOTAL CHECKS:", len(checks))
    print("FAILED:", len(failed))
    print("RESULT:", RESULT)
    print("REPORT:", REPORT)
    print("MANIFEST:", MANIFEST)
    print("=" * 75)

    if failed:
        raise SystemExit(2)


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
