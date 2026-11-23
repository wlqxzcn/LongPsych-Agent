import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/phase_s6"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "DENSE_101",
    "RANK_ONLY",
    "R_FUTURE",
    "QR_FUTURE",
]

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]
KS = [1, 5, 10, 20]
N_BOOT = 5000
BOOT_SEED = 42

INPUTS = {
    "protocol": (
        "protocols/fsar_s6a_oof_retrieval_reveal_protocol_v1.json",
        "d0cd870354e2c4285bf9bbc0a5c26fea016853293d9364db935ab5883b552e7c",
    ),
    "master": (
        "data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl",
        "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",
    ),
    "fold": (
        "data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy",
        "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",
    ),
    "reservoir": (
        "cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz",
        "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",
    ),
    "rank_oof": (
        "results/phase_s4/s4_rank_only_oof_scores_v1.npz",
        "fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4",
    ),
    "rank_summary": (
        "results/phase_s4/s4_rank_only_training_summary_v1.json",
        "be30581e5804c9b7368ed3f23037f79dc78b53659e3ca9abde63a2e4387de6d0",
    ),
    "r_oof": (
        "results/phase_s5/s5_r_future_oof_scores_v1.npz",
        "12deca452373075b19b3699d037578c049d74cb7f49bdc88aebb36e42e9d4ecc",
    ),
    "r_summary": (
        "results/phase_s5/s5_r_future_training_summary_v1.json",
        "ea73d0b84fba2d9415aa23c8f98c800e47b41be1c00e430f0806ad93000edf4c",
    ),
    "qr_oof": (
        "results/phase_s5/s5_qr_future_oof_scores_v1.npz",
        "8928ebb960bc6eb821f7419e188b83bdd306be1dfcbf96764b35381f633ad376",
    ),
    "qr_summary": (
        "results/phase_s5/s5_qr_future_training_summary_v1.json",
        "95564e659e1f53694e7bfb72d39f6b5575420ef5bced6995bbaf10cc52f9fa0a",
    ),
}

OUTPUTS = [
    "s6a_oof_retrieval_ranks_v1.npz",
    "s6a_dialogue_macro_metrics_v1.npz",
    "s6a_mrr_dialogue_bootstrap_v1.npz",
    "s6a_oof_retrieval_reveal_summary_v1.json",
    "s6a_oof_retrieval_reveal_v1.txt",
]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    print("=" * 72)
    print("FSAR PHASE S6-A OOF RETRIEVAL REVEAL")
    print("=" * 72)

    # 1. Protect all formal outputs.
    collisions = [name for name in OUTPUTS if (OUT / name).exists()]
    require(not collisions, f"FORMAL_OUTPUT_COLLISION: {collisions}")
    print("OUTPUT_PROTECTION = PASS")

    # 2. Verify every frozen input against the pre-registered SHA.
    actual_sha = {}
    for name, (relative, expected) in INPUTS.items():
        path = ROOT / relative
        require(path.is_file(), f"MISSING_INPUT: {relative}")
        actual = sha256(path)
        actual_sha[name] = actual
        require(
            actual == expected,
            f"INPUT_SHA_MISMATCH: {relative}; expected={expected}; actual={actual}",
        )
        print(f"SHA_PASS {name} {actual}")

    print("FROZEN_INPUT_SHA_GATE = PASS")

    # 3. Check the frozen protocol.
    protocol = json.loads(
        (ROOT / INPUTS["protocol"][0]).read_text(encoding="utf-8")
    )
    require(
        protocol["status"] == "FROZEN_BEFORE_FIRST_RETRIEVAL_METRIC_COMPUTATION",
        "Unexpected S6-A protocol status",
    )
    require(protocol["evaluation_task"]["queries"] == 62889, "Query count mismatch")
    require(
        protocol["evaluation_task"]["candidate_set_per_query"] == 101,
        "Candidate count mismatch",
    )
    require(protocol["conditions"] == CONDITIONS, "Condition definition mismatch")
    require(
        protocol["bootstrap"]["replicates"] == N_BOOT
        and protocol["bootstrap"]["seed"] == BOOT_SEED,
        "Bootstrap settings differ from frozen protocol",
    )
    require(
        protocol["confirmatory_contrast"]["name"]
        == "QR_FUTURE_MINUS_RANK_ONLY",
        "Primary contrast mismatch",
    )

    # 4. Load the canonical training-only query index.
    master_path = ROOT / INPUTS["master"][0]
    with master_path.open("r", encoding="utf-8") as f:
        master = [json.loads(line) for line in f]

    n = len(master)
    require(n == 62889, f"Unexpected master rows: {n}")

    ranking_row = np.asarray([x["ranking_row"] for x in master], dtype=np.int64)
    require(
        np.array_equal(ranking_row, np.arange(n)),
        "Master ranking_row order mismatch",
    )

    query_ids = np.asarray([x["query_id"] for x in master], dtype=str)
    dialogue_ids = np.asarray(
        [x["dialogue_id"] for x in master], dtype=np.int64
    )
    master_gold_rows = np.asarray(
        [x["gold_response_row"] for x in master], dtype=np.int32
    )

    require(len(np.unique(query_ids)) == n, "Duplicate query IDs")
    unique_dialogues, dialogue_inverse = np.unique(
        dialogue_ids, return_inverse=True
    )
    n_dialogues = len(unique_dialogues)
    require(n_dialogues == 1666, f"Unexpected dialogue count: {n_dialogues}")

    print(f"MASTER_ROWS = {n}")
    print(f"DIALOGUES = {n_dialogues}")
    print("MASTER_ROW_AND_IDENTITY_CHECK = PASS")

    # 5. Verify candidate reservoir and gold mapping.
    reservoir = load_npz(ROOT / INPUTS["reservoir"][0])
    negative_scores = reservoir["candidate_similarity"]
    gold_similarity = reservoir["gold_similarity"]
    negative_rows = reservoir["candidate_response_row"]
    reservoir_gold_rows = reservoir["gold_response_row"]

    require(negative_scores.shape == (n, 100), "Reservoir score shape mismatch")
    require(negative_rows.shape == (n, 100), "Reservoir identity shape mismatch")
    require(gold_similarity.shape == (n,), "Gold score shape mismatch")
    require(
        np.array_equal(master_gold_rows, reservoir_gold_rows),
        "Master/reservoir gold response row mismatch",
    )
    require(
        not np.any(negative_rows == reservoir_gold_rows[:, None]),
        "Gold response appears in negative reservoir",
    )

    dense_scores = np.concatenate(
        [gold_similarity[:, None], negative_scores], axis=1
    ).astype(np.float32, copy=False)

    require(dense_scores.shape == (n, 101), "Dense candidate shape mismatch")
    require(np.isfinite(dense_scores).all(), "Non-finite dense scores")

    print("DENSE_CANDIDATE_CONSTRUCTION = PASS")

    # 6. Load frozen OOF scores and check row/fold alignment.
    fold = np.load(ROOT / INPUTS["fold"][0], allow_pickle=False)
    require(fold.shape == (n,), "Fold assignment shape mismatch")

    rank_data = load_npz(ROOT / INPUTS["rank_oof"][0])
    r_data = load_npz(ROOT / INPUTS["r_oof"][0])
    qr_data = load_npz(ROOT / INPUTS["qr_oof"][0])

    learned = [rank_data, r_data, qr_data]
    for label, data in zip(
        ["RANK_ONLY", "R_FUTURE", "QR_FUTURE"], learned
    ):
        scores = data["candidate_scores"]
        require(scores.shape == (n, 101), f"{label} score shape mismatch")
        require(np.isfinite(scores).all(), f"{label} has non-finite scores")
        require(
            np.array_equal(data["fold_id"], fold),
            f"{label} fold alignment mismatch",
        )
        require(
            np.all(data["gold_position"] == 0),
            f"{label} gold position is not zero",
        )
        require(data["ranking_ce"].shape == (n,), f"{label} CE shape mismatch")
        require(np.isfinite(data["ranking_ce"]).all(), f"{label} CE non-finite")

    # Verify frozen training summaries were marked successful.
    for key in ["rank_summary", "r_summary", "qr_summary"]:
        summary = json.loads(
            (ROOT / INPUTS[key][0]).read_text(encoding="utf-8")
        )
        require(summary.get("status") == "PASS", f"{key} status is not PASS")

    print("LEARNED_OOF_SCORE_AND_FOLD_ALIGNMENT = PASS")

    # 7. Compute ranks using the frozen STRICT_GREATER tie policy.
    score_matrices = [
        dense_scores,
        rank_data["candidate_scores"],
        r_data["candidate_scores"],
        qr_data["candidate_scores"],
    ]

    ranks = np.empty((4, n), dtype=np.int16)
    tie_counts = np.empty((4, n), dtype=np.int16)

    for c, scores in enumerate(score_matrices):
        gold = scores[:, 0:1]
        negatives = scores[:, 1:]
        ranks[c] = 1 + np.sum(negatives > gold, axis=1)
        tie_counts[c] = np.sum(negatives == gold, axis=1)

        require(
            np.all((ranks[c] >= 1) & (ranks[c] <= 101)),
            f"Invalid rank values for {CONDITIONS[c]}",
        )

    print("\n===== OOF QUERY-MICRO METRICS =====")
    query_metrics = np.empty((n, 4, 5), dtype=np.float64)

    for c, condition in enumerate(CONDITIONS):
        query_metrics[:, c, 0] = 1.0 / ranks[c]
        for j, k in enumerate(KS, start=1):
            query_metrics[:, c, j] = ranks[c] <= k

        print(f"\n{condition}")
        print(f"  MRR  = {query_metrics[:, c, 0].mean():.8f}")
        for j, k in enumerate(KS, start=1):
            print(f"  R@{k:<2} = {query_metrics[:, c, j].mean():.8f}")
        print(
            "  queries_with_gold_ties =",
            int(np.count_nonzero(tie_counts[c] > 0)),
        )
        print("  total_gold_tied_candidates =", int(tie_counts[c].sum()))
        print("  max_gold_ties_per_query =", int(tie_counts[c].max()))

    # 8. Aggregate per query, then equally across dialogues.
    # The primary metric is the unweighted dialogue macro-average.
    dialogue_counts = np.bincount(
        dialogue_inverse, minlength=n_dialogues
    ).astype(np.int64)

    dialogue_metrics = np.empty(
        (n_dialogues, 4, 5), dtype=np.float64
    )

    for c in range(4):
        for m in range(5):
            sums = np.bincount(
                dialogue_inverse,
                weights=query_metrics[:, c, m],
                minlength=n_dialogues,
            )
            dialogue_metrics[:, c, m] = sums / dialogue_counts

    dialogue_macro = dialogue_metrics.mean(axis=0)
    query_micro = query_metrics.mean(axis=0)

    print("\n===== DIALOGUE-MACRO METRICS =====")
    for c, condition in enumerate(CONDITIONS):
        print(f"\n{condition}")
        for m, metric in enumerate(METRICS):
            print(f"  {metric} = {dialogue_macro[c, m]:.8f}")

    # 9. Paired dialogue bootstrap. Every condition uses the same sampled
    # dialogue indices within each replicate.
    mrr_by_dialogue = dialogue_metrics[:, :, 0]
    rng = np.random.default_rng(BOOT_SEED)
    bootstrap_means = np.empty((N_BOOT, 4), dtype=np.float64)

    for b in range(N_BOOT):
        sampled_dialogues = rng.integers(
            0, n_dialogues, size=n_dialogues
        )
        bootstrap_means[b] = mrr_by_dialogue[sampled_dialogues].mean(axis=0)

    contrasts = [
        ("QR_FUTURE_MINUS_RANK_ONLY", 3, 1),
        ("R_FUTURE_MINUS_RANK_ONLY", 2, 1),
        ("QR_FUTURE_MINUS_R_FUTURE", 3, 2),
        ("RANK_ONLY_MINUS_DENSE_101", 1, 0),
    ]

    contrast_names = []
    bootstrap_deltas = []
    point_estimates = []
    ci_lower = []
    ci_upper = []

    print("\n===== PAIRED DIALOGUE BOOTSTRAP: MRR =====")

    for name, a, b in contrasts:
        delta_samples = bootstrap_means[:, a] - bootstrap_means[:, b]
        point = float(dialogue_macro[a, 0] - dialogue_macro[b, 0])
        lo, hi = np.quantile(delta_samples, [0.025, 0.975])

        contrast_names.append(name)
        bootstrap_deltas.append(delta_samples)
        point_estimates.append(point)
        ci_lower.append(float(lo))
        ci_upper.append(float(hi))

        print(f"\n{name}")
        print(f"  delta_mrr = {point:.8f}")
        print(f"  ci95_lower = {lo:.8f}")
        print(f"  ci95_upper = {hi:.8f}")

    bootstrap_deltas = np.asarray(bootstrap_deltas)
    point_estimates = np.asarray(point_estimates, dtype=np.float64)
    ci_lower = np.asarray(ci_lower, dtype=np.float64)
    ci_upper = np.asarray(ci_upper, dtype=np.float64)

    primary_lower = float(ci_lower[0])
    primary_success = primary_lower > 0.0

    print("\n===== PRIMARY CONFIRMATORY GATE =====")
    print("primary_contrast = QR_FUTURE_MINUS_RANK_ONLY")
    print(f"primary_ci95_lower = {primary_lower:.8f}")
    print("success_gate = lower_bound > 0")
    print(f"PRIMARY_SUCCESS = {primary_success}")

    # 10. Construct formal artifacts in a staging directory.
    stage = Path(tempfile.mkdtemp(prefix=".s6a_stage_", dir=OUT))

    try:
        np.savez_compressed(
            stage / OUTPUTS[0],
            conditions=np.asarray(CONDITIONS, dtype="U16"),
            query_ids=query_ids,
            dialogue_ids=dialogue_ids,
            fold_id=fold,
            ranks=ranks,
            gold_tie_counts=tie_counts,
            gold_position=np.zeros(n, dtype=np.int8),
        )

        np.savez_compressed(
            stage / OUTPUTS[1],
            conditions=np.asarray(CONDITIONS, dtype="U16"),
            metric_names=np.asarray(METRICS, dtype="U8"),
            dialogue_ids=unique_dialogues,
            dialogue_query_counts=dialogue_counts,
            dialogue_metrics=dialogue_metrics,
            dialogue_macro=dialogue_macro,
            query_micro=query_micro,
        )

        np.savez_compressed(
            stage / OUTPUTS[2],
            conditions=np.asarray(CONDITIONS, dtype="U16"),
            contrast_names=np.asarray(contrast_names, dtype="U32"),
            bootstrap_seed=np.asarray(BOOT_SEED, dtype=np.int64),
            bootstrap_replicates=np.asarray(N_BOOT, dtype=np.int64),
            bootstrap_means=bootstrap_means,
            bootstrap_deltas=bootstrap_deltas,
            point_estimates=point_estimates,
            ci95_lower=ci_lower,
            ci95_upper=ci_upper,
        )

        condition_summary = {}
        tie_summary = {}

        for c, condition in enumerate(CONDITIONS):
            condition_summary[condition] = {
                "dialogue_macro": {
                    METRICS[m]: float(dialogue_macro[c, m])
                    for m in range(5)
                },
                "query_micro": {
                    METRICS[m]: float(query_micro[c, m])
                    for m in range(5)
                },
            }
            tie_summary[condition] = {
                "queries_with_gold_ties": int(
                    np.count_nonzero(tie_counts[c] > 0)
                ),
                "total_tied_non_gold_candidates": int(tie_counts[c].sum()),
                "max_tied_candidates_in_one_query": int(tie_counts[c].max()),
            }

        contrast_summary = []
        for i, (name, a, b) in enumerate(contrasts):
            contrast_summary.append({
                "name": name,
                "metric": "DIALOGUE_MACRO_MRR",
                "condition_a": CONDITIONS[a],
                "condition_b": CONDITIONS[b],
                "delta_a_minus_b": float(point_estimates[i]),
                "ci95_lower": float(ci_lower[i]),
                "ci95_upper": float(ci_upper[i]),
                "primary_confirmatory": i == 0,
            })

        summary = {
            "artifact": "FSAR S6-A Unified OOF Retrieval Reveal V1",
            "status": "PASS",
            "scope": "TRAIN_INTERNAL_OOF_ONLY",
            "evaluation_task": "CANDIDATE_CONDITIONED_RERANKING",
            "query_rows": int(n),
            "dialogues": int(n_dialogues),
            "candidates_per_query": 101,
            "candidate_set": "1_GOLD_PLUS_FROZEN_DENSE_TOP100_NON_GOLD",
            "conditions": CONDITIONS,
            "rank_definition": "1 + COUNT(candidate_score > gold_score)",
            "tie_policy": "STRICT_GREATER",
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "condition_metrics": condition_summary,
            "tie_diagnostics": tie_summary,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": N_BOOT,
                "seed": BOOT_SEED,
                "ci": 0.95,
                "method": "PERCENTILE",
                "contrasts": contrast_summary,
            },
            "primary_confirmatory_result": {
                "contrast": "QR_FUTURE_MINUS_RANK_ONLY",
                "delta_mrr": float(point_estimates[0]),
                "ci95_lower": float(ci_lower[0]),
                "ci95_upper": float(ci_upper[0]),
                "success_gate": "CI95_LOWER_GT_0",
                "success": bool(primary_success),
            },
            "retrieval_metrics_computed": True,
            "future_metrics_computed": False,
            "dev_payload_used": False,
            "test_payload_used": False,
            "reranker_training_performed": False,
            "frozen_input_sha256": actual_sha,
            "candidate_column_identity_note": (
                "Candidate column order is inherited from the frozen "
                "S4/S5 training protocol and shared reservoir. OOF score "
                "arrays do not independently store candidate IDs per column."
            ),
        }

        with (stage / OUTPUTS[3]).open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

        # Refuse promotion if an output unexpectedly appeared meanwhile.
        require(
            not any((OUT / name).exists() for name in OUTPUTS),
            "Formal output appeared during evaluation; refusing promotion",
        )

        for name in OUTPUTS[:4]:
            (stage / name).replace(OUT / name)

    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("\n===== FORMAL ARTIFACT PROMOTION =====")
    for name in OUTPUTS[:4]:
        path = OUT / name
        require(path.is_file() and path.stat().st_size > 0,
                f"Missing/empty promoted output: {name}")
        print(f"PROMOTED {name} bytes={path.stat().st_size}")

    print("FORMAL_ARTIFACT_PROMOTION = PASS")
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("RERANKER_TRAINING_PERFORMED = NO")
    print("PHASE_S6A_OOF_RETRIEVAL_REVEAL = PASS")


if __name__ == "__main__":
    main()
