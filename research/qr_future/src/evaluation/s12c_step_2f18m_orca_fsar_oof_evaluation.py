import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

TRAIN = ROOT / "results/phase_s12c/step_2f18_orca_fsar_training_v1"

MASTER = ROOT / "data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl"
FOLD = ROOT / "data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy"
MASK = ROOT / "data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz"

CONDITIONS = [
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

SLUGS = {c: c.lower() for c in CONDITIONS}

OOF = {
    c: TRAIN / "conditions" / SLUGS[c] /
       f"{SLUGS[c]}_oof_scores_v1.npz"
    for c in CONDITIONS
}

OUT = ROOT / "results/phase_s12c/step_2f18m_orca_fsar_oof_evaluation_v1"
PROTO = OUT / "evaluation_protocol_v1.json"

FILES = [
    "evaluation_summary_v1.json",
    "evaluation_arrays_v1.npz",
    "evaluation_report_v1.txt",
]

N = 62889
NC = 101
NFOLDS = 5
NBOOT = 5000
SEED = 42

METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20", "nDCG@10"]
KS = [1, 5, 10, 20]

PAIR_CONTRASTS = [
    ("R_FUTURE_FULL_MINUS_RANK_ONLY_FULL",
     "R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),

    ("QR_FUTURE_FULL_MINUS_RANK_ONLY_FULL",
     "QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),

    ("QR_FUTURE_FULL_MINUS_R_FUTURE_FULL",
     "QR_FUTURE_FULL", "R_FUTURE_FULL"),

    ("R_FUTURE_NO_PRODUCT_MINUS_RANK_ONLY_NO_PRODUCT",
     "R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),

    ("QR_FUTURE_NO_PRODUCT_MINUS_RANK_ONLY_NO_PRODUCT",
     "QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),

    ("QR_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_NO_PRODUCT",
     "QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),

    ("R_FUTURE_NO_PRODUCT_MINUS_R_FUTURE_FULL",
     "R_FUTURE_NO_PRODUCT", "R_FUTURE_FULL"),

    ("QR_FUTURE_NO_PRODUCT_MINUS_QR_FUTURE_FULL",
     "QR_FUTURE_NO_PRODUCT", "QR_FUTURE_FULL"),
]

INPUTS = {
    "master": MASTER,
    "fold": FOLD,
    "future_mask": MASK,
    "training_protocol": TRAIN / "training_protocol_v1.json",
    "run_manifest": TRAIN / "run_manifest.json",
}

for c, p in OOF.items():
    INPUTS["oof_" + SLUGS[c]] = p


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def load_base():
    with MASTER.open(encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]

    require(len(rows) == N, "MASTER_ROW_COUNT_MISMATCH")

    qids = np.asarray(
        [str(x["query_id"]) for x in rows], dtype=str
    )
    dids = np.asarray(
        [int(x["dialogue_id"]) for x in rows], dtype=np.int64
    )

    fold = np.load(FOLD, allow_pickle=False)

    require(fold.shape == (N,), "FOLD_SHAPE_MISMATCH")
    require(np.unique(dids).size == 1666, "DIALOGUE_COUNT_MISMATCH")
    require(set(np.unique(fold).tolist()) == set(range(NFOLDS)),
            "FOLD_VALUES_MISMATCH")

    unique_d, inv = np.unique(dids, return_inverse=True)

    dialogue_fold = np.empty(len(unique_d), dtype=np.int8)

    for i in range(len(unique_d)):
        fs = np.unique(fold[inv == i])
        require(len(fs) == 1, "DIALOGUE_FOLD_CONFLICT")
        dialogue_fold[i] = fs[0]

    with np.load(MASK, allow_pickle=False) as z:
        available = z["future_available"].astype(bool)
        labels = z["future_label"].astype(np.int64)

    require(available.shape == (N,), "FUTURE_MASK_SHAPE")
    require(labels.shape == (N,), "FUTURE_LABEL_SHAPE")
    require(int(available.sum()) == 17458,
            "FUTURE_AVAILABLE_COUNT_MISMATCH")

    return qids, dids, fold, unique_d, inv, dialogue_fold, available, labels


def load_and_validate_oof(base, compute_ranks=False):
    qids, dids, fold, unique_d, inv, dialogue_fold, available, labels = base

    ranks = np.empty((6, N), dtype=np.int16)
    ties = np.empty((6, N), dtype=np.int16)

    for i, c in enumerate(CONDITIONS):

        path = OOF[c]
        require(path.is_file(), f"MISSING_OOF: {c}")

        with np.load(path, allow_pickle=False) as z:

            require("candidate_scores" in z.files, c + ":SCORE_MISSING")

            scores = z["candidate_scores"]

            require(scores.shape == (N, NC), c + ":SCORE_SHAPE")
            require(np.isfinite(scores).all(), c + ":NONFINITE_SCORE")
            require(np.array_equal(z["fold_id"], fold),
                    c + ":FOLD_MISMATCH")
            require(np.all(z["gold_position"] == 0),
                    c + ":GOLD_POSITION_MISMATCH")

            if c.startswith("RANK_ONLY"):
                require("future_logits" not in z.files,
                        c + ":UNEXPECTED_FUTURE_HEAD")
            else:
                require(z["future_logits"].shape == (N, 4),
                        c + ":FUTURE_LOGITS_SHAPE")
                require(np.isfinite(z["future_logits"][available]).all(),
                        c + ":NONFINITE_FUTURE_LOGITS")
                require(np.array_equal(
                    z["future_available"].astype(bool), available
                ), c + ":FUTURE_MASK_MISMATCH")
                require(np.array_equal(
                    z["future_value_label"][available],
                    labels[available]
                ), c + ":FUTURE_LABEL_MISMATCH")

            if compute_ranks:
                gold = scores[:, 0:1]
                neg = scores[:, 1:]

                ranks[i] = 1 + (neg > gold).sum(axis=1)
                ties[i] = (neg == gold).sum(axis=1)

        print("OOF_VALIDATION_PASS:", c, flush=True)

    if compute_ranks:
        require(np.all((ranks >= 1) & (ranks <= NC)),
                "INVALID_RANK")
        return ranks, ties

    return None


def input_hashes():
    result = {}

    for name, path in INPUTS.items():
        require(path.is_file(), f"INPUT_MISSING: {name}")
        result[name] = sha(path)
        print("SHA_PASS:", name, result[name], flush=True)

    return result


def prepare():
    require(not OUT.exists(), "OUTPUT_DIRECTORY_EXISTS_REFUSE_OVERWRITE")

    actual = input_hashes()

    base = load_base()
    load_and_validate_oof(base, False)

    protocol = {
        "artifact": "ORCA-FSAR STEP 2F18M Unified Six-Condition OOF Evaluation",
        "status": "FROZEN_BEFORE_METRIC_REVEAL",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "interpretation": "EXPLORATORY_COMPONENT_ANALYSIS_NO_CONFIRMATORY_GATE",
        "script_sha256": sha(Path(__file__).resolve()),
        "inputs": {
            k: {"path": str(v.relative_to(ROOT)), "sha256": actual[k]}
            for k, v in INPUTS.items()
        },
        "data": {
            "queries": N,
            "dialogues": 1666,
            "candidate_count": NC,
            "folds": NFOLDS,
            "future_available": 17458,
        },
        "conditions": CONDITIONS,
        "retrieval": {
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "metrics": METRICS,
            "rank_definition": "1 + COUNT(negative_score > gold_score)",
            "tie_policy": "STRICT_GREATER",
            "ndcg_definition": "Single relevant Gold; DCG@10 / IDCG@10",
            "aggregation": "Query mean within dialogue, then unweighted dialogue mean",
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "same_resampled_dialogues_for_all_conditions": True,
                "replicates": NBOOT,
                "seed": SEED,
                "ci": 0.95,
                "method": "PERCENTILE",
            },
        },
        "planned_contrasts": [
            {"name": x[0], "A": x[1], "B": x[2]}
            for x in PAIR_CONTRASTS
        ] + [{
            "name": "PRODUCT_EFFECT_DIFFERENCE_IN_DIFFERENCES",
            "formula": "(QR_NO_PRODUCT - QR_FULL) - (R_NO_PRODUCT - R_FULL)",
        }],
        "dev_payload_used": False,
        "test_payload_used": False,
        "training": False,
        "new_inference": False,
        "no_posthoc_success_gate": True,
        "overwrite_policy": "REFUSE_IF_OUTPUT_DIRECTORY_EXISTS",
    }

    OUT.mkdir(parents=True)

    PROTO.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSTEP_2F18M_PREPARE_COMPLETE")
    print("PROTOCOL =", PROTO)
    print("PROTOCOL_SHA256 =", sha(PROTO))
    print("SCRIPT_SHA256 =", sha(Path(__file__).resolve()))
    print("METRIC_COMPUTATION = NO")
    print("TRAINING = NO")
    print("FILES_MODIFIED = new evaluation protocol only")


def bootstrap_dialogues(dialogue_metrics):
    nd = dialogue_metrics.shape[0]
    rng = np.random.default_rng(SEED)

    boot = np.empty(
        (NBOOT, len(CONDITIONS), len(METRICS)),
        dtype=np.float64,
    )

    # Small batches keep CPU memory usage low.
    batch_size = 32

    for start in range(0, NBOOT, batch_size):
        end = min(start + batch_size, NBOOT)
        b = end - start

        idx = rng.integers(0, nd, size=(b, nd))

        boot[start:end] = dialogue_metrics[idx].mean(axis=1)

        if start % 1024 == 0:
            print("BOOTSTRAP_PROGRESS:", end, "/", NBOOT, flush=True)

    return boot


def evaluate():
    require(PROTO.is_file(), "FROZEN_PROTOCOL_MISSING")

    protocol = json.loads(PROTO.read_text(encoding="utf-8"))

    require(protocol["status"] == "FROZEN_BEFORE_METRIC_REVEAL",
            "PROTOCOL_STATUS_INVALID")

    require(
        protocol["script_sha256"] == sha(Path(__file__).resolve()),
        "EVALUATION_SCRIPT_HASH_MISMATCH",
    )

    for name, info in protocol["inputs"].items():
        path = ROOT / info["path"]
        got = sha(path)
        require(got == info["sha256"], f"INPUT_HASH_MISMATCH: {name}")
        print("FROZEN_INPUT_PASS:", name, flush=True)

    require(not any((OUT / f).exists() for f in FILES),
            "OUTPUT_FILE_EXISTS_REFUSE_OVERWRITE")

    base = load_base()
    qids, dids, fold, unique_d, inv, dialogue_fold, available, labels = base

    ranks, ties = load_and_validate_oof(base, True)

    print("\nOOF_RANKING_COMPUTATION_PASS")

    # Query-level metrics: [query, condition, metric]
    r = ranks.T.astype(np.float64)

    qmetrics = np.empty((N, 6, len(METRICS)), dtype=np.float64)

    qmetrics[:, :, 0] = 1.0 / r

    for j, k in enumerate(KS, 1):
        qmetrics[:, :, j] = (r <= k).astype(np.float64)

    qmetrics[:, :, 5] = np.where(
        r <= 10,
        1.0 / np.log2(r + 1.0),
        0.0,
    )

    # Aggregate queries within each dialogue.
    nd = len(unique_d)
    counts = np.bincount(inv, minlength=nd)

    dialogue_metrics = np.empty(
        (nd, 6, len(METRICS)),
        dtype=np.float64,
    )

    for c in range(6):
        for m in range(len(METRICS)):
            sums = np.bincount(
                inv,
                weights=qmetrics[:, c, m],
                minlength=nd,
            )
            dialogue_metrics[:, c, m] = sums / counts

    dialogue_macro = dialogue_metrics.mean(axis=0)
    query_micro = qmetrics.mean(axis=0)

    print("DIALOGUE_AGGREGATION_PASS")
    print("DIALOGUES =", nd)

    # Fold-level descriptive metrics.
    fold_macro = {}

    for f in range(NFOLDS):
        select = dialogue_fold == f
        fold_macro[str(f)] = dialogue_metrics[select].mean(axis=0)

    # Paired dialogue bootstrap.
    bootstrap = bootstrap_dialogues(dialogue_metrics)

    print("BOOTSTRAP_COMPLETE")

    idx = {c: i for i, c in enumerate(CONDITIONS)}

    contrasts = {}

    def add_contrast(name, point, boot_delta):
        metrics = {}

        for j, metric in enumerate(METRICS):
            lo, hi = np.quantile(
                boot_delta[:, j], [0.025, 0.975]
            )

            metrics[metric] = {
                "delta": float(point[j]),
                "ci95_lower": float(lo),
                "ci95_upper": float(hi),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            }

        contrasts[name] = metrics

    for name, a, b in PAIR_CONTRASTS:
        ia, ib = idx[a], idx[b]

        add_contrast(
            name,
            dialogue_macro[ia] - dialogue_macro[ib],
            bootstrap[:, ia] - bootstrap[:, ib],
        )

    # Difference-in-differences:
    # (QR_NO_PRODUCT - QR_FULL) - (R_NO_PRODUCT - R_FULL)
    qr_np = idx["QR_FUTURE_NO_PRODUCT"]
    qr_full = idx["QR_FUTURE_FULL"]
    r_np = idx["R_FUTURE_NO_PRODUCT"]
    r_full = idx["R_FUTURE_FULL"]

    did_point = (
        dialogue_macro[qr_np] - dialogue_macro[qr_full]
        - dialogue_macro[r_np] + dialogue_macro[r_full]
    )

    did_boot = (
        bootstrap[:, qr_np] - bootstrap[:, qr_full]
        - bootstrap[:, r_np] + bootstrap[:, r_full]
    )

    add_contrast(
        "PRODUCT_EFFECT_DIFFERENCE_IN_DIFFERENCES",
        did_point,
        did_boot,
    )

    condition_summary = {}

    for i, c in enumerate(CONDITIONS):
        condition_summary[c] = {
            "dialogue_macro": {
                m: float(dialogue_macro[i, j])
                for j, m in enumerate(METRICS)
            },
            "query_micro_descriptive": {
                m: float(query_micro[i, j])
                for j, m in enumerate(METRICS)
            },
            "rank_distribution": {
                "median": float(np.median(ranks[i])),
                "p90": float(np.percentile(ranks[i], 90)),
                "top1": float(np.mean(ranks[i] == 1)),
                "top5": float(np.mean(ranks[i] <= 5)),
            },
            "tie_diagnostics": {
                "queries_with_gold_ties": int((ties[i] > 0).sum()),
                "total_tied_non_gold_candidates": int(ties[i].sum()),
                "max_tied_candidates_per_query": int(ties[i].max()),
            },
        }

    fold_summary = {}

    for f in range(NFOLDS):
        fold_summary[str(f)] = {
            c: {
                m: float(fold_macro[str(f)][i, j])
                for j, m in enumerate(METRICS)
            }
            for i, c in enumerate(CONDITIONS)
        }

    summary = {
        "artifact": "ORCA-FSAR STEP 2F18M Unified OOF Evaluation",
        "status": "PASS",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "protocol_sha256": sha(PROTO),
        "evaluation_script_sha256": sha(Path(__file__).resolve()),
        "input_sha256": {
            k: v["sha256"] for k, v in protocol["inputs"].items()
        },
        "query_rows": N,
        "dialogues": nd,
        "candidate_count": NC,
        "folds": NFOLDS,
        "primary_metric": "DIALOGUE_MACRO_MRR",
        "condition_metrics": condition_summary,
        "fold_dialogue_macro": fold_summary,
        "contrasts": contrasts,
        "bootstrap": protocol["retrieval"]["bootstrap"],
        "rank_definition": protocol["retrieval"]["rank_definition"],
        "tie_policy": "STRICT_GREATER",
        "dev_payload_used": False,
        "test_payload_used": False,
        "training": False,
        "future_state_metrics_computed": False,
        "no_posthoc_success_gate": True,
    }

    # Write atomically.
    tmp_json = OUT / "evaluation_summary_v1.tmp.json"

    tmp_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    os.replace(tmp_json, OUT / FILES[0])

    tmp_npz = OUT / "evaluation_arrays_v1.tmp.npz"

    np.savez_compressed(
        tmp_npz,
        conditions=np.asarray(CONDITIONS, dtype="U40"),
        metrics=np.asarray(METRICS, dtype="U16"),
        query_ids=qids,
        dialogue_ids=dids,
        fold_id=fold,
        ranks=ranks,
        gold_tie_counts=ties,
        dialogue_query_counts=counts,
        dialogue_metrics=dialogue_metrics,
        dialogue_macro=dialogue_macro,
        query_micro=query_micro,
        bootstrap_means=bootstrap,
    )

    os.replace(tmp_npz, OUT / FILES[1])

    lines = [
        "ORCA-FSAR STEP 2F18M",
        "Status: PASS",
        "Scope: TRAIN_INTERNAL_OOF_ONLY",
        f"Queries={N}; dialogues={nd}; candidates={NC}",
        "",
        "DIALOGUE-MACRO RETRIEVAL METRICS",
    ]

    for c in CONDITIONS:
        vals = condition_summary[c]["dialogue_macro"]
        lines.append(
            c + ": " + ", ".join(
                f"{m}={vals[m]:.8f}" for m in METRICS
            )
        )

    lines += [
        "",
        "PAIRED DIALOGUE BOOTSTRAP CONTRASTS",
        "Primary metric: MRR",
    ]

    for name, result in contrasts.items():
        x = result["MRR"]
        lines.append(
            f"{name}: delta={x['delta']:+.8f}, "
            f"CI95=[{x['ci95_lower']:+.8f},"
            f"{x['ci95_upper']:+.8f}]"
        )

    lines += [
        "",
        "Fold-level metrics saved in evaluation_summary_v1.json.",
        "Tie diagnostics saved in evaluation_summary_v1.json.",
        "Exploratory analysis; no confirmatory success gate.",
        "No dev/test payload used.",
        "No training or parameter updates.",
    ]

    tmp_txt = OUT / "evaluation_report_v1.tmp.txt"
    tmp_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp_txt, OUT / FILES[2])

    print("\n" + "=" * 84)
    print("STEP_2F18M_EVALUATION_COMPLETE")
    print("QUERIES =", N)
    print("DIALOGUES =", nd)
    print("CONDITIONS =", len(CONDITIONS))
    print("BOOTSTRAP =", NBOOT)
    print("PRIMARY_METRIC = DIALOGUE_MACRO_MRR")
    print("OUTPUT =", OUT)
    print("STATUS = PASS")
    print("=" * 84)

    for name, result in contrasts.items():
        x = result["MRR"]
        print(
            name,
            "DELTA =", round(x["delta"], 8),
            "CI95 =",
            (round(x["ci95_lower"], 8),
             round(x["ci95_upper"], 8)),
        )


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
