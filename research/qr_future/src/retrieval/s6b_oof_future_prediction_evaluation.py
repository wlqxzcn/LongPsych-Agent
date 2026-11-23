import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_REL = "protocols/fsar_s6b_oof_future_prediction_evaluation_v1.json"
EXPECTED_PROTOCOL_SHA = (
    "72928cc629b7f32034628f26022af61903c1ee00a7e58e73624a38c7f73d9d58"
)
CLASSES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
THRESHOLDS = [-2, -1, 0, 1]
N_CLASSES = 5

SUMMARY_PATH = ROOT / (
    "results/phase_s6/s6b_oof_future_prediction_metrics_v1.json"
)
PREDICTIONS_PATH = ROOT / (
    "results/phase_s6/s6b_oof_future_prediction_predictions_v1.npz"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def principal_metrics(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(
            y, pred, labels=CLASSES, average="macro",
            zero_division=0
        )),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "weighted_f1": float(f1_score(
            y, pred, labels=CLASSES, average="weighted",
            zero_division=0
        )),
        "ordinal_mae": float(mean_absolute_error(y, pred)),
    }


def weighted_confusion(y, pred, weights):
    yi = y.astype(np.int64) - int(CLASSES[0])
    pi = pred.astype(np.int64) - int(CLASSES[0])
    check(
        np.all((yi >= 0) & (yi < N_CLASSES)),
        "Invalid true class index"
    )
    check(
        np.all((pi >= 0) & (pi < N_CLASSES)),
        "Invalid predicted class index"
    )
    return np.bincount(
        yi * N_CLASSES + pi,
        weights=weights,
        minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)


def metrics_from_confusion(cm):
    total = float(cm.sum())
    check(total > 0, "Empty bootstrap sample")

    support = cm.sum(axis=1)
    predicted_support = cm.sum(axis=0)
    tp = np.diag(cm)

    precision = np.divide(
        tp, predicted_support,
        out=np.zeros(N_CLASSES, dtype=float),
        where=predicted_support > 0,
    )
    recall = np.divide(
        tp, support,
        out=np.zeros(N_CLASSES, dtype=float),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(N_CLASSES, dtype=float),
        where=(precision + recall) > 0,
    )

    costs = np.abs(
        np.arange(N_CLASSES)[:, None]
        - np.arange(N_CLASSES)[None, :]
    )

    return {
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "ordinal_mae": float((cm * costs).sum() / total),
    }


def evaluate_model(name, logits, y, available, fold_ids, dialogue_ids):
    eligible_logits = logits[available]
    pred = (
        int(CLASSES[0])
        + (eligible_logits >= 0).sum(axis=1)
    ).astype(np.int64)

    yy = y[available]
    ff = fold_ids[available]
    dd = dialogue_ids[available]

    metrics = principal_metrics(yy, pred)
    cm = confusion_matrix(yy, pred, labels=CLASSES)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            yy, pred, labels=CLASSES, zero_division=0
        )
    )

    per_class = {}
    for i, c in enumerate(CLASSES):
        per_class[str(int(c))] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    per_threshold = {}
    for j, threshold in enumerate(THRESHOLDS):
        target = (yy > threshold).astype(np.int64)
        binary_pred = (eligible_logits[:, j] >= 0).astype(np.int64)

        item = {
            "positive_support": int(target.sum()),
            "negative_support": int(len(target) - target.sum()),
            "accuracy": float(accuracy_score(target, binary_pred)),
            "balanced_accuracy": float(
                balanced_accuracy_score(target, binary_pred)
            ),
            "f1": float(f1_score(
                target, binary_pred, zero_division=0
            )),
            "roc_auc": float(
                roc_auc_score(target, eligible_logits[:, j])
            ),
            "average_precision": float(
                average_precision_score(
                    target, eligible_logits[:, j]
                )
            ),
        }
        per_threshold[str(threshold)] = item

    per_fold = {}
    for fold in sorted(np.unique(ff).tolist()):
        m = ff == fold
        per_fold[str(int(fold))] = {
            "rows": int(m.sum()),
            "dialogues": int(np.unique(dd[m]).size),
            "metrics": principal_metrics(yy[m], pred[m]),
            "class_support": {
                str(int(c)): int(np.sum(yy[m] == c))
                for c in CLASSES
            },
        }

    # Thresholds [-2, -1, 0, 1] imply non-increasing logits.
    # Report violations; do not repair or reorder logits.
    violations = np.any(
        eligible_logits[:, :-1] < eligible_logits[:, 1:],
        axis=1,
    )

    report = {
        "n": int(len(yy)),
        "labeled_dialogues": int(np.unique(dd).size),
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix_labels": CLASSES.tolist(),
        "confusion_matrix": cm.tolist(),
        "per_threshold": per_threshold,
        "per_fold": per_fold,
        "ordinal_logit_monotonicity": {
            "violating_rows": int(violations.sum()),
            "eligible_rows": int(len(violations)),
            "violation_rate": float(violations.mean()),
            "expected_order": "non-increasing across ascending thresholds",
        },
    }

    return report, pred


def main():
    protocol_path = ROOT / PROTOCOL_REL
    check(protocol_path.exists(), "Missing frozen S6-B protocol")

    protocol_sha = sha256_file(protocol_path)
    check(
        protocol_sha == EXPECTED_PROTOCOL_SHA,
        f"Protocol SHA mismatch: {protocol_sha}",
    )

    with protocol_path.open("r", encoding="utf-8") as f:
        protocol = json.load(f)

    print("=" * 76)
    print("FSAR S6-B OOF FUTURE-STATE PREDICTION EVALUATION")
    print("=" * 76)
    print("PROTOCOL_SHA256 =", protocol_sha)

    # Verify every input recorded in the frozen protocol.
    input_hashes = {}
    for name, info in protocol["inputs"].items():
        path = ROOT / info["path"]
        check(path.is_file(), f"Missing input: {info['path']}")
        actual = sha256_file(path)
        expected = info["sha256"]
        check(
            actual == expected,
            f"SHA mismatch for {name}: expected={expected}, actual={actual}",
        )
        input_hashes[name] = actual
        print("SHA_PASS", name, actual)

    check(
        not SUMMARY_PATH.exists() and not PREDICTIONS_PATH.exists(),
        "Formal JSON/NPZ output already exists; refusing overwrite",
    )

    with (ROOT / protocol["inputs"]["master_index"]["path"]).open(
        "r", encoding="utf-8"
    ) as f:
        master = [json.loads(line) for line in f]

    n = len(master)
    available = np.array(
        [bool(r["future_value_available"]) for r in master],
        dtype=bool,
    )
    y = np.array([
        int(r["future_value_label"]) if available[i] else -2
        for i, r in enumerate(master)
    ], dtype=np.int64)
    dialogue_ids = np.array(
        [int(r["dialogue_id"]) for r in master],
        dtype=np.int64,
    )
    query_ids = np.array(
        [str(r["query_id"]) for r in master],
        dtype=str,
    )

    fold_path = ROOT / protocol["inputs"]["fold_assignment"]["path"]
    frozen_folds = np.load(fold_path, allow_pickle=False)

    check(n == 62889, f"Unexpected master row count: {n}")
    check(available.sum() == 17458, "Unexpected labeled-row count")
    check(
        np.unique(dialogue_ids[available]).size == 1653,
        "Unexpected labeled-dialogue count",
    )
    check(
        np.array_equal(
            np.array([
                int(r["future_value_label"])
                for i, r in enumerate(master) if available[i]
            ]),
            y[available],
        ),
        "Master label construction mismatch",
    )
    check(frozen_folds.shape == (n,), "Invalid frozen fold shape")

    expected_counts = {
        int(k): int(v)
        for k, v in protocol["cohort"]["expected_class_counts"].items()
    }
    observed_counts = {
        int(k): int(v)
        for k, v in Counter(y[available].tolist()).items()
    }
    check(
        observed_counts == expected_counts,
        f"Class-count mismatch: {observed_counts}",
    )

    print("\nMASTER_ROWS =", n)
    print("ELIGIBLE_ROWS =", int(available.sum()))
    print("ELIGIBLE_DIALOGUES =", int(np.unique(dialogue_ids[available]).size))
    print("CLASS_COUNTS =", observed_counts)

    models = {}
    for name in ["R_FUTURE", "QR_FUTURE"]:
        key = "R_FUTURE_OOF" if name == "R_FUTURE" else "QR_FUTURE_OOF"
        path = ROOT / protocol["inputs"][key]["path"]

        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k].copy() for k in z.files}

        for required in [
            "future_logits",
            "future_available",
            "future_value_label",
            "fold_id",
        ]:
            check(required in a, f"{name} missing key {required}")

        logits = a["future_logits"]
        stored_available = a["future_available"]
        stored_y = a["future_value_label"]
        stored_folds = a["fold_id"]

        check(logits.shape == (n, 4), f"{name}: invalid logits shape")
        check(
            np.array_equal(stored_available, available),
            f"{name}: availability alignment failure",
        )
        check(
            np.array_equal(stored_folds, frozen_folds),
            f"{name}: fold alignment failure",
        )
        check(
            np.array_equal(stored_y[available], y[available]),
            f"{name}: eligible label mismatch",
        )
        check(
            np.all(stored_y[~available] == -2),
            f"{name}: unexpected unavailable-row sentinel",
        )
        check(np.isfinite(logits).all(), f"{name}: nonfinite logits")

        models[name] = logits
        print(f"{name}_ALIGNMENT = PASS")
        print(f"{name}_LOGITS_SHAPE = {logits.shape}")

    print(
        "FUTURE_LOGITS_IDENTICAL =",
        bool(np.array_equal(models["R_FUTURE"], models["QR_FUTURE"])),
    )

    reports = {}
    predictions = {}

    for name in ["R_FUTURE", "QR_FUTURE"]:
        report, pred = evaluate_model(
            name, models[name], y, available,
            frozen_folds, dialogue_ids,
        )
        reports[name] = report
        predictions[name] = pred

        print("\n" + "=" * 76)
        print(name)
        print("=" * 76)
        print("METRICS =", json.dumps(
            report["metrics"], ensure_ascii=False
        ))
        print("PER_CLASS =", json.dumps(
            report["per_class"], ensure_ascii=False
        ))
        print("CONFUSION_MATRIX =", report["confusion_matrix"])
        print("PER_THRESHOLD =", json.dumps(
            report["per_threshold"], ensure_ascii=False
        ))
        print("LOGIT_MONOTONICITY =", json.dumps(
            report["ordinal_logit_monotonicity"]
        ))

        for fold, fold_report in report["per_fold"].items():
            print(
                f"FOLD={fold} rows={fold_report['rows']} "
                f"dialogues={fold_report['dialogues']} "
                f"metrics={fold_report['metrics']}"
            )

    # Paired dialogue-cluster bootstrap. Both models use identical draws.
    eligible_idx = np.flatnonzero(available)
    yy = y[eligible_idx]
    dd = dialogue_ids[eligible_idx]
    pred_r = predictions["R_FUTURE"]
    pred_qr = predictions["QR_FUTURE"]

    unique_dialogues, row_group = np.unique(dd, return_inverse=True)
    group_count = len(unique_dialogues)

    bootstrap_replicates = int(
        protocol["comparison"]["bootstrap_replicates"]
    )
    bootstrap_seed = int(protocol["comparison"]["bootstrap_seed"])
    rng = np.random.default_rng(bootstrap_seed)

    deltas = {
        "macro_f1": np.empty(bootstrap_replicates, dtype=np.float64),
        "balanced_accuracy": np.empty(bootstrap_replicates, dtype=np.float64),
        "ordinal_mae": np.empty(bootstrap_replicates, dtype=np.float64),
    }

    print("\nBOOTSTRAP_REPLICATES =", bootstrap_replicates)
    print("BOOTSTRAP_DIALOGUES =", group_count)
    print("BOOTSTRAP_SEED =", bootstrap_seed)

    for b in range(bootstrap_replicates):
        sampled_groups = rng.integers(
            0, group_count, size=group_count
        )
        multiplicity = np.bincount(
            sampled_groups, minlength=group_count
        )
        row_weights = multiplicity[row_group].astype(np.float64)

        cm_r = weighted_confusion(yy, pred_r, row_weights)
        cm_qr = weighted_confusion(yy, pred_qr, row_weights)

        mr = metrics_from_confusion(cm_r)
        mqr = metrics_from_confusion(cm_qr)

        for metric in deltas:
            deltas[metric][b] = mqr[metric] - mr[metric]

        if (b + 1) % 1000 == 0:
            print(f"BOOTSTRAP_PROGRESS = {b + 1}/{bootstrap_replicates}")

    comparison = {}
    point_r = reports["R_FUTURE"]["metrics"]
    point_qr = reports["QR_FUTURE"]["metrics"]

    for metric, samples in deltas.items():
        comparison[metric] = {
            "R_FUTURE": point_r[metric],
            "QR_FUTURE": point_qr[metric],
            "delta_QR_minus_R": float(point_qr[metric] - point_r[metric]),
            "ci95_lower": float(np.quantile(samples, 0.025)),
            "ci95_upper": float(np.quantile(samples, 0.975)),
            "bootstrap_replicates": bootstrap_replicates,
            "resampling_unit": "dialogue_id",
        }

    print("\nPAIRED_DIALOGUE_BOOTSTRAP =")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))

    summary = {
        "artifact": "FSAR S6-B OOF Future-State Prediction Evaluation V1",
        "status": "PASS",
        "protocol_sha256": protocol_sha,
        "input_sha256": input_hashes,
        "cohort": {
            "master_rows": int(n),
            "eligible_rows": int(available.sum()),
            "eligible_dialogues": int(
                np.unique(dialogue_ids[available]).size
            ),
            "class_counts": {
                str(k): v for k, v in sorted(observed_counts.items())
            },
            "filter": "future_available == true",
        },
        "prediction_policy": {
            "thresholds": THRESHOLDS,
            "binary_decision": "logit >= 0",
            "class_decoding": "-2 + number of nonnegative logits",
            "logits_repaired": False,
        },
        "models": reports,
        "comparison": {
            "contrast": "QR_FUTURE minus R_FUTURE",
            "paired_dialogue_bootstrap": comparison,
        },
        "training_performed": False,
        "dev_payload_used": False,
        "test_payload_used": False,
        "s6a_primary_result_overridden": False,
    }

    # Save only after all gates and metrics have completed successfully.
    np.savez_compressed(
        PREDICTIONS_PATH,
        eligible_row_indices=eligible_idx,
        query_ids=query_ids[eligible_idx],
        dialogue_ids=dialogue_ids[eligible_idx],
        fold_id=frozen_folds[eligible_idx],
        y_true=y[eligible_idx],
        r_future_logits=models["R_FUTURE"][eligible_idx],
        qr_future_logits=models["QR_FUTURE"][eligible_idx],
        r_future_pred=predictions["R_FUTURE"],
        qr_future_pred=predictions["QR_FUTURE"],
    )

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print("\nSUMMARY_JSON =", SUMMARY_PATH)
    print("PREDICTIONS_NPZ =", PREDICTIONS_PATH)
    print("S6B_INPUT_SHA_GATE = PASS")
    print("S6B_OOF_FUTURE_PREDICTION_EVALUATION = PASS")
    print("TRAINING_PERFORMED = NO")
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")


if __name__ == "__main__":
    main()
