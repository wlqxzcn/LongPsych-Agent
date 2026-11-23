#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/evaluation"))

import s7_beta8_oof_evaluation as legacy

PROTO = ROOT / "protocols/fsar_s12b_component_ablation_evaluation_v1.json"
OUT = ROOT / "results/phase_s12b/evaluation_v1"

NBOOT = 5000
CLASSES = np.array([-2, -1, 0, 1, 2])

CONDITIONS = [
    ("S7_R", "s7_r_oof", "s7_r_summary"),
    ("S7_QR", "s7_qr_oof", "s7_qr_summary"),
    ("S12B_R", "s12b_r_oof", "s12b_r_summary"),
    ("S12B_QR", "s12b_qr_oof", "s12b_qr_summary"),
]

CONTRASTS = [
    ("S12B_R_MINUS_S7_R", 2, 0),
    ("S12B_QR_MINUS_S7_QR", 3, 1),
    ("S7_QR_MINUS_S7_R", 1, 0),
    ("S12B_QR_MINUS_S12B_R", 3, 2),
]

RETRIEVAL_METRICS = [
    "MRR", "R@1", "R@5", "R@10", "R@20"
]

FUTURE_METRICS = [
    "brier",
    "ordinal_logloss",
    "expected_state_mae",
    "threshold_accuracy",
]

FILES = [
    "s12b_component_ablation_evaluation_summary_v1.json",
    "s12b_component_ablation_evaluation_arrays_v1.npz",
    "s12b_component_ablation_evaluation_report_v1.txt",
]


def req(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_protocol():

    req(not PROTO.exists(), "Protocol already exists; refusing overwrite")

    inputs = {
        "master": {
            "path": "data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl",
            "sha256": "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be"
        },
        "fold": {
            "path": "data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy",
            "sha256": "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9"
        },
        "future_mask": {
            "path": "data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz",
            "sha256": "84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d"
        },
        "reservoir": {
            "path": "cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz",
            "sha256": "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0"
        },

        "s7_r_oof": {
            "path": "results/phase_s7/beta_8/s7_beta8_r_future_oof_scores_v1.npz",
            "sha256": "73a0fa73d43797ed0c28b1f3dfd30c82429da72fdf654fc69e8738d939f9543f"
        },
        "s7_r_summary": {
            "path": "results/phase_s7/beta_8/s7_beta8_r_future_training_summary_v1.json",
            "sha256": "92e06e2f09a95352af81cfc41bb88728f10457bc1f900d9bb17e97d4ee6235cf"
        },
        "s7_qr_oof": {
            "path": "results/phase_s7/beta_8/s7_beta8_qr_future_oof_scores_v1.npz",
            "sha256": "62bafd483fd939134e686f67ab71f7ce116c771d0d2bcb5f85e4b323b9b6d51e"
        },
        "s7_qr_summary": {
            "path": "results/phase_s7/beta_8/s7_beta8_qr_future_training_summary_v1.json",
            "sha256": "c5f061d7e0fef33ea1ecf4db6bbe150a885e2c891a59efb32f698a8dfff687ff"
        },

        "s12b_r_oof": {
            "path": "results/phase_s12b/s12b_beta8_r_future_oof_scores_v1.npz",
            "sha256": "7fa4e32c0678b7f9a102e7682b7cf75b3a756703cc7b702286abaaa21415e585"
        },
        "s12b_r_summary": {
            "path": "results/phase_s12b/s12b_beta8_r_future_training_summary_v1.json",
            "sha256": "893a275cd4b1e66579088565825282e1a42be7fe96dc45bda8e1b3dde46e0f95"
        },
        "s12b_qr_oof": {
            "path": "results/phase_s12b/s12b_beta8_qr_future_oof_scores_v1.npz",
            "sha256": "e82d6ac591285d362577f5e607d812b7f1b4f38d074ac00bf81372444a21988d"
        },
        "s12b_qr_summary": {
            "path": "results/phase_s12b/s12b_beta8_qr_future_training_summary_v1.json",
            "sha256": "c4a14d0d5da1a0baff2182a8c8ffce4a3f0d57c84c0e467f5ad74c7f80edac31"
        },

        "s7_protocol": {
            "path": "protocols/fsar_s7_future_aux_beta_sensitivity_v1.json",
            "sha256": "6f3924f74a1b3168ea6b75f7f2383ce41b0b2715c236b66ec53f38341edcb23d"
        },
        "s7_training_script": {
            "path": "src/training/s7_train_future_aux_beta_sensitivity.py",
            "sha256": "afbebfa650d5ada14b3e5ad23b7935c9a91c0d7fe561f50e3ede31f02dd9e9a7"
        },
        "s7_eval_script": {
            "path": "src/evaluation/s7_beta8_oof_evaluation.py",
            "sha256": "dacbc93f025dfeb2e90f79919fd7ae91899ea58f4a906fb5d7ebe34606b81536"
        },
        "s12b_model": {
            "path": "src/models/s12b_fsar_ablation_models.py",
            "sha256": "10878111132b0e9ec46032c4bcd3e613e6a0a0742b561776bee34a0dc6c8ca4f"
        },
        "s12b_protocol": {
            "path": "protocols/fsar_s12b_no_product_ablation_v1.json",
            "sha256": "b9edb00d1860e0acbe52d76db54fa5c343c79afec4f3bf5eb6444cca0649f974"
        },
        "s12b_training_script": {
            "path": "src/training/s12b_train_no_product_ablation_beta8_5fold.py",
            "sha256": "53a8dbf9c221d6727b0df5e496efa27e733d06937b0c6d3ea586a2859e7230f3"
        },
    }

    protocol = {
        "project": "PsychResponse_FSAR",
        "phase": "S12B_COMPONENT_ABLATION_OOF_EVALUATION",
        "artifact": "FSAR S12B Four-Model Component Ablation Evaluation V1",
        "status": "FROZEN_BEFORE_S12B_COMPONENT_METRIC_REVEAL",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "interpretation": "EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE",
        "evaluation_script": "src/evaluation/s12b_component_ablation_oof_evaluation.py",
        "evaluation_script_sha256": sha(Path(__file__)),
        "inputs": inputs,

        "retrieval": {
            "conditions": ["S7_R", "S7_QR", "S12B_R", "S12B_QR"],
            "candidate_count": 101,
            "rank_definition": "1 + COUNT(candidate_score > gold_score)",
            "tie_policy": "STRICT_GREATER",
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "metrics": RETRIEVAL_METRICS,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": 5000,
                "seed": 42,
                "ci": 0.95,
                "method": "PERCENTILE"
            }
        },

        "future_state": {
            "eligible_rows": 17458,
            "eligible_dialogues": 1653,
            "classes": [-2, -1, 0, 1, 2],
            "thresholds": [-2, -1, 0, 1],
            "classification_metrics": [
                "accuracy", "macro_f1", "balanced_accuracy",
                "weighted_f1", "ordinal_mae"
            ],
            "probability_metrics": FUTURE_METRICS,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": 5000,
                "seed": 43,
                "ci": 0.95,
                "method": "PERCENTILE"
            }
        },

        "contrasts": [
            "S12B_R_MINUS_S7_R",
            "S12B_QR_MINUS_S7_QR",
            "S7_QR_MINUS_S7_R",
            "S12B_QR_MINUS_S12B_R",
            "DIFFERENCE_IN_DIFFERENCES"
        ],

        "outputs": [
            "results/phase_s12b/evaluation_v1/" + x
            for x in FILES
        ],

        "dev_payload_used": False,
        "test_payload_used": False,
        "no_posthoc_success_gate": True,
        "overwrite_policy": "REFUSE_IF_OUTPUT_DIRECTORY_EXISTS"
    }

    PROTO.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    print("PROTOCOL_CREATED =", PROTO)
    print("PROTOCOL_SHA256 =", sha(PROTO))
    print("SCRIPT_SHA256 =", sha(Path(__file__)))


def preflight():

    req(PROTO.is_file(), "Missing frozen evaluation protocol")

    p = read_json(PROTO)

    req(
        p["status"] == "FROZEN_BEFORE_S12B_COMPONENT_METRIC_REVEAL",
        "Protocol status mismatch"
    )

    req(
        p["evaluation_script_sha256"] == sha(Path(__file__)),
        "Evaluation script SHA mismatch"
    )

    req(not OUT.exists(), "Output directory collision; refusing overwrite")

    actual = {}

    for name, info in p["inputs"].items():
        path = ROOT / info["path"]

        req(path.is_file(), f"Missing input: {name}")

        got = sha(path)

        req(
            got == info["sha256"],
            f"SHA mismatch: {name}; actual={got}"
        )

        actual[name] = got

        print("SHA_PASS", name, got)

    master_path = ROOT / p["inputs"]["master"]["path"]

    with master_path.open(encoding="utf-8") as f:
        master = [json.loads(x) for x in f]

    n = len(master)

    req(n == 62889, f"Unexpected master size: {n}")

    dids = np.asarray(
        [int(x["dialogue_id"]) for x in master],
        dtype=np.int64
    )

    avail = np.asarray(
        [bool(x["future_value_available"]) for x in master],
        dtype=bool
    )

    y = np.asarray([
        int(x["future_value_label"]) if avail[i] else -2
        for i, x in enumerate(master)
    ], dtype=np.int64)

    fold = np.load(
        ROOT / p["inputs"]["fold"]["path"],
        allow_pickle=False
    )

    req(fold.shape == (n,), "Fold shape mismatch")
    req(avail.sum() == 17458, "Future eligible rows mismatch")
    req(np.unique(dids[avail]).size == 1653,
        "Future eligible dialogues mismatch")

    res = legacy.npz(
        ROOT / p["inputs"]["reservoir"]["path"]
    )

    negrows = res["candidate_response_row"]
    goldrows = res["gold_response_row"]

    master_gold = np.asarray(
        [int(x["gold_response_row"]) for x in master],
        dtype=np.int32
    )

    req(
        negrows.shape == (n, 100),
        "Negative reservoir shape mismatch"
    )

    req(
        np.array_equal(goldrows, master_gold),
        "Reservoir gold response mismatch"
    )

    req(
        not np.any(negrows == master_gold[:, None]),
        "Gold response found in negative reservoir"
    )

    loaded = {}
    summaries = {}

    for name, oof_key, summary_key in CONDITIONS:

        d = legacy.npz(
            ROOT / p["inputs"][oof_key]["path"]
        )

        req(
            d["candidate_scores"].shape == (n, 101),
            f"{name}: candidate score shape mismatch"
        )

        req(
            np.isfinite(d["candidate_scores"]).all(),
            f"{name}: nonfinite candidate scores"
        )

        req(
            d["ranking_ce"].shape == (n,),
            f"{name}: ranking CE shape mismatch"
        )

        req(
            np.isfinite(d["ranking_ce"]).all(),
            f"{name}: nonfinite ranking CE"
        )

        req(
            d["future_logits"].shape == (n, 4),
            f"{name}: future logits shape mismatch"
        )

        req(
            np.isfinite(d["future_logits"]).all(),
            f"{name}: nonfinite future logits"
        )

        req(
            np.array_equal(d["fold_id"], fold),
            f"{name}: fold mismatch"
        )

        req(
            np.all(d["gold_position"] == 0),
            f"{name}: gold position mismatch"
        )

        req(
            np.array_equal(d["future_available"], avail),
            f"{name}: future availability mismatch"
        )

        req(
            np.array_equal(d["future_value_label"][avail], y[avail]),
            f"{name}: future label mismatch"
        )

        s = read_json(
            ROOT / p["inputs"][summary_key]["path"]
        )

        req(s.get("status") == "PASS", f"{name}: training failed")
        req(float(s.get("beta")) == 8.0, f"{name}: beta mismatch")
        req(s.get("dev_payload_used") is False, f"{name}: dev payload used")
        req(s.get("test_payload_used") is False, f"{name}: test payload used")
        req(s.get("formal_outputs_promoted") is True,
            f"{name}: outputs not promoted")
        req(s.get("ranking_rows") == n, f"{name}: ranking row mismatch")
        req(s.get("future_value_rows") == int(avail.sum()),
            f"{name}: future row mismatch")

        loaded[name] = d
        summaries[name] = s

    shared_keys = [
        "query_embedding", "response_embedding",
        "master", "fold", "future_mask", "reservoir",
        "training_protocol", "s5_protocol", "s7_protocol",
        "s4_summary", "s4_oof"
    ]

    for key in shared_keys:
        values = [
            summaries[name]["input_sha256"][key]
            for name, _, _ in CONDITIONS
        ]

        req(
            len(set(values)) == 1,
            f"Training summary input mismatch: {key}"
        )

    req(
        summaries["S7_R"]["input_sha256"]["model_code"]
        == summaries["S7_QR"]["input_sha256"]["model_code"],
        "S7 model code mismatch"
    )

    req(
        summaries["S12B_R"]["input_sha256"]["model_code"]
        == summaries["S12B_QR"]["input_sha256"]["model_code"],
        "S12B model code mismatch"
    )

    req(
        summaries["S12B_R"]["input_sha256"]["model_code"]
        == p["inputs"]["s12b_model"]["sha256"],
        "S12B model provenance mismatch"
    )

    req(
        summaries["S12B_R"]["input_sha256"]["architecture_protocol"]
        == p["inputs"]["s12b_protocol"]["sha256"],
        "S12B architecture protocol mismatch"
    )

    for key in [
        "fold_id", "future_available",
        "future_value_label", "gold_position"
    ]:
        reference = loaded["S7_R"][key]

        for name, _, _ in CONDITIONS[1:]:
            req(
                np.array_equal(reference, loaded[name][key]),
                f"Cross-model alignment failure: {key}/{name}"
            )

    print("\nS12B_PRELIGHT_HASH_GATE = PASS")
    print("QUERY_ROWS =", n)
    print("DIALOGUES =", len(np.unique(dids)))
    print("FUTURE_ROWS =", int(avail.sum()))
    print("FUTURE_DIALOGUES =", int(np.unique(dids[avail]).size))
    print("MODELS = 4")
    print("CANDIDATES_PER_QUERY = 101")
    print("OUTPUT_COLLISION = NONE")
    print("METRIC_COMPUTATION = NO")

    return p, actual, master, dids, fold, avail, y, loaded


def add_did(boots, points, names, directions):

    delta_boot = boots[:, 3, :] - boots[:, 2, :] \
        - boots[:, 1, :] + boots[:, 0, :]

    delta_point = points[3] - points[2] - points[1] + points[0]

    result = []

    for j, metric in enumerate(names):

        lo, hi = np.quantile(
            delta_boot[:, j], [0.025, 0.975]
        )

        result.append({
            "contrast": "DIFFERENCE_IN_DIFFERENCES",
            "metric": metric,
            "delta": float(delta_point[j]),
            "ci95_lower": float(lo),
            "ci95_upper": float(hi),
            "direction_of_improvement": directions[metric],
            "status": "EXPLORATORY_NO_CONFIRMATORY_GATE"
        })

    return result


def evaluate(data):

    p, actual, master, dids, fold, avail, y, loaded = data

    n = len(master)

    qids = np.asarray(
        [str(x["query_id"]) for x in master],
        dtype=str
    )

    scores = [
        loaded[name]["candidate_scores"]
        for name, _, _ in CONDITIONS
    ]

    ranks = np.empty((4, n), dtype=np.int16)
    ties = np.empty_like(ranks)
    qm = np.empty((n, 4, 5), dtype=np.float64)

    for c, s in enumerate(scores):

        ranks[c] = 1 + (
            s[:, 1:] > s[:, 0:1]
        ).sum(axis=1)

        ties[c] = (
            s[:, 1:] == s[:, 0:1]
        ).sum(axis=1)

        qm[:, c, 0] = 1.0 / ranks[c]

        for j, k in enumerate([1, 5, 10, 20], 1):
            qm[:, c, j] = ranks[c] <= k

    unique_d, inv = np.unique(dids, return_inverse=True)

    counts = np.bincount(inv, minlength=len(unique_d))

    dlg = np.empty((len(unique_d), 4, 5), dtype=np.float64)

    for c in range(4):
        for m in range(5):
            dlg[:, c, m] = (
                np.bincount(
                    inv,
                    weights=qm[:, c, m],
                    minlength=len(unique_d)
                ) / counts
            )

    macro = dlg.mean(axis=0)
    micro = qm.mean(axis=0)

    rb = legacy.boot_means(dlg, 42)

    rdirs = {m: "higher_is_better" for m in RETRIEVAL_METRICS}

    rcontr = legacy.contrasts(
        rb, macro, CONTRASTS,
        RETRIEVAL_METRICS, rdirs
    )

    rcontr += add_did(
        rb, macro, RETRIEVAL_METRICS, rdirs
    )

    # Future state: all models use the same eligible dialogue universe.

    freports = {}
    fdmetrics = []
    fpreds = []
    fdlg_ids = None

    for name, _, _ in CONDITIONS:

        rep, pred, dlgmet, ud, rows = legacy.future_metrics(
            name,
            loaded[name]["future_logits"],
            y, avail, fold, dids
        )

        if fdlg_ids is None:
            fdlg_ids = ud
        else:
            req(
                np.array_equal(fdlg_ids, ud),
                "Future dialogue universe mismatch"
            )

        freports[name] = rep
        fdmetrics.append(dlgmet)
        fpreds.append(pred)

    fmacro = np.stack(
        [x.mean(axis=0) for x in fdmetrics],
        axis=0
    )

    fd_all = np.stack(fdmetrics, axis=1)

    fb = np.empty((NBOOT, 4, 4), dtype=np.float64)
    rng = np.random.default_rng(43)

    nd = len(fdlg_ids)

    for b in range(NBOOT):
        fb[b] = fd_all[
            rng.integers(0, nd, size=nd)
        ].mean(axis=0)

    fdirs = {
        "brier": "lower_is_better",
        "ordinal_logloss": "lower_is_better",
        "expected_state_mae": "lower_is_better",
        "threshold_accuracy": "higher_is_better"
    }

    fcontr = legacy.contrasts(
        fb, fmacro, CONTRASTS,
        FUTURE_METRICS, fdirs
    )

    fcontr += add_did(
        fb, fmacro, FUTURE_METRICS, fdirs
    )

    retrieval_summary = {}
    tie_summary = {}

    for i, (name, _, _) in enumerate(CONDITIONS):

        retrieval_summary[name] = {
            "dialogue_macro": {
                m: float(macro[i, j])
                for j, m in enumerate(RETRIEVAL_METRICS)
            },
            "query_micro": {
                m: float(micro[i, j])
                for j, m in enumerate(RETRIEVAL_METRICS)
            }
        }

        tie_summary[name] = {
            "queries_with_gold_ties": int((ties[i] > 0).sum()),
            "total_tied_non_gold_candidates": int(ties[i].sum()),
            "max_tied_candidates_per_query": int(ties[i].max())
        }

        freports[name]["dialogue_macro_probability_metrics"] = {
            m: float(fmacro[i, j])
            for j, m in enumerate(FUTURE_METRICS)
        }

    summary = {
        "artifact": "FSAR S12B Four-Model Component Ablation Evaluation V1",
        "status": "PASS",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "interpretation": "EXPLORATORY_COMPONENT_ABLATION_NO_CONFIRMATORY_GATE",
        "protocol_sha256": sha(PROTO),
        "evaluation_script_sha256": sha(Path(__file__)),
        "input_sha256": actual,
        "query_rows": int(n),
        "dialogues": int(len(unique_d)),
        "candidate_count": 101,
        "future_eligible_rows": int(avail.sum()),
        "future_eligible_dialogues": int(len(fdlg_ids)),

        "retrieval": {
            "conditions": [x[0] for x in CONDITIONS],
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "rank_definition": "1 + COUNT(candidate_score > gold_score)",
            "tie_policy": "STRICT_GREATER",
            "condition_metrics": retrieval_summary,
            "tie_diagnostics": tie_summary,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 42,
                "ci": 0.95,
                "method": "PERCENTILE",
                "contrasts": rcontr
            }
        },

        "future_state": {
            "conditions": [x[0] for x in CONDITIONS],
            "classification_metrics": freports,
            "probability_metric_names": FUTURE_METRICS,
            "probability_metric_dialogue_macro": {
                name: {
                    m: float(fmacro[i, j])
                    for j, m in enumerate(FUTURE_METRICS)
                }
                for i, (name, _, _) in enumerate(CONDITIONS)
            },
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 43,
                "ci": 0.95,
                "method": "PERCENTILE",
                "contrasts": fcontr
            }
        },

        "dev_payload_used": False,
        "test_payload_used": False,
        "no_posthoc_success_gate": True
    }

    OUT.mkdir(parents=True, exist_ok=False)

    stage = Path(
        tempfile.mkdtemp(prefix=".stage_", dir=OUT)
    )

    try:

        (stage / FILES[0]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )

        future_logits = np.stack([
            loaded[name]["future_logits"]
            for name, _, _ in CONDITIONS
        ])

        np.savez_compressed(
            stage / FILES[1],
            retrieval_conditions=np.asarray(
                [x[0] for x in CONDITIONS], dtype="U32"
            ),
            query_ids=qids,
            dialogue_ids=dids,
            fold_id=fold,
            ranks=ranks,
            gold_tie_counts=ties,
            dialogue_query_counts=counts,
            dialogue_metrics=dlg,
            dialogue_macro=macro,
            query_micro=micro,
            retrieval_bootstrap_means=rb,
            future_conditions=np.asarray(
                [x[0] for x in CONDITIONS], dtype="U32"
            ),
            future_eligible_row_indices=np.flatnonzero(avail),
            future_available=avail,
            future_value_label=y,
            future_logits=future_logits,
            future_predictions=np.asarray(fpreds),
            future_bootstrap_means=fb
        )

        lines = [
            "FSAR S12B FOUR-MODEL COMPONENT ABLATION",
            "Status: PASS",
            "Scope: TRAIN_INTERNAL_OOF_ONLY",
            f"Queries={n}; dialogues={len(unique_d)}; candidates=101",
            f"Future eligible rows={int(avail.sum())}",
            f"Future eligible dialogues={len(fdlg_ids)}",
            "",
            "RETRIEVAL DIALOGUE-MACRO"
        ]

        for name, _, _ in CONDITIONS:
            lines.append(
                name + ": " + ", ".join(
                    f"{k}={v:.8f}"
                    for k, v in retrieval_summary[name]["dialogue_macro"].items()
                )
            )

        lines += ["", "RETRIEVAL CONTRASTS"]

        for x in rcontr:
            if x["metric"] == "MRR":
                lines.append(json.dumps(x))

        lines += ["", "FUTURE-STATE CLASSIFICATION"]

        for name, _, _ in CONDITIONS:
            lines.append(
                name + ": " +
                json.dumps(freports[name]["metrics"])
            )

        lines += ["", "FUTURE-STATE PROBABILITY METRICS"]

        for name, _, _ in CONDITIONS:
            lines.append(
                name + ": " +
                json.dumps(
                    freports[name]["dialogue_macro_probability_metrics"]
                )
            )

        lines += [
            "",
            "FUTURE-STATE CONTRASTS"
        ]

        for x in fcontr:
            lines.append(json.dumps(x))

        lines += [
            "",
            "Exploratory analysis only.",
            "No confirmatory success gate.",
            "No dev/test payload used."
        ]

        (stage / FILES[2]).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

        for filename in FILES:
            os.replace(stage / filename, OUT / filename)

    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("\nS12B_COMPONENT_ABLATION_EVALUATION_PASS")

    for name, _, _ in CONDITIONS:
        print(name, retrieval_summary[name]["dialogue_macro"])

    print("OUTPUT =", OUT)


def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--freeze-protocol", action="store_true")
    group.add_argument("--preflight-only", action="store_true")
    group.add_argument("--run-evaluation", action="store_true")

    args = parser.parse_args()

    if args.freeze_protocol:
        freeze_protocol()
        return

    data = preflight()

    if args.preflight_only:
        print("\nS12B_PREFLIGHT_ONLY_PASS")
        return

    if args.run_evaluation:
        evaluate(data)


if __name__ == "__main__":
    main()
