#!/usr/bin/env python3
from __future__ import annotations

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
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import s7_beta8_oof_evaluation as legacy

PROTO = ROOT / "protocols/fsar_s12c_future_input_source_oof_evaluation_v1.json"
OUT = ROOT / "results/phase_s12c/future_input_source_v1/evaluation_v1"

FILES = [
    "s12c_future_input_source_oof_evaluation_summary_v1.json",
    "s12c_future_input_source_oof_evaluation_arrays_v1.npz",
    "s12c_future_input_source_oof_evaluation_report_v1.txt",
]

N = 62889
NFOLDS = 5
NBOOT = 5000

CLASSES = np.array([-2, -1, 0, 1, 2])
F_METRICS = [
    "brier", "ordinal_logloss",
    "expected_state_mae", "threshold_accuracy"
]
R_METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]
KS = [1, 5, 10, 20]
ALPHA = 1.0

PATHS = {
    "master": "data/supervision/fsar_s3/fsar_s3a_ranking_future_master_index_v1.jsonl",
    "fold": "data/supervision/fsar_s3/fsar_s3b_internal_fold_assignment_v1.npy",
    "future_mask": "data/supervision/fsar_s3/fsar_s3c0_future_loss_eligibility_by_fold_v1.npz",
    "reservoir": "cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz",
    "model_code": "src/models/s12c_future_input_source_models.py",

    "r_only_oof": "results/phase_s12c/future_input_source_v1/oof/s12c_beta8_r_only_oof_scores_v1.npz",
    "q_only_oof": "results/phase_s12c/future_input_source_v1/oof/s12c_beta8_q_only_oof_scores_v1.npz",
    "qr_add_oof": "results/phase_s12c/future_input_source_v1/oof/s12c_beta8_qr_add_oof_scores_v1.npz",

    "r_only_summary": "results/phase_s12c/future_input_source_v1/summaries/s12c_beta8_r_only_training_summary_v1.json",
    "q_only_summary": "results/phase_s12c/future_input_source_v1/summaries/s12c_beta8_q_only_training_summary_v1.json",
    "qr_add_summary": "results/phase_s12c/future_input_source_v1/summaries/s12c_beta8_qr_add_training_summary_v1.json",

    "rank_only_oof": "results/phase_s4/s4_rank_only_oof_scores_v1.npz",
}

EXPECTED = {
    "master": "d9196a5ea11a9433c0550db6ebcfce76a6965ebec14ae9e3fb6660cc1e6ab3be",
    "fold": "a4f129fb6b5e5d23aad4215d261d33e4be2dc1ff34f8dcb7e4f0c92d430bcdf9",
    "future_mask": "84ee56cf9549a9842dc717bfd6a8a3cad41c457298356a8796dbfbe7a9b85a3d",
    "reservoir": "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",
    "model_code": "67ff3cc1a7af8248aef45ebe01e85b3b98d6000257aefe40c9f68250eb8d6c36",

    "r_only_oof": "c2b9cabf149f7517cc6457f81fe0a8defa6b74ec43b83a5b22725213f24732ac",
    "q_only_oof": "34a71e4ec56063855728c928118913676f3f76cd6feeee8a8cf5bb18a401f75f",
    "qr_add_oof": "6e304f2e5666b71648890af8a1bf45c8378d56729877a0aade026bae7d669f07",

    "r_only_summary": "1889ce7dd0ccaee94c2d6fdc14dbec14c4a36c9485e18b84f8f2ca2faadcb140",
    "q_only_summary": "71fa5f755be92477d3cd1467315923edfd28dcca01f95194e1c1c6f5698a1c93",
    "qr_add_summary": "80f3ae74ec1df6e1ae461568f921cd2cdff594e951bebb01f7fab3663b07c714",

    "rank_only_oof": "fbcb8947ebf72b1364e535a15872ac3415fdb56959b5f00091c464a75fc092f4",
}

OOF_KEY = {
    "R_ONLY": "r_only_oof",
    "Q_ONLY": "q_only_oof",
    "QR_ADD": "qr_add_oof",
}

SUM_KEY = {
    "R_ONLY": "r_only_summary",
    "Q_ONLY": "q_only_summary",
    "QR_ADD": "qr_add_summary",
}


def req(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def legacy_dependency():
    path = Path(legacy.__file__).resolve()

    req(path.is_file(), "Legacy metric source missing.")

    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "Legacy metric source is outside project root."
        ) from exc

    return {
        "path": rel,
        "sha256": sha(path),
    }


def npload(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def verify_hashes():
    actual = {}

    for name, rel in PATHS.items():
        path = ROOT / rel
        req(path.is_file(), f"Missing {name}: {path}")

        got = sha(path)
        req(
            got == EXPECTED[name],
            f"SHA mismatch {name}: expected={EXPECTED[name]} actual={got}"
        )

        actual[name] = got
        print("SHA_PASS", name, got)

    return actual


def load_data():
    with (ROOT / PATHS["master"]).open(encoding="utf-8") as f:
        master = [json.loads(x) for x in f if x.strip()]

    n = len(master)
    req(n == N, f"master rows={n}")

    qids = np.asarray([str(x["query_id"]) for x in master])
    dids = np.asarray([int(x["dialogue_id"]) for x in master])
    avail = np.asarray(
        [bool(x["future_value_available"]) for x in master]
    )

    y = np.asarray([
        int(x["future_value_label"]) if avail[i] else -2
        for i, x in enumerate(master)
    ])

    req(np.unique(qids).size == n, "Duplicate query_id")
    req(int(avail.sum()) == 17458, "Future row count mismatch")
    req(np.unique(dids[avail]).size == 1653, "Future dialogue mismatch")

    fold = np.load(ROOT / PATHS["fold"], allow_pickle=False)

    req(
        fold.shape == (n,)
        and set(np.unique(fold).tolist()) == set(range(NFOLDS)),
        "Fold shape/id mismatch"
    )

    mask = npload(ROOT / PATHS["future_mask"])
    req(
        "future_loss_eligible_by_fold" in mask,
        "Future mask key missing"
    )

    eligible = np.asarray(
        mask["future_loss_eligible_by_fold"],
        dtype=bool
    )

    req(eligible.shape == (NFOLDS, n), "Future mask shape mismatch")

    reservoir = npload(ROOT / PATHS["reservoir"])

    dense = np.column_stack([
        reservoir["gold_similarity"],
        reservoir["candidate_similarity"]
    ]).astype(np.float32)

    req(
        dense.shape == (n, 101) and np.isfinite(dense).all(),
        "Dense score mismatch"
    )

    rank = npload(ROOT / PATHS["rank_only_oof"])["candidate_scores"]

    req(
        rank.shape == (n, 101) and np.isfinite(rank).all(),
        "Rank-only OOF mismatch"
    )

    oof = {}

    for cond in OOF_KEY:
        a = npload(ROOT / PATHS[OOF_KEY[cond]])

        for k in [
            "candidate_scores", "ranking_ce", "future_logits",
            "fold_id", "future_available",
            "future_value_label", "gold_position"
        ]:
            req(k in a, f"{cond} OOF missing {k}")

        req(
            a["candidate_scores"].shape == (n, 101)
            and np.isfinite(a["candidate_scores"]).all(),
            f"{cond} rank scores invalid"
        )

        req(
            a["future_logits"].shape == (n, 4)
            and np.isfinite(a["future_logits"]).all(),
            f"{cond} future logits invalid"
        )

        req(np.array_equal(a["fold_id"], fold), f"{cond} fold mismatch")
        req(np.array_equal(a["future_available"], avail), f"{cond} availability mismatch")
        req(np.array_equal(a["future_value_label"][avail], y[avail]), f"{cond} label mismatch")
        req(np.all(a["gold_position"] == 0), f"{cond} gold position mismatch")

        s = jload(ROOT / PATHS[SUM_KEY[cond]])

        req(s.get("status") == "PASS", f"{cond} summary status invalid")
        req(s.get("condition") == cond, f"{cond} summary condition mismatch")
        req(float(s.get("beta", -1)) == 8.0, f"{cond} beta mismatch")
        req(s.get("formal_outputs_promoted") is True, f"{cond} promotion invalid")
        req(s.get("dev_payload_used") is False, f"{cond} used Dev")
        req(s.get("test_payload_used") is False, f"{cond} used Test")

        oof[cond] = a
        print("OOF_ALIGNMENT_PASS", cond)

    return {
        "qids": qids,
        "dids": dids,
        "avail": avail,
        "y": y,
        "fold": fold,
        "eligible": eligible,
        "dense": dense,
        "rank": rank,
        "oof": oof,
    }


def prepare():
    req(not PROTO.exists(), "Protocol already exists; inspect first")
    req(not OUT.exists(), "Evaluation output already exists; inspect first")

    metric_dependency = legacy_dependency()

    actual = verify_hashes()
    d = load_data()

    protocol = {
        "project": "PsychResponse_FSAR",
        "phase": "S12C_FUTURE_INPUT_SOURCE_OOF_EVALUATION",
        "artifact": "FSAR S12C Future Input Source OOF Evaluation V1",
        "status": "FROZEN_BEFORE_METRIC_REVEAL",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "interpretation": "EXPLORATORY_INPUT_SOURCE_ABLATION_NO_CONFIRMATORY_GATE",
        "evaluation_script": "src/evaluation/s12c_future_input_source_oof_evaluation.py",
        "evaluation_script_sha256": sha(Path(__file__).resolve()),
        "legacy_metric_dependency": metric_dependency,
        "inputs": {
            k: {"path": PATHS[k], "sha256": actual[k]}
            for k in PATHS
        },
        "data": {
            "ranking_rows": N,
            "dialogues": int(np.unique(d["dids"]).size),
            "future_eligible_rows": 17458,
            "future_eligible_dialogues": 1653,
            "candidate_count": 101,
            "folds": NFOLDS,
            "split_unit": "dialogue",
        },
        "conditions": {
            "R_ONLY": "[z_r, zeros]",
            "Q_ONLY": "[zeros, z_q]",
            "QR_ADD": "[z_r, z_q]",
        },
        "future_state": {
            "classes": CLASSES.tolist(),
            "thresholds": [-2, -1, 0, 1],
            "classification_metrics": [
                "accuracy", "macro_f1", "balanced_accuracy",
                "weighted_f1", "ordinal_mae"
            ],
            "probability_metrics": F_METRICS,
            "primary_contrast": "QR_ADD - Q_ONLY",
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 43,
                "ci": 0.95,
                "method": "PERCENTILE",
            },
        },
        "fold_aware_prevalence_baseline": {
            "training_rows": "future_available AND future_loss_eligible_by_fold[f] AND fold_id != f",
            "estimator": "five-class empirical prevalence with add-one Laplace smoothing",
            "alpha_per_class": ALPHA,
            "validation_labels_used": False,
        },
        "retrieval": {
            "conditions": ["DENSE_101", "RANK_ONLY", "R_ONLY", "Q_ONLY", "QR_ADD"],
            "metrics": R_METRICS,
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "rank_definition": "1 + COUNT(negative_score > gold_score)",
            "tie_policy": "STRICT_GREATER",
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 42,
                "ci": 0.95,
                "method": "PERCENTILE",
            },
            "role": "SECONDARY",
        },
        "dev_payload_used": False,
        "test_payload_used": False,
        "training": False,
        "new_inference": False,
        "no_posthoc_success_gate": True,
        "multiple_comparison_adjustment": "NONE; exploratory contrasts",
        "overwrite_policy": "REFUSE_IF_PROTOCOL_OR_OUTPUT_EXISTS",
    }

    PROTO.parent.mkdir(parents=True, exist_ok=True)
    tmp = PROTO.with_suffix(PROTO.suffix + ".tmp")

    tmp.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    os.replace(tmp, PROTO)

    print("PROTOCOL_CREATED =", PROTO)
    print("PROTOCOL_SHA256 =", sha(PROTO))
    print("METRICS_COMPUTED = NO")
    print("TRAINING = NO")
    print("NEW_INFERENCE = NO")
    print("STEP_2F8E_PREPARE = PASS")


def boot(dlg, seed):
    nd, nc, nm = dlg.shape
    rng = np.random.default_rng(seed)
    out = np.empty((NBOOT, nc, nm), dtype=np.float64)

    for b in range(NBOOT):
        idx = rng.integers(0, nd, size=nd)
        out[b] = dlg[idx].mean(axis=0)

    return out


def contrasts(boots, points, defs, metrics, directions):
    out = []

    for name, a, b in defs:
        for j, m in enumerate(metrics):
            z = boots[:, a, j] - boots[:, b, j]
            lo, hi = np.quantile(z, [0.025, 0.975])

            out.append({
                "contrast": name,
                "metric": m,
                "delta_a_minus_b": float(points[a, j] - points[b, j]),
                "ci95_lower": float(lo),
                "ci95_upper": float(hi),
                "direction_of_improvement": directions[m],
                "status": "EXPLORATORY_NO_CONFIRMATORY_GATE",
            })

    return out


def prevalence(d):
    logits = np.zeros((N, 4), dtype=np.float32)

    counts_all, probs_all, thresholds_all = [], [], []
    train_rows, train_dialogues, valid_rows = [], [], []

    for f in range(NFOLDS):
        tr = d["avail"] & d["eligible"][f] & (d["fold"] != f)
        va = d["fold"] == f

        req(tr.any(), f"Fold {f}: empty baseline training set")
        req((d["avail"] & va).any(), f"Fold {f}: empty validation set")

        counts = np.bincount(d["y"][tr] + 2, minlength=5)
        p = (counts.astype(np.float64) + ALPHA) / (
            counts.sum() + 5 * ALPHA
        )

        tp = np.array([
            p[1:].sum(), p[2:].sum(),
            p[3:].sum(), p[4]
        ])
        tp = np.clip(tp, 1e-8, 1 - 1e-8)

        logits[va] = np.log(tp / (1 - tp)).astype(np.float32)

        counts_all.append(counts)
        probs_all.append(p)
        thresholds_all.append(tp)

        train_rows.append(int(tr.sum()))
        train_dialogues.append(int(np.unique(d["dids"][tr]).size))
        valid_rows.append(int((d["avail"] & va).sum()))

    req(np.isfinite(logits).all(), "Baseline logits nonfinite")
    req(sum(valid_rows) == 17458, "Baseline coverage mismatch")

    return (
        logits, np.asarray(counts_all), np.asarray(probs_all),
        np.asarray(thresholds_all), train_rows,
        train_dialogues, valid_rows
    )


def retrieval_eval(d):
    scores = {
        "DENSE_101": d["dense"],
        "RANK_ONLY": d["rank"],
    }

    for c in OOF_KEY:
        scores[c] = d["oof"][c]["candidate_scores"]

    names = list(scores)
    n = len(d["dids"])

    ranks = np.empty((len(names), n), dtype=np.int16)
    ties = np.empty_like(ranks)
    qm = np.empty((n, len(names), 5), dtype=np.float64)

    for i, name in enumerate(names):
        s = scores[name]
        ranks[i] = 1 + (s[:, 1:] > s[:, 0:1]).sum(axis=1)
        ties[i] = (s[:, 1:] == s[:, 0:1]).sum(axis=1)

        qm[:, i, 0] = 1 / ranks[i]

        for j, k in enumerate(KS, 1):
            qm[:, i, j] = ranks[i] <= k

    ud, inv = np.unique(d["dids"], return_inverse=True)
    cnt = np.bincount(inv, minlength=len(ud))

    dlg = np.empty((len(ud), len(names), 5), dtype=np.float64)

    for i in range(len(names)):
        for j in range(5):
            dlg[:, i, j] = (
                np.bincount(inv, weights=qm[:, i, j], minlength=len(ud)) / cnt
            )

    macro = dlg.mean(axis=0)
    micro = qm.mean(axis=0)
    b = boot(dlg, 42)

    ix = {x: i for i, x in enumerate(names)}

    defs = [
        ("QR_ADD_MINUS_Q_ONLY", ix["QR_ADD"], ix["Q_ONLY"]),
        ("QR_ADD_MINUS_R_ONLY", ix["QR_ADD"], ix["R_ONLY"]),
        ("R_ONLY_MINUS_Q_ONLY", ix["R_ONLY"], ix["Q_ONLY"]),
        ("R_ONLY_MINUS_RANK_ONLY", ix["R_ONLY"], ix["RANK_ONLY"]),
        ("Q_ONLY_MINUS_RANK_ONLY", ix["Q_ONLY"], ix["RANK_ONLY"]),
        ("QR_ADD_MINUS_RANK_ONLY", ix["QR_ADD"], ix["RANK_ONLY"]),
    ]

    con = contrasts(
        b, macro, defs, R_METRICS,
        {m: "higher_is_better" for m in R_METRICS}
    )

    summ, tie = {}, {}

    for i, name in enumerate(names):
        summ[name] = {
            "dialogue_macro": {
                m: float(macro[i, j]) for j, m in enumerate(R_METRICS)
            },
            "query_micro": {
                m: float(micro[i, j]) for j, m in enumerate(R_METRICS)
            },
        }

        tie[name] = {
            "queries_with_gold_ties": int((ties[i] > 0).sum()),
            "total_tied_non_gold_candidates": int(ties[i].sum()),
            "max_tied_candidates_per_query": int(ties[i].max()),
        }

    return names, ranks, ties, qm, dlg, macro, micro, b, summ, tie, con, cnt


def evaluate():
    req(PROTO.is_file(), f"Missing protocol: {PROTO}")

    p = jload(PROTO)

    req(
        p.get("legacy_metric_dependency") == legacy_dependency(),
        "Legacy metric dependency SHA mismatch"
    )

    req(
        p.get("status") == "FROZEN_BEFORE_METRIC_REVEAL",
        "Protocol not frozen"
    )
    req(
        p.get("evaluation_script_sha256") == sha(Path(__file__).resolve()),
        "Evaluation script SHA mismatch"
    )
    req(not OUT.exists(), f"Output exists; refusing overwrite: {OUT}")

    actual = verify_hashes()

    req(
        actual == {k: v["sha256"] for k, v in p["inputs"].items()},
        "Frozen input hash mismatch"
    )

    d = load_data()

    base, bc, bp, bt, btr, btd, bvr = prevalence(d)

    fnames = ["R_ONLY", "Q_ONLY", "QR_ADD", "FOLD_AWARE_PREVALENCE"]

    flogs = np.stack([
        d["oof"][x]["future_logits"]
        for x in ["R_ONLY", "Q_ONLY", "QR_ADD"]
    ] + [base])

    freports, fpreds, fdlg, frows = {}, [], [], []
    fids = None

    for i, name in enumerate(fnames):
        rep, pred, dlg, ids, rows = legacy.future_metrics(
            name, flogs[i], d["y"], d["avail"], d["fold"], d["dids"]
        )

        if fids is None:
            fids = ids
        else:
            req(np.array_equal(fids, ids), "Future dialogue universe mismatch")

        freports[name] = rep
        fpreds.append(pred)
        fdlg.append(dlg)
        frows.append(rows)

    fdall = np.stack(fdlg, axis=1)
    fmacro = fdall.mean(axis=0)
    fb = boot(fdall, 43)

    ix = {x: i for i, x in enumerate(fnames)}

    fdefs = [
        ("QR_ADD_MINUS_Q_ONLY", ix["QR_ADD"], ix["Q_ONLY"]),
        ("QR_ADD_MINUS_R_ONLY", ix["QR_ADD"], ix["R_ONLY"]),
        ("R_ONLY_MINUS_Q_ONLY", ix["R_ONLY"], ix["Q_ONLY"]),
        ("R_ONLY_MINUS_PREVALENCE", ix["R_ONLY"], ix["FOLD_AWARE_PREVALENCE"]),
        ("Q_ONLY_MINUS_PREVALENCE", ix["Q_ONLY"], ix["FOLD_AWARE_PREVALENCE"]),
        ("QR_ADD_MINUS_PREVALENCE", ix["QR_ADD"], ix["FOLD_AWARE_PREVALENCE"]),
    ]

    fdirs = {
        "brier": "lower_is_better",
        "ordinal_logloss": "lower_is_better",
        "expected_state_mae": "lower_is_better",
        "threshold_accuracy": "higher_is_better",
    }

    fcon = contrasts(fb, fmacro, fdefs, F_METRICS, fdirs)

    for i, name in enumerate(fnames):
        freports[name]["dialogue_macro_probability_metrics"] = {
            m: float(fmacro[i, j]) for j, m in enumerate(F_METRICS)
        }

    r = retrieval_eval(d)
    rnames, rranks, rties, rqm, rdlg, rmacro, rmicro, rb, rsumm, rtie, rcon, rcounts = r

    summary = {
        "artifact": "FSAR S12C Future Input Source OOF Evaluation V1",
        "status": "PASS",
        "scope": "TRAIN_INTERNAL_OOF_ONLY",
        "interpretation": "EXPLORATORY_INPUT_SOURCE_ABLATION_NO_CONFIRMATORY_GATE",
        "protocol_sha256": sha(PROTO),
        "evaluation_script_sha256": sha(Path(__file__).resolve()),
        "input_sha256": actual,
        "query_rows": N,
        "dialogues": int(np.unique(d["dids"]).size),
        "future_eligible_rows": int(d["avail"].sum()),
        "future_eligible_dialogues": int(np.unique(d["dids"][d["avail"]]).size),
        "class_counts": {
            str(int(c)): int((d["y"][d["avail"]] == c).sum())
            for c in CLASSES
        },
        "future_state": {
            "conditions": fnames,
            "metrics": freports,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 43,
                "ci": 0.95,
                "method": "PERCENTILE",
                "contrasts": fcon,
            },
            "fold_aware_prevalence_baseline": {
                "alpha_per_class": ALPHA,
                "training_rows": btr,
                "training_dialogues": btd,
                "validation_rows": bvr,
                "class_counts": bc.tolist(),
                "class_probabilities": bp.tolist(),
                "threshold_probabilities": bt.tolist(),
                "validation_labels_used": False,
            },
        },
        "retrieval": {
            "conditions": rnames,
            "primary_metric": "DIALOGUE_MACRO_MRR",
            "condition_metrics": rsumm,
            "tie_diagnostics": rtie,
            "bootstrap": {
                "unit": "dialogue_id",
                "paired": True,
                "replicates": NBOOT,
                "seed": 42,
                "ci": 0.95,
                "method": "PERCENTILE",
                "contrasts": rcon,
            },
        },
        "dev_payload_used": False,
        "test_payload_used": False,
        "training": False,
        "new_inference": False,
        "no_posthoc_success_gate": True,
        "multiple_comparison_adjustment": "NONE; exploratory contrasts",
    }

    OUT.mkdir(parents=True, exist_ok=False)
    stage = Path(tempfile.mkdtemp(prefix=".stage_", dir=OUT))

    try:
        (stage / FILES[0]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8"
        )

        np.savez_compressed(
            stage / FILES[1],
            retrieval_conditions=np.asarray(rnames, dtype="U32"),
            query_ids=d["qids"],
            dialogue_ids=d["dids"],
            fold_id=d["fold"],
            ranks=rranks,
            gold_tie_counts=rties,
            query_metrics=rqm,
            dialogue_metrics=rdlg,
            dialogue_macro=rmacro,
            query_micro=rmicro,
            retrieval_bootstrap_means=rb,
            dialogue_query_counts=rcounts,
            future_conditions=np.asarray(fnames, dtype="U32"),
            future_available=d["avail"],
            future_value_label=d["y"],
            future_eligible_row_indices=np.flatnonzero(d["avail"]),
            future_logits=flogs,
            future_predictions=np.asarray(fpreds),
            future_row_metrics=np.asarray(frows),
            future_dialogue_ids=fids,
            future_dialogue_metrics=fdall,
            future_dialogue_macro=fmacro,
            future_bootstrap_means=fb,
            prevalence_fold_class_counts=bc,
            prevalence_fold_class_probabilities=bp,
            prevalence_fold_threshold_probabilities=bt,
        )

        lines = [
            "FSAR S12C Future Input Source OOF Evaluation V1",
            "Status: PASS",
            "Scope: TRAIN_INTERNAL_OOF_ONLY",
            "",
            "FUTURE-STATE CLASSIFICATION",
        ]

        for name in fnames:
            lines.append(
                name + ": " + ", ".join(
                    f"{k}={v:.8f}"
                    for k, v in freports[name]["metrics"].items()
                )
            )

        lines += ["", "FUTURE-STATE PROBABILITY METRICS"]

        for name in fnames:
            lines.append(
                name + ": " + ", ".join(
                    f"{k}={v:.8f}"
                    for k, v in freports[name]["dialogue_macro_probability_metrics"].items()
                )
            )

        lines += ["", "FUTURE-STATE PAIRED CONTRASTS"]

        for x in fcon:
            lines.append(
                f"{x['contrast']} / {x['metric']}: "
                f"delta={x['delta_a_minus_b']:+.8f}, "
                f"CI95=[{x['ci95_lower']:+.8f},{x['ci95_upper']:+.8f}]"
            )

        lines += ["", "RETRIEVAL DIALOGUE-MACRO (SECONDARY)"]

        for name in rnames:
            lines.append(
                name + ": " + ", ".join(
                    f"{k}={v:.8f}"
                    for k, v in rsumm[name]["dialogue_macro"].items()
                )
            )

        lines += [
            "",
            "Fold-aware prevalence baseline: five-class training-fold prevalence, add-one smoothing.",
            "Validation labels were not used to construct baseline predictions.",
            "Exploratory only; no confirmatory success gate.",
            "No Dev/Test payload used. No training or new inference performed.",
        ]

        (stage / FILES[2]).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8"
        )

        for fn in FILES:
            os.replace(stage / fn, OUT / fn)

    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("STEP_2F8E_EVALUATION = PASS")

    for fn in FILES:
        print(sha(OUT / fn), OUT / fn)


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
