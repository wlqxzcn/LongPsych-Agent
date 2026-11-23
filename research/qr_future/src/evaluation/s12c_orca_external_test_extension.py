#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

BASE_SCRIPT = ROOT / "src/evaluation/s12c_external_test_evaluation.py"

spec = importlib.util.spec_from_file_location(
    "s12c_base_evaluator", BASE_SCRIPT
)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def project_normalize(x, basis, batch_size=4096):
    """
    Original ORCA projection:
        x - (x @ B) @ B.T

    No centering.
    Followed by L2 normalization.
    """
    out = np.empty(x.shape, dtype=np.float32)

    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))

        xb = np.asarray(x[start:end], dtype=np.float32)

        y = xb - (xb @ basis) @ basis.T

        norms = np.linalg.norm(
            y, axis=1, keepdims=True
        ).astype(np.float32)

        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite projected embeddings")

        if np.any(norms < 1e-12):
            raise RuntimeError("Zero-norm projected embedding")

        y = y / norms
        out[start:end] = y

    return out


def load_existing_predictions(rows, li, match, counts):
    pred_path = base.OUT["predictions"]
    summary_path = base.OUT["summary"]
    manifest_path = base.OUT["manifest"]

    for p in [pred_path, summary_path, manifest_path]:
        if not p.exists():
            raise RuntimeError(f"Missing existing evaluation artifact: {p}")

    old_summary = read_json(summary_path)
    old_manifest = read_json(manifest_path)

    expected_names = ["DENSE_BGE"] + list(base.CONDITIONS)

    if old_summary.get("status") != "COMPLETED":
        raise RuntimeError("Existing baseline evaluation is incomplete")

    if old_manifest.get("status") != "COMPLETED":
        raise RuntimeError("Existing baseline run manifest is incomplete")

    protocol_sha = base.EXPECTED[str(base.P)]

    if old_summary.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Existing summary protocol hash mismatch")

    if old_manifest.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Existing run manifest protocol hash mismatch")

    if old_manifest.get("conditions") != expected_names:
        raise RuntimeError("Existing baseline condition list mismatch")

    with np.load(pred_path, allow_pickle=False) as z:
        pred = {key: z[key] for key in z.files}

    sample_ids = np.asarray(
        [str(x["sample_id"]) for x in rows], dtype=str
    )
    dialogue_ids = np.asarray(
        [str(x["dialogue_id"]) for x in rows], dtype=str
    )

    identity_checks = {
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "long20_indices": li,
        "gold_match_idx": match,
        "candidate_counts": counts,
    }

    for key, expected in identity_checks.items():
        if key not in pred:
            raise RuntimeError(f"Missing prediction metadata: {key}")

        if not np.array_equal(pred[key], expected):
            raise RuntimeError(f"Prediction identity mismatch: {key}")

    ranks = {}

    for name in expected_names:
        key = f"rank_{name}"

        if key not in pred:
            raise RuntimeError(f"Missing baseline ranking: {key}")

        arr = pred[key]

        if arr.shape != (len(rows),):
            raise RuntimeError(f"Invalid rank shape: {key}")

        if arr.dtype != np.int32:
            raise RuntimeError(f"Invalid rank dtype: {key}")

        if np.any(arr < 1) or np.any(arr > counts):
            raise RuntimeError(f"Invalid ranking values: {key}")

        ranks[name] = arr.copy()

    print("EXISTING_BASELINES =", list(ranks))
    print("BASELINE_PREDICTION_REUSE=PASS")
    print("BASELINE_IDENTITY_ALIGNMENT=PASS")

    return ranks, old_summary


def load_orca_basis():
    d = ROOT / "results/phase_s12c/orca_baseline_v1"

    basis_path = d / "orca_source_basis_k2_v1.npy"
    manifest_path = d / "orca_source_basis_manifest_v1.json"
    selection_path = d / "orca_source_selection_v1.json"
    indices_path = d / "orca_source_train_indices_v1.npy"

    for p in [basis_path, manifest_path, selection_path, indices_path]:
        if not p.exists():
            raise RuntimeError(f"Missing ORCA artifact: {p}")

    manifest = read_json(manifest_path)

    required = {
        "status": "PASS",
        "scope": "TRAIN_ONLY",
        "encoder": "BAAI/bge-large-zh-v1.5",
        "source_mode": "affective_user",
        "source_count": 5000,
        "k": 2,
        "svd_method": "RAW_UNCENTERED_SVD",
        "dev_used": False,
        "test_used": False,
        "future_state_supervision_used": False,
        "retrieval_evaluation_performed": False,
    }

    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"ORCA manifest mismatch: {key}, "
                f"actual={manifest.get(key)}, expected={expected}"
            )

    if sha256_file(basis_path) != manifest["basis_sha256"]:
        raise RuntimeError("ORCA basis SHA256 mismatch")

    if sha256_file(selection_path) != manifest["source_selection_sha256"]:
        raise RuntimeError("ORCA source selection SHA256 mismatch")

    if sha256_file(indices_path) != manifest["source_indices_sha256"]:
        raise RuntimeError("ORCA source indices SHA256 mismatch")

    basis = np.load(basis_path, allow_pickle=False)

    if basis.shape != (1024, 2):
        raise RuntimeError(f"Unexpected ORCA basis shape: {basis.shape}")

    if basis.dtype != np.float32:
        raise RuntimeError("ORCA basis dtype mismatch")

    if not np.isfinite(basis).all():
        raise RuntimeError("ORCA basis contains non-finite values")

    orth_error = float(
        np.linalg.norm(basis.T @ basis - np.eye(2))
    )

    if orth_error > 1e-4:
        raise RuntimeError("ORCA basis orthogonality failure")

    print("ORCA_BASIS_HASH=PASS")
    print("ORCA_SOURCE_MANIFEST=PASS")
    print("ORCA_TRAIN_ONLY=PASS")
    print("ORCA_BASIS_SHAPE =", basis.shape)
    print("ORTHOGONALITY_ERROR =", orth_error)

    return basis, manifest


def evaluate_subset_extended(ranks, dids, indices, rng):

    did = dids[indices].astype(str)
    unique = np.unique(did)

    grouped = {}

    for name, arr in ranks.items():
        ids, vals = base.group_values(arr[indices], did)

        if not np.array_equal(ids, unique):
            raise RuntimeError("Dialogue grouping mismatch")

        grouped[name] = vals

    boot_idx = rng.integers(
        0, len(unique),
        size=(base.N_BOOT, len(unique))
    )

    def ci(v):
        est = v.mean(0)
        boot = v[boot_idx].mean(1)

        lo = np.quantile(boot, 0.025, axis=0)
        hi = np.quantile(boot, 0.975, axis=0)

        return {
            metric: {
                "estimate": float(est[j]),
                "ci95_lower": float(lo[j]),
                "ci95_upper": float(hi[j]),
            }
            for j, metric in enumerate(base.METRICS)
        }

    result = {
        "n_queries": int(len(indices)),
        "n_dialogues": int(len(unique)),
        "models": {},
        "contrasts": {},
    }

    for name, vals in grouped.items():
        result["models"][name] = {
            "dialogue_macro": ci(vals)
        }

    def contrast(category, name, reference):
        v = grouped[name] - grouped[reference]

        est = v.mean(0)
        boot = v[boot_idx].mean(1)

        lo = np.quantile(boot, 0.025, axis=0)
        hi = np.quantile(boot, 0.975, axis=0)

        result["contrasts"].setdefault(category, {})[
            f"{name}_MINUS_{reference}"
        ] = {
            metric: {
                "delta": float(est[j]),
                "ci95_lower": float(lo[j]),
                "ci95_upper": float(hi[j]),
            }
            for j, metric in enumerate(base.METRICS)
        }

    # Original frozen-protocol contrasts
    for c in base.CONDITIONS:
        contrast("learned_vs_dense_bge", c, "DENSE_BGE")

    for c in base.CONDITIONS:
        if c != "RANK_ONLY":
            contrast("future_vs_rank_only", c, "RANK_ONLY")

    contrast(
        "product_ablation",
        "R_FUTURE_NO_PRODUCT",
        "R_FUTURE_FULL"
    )

    contrast(
        "product_ablation",
        "QR_FUTURE_NO_PRODUCT",
        "QR_FUTURE_FULL"
    )

    # New ORCA extension contrasts
    contrast("orca_vs_baselines", "ORCA_K2", "DENSE_BGE")
    contrast("orca_vs_baselines", "ORCA_K2", "RANK_ONLY")

    for c in base.CONDITIONS:
        if c not in ("RANK_ONLY",):
            contrast("orca_vs_future_conditions", "ORCA_K2", c)

    return result


def validate_baseline_reproduction(new_data, old_data, dataset_name):

    for name in ["DENSE_BGE"] + list(base.CONDITIONS):
        for metric in base.METRICS:

            new = new_data["models"][name]["dialogue_macro"][metric]
            old = old_data["models"][name]["dialogue_macro"][metric]

            for field in [
                "estimate",
                "ci95_lower",
                "ci95_upper"
            ]:
                if not np.isclose(
                    new[field], old[field],
                    rtol=0, atol=1e-9
                ):
                    raise RuntimeError(
                        f"Baseline reproduction mismatch: "
                        f"{dataset_name}/{name}/{metric}/{field}"
                    )

    print("BASELINE_REPRODUCTION_PASS =", dataset_name)


def main():

    print("=" * 76)
    print("STEP 2F16AF: ORCA EXTERNAL TEST EXTENSION")
    print("=" * 76)

    print("HOST =", __import__("socket").gethostname())

    # 1. Load and validate original frozen inputs
    protocol, rows, li, q, g, r, match, counts = base.load_inputs()

    print("\n[1] FROZEN INPUT VALIDATION PASS")

    # 2. Reuse the existing six baseline predictions
    ranks, old_summary = load_existing_predictions(
        rows, li, match, counts
    )

    # 3. Load frozen ORCA k=2 basis
    basis, basis_manifest = load_orca_basis()

    # 4. Prepare independent ORCA projection
    print("\n[2] ORCA EMBEDDING PROJECTION")

    q_orca = project_normalize(q, basis)
    g_orca = project_normalize(g, basis)
    r_orca = project_normalize(r, basis)

    print("QUERY_SHAPE =", q_orca.shape)
    print("GOLD_SHAPE =", g_orca.shape)
    print("RESPONSE_SHAPE =", r_orca.shape)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ORCA ranking")

    device = torch.device("cuda")

    print("GPU =", torch.cuda.get_device_name(device))

    # 5. ORCA ranking inference only
    print("\n[3] ORCA RANKING INFERENCE")

    r_orca_gpu = torch.from_numpy(r_orca).to(device)
    del r_orca
    torch.cuda.empty_cache()

    orca_ranks = base.rank_dense(
        q_orca,
        g_orca,
        r_orca_gpu,
        match,
        device
    )

    del r_orca_gpu
    torch.cuda.empty_cache()

    if orca_ranks.shape != (len(rows),):
        raise RuntimeError("ORCA ranking shape mismatch")

    if np.any(orca_ranks < 1) or np.any(orca_ranks > counts):
        raise RuntimeError("ORCA ranking range failure")

    ranks["ORCA_K2"] = orca_ranks.astype(np.int32)

    print("ORCA_RANKING_COMPLETE")
    print("ORCA_RANK_MIN =", int(orca_ranks.min()))
    print("ORCA_RANK_MAX =", int(orca_ranks.max()))

    # 6. Recalculate all conditions with exactly shared bootstrap indices
    print("\n[4] JOINT PAIRED EVALUATION")

    dids = np.asarray(
        [str(x["dialogue_id"]) for x in rows],
        dtype=str
    )

    rng = np.random.default_rng(base.BOOT_SEED)

    all_idx = np.arange(len(rows), dtype=np.int64)

    datasets = {}

    datasets["TEST_ALL"] = evaluate_subset_extended(
        ranks, dids, all_idx, rng
    )

    datasets["TEST_LONG20"] = evaluate_subset_extended(
        ranks, dids, li, rng
    )

    # Verify that the six existing conditions reproduce exactly.
    validate_baseline_reproduction(
        datasets["TEST_ALL"],
        old_summary["datasets"]["TEST_ALL"],
        "TEST_ALL"
    )

    validate_baseline_reproduction(
        datasets["TEST_LONG20"],
        old_summary["datasets"]["TEST_LONG20"],
        "TEST_LONG20"
    )

    # 7. Construct extension summary
    candidate_dist = {
        str(k): int(v)
        for k, v in zip(
            *np.unique(counts, return_counts=True)
        )
    }

    result = {
        "artifact": "FSAR S12C ORCA K2 External Test Extension V1",
        "status": "COMPLETED",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": base.EXPECTED[str(base.P)],

        "evaluation_scope": "POST_FROZEN_PROTOCOL_BASELINE_EXTENSION",

        "protocol_extension": {
            "original_protocol_modified": False,
            "orca_was_original_protocol_condition": False,
            "added_condition": "ORCA_K2",
            "basis_scope": "TRAIN_ONLY",
            "test_based_tuning": False,
            "dev_based_tuning": False,
        },

        "conditions": list(ranks.keys()),

        "reused_baseline_predictions": True,
        "baseline_reinference": False,

        "orca_basis": {
            "k": 2,
            "basis_sha256": basis_manifest["basis_sha256"],
            "source_mode": basis_manifest["source_mode"],
            "source_count": basis_manifest["source_count"],
            "top2_source_energy_ratio": basis_manifest[
                "top2_source_energy_ratio"
            ],
            "svd_method": "RAW_UNCENTERED_SVD",
            "centering": False,
            "projection": "x - (x @ B) @ B.T",
            "post_projection_normalization": "L2",
        },

        "candidate_count_distribution": candidate_dist,

        "gold_matches_train": int((match >= 0).sum()),
        "gold_not_in_train": int((match < 0).sum()),

        "ranking": {
            "tie_rule": "negative score >= Gold ranks before Gold",
            "rank_formula": "1 + count(negative_score >= gold_score)",
            "metrics": base.METRICS,
        },

        "statistics": {
            "method": "paired_dialogue_cluster_bootstrap",
            "replicates": base.N_BOOT,
            "seed": base.BOOT_SEED,
            "confidence_level": 0.95,
            "resampling_unit": "dialogue",
            "shared_resampling_indices_across_conditions": True,
        },

        "datasets": datasets,

        "interpretation_limit": (
            "Gold-injected candidate-set ranking generalization, "
            "not retrieval coverage over the TRAIN response library alone. "
            "ORCA is an additional post-freeze baseline extension."
        ),

        "training": False,
        "test_time_fitting": False,
        "checkpoint_reselection": False,
    }

    # 8. Generate independent output artifacts
    out_dir = ROOT / "results/phase_s12c/orca_external_test_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "predictions": out_dir / "orca_external_test_ranking_predictions_v1.npz",
        "summary": out_dir / "orca_external_test_ranking_summary_v1.json",
        "report": out_dir / "orca_external_test_ranking_report_v1.txt",
        "manifest": out_dir / "orca_external_test_ranking_run_manifest_v1.json",
    }

    for p in outputs.values():
        if p.exists() or p.with_suffix(p.suffix + ".tmp").exists():
            raise RuntimeError(
                f"Output collision. Refusing overwrite: {p}"
            )

    report = [
        "ORCA K2 EXTERNAL TEST EXTENSION",
        "=" * 76,
        "Original six baseline predictions reused.",
        "ORCA K2 independently ranked.",
        "Gold-injected candidate-set ranking generalization.",
        "Tie rule: negative score >= Gold ranks before Gold.",
        "Bootstrap: paired dialogue-cluster, 2000 replicates.",
        "ORCA is a post-frozen-protocol extension.",
    ]

    for ds, data in datasets.items():
        report += ["", "=" * 76, ds, "=" * 76]

        for model, md in data["models"].items():
            fields = []

            for metric, x in md["dialogue_macro"].items():
                fields.append(
                    f"{metric}={x['estimate']:.6f} "
                    f"[{x['ci95_lower']:.6f},{x['ci95_upper']:.6f}]"
                )

            report.append(model + ": " + " | ".join(fields))

        report.append("PAIRED CONTRASTS")

        for category, contrasts in data["contrasts"].items():
            report.append("[" + category + "]")

            for cname, vals in contrasts.items():
                fields = [
                    f"{m}={x['delta']:.6f} "
                    f"[{x['ci95_lower']:.6f},{x['ci95_upper']:.6f}]"
                    for m, x in vals.items()
                ]

                report.append(cname + ": " + " | ".join(fields))

    report += [
        "",
        "TRAINING = NO",
        "BASELINE_REINFERENCE = NO",
        "TEST_TIME_FITTING = NO",
        "CHECKPOINT_RESELECTION = NO",
        "ORCA_RANKING_INFERENCE = COMPLETED",
    ]

    run_manifest = {
        "artifact": "ORCA K2 External Test Extension Run Manifest V1",
        "status": "COMPLETED",
        "host": __import__("socket").gethostname(),

        "protocol_sha256": base.EXPECTED[str(base.P)],
        "evaluator_script_sha256": sha256_file(BASE_SCRIPT),

        "baseline_prediction_sha256": sha256_file(
            base.OUT["predictions"]
        ),
        "baseline_summary_sha256": sha256_file(
            base.OUT["summary"]
        ),
        "baseline_run_manifest_sha256": sha256_file(
            base.OUT["manifest"]
        ),

        "orca_basis_sha256": basis_manifest["basis_sha256"],

        "conditions": list(ranks.keys()),
        "test_all_queries": len(rows),
        "test_long20_queries": len(li),

        "baseline_predictions_reused": True,
        "baseline_reinference": False,

        "training": False,
        "test_time_fitting": False,
        "checkpoint_reselection": False,
        "ranking_inference": True,
        "protocol_modified": False,
        "post_frozen_protocol_extension": True,
    }

    # Atomic output generation
    tmp = {
        key: p.with_suffix(p.suffix + ".tmp")
        for key, p in outputs.items()
    }

    with tmp["predictions"].open("wb") as f:
        np.savez_compressed(
            f,
            sample_ids=np.asarray(
                [str(x["sample_id"]) for x in rows],
                dtype=str
            ),
            dialogue_ids=dids,
            long20_indices=li,
            gold_match_idx=match,
            candidate_counts=counts,
            **{
                f"rank_{k}": v.astype(np.int32)
                for k, v in ranks.items()
            }
        )

    write_json(tmp["summary"], result)

    tmp["report"].write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    write_json(tmp["manifest"], run_manifest)

    for key in outputs:
        os.replace(tmp[key], outputs[key])

    print("\n" + "=" * 76)
    print("STEP_2F16AF_ORCA_EXTERNAL_TEST_EXTENSION_COMPLETE")
    print("=" * 76)

    for key, path in outputs.items():
        print(key.upper(), "=", path)

    print("BASELINE_REINFERENCE=NO")
    print("TRAINING=NO")
    print("TEST_TIME_FITTING=NO")
    print("CHECKPOINT_RESELECTION=NO")
    print("ORCA_RANKING_INFERENCE=YES")
    print("PROTOCOL_MODIFIED=NO")
    print("=" * 76)


if __name__ == "__main__":
    main()
