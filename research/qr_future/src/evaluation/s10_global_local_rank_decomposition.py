#!/usr/bin/env python3
"""FSAR S10: exploratory decomposition of full-pool vs Top-100 ranks.

Run from the project root after uploading this file to src/evaluation/:
  python -u src/evaluation/s10_global_local_rank_decomposition.py --prepare
  python -u src/evaluation/s10_global_local_rank_decomposition.py --evaluate

This is a post-hoc, Dev-only diagnostic over frozen S8/S9 prediction artifacts.
It does not load text payloads, run model inference, train models, or read Test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

S8_PROTOCOL = ROOT / "protocols/fsar_s8_dev_retrieval_evaluation_v1.json"
S9_PROTOCOL = ROOT / "protocols/fsar_s9_dev_top100_reranking_v1.json"
S8_SUMMARY = ROOT / "results/phase_s8/s8_dev_retrieval_summary_v1.json"
S9_SUMMARY = ROOT / "results/phase_s9/s9_dev_top100_reranking_summary_v1.json"
S8_PREDICTIONS = ROOT / "results/phase_s8/s8_dev_retrieval_predictions_v1.npz"
S9_PREDICTIONS = ROOT / "results/phase_s9/s9_dev_top100_reranking_predictions_v1.npz"
S8_SCRIPT = ROOT / "src/evaluation/s8_dev_retrieval_evaluation.py"
S9_SCRIPT = ROOT / "src/evaluation/s9_dev_top100_reranking_diagnostic.py"
S10_SCRIPT = ROOT / "src/evaluation/s10_global_local_rank_decomposition.py"

PROTOCOL_OUT = ROOT / "protocols/fsar_s10_global_local_rank_decomposition_v1.json"
OUT_DIR = ROOT / "results/phase_s10"
SUMMARY_OUT = OUT_DIR / "s10_global_local_rank_decomposition_summary_v1.json"
REPORT_OUT = OUT_DIR / "s10_global_local_rank_decomposition_report_v1.txt"
PREDICTIONS_OUT = OUT_DIR / "s10_global_local_rank_decomposition_predictions_v1.csv"

MODELS = [
    "DENSE_BGE",
    "RANK_ONLY",
    "R_FUTURE_BETA1", "R_FUTURE_BETA2", "R_FUTURE_BETA4", "R_FUTURE_BETA8",
    "QR_FUTURE_BETA1", "QR_FUTURE_BETA2", "QR_FUTURE_BETA4", "QR_FUTURE_BETA8",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def input_paths() -> dict[str, Path]:
    return {
        "s8_protocol": S8_PROTOCOL,
        "s9_protocol": S9_PROTOCOL,
        "s8_summary": S8_SUMMARY,
        "s9_summary": S9_SUMMARY,
        "s8_predictions": S8_PREDICTIONS,
        "s9_predictions": S9_PREDICTIONS,
        "s8_script": S8_SCRIPT,
        "s9_script": S9_SCRIPT,
        "s10_script": S10_SCRIPT,
    }


def input_hashes() -> dict[str, str]:
    paths = input_paths()
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required S8/S9 inputs are missing:\n" + "\n".join(missing))
    return {name: sha256_file(path) for name, path in paths.items()}


def load_predictions() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with np.load(S8_PREDICTIONS, allow_pickle=False) as z8:
        p8 = {key: z8[key] for key in z8.files}
    with np.load(S9_PREDICTIONS, allow_pickle=False) as z9:
        p9 = {key: z9[key] for key in z9.files}
    return p8, p9


def validate_artifacts(p8: dict[str, np.ndarray], p9: dict[str, np.ndarray]) -> dict[str, Any]:
    for key in ("sample_ids", "dialogue_ids", "dev_long20_mask"):
        if key not in p8 or key not in p9:
            raise RuntimeError(f"Missing alignment field {key} in S8 or S9 predictions")
        if not np.array_equal(p8[key], p9[key]):
            raise RuntimeError(f"S8/S9 prediction alignment mismatch: {key}")

    if "match_idx" not in p8 or "gold_match_idx" not in p9:
        raise RuntimeError("Missing S8 match_idx or S9 gold_match_idx")
    if not np.array_equal(p8["match_idx"], p9["gold_match_idx"]):
        raise RuntimeError("S8/S9 Gold identity alignment mismatch")

    n = len(p8["sample_ids"])
    if n != 9070:
        raise RuntimeError(f"Expected frozen Dev row count 9070, got {n}")
    if "candidate_response_rows" not in p9:
        raise RuntimeError("S9 candidate_response_rows missing")
    candidates = p9["candidate_response_rows"]
    if candidates.shape != (n, 100):
        raise RuntimeError(f"Expected S9 candidate matrix {(n, 100)}, got {candidates.shape}")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise RuntimeError(f"Candidate row IDs must be integer, got {candidates.dtype}")
    if np.any(np.diff(np.sort(candidates, axis=1), axis=1) == 0):
        raise RuntimeError("Duplicate candidate response row detected within an S9 query")

    match_idx = p8["match_idx"].astype(np.int64, copy=False)
    valid_match = match_idx >= 0
    if np.any(valid_match):
        leaked = np.any(candidates[valid_match] == match_idx[valid_match, None], axis=1)
        if np.any(leaked):
            raise RuntimeError("Gold response identity appears in S9 Top-100 negative candidates")

    missing_keys = []
    for model in MODELS:
        k = f"ranks__{model}"
        if k not in p8 or k not in p9:
            missing_keys.append(k)
        else:
            if p8[k].shape != (n,) or p9[k].shape != (n,):
                raise RuntimeError(f"Unexpected rank array shape for {model}")
            if np.any(p8[k] < 1) or np.any(p9[k] < 1) or np.any(p9[k] > 101):
                raise RuntimeError(f"Invalid rank values for {model}")
            if np.any(p8[k] < p9[k]):
                bad = int(np.flatnonzero(p8[k] < p9[k])[0])
                raise RuntimeError(
                    f"Full-pool rank is better than subset rank for {model} at row {bad}; "
                    "rank decomposition is inconsistent"
                )
    if missing_keys:
        raise RuntimeError("Missing model rank arrays: " + ", ".join(missing_keys))

    return {
        "dev_rows": n,
        "dev_dialogues": int(len(np.unique(p8["dialogue_ids"]))),
        "dev_long20_rows": int(np.sum(p8["dev_long20_mask"])),
        "dev_long20_dialogues": int(len(np.unique(p8["dialogue_ids"][p8["dev_long20_mask"]]))),
        "candidate_count_per_query": int(candidates.shape[1]),
        "sample_id_alignment": True,
        "dialogue_id_alignment": True,
        "gold_identity_alignment": True,
        "gold_identity_excluded_from_top100": True,
        "rank_subset_consistency": True,
    }


def prepare() -> None:
    if PROTOCOL_OUT.exists():
        raise RuntimeError(f"S10 protocol already exists; refusing overwrite: {PROTOCOL_OUT}")
    for path in (SUMMARY_OUT, REPORT_OUT, PREDICTIONS_OUT):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {path}")

    p8, p9 = load_predictions()
    audit = validate_artifacts(p8, p9)
    protocol = {
        "artifact": "FSAR S10 Global-vs-Local Rank Decomposition V1",
        "status": "FROZEN_BEFORE_ANALYSIS",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "purpose": "Decompose each frozen S8 full-library Gold rank into S9 local Top-100 rank plus the count of outside-Top-100 candidates ranked ahead of Gold.",
        "input_sha256": input_hashes(),
        "alignment_audit": audit,
        "models": MODELS,
        "decomposition": "outside_top100_ahead_count = S8 full_library_rank - S9 Top100_plus_injected_Gold_rank",
        "interpretation": "Descriptive post-hoc diagnostic, not a confirmatory test and not a new performance claim.",
        "test_payload_used": False,
        "model_inference_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
    }
    atomic_json(PROTOCOL_OUT, protocol)
    print("S10_PROTOCOL_FROZEN = PASS")
    print("PROTOCOL =", PROTOCOL_OUT)
    print("PROTOCOL_SHA256 =", sha256_file(PROTOCOL_OUT))
    print("ALIGNMENT_AUDIT =", json.dumps(audit, ensure_ascii=False))
    print("TEST_PAYLOAD_READ = NO")
    print("ANALYSIS_STARTED = NO")


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_OUT.is_file():
        raise FileNotFoundError("Run --prepare first; S10 protocol is missing")
    protocol = json.loads(PROTOCOL_OUT.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_ANALYSIS":
        raise RuntimeError(f"Unexpected protocol status: {protocol.get('status')}")
    current = input_hashes()
    frozen = protocol.get("input_sha256", {})
    if current != frozen:
        changed = [k for k in sorted(set(current) | set(frozen)) if current.get(k) != frozen.get(k)]
        raise RuntimeError("Frozen S10 input hash mismatch: " + ", ".join(changed))
    return protocol


def summarize_ranks(full: np.ndarray, local: np.ndarray, subset: np.ndarray) -> dict[str, Any]:
    full = full[subset].astype(np.int64, copy=False)
    local = local[subset].astype(np.int64, copy=False)
    outside = full - local
    if np.any(outside < 0):
        raise RuntimeError("Negative outside-Top-100 count encountered")

    def quantiles(x: np.ndarray) -> dict[str, float]:
        return {
            "p10": float(np.quantile(x, 0.10)),
            "median": float(np.quantile(x, 0.50)),
            "p90": float(np.quantile(x, 0.90)),
            "mean": float(np.mean(x)),
        }

    return {
        "n_queries": int(len(full)),
        "full_library_rank": quantiles(full),
        "local_top100_plus_gold_rank": quantiles(local),
        "outside_top100_candidates_ahead_of_gold": quantiles(outside),
        "local_R@1": float(np.mean(local <= 1)),
        "local_R@5": float(np.mean(local <= 5)),
        "local_R@20": float(np.mean(local <= 20)),
        "full_library_R@1": float(np.mean(full <= 1)),
        "full_library_R@5": float(np.mean(full <= 5)),
        "full_library_R@20": float(np.mean(full <= 20)),
        "full_rank_over_100_rate": float(np.mean(full > 100)),
        "local_rank_le_5_but_full_rank_over_100_rate": float(np.mean((local <= 5) & (full > 100))),
        "outside_top100_count_positive_rate": float(np.mean(outside > 0)),
    }


def evaluate() -> None:
    verify_protocol()
    for path in (SUMMARY_OUT, REPORT_OUT, PREDICTIONS_OUT):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise RuntimeError(f"Output collision; refusing overwrite: {path}")

    p8, p9 = load_predictions()
    audit = validate_artifacts(p8, p9)
    n = len(p8["sample_ids"])
    long_mask = p8["dev_long20_mask"].astype(bool, copy=False)
    masks = {"dev_all": np.ones(n, dtype=bool), "dev_long20": long_mask}

    summary: dict[str, Any] = {
        "artifact": "FSAR S10 Global-vs-Local Rank Decomposition Results V1",
        "status": "PASS",
        "scope": "EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "protocol_path": str(PROTOCOL_OUT.relative_to(ROOT)),
        "protocol_sha256": sha256_file(PROTOCOL_OUT),
        "alignment_audit": audit,
        "interpretation": (
            "For each query and model, S8 full-library rank minus S9 rank on the same query's "
            "BGE Top-100 negatives plus injected Gold estimates how many candidates outside the "
            "Top-100 are scored at least as highly as Gold, assuming the shared score and tie rules. "
            "This is a post-hoc descriptive decomposition, not confirmatory evidence."
        ),
        "datasets": {},
        "test_payload_used": False,
        "model_inference_performed": False,
        "training_artifacts_modified": False,
        "checkpoints_modified": False,
    }

    for ds_name, mask in masks.items():
        summary["datasets"][ds_name] = {}
        for model in MODELS:
            full = p8[f"ranks__{model}"].astype(np.int64, copy=False)
            local = p9[f"ranks__{model}"].astype(np.int64, copy=False)
            summary["datasets"][ds_name][model] = summarize_ranks(full, local, mask)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_tmp = SUMMARY_OUT.with_suffix(SUMMARY_OUT.suffix + ".tmp")
    report_tmp = REPORT_OUT.with_suffix(REPORT_OUT.suffix + ".tmp")
    pred_tmp = PREDICTIONS_OUT.with_suffix(PREDICTIONS_OUT.suffix + ".tmp")
    atomic_json(summary_tmp, summary)

    with pred_tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["sample_id", "dialogue_id", "dev_long20"]
        for model in MODELS:
            header.extend([f"full_rank__{model}", f"local_rank__{model}", f"outside_top100_ahead__{model}"])
        writer.writerow(header)
        for i in range(n):
            row: list[Any] = [str(p8["sample_ids"][i]), str(p8["dialogue_ids"][i]), int(long_mask[i])]
            for model in MODELS:
                full = int(p8[f"ranks__{model}"][i])
                local = int(p9[f"ranks__{model}"][i])
                row.extend([full, local, full - local])
            writer.writerow(row)

    lines = [
        "FSAR S10 GLOBAL-vs-LOCAL RANK DECOMPOSITION",
        f"protocol_sha256 = {summary['protocol_sha256']}",
        "scope = EXPLORATORY_POSTHOC_DEV_DIAGNOSTIC_ONLY",
        "decomposition = full_library_rank - local_top100_plus_gold_rank",
        "interpretation = outside-Top-100 candidates ranked at least as highly as Gold, under shared scoring/tie rules",
        "test_payload_used = NO",
        "model_inference_performed = NO",
        "",
    ]
    for ds_name, models in summary["datasets"].items():
        lines.append(f"===== {ds_name} =====")
        lines.append("model | full median | local median | outside median | outside mean | local R@1 | full R@1 | local<=5 & full>100")
        for model, vals in models.items():
            lines.append(
                f"{model} | {vals['full_library_rank']['median']:.1f} | "
                f"{vals['local_top100_plus_gold_rank']['median']:.1f} | "
                f"{vals['outside_top100_candidates_ahead_of_gold']['median']:.1f} | "
                f"{vals['outside_top100_candidates_ahead_of_gold']['mean']:.2f} | "
                f"{vals['local_R@1']:.6f} | {vals['full_library_R@1']:.6f} | "
                f"{vals['local_rank_le_5_but_full_rank_over_100_rate']:.6f}"
            )
        lines.append("")
    report_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(summary_tmp, SUMMARY_OUT)
    os.replace(pred_tmp, PREDICTIONS_OUT)
    os.replace(report_tmp, REPORT_OUT)

    print("S10_ANALYSIS = PASS")
    print("SUMMARY =", SUMMARY_OUT)
    print("REPORT =", REPORT_OUT)
    print("PREDICTIONS =", PREDICTIONS_OUT)
    print("TEST_PAYLOAD_READ = NO")
    print("MODEL_INFERENCE_PERFORMED = NO")
    print("TRAINING_ARTIFACTS_MODIFIED = NO")
    print("CHECKPOINTS_MODIFIED = NO")
    for ds_name, models in summary["datasets"].items():
        print(f"\n[{ds_name}]")
        for model in MODELS:
            vals = models[model]
            print(
                model,
                json.dumps({
                    "full_rank_median": vals["full_library_rank"]["median"],
                    "local_rank_median": vals["local_top100_plus_gold_rank"]["median"],
                    "outside_top100_ahead_median": vals["outside_top100_candidates_ahead_of_gold"]["median"],
                    "outside_top100_ahead_mean": vals["outside_top100_candidates_ahead_of_gold"]["mean"],
                    "local_R@1": vals["local_R@1"],
                    "full_R@1": vals["full_library_R@1"],
                    "local_le5_full_gt100": vals["local_rank_le_5_but_full_rank_over_100_rate"],
                }, ensure_ascii=False),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true", help="Validate S8/S9 outputs and freeze S10 protocol")
    group.add_argument("--evaluate", action="store_true", help="Run the read-only rank decomposition")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
