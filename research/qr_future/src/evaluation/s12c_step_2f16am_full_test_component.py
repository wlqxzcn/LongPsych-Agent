#!/usr/bin/env python3
from pathlib import Path
import sys
import json
import csv
import gc
import hashlib
import os
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.evaluation import s12c_external_test_evaluation as base

OUT = ROOT / "results/phase_s12c/step_2f16am_full_test_component_v1"

DIM = 1024

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def atomic_savez(path, arrays):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def calculate_condition(condition, models, q, g, r_gpu, match, device):

    n = len(q)
    has_product = "NO_PRODUCT" not in condition

    terms = [
        "RESPONSE",
        "ABS_DIFF",
        "RESPONSE_PLUS_ABS",
        "FULL",
    ]

    if has_product:
        terms.append("PRODUCT")

    zqs = []
    zrs = []
    weights = []

    gold_parts = {x: [] for x in terms}
    max_fast_error = 0.0

    # --------------------------------------------------
    # 1. Frozen model representations and Gold scores
    # --------------------------------------------------

    with torch.inference_mode():

        for fold, model in enumerate(models):

            max_fast_error = max(
                max_fast_error,
                base.validate_scorer(model, device)
            )

            ranker = model.ranker

            zq = ranker.adapt_query(q)
            zg = ranker.adapt_response(g)
            zr = ranker.adapt_response(r_gpu)

            w = ranker.rank_head.weight[0]

            wr = w[DIM:2*DIM]
            wd = w[2*DIM:3*DIM]
            wp = w[3*DIM:4*DIM] if has_product else None

            gr = (zg * wr).sum(-1)

            gd = (
                torch.abs(zq - zg) * wd
            ).sum(-1)

            gold_parts["RESPONSE"].append(gr)
            gold_parts["ABS_DIFF"].append(gd)
            gold_parts["RESPONSE_PLUS_ABS"].append(gr + gd)

            # Official Gold scoring path
            gold_parts["FULL"].append(
                ranker.score_from_adapted(zq, zg)
            )

            if has_product:
                gp = ((zq * wp) * zg).sum(-1)
                gold_parts["PRODUCT"].append(gp)

            zqs.append(zq)
            zrs.append(zr)
            weights.append((wr, wd, wp))

            print(
                f"FOLD_{fold}=PASS "
                f"FAST_REFERENCE_ERROR={max_fast_error:.3e}",
                flush=True
            )

    gold_scores = {
        name: torch.stack(values).mean(0)
        for name, values in gold_parts.items()
    }

    ranks = {
        name: np.zeros(n, dtype=np.int32)
        for name in terms
    }

    gold_output = {
        name: value.cpu().numpy()
        for name, value in gold_scores.items()
    }

    # --------------------------------------------------
    # 2. Official batch and chunk sizes
    # --------------------------------------------------

    QB = base.QB
    CC = base.CC

    with torch.inference_mode():

        for qs in range(0, n, QB):

            qe = min(qs + QB, n)
            qn = qe - qs

            counters = {
                name: torch.zeros(
                    qn, dtype=torch.int64, device=device
                )
                for name in terms
            }

            gs = {
                name: gold_scores[name][qs:qe]
                for name in terms
            }

            for cs in range(0, r_gpu.shape[0], CC):

                ce = min(cs + CC, r_gpu.shape[0])

                score_sums = {
                    name: None for name in terms
                }

                # Fold order must match official evaluator.
                for fi, model in enumerate(models):

                    zqb = zqs[fi][qs:qe]
                    zrb = zrs[fi][cs:ce]

                    wr, wd, wp = weights[fi]

                    response = (
                        (zrb * wr).sum(-1)
                        .unsqueeze(0)
                        .expand(qn, -1)
                    )

                    diff = torch.abs(
                        zqb[:, None, :] -
                        zrb[None, :, :]
                    )

                    abs_score = (diff * wd).sum(-1)
                    del diff

                    current = {
                        "RESPONSE": response,
                        "ABS_DIFF": abs_score,
                        "RESPONSE_PLUS_ABS": response + abs_score,

                        # Critical: FULL uses the official
                        # fast_score implementation directly.
                        "FULL": base.fast_score(
                            model, zqb, zrb
                        ),
                    }

                    if has_product:
                        current["PRODUCT"] = (
                            (zqb * wp) @ zrb.T
                        )

                    for name in terms:

                        value = current[name]

                        if score_sums[name] is None:
                            score_sums[name] = value
                        else:
                            score_sums[name] = (
                                score_sums[name] + value
                            )

                # Same fold averaging as official evaluation.
                scores = {
                    name: score_sums[name] / len(models)
                    for name in terms
                }

                # --------------------------------------------------
                # 3. Pairwise ranking counts
                # --------------------------------------------------

                for name in terms:

                    counters[name] += (
                        scores[name] >= gs[name][:, None]
                    ).sum(1)

                mi = match[qs:qe]

                mask = (mi >= cs) & (mi < ce)

                # Exclude the Gold candidate itself when present.
                if np.any(mask):

                    ri = torch.from_numpy(
                        np.flatnonzero(mask)
                    ).to(device)

                    ci = torch.from_numpy(
                        (mi[mask] - cs).astype(np.int64)
                    ).to(device)

                    for name in terms:

                        counters[name][ri] -= (
                            scores[name][ri, ci] >= gs[name][ri]
                        ).to(torch.int64)

            for name in terms:

                ranks[name][qs:qe] = (
                    counters[name] + 1
                ).cpu().numpy().astype(np.int32)

            if (qe % 512 == 0) or qe == n:
                print(
                    f"PROGRESS {condition}: {qe}/{n}",
                    flush=True
                )

    return ranks, gold_output, max_fast_error


def main():

    if OUT.exists():
        raise RuntimeError(
            f"OUTPUT_EXISTS: refusing overwrite: {OUT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda")

    print("=" * 76)
    print("STEP 2F16AM: OFFICIAL FULL TEST COMPONENT ATTRIBUTION")
    print("=" * 76)

    print("HOST =", __import__("socket").gethostname())
    print("DEVICE =", torch.cuda.get_device_name(0))

    # --------------------------------------------------
    # 1. Official frozen input loading
    # --------------------------------------------------

    (
        protocol,
        rows,
        long_idx,
        q_np,
        g_np,
        r_np,
        match,
        candidate_counts,
    ) = base.load_inputs()

    pred = np.load(
        base.OUT["predictions"],
        allow_pickle=False
    )

    sample_ids = np.asarray(
        [str(x["sample_id"]) for x in rows]
    )

    dialogue_ids = np.asarray(
        [str(x["dialogue_id"]) for x in rows]
    )

    assert np.array_equal(sample_ids, pred["sample_ids"])
    assert np.array_equal(dialogue_ids, pred["dialogue_ids"])
    assert np.array_equal(match, pred["gold_match_idx"])
    assert np.array_equal(candidate_counts, pred["candidate_counts"])

    n = len(rows)

    print("QUERY_COUNT =", n)
    print("RESPONSE_POOL =", len(r_np))
    print("MATCHED =", int((match >= 0).sum()))
    print("UNMATCHED =", int((match < 0).sum()))
    print("LONG20 =", len(long_idx))

    OUT.mkdir(parents=True, exist_ok=False)
    print("OUTPUT_DIR_READY =", OUT, flush=True)

    q = torch.from_numpy(q_np).to(device)
    g = torch.from_numpy(g_np).to(device)
    r_gpu = torch.from_numpy(r_np).to(device)

    # --------------------------------------------------
    # 2. Full test component evaluation
    # --------------------------------------------------

    ranks_by_condition = {}
    gold_by_condition = {}
    reproduction = {}
    scorer_errors = {}

    for condition in base.CONDITIONS:

        print("\n" + "=" * 70)
        print("CONDITION =", condition)
        print("=" * 70, flush=True)

        models = base.load_models(
            condition, protocol, device
        )

        ranks, gold_scores, fast_error = calculate_condition(
            condition,
            models,
            q,
            g,
            r_gpu,
            match,
            device
        )

        ranks_by_condition[condition] = ranks
        gold_by_condition[condition] = gold_scores
        scorer_errors[condition] = fast_error

        # Incremental checkpoint after each completed condition.
        partial_arrays = {
            "sample_ids": sample_ids,
            "dialogue_ids": dialogue_ids,
            "gold_match_idx": match,
            "candidate_counts": candidate_counts,
            "long20_indices": long_idx,
        }

        for done_condition, done_components in ranks_by_condition.items():
            for component, arr in done_components.items():
                partial_arrays[
                    f"rank_{done_condition}__{component}"
                ] = arr

        atomic_savez(
            OUT / "rank_predictions_partial.npz",
            partial_arrays
        )

        print(
            "PARTIAL_SAVE_PASS:",
            condition,
            flush=True
        )

        # --------------------------------------------------
        # 3. Exact official FULL rank reproduction
        # --------------------------------------------------

        official = pred[f"rank_{condition}"]
        reproduced = ranks["FULL"]

        diff = np.abs(
            reproduced.astype(np.int64) -
            official.astype(np.int64)
        )

        bad = np.flatnonzero(diff != 0)

        reproduction[condition] = {
            "mismatch_count": int(len(bad)),
            "max_absolute_difference": int(diff.max()),
            "exact": bool(len(bad) == 0),
            "examples": [
                {
                    "sample_id": str(sample_ids[i]),
                    "new_rank": int(reproduced[i]),
                    "official_rank": int(official[i]),
                    "difference": int(
                        reproduced[i] - official[i]
                    )
                }
                for i in bad[:20]
            ]
        }

        print(
            "OFFICIAL_RANK_EXACT_MATCH =",
            len(bad) == 0
        )
        print(
            "MISMATCH_COUNT =",
            len(bad)
        )
        print(
            "MAX_DIFF =",
            int(diff.max())
        )

        del models
        gc.collect()
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # 4. Paired dialogue-cluster evaluation
    # --------------------------------------------------

    # Official evaluation requires standard condition keys.
    rank_map = {
        "DENSE_BGE": np.asarray(
            pred["rank_DENSE_BGE"],
            dtype=np.int32
        )
    }

    for condition, components in ranks_by_condition.items():

        # FULL is the official model ranking.
        rank_map[condition] = components["FULL"]

        for component, arr in components.items():
            rank_map[
                f"{condition}::{component}"
            ] = arr

    required_keys = ["DENSE_BGE"] + list(base.CONDITIONS)

    missing = [
        key for key in required_keys
        if key not in rank_map
    ]

    if missing:
        raise RuntimeError(
            f"EVALUATION_INPUT_MISSING: {missing}"
        )

    print(
        "EVALUATION_INPUT_ALIGNMENT=PASS",
        "KEYS =", len(rank_map),
        flush=True
    )

    rng_all = np.random.default_rng(base.BOOT_SEED)

    all_eval = base.evaluate_subset(
        rank_map,
        dialogue_ids,
        np.arange(n),
        rng_all
    )

    rng_long = np.random.default_rng(base.BOOT_SEED + 1)

    long_eval = base.evaluate_subset(
        rank_map,
        dialogue_ids,
        long_idx,
        rng_long
    )

    # --------------------------------------------------
    # 5. Save independent artifacts
    # --------------------------------------------------
    # Output directory was created before inference.

    prediction_arrays = {
        "sample_ids": sample_ids,
        "dialogue_ids": dialogue_ids,
        "gold_match_idx": match,
        "candidate_counts": candidate_counts,
        "long20_indices": long_idx,
    }

    for condition, components in ranks_by_condition.items():

        for component, arr in components.items():

            key = f"rank_{condition}__{component}"
            prediction_arrays[key] = arr

    atomic_savez(
        OUT / "rank_predictions.npz",
        prediction_arrays
    )

    with open(
        OUT / "per_query.csv",
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "condition",
            "component",
            "sample_id",
            "dialogue_id",
            "matched",
            "candidate_count",
            "rank",
            "rank_percentile",
            "pairwise_ge_gold",
            "gold_score",
        ])

        for condition, components in ranks_by_condition.items():

            for component, arr in components.items():

                gold = gold_by_condition[condition][component]

                for i in range(n):

                    count = int(candidate_counts[i])
                    rank = int(arr[i])

                    writer.writerow([
                        condition,
                        component,
                        sample_ids[i],
                        dialogue_ids[i],
                        int(match[i] >= 0),
                        count,
                        rank,
                        rank / count,
                        (rank - 1) / max(count - 1, 1),
                        float(gold[i]),
                    ])

    all_exact = all(
        x["exact"] for x in reproduction.values()
    )

    summary = {
        "artifact": "S12C Official Full Test Component Attribution V1",
        "status": (
            "COMPLETED"
            if all_exact
            else "COMPLETED_WITH_RANK_REPRODUCTION_FAILURE"
        ),
        "scope": "EXTERNAL_TEST_FULL_POOL_COMPONENT_ANALYSIS",
        "queries": n,
        "response_pool": len(r_np),
        "matched": int((match >= 0).sum()),
        "unmatched": int((match < 0).sum()),
        "long20_queries": len(long_idx),
        "checkpoint_folds": 5,
        "components": {
            c: list(v.keys())
            for c, v in ranks_by_condition.items()
        },
        "official_full_rank_reproduction": reproduction,
        "fast_scorer_validation_errors": scorer_errors,
        "test_all_dialogue_bootstrap": all_eval,
        "test_long20_dialogue_bootstrap": long_eval,
        "training": False,
        "checkpoint_modification": False,
        "test_time_fitting": False,
        "checkpoint_reselection": False,
        "official_full_scorer_reused": True,
    }

    with open(
        OUT / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest_paths = {
        "protocol": base.P,
        "model_code": base.MODEL_CODE,
        "response_embeddings": base.RESP_EMB,
        "query_embeddings": base.Q_EMB,
        "gold_embeddings": base.G_EMB,
        "official_predictions": base.OUT["predictions"],
    }

    manifest = {
        "artifact": "STEP_2F16AM",
        "status": summary["status"],
        "input_sha256": {
            name: sha(path)
            for name, path in manifest_paths.items()
        },
        "checkpoint_manifest": protocol["checkpoint_manifest"],
        "query_batch": base.QB,
        "candidate_chunk": base.CC,
        "bootstrap_replicates": base.N_BOOT,
        "bootstrap_seed": base.BOOT_SEED,
        "training": False,
        "test_time_fitting": False,
        "checkpoint_modification": False,
    }

    with open(
        OUT / "run_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = [
        "STEP 2F16AM: OFFICIAL FULL TEST COMPONENT ATTRIBUTION",
        "=" * 76,
        f"STATUS = {summary['status']}",
        f"QUERIES = {n}",
        f"RESPONSE_POOL = {len(r_np)}",
        f"MATCHED = {summary['matched']}",
        f"UNMATCHED = {summary['unmatched']}",
        f"LONG20 = {len(long_idx)}",
        "",
        "OFFICIAL FULL RANK REPRODUCTION",
    ]

    for condition, result in reproduction.items():

        report.append(
            f"{condition}: exact={result['exact']}, "
            f"mismatch={result['mismatch_count']}, "
            f"max_diff={result['max_absolute_difference']}"
        )

    report += [
        "",
        "TRAINING = NO",
        "CHECKPOINT_MODIFICATION = NO",
        "TEST_TIME_FITTING = NO",
        "OFFICIAL_FULL_SCORER_REUSED = YES",
    ]

    (OUT / "report.txt").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\n" + "\n".join(report))
    print("\nOUTPUT =", OUT)
    print("STEP_2F16AM_COMPLETE")


if __name__ == "__main__":
    main()
