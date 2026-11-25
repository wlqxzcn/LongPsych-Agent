import sys
import json
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.s12c_matched_ablation_models import (
    QRFutureFullMatchedModel
)

QUERY_INDEX = 3148
EMPTY_ROW = 54735
CHUNK_SIZE = 2048

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sha256_array(arr):
    return hashlib.sha256(
        np.ascontiguousarray(arr).tobytes()
    ).hexdigest()

# --------------------------------------------------
# Load frozen external test data
# --------------------------------------------------

script = (
    ROOT / "research/qr_future/src/evaluation/"
    "s12c_step_2f18t_orca_fsar_test_evaluation.py"
)

spec = importlib.util.spec_from_file_location(
    "external_test_audit", script
)

mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

dev, test_base = mod.modules()
data = mod.load_test_data(dev, test_base)

q = data["q"][QUERY_INDEX:QUERY_INDEX+1]
gold = data["gold"][QUERY_INDEX:QUERY_INDEX+1]
responses = data["responses"]
match_idx = int(data["match"][QUERY_INDEX])

assert q.shape == (1, 1024)
assert gold.shape == (1, 1024)
assert responses.shape == (61502, 1024)
assert match_idx == -1

print("\n========== TARGET ==========")
print("QUERY_INDEX:", QUERY_INDEX)
print("SAMPLE_ID:", data["sample_ids"][QUERY_INDEX])
print("DIALOGUE_ID:", data["dialogue_ids"][QUERY_INDEX])
print("GOLD_MATCH_INDEX:", match_idx)

# --------------------------------------------------
# Load deployment manifest
# --------------------------------------------------

manifest_path = ROOT / "data/qr_future/manifest.json"

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

assert manifest["status"] == "PASS"
assert manifest["condition"] == "QR_FUTURE_FULL"

# --------------------------------------------------
# Projection
# --------------------------------------------------

def project(x, basis):
    return F.normalize(
        x - (x @ basis) @ basis.T,
        p=2,
        dim=-1
    )

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

q_gpu = torch.from_numpy(q.copy()).to(DEVICE)
gold_gpu = torch.from_numpy(gold.copy()).to(DEVICE)

results = []

# --------------------------------------------------
# Five-fold evaluation
# --------------------------------------------------

for fold in range(5):

    print("\n" + "=" * 60)
    print("FOLD", fold)
    print("=" * 60)

    fold_info = manifest["folds"][str(fold)]

    basis_path = ROOT / "data/qr_future" / fold_info["basis_file"]
    checkpoint_path = ROOT / "data/qr_future" / fold_info["checkpoint_file"]

    assert sha256_file(basis_path) == fold_info["basis_file_sha256"]
    assert sha256_file(checkpoint_path) == fold_info["checkpoint_sha256"]

    basis_np = np.load(basis_path, allow_pickle=False)

    assert basis_np.shape == (1024, 2)

    basis_sha = sha256_array(basis_np)

    assert basis_sha == fold_info["basis_array_sha256"]

    basis = torch.from_numpy(
        basis_np.copy()
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    assert checkpoint["condition"] == "QR_FUTURE_FULL"
    assert checkpoint["input_space"] == "ORCA_K2_FOLD_SPECIFIC"
    assert int(checkpoint["orca_fold"]) == fold
    assert checkpoint["orca_basis_sha256"] == basis_sha

    model = QRFutureFullMatchedModel().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    q_proj = project(q_gpu, basis)
    gold_proj = project(gold_gpu, basis)

    # Score all original response candidates.
    response_scores = np.empty(
        len(responses), dtype=np.float32
    )

    with torch.inference_mode():

        for start in range(0, len(responses), CHUNK_SIZE):

            end = min(start + CHUNK_SIZE, len(responses))

            r_gpu = torch.from_numpy(
                responses[start:end].copy()
            ).to(DEVICE)

            r_proj = project(r_gpu, basis)

            scores = model.rank_scores(
                q_proj,
                r_proj.unsqueeze(0)
            ).squeeze(0)

            response_scores[start:end] = (
                scores.float().cpu().numpy()
            )

    # Append Gold only when it is absent from response corpus.
    gold_score = model.rank_scores(
        q_proj,
        gold_proj.unsqueeze(1)
    ).item()

    all_scores = np.concatenate([
        response_scores,
        np.asarray([gold_score], dtype=np.float32)
    ])

    gold_idx = len(responses)

    assert len(all_scores) == 61503
    assert gold_idx == 61502

    # Gold rank
    greater = int(np.sum(all_scores > gold_score))
    equal = int(np.sum(all_scores == gold_score))

    gold_rank_min = greater + 1
    gold_rank_max = greater + equal

    # Empty response rank
    empty_score = float(all_scores[EMPTY_ROW])

    empty_greater = int(
        np.sum(all_scores > empty_score)
    )

    empty_equal = int(
        np.sum(all_scores == empty_score)
    )

    empty_rank_min = empty_greater + 1
    empty_rank_max = empty_greater + empty_equal

    margin = float(gold_score - empty_score)

    result = {
        "fold": fold,
        "best_epoch": int(checkpoint["best_epoch"]),
        "candidate_count": len(all_scores),
        "gold_score": float(gold_score),
        "gold_rank_min": gold_rank_min,
        "gold_rank_max": gold_rank_max,
        "gold_top1": gold_rank_min == 1,
        "gold_top5": gold_rank_min <= 5,
        "gold_top10": gold_rank_min <= 10,
        "gold_top100": gold_rank_min <= 100,
        "empty_score": empty_score,
        "empty_rank_min": empty_rank_min,
        "empty_rank_max": empty_rank_max,
        "empty_top100": empty_rank_min <= 100,
        "gold_empty_margin": margin,
    }

    results.append(result)

    print("GOLD_SCORE:", gold_score)
    print("GOLD_RANK:", gold_rank_min, "-", gold_rank_max)
    print("GOLD_TOP1:", result["gold_top1"])
    print("GOLD_TOP5:", result["gold_top5"])
    print("GOLD_TOP10:", result["gold_top10"])
    print("GOLD_TOP100:", result["gold_top100"])

    print("EMPTY_SCORE:", empty_score)
    print("EMPTY_RANK:", empty_rank_min, "-", empty_rank_max)
    print("GOLD_EMPTY_MARGIN:", margin)

    del model, basis, q_proj, gold_proj
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("FIVE-FOLD GOLD AUDIT SUMMARY")
print("=" * 60)

for r in results:
    print(
        f"FOLD={r['fold']} "
        f"GOLD_RANK={r['gold_rank_min']}-{r['gold_rank_max']} "
        f"EMPTY_RANK={r['empty_rank_min']}-{r['empty_rank_max']} "
        f"MARGIN={r['gold_empty_margin']:.6f}"
    )

summary = {
    "query_index": QUERY_INDEX,
    "sample_id": str(data["sample_ids"][QUERY_INDEX]),
    "dialogue_id": str(data["dialogue_ids"][QUERY_INDEX]),
    "gold_match_index": match_idx,
    "results": results,
    "gold_top1_folds": sum(r["gold_top1"] for r in results),
    "gold_top5_folds": sum(r["gold_top5"] for r in results),
    "gold_top10_folds": sum(r["gold_top10"] for r in results),
    "gold_top100_folds": sum(r["gold_top100"] for r in results),
    "empty_top100_folds": sum(r["empty_top100"] for r in results),
}

output = ROOT / "logs/qr_future_gold_case_audit_3148.json"

output.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nOUTPUT:", output)
print("GOLD_CASE_AUDIT_COMPLETE")
