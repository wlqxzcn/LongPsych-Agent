import sys
import json
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
CHUNK = 2048
TOP_K = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def project(x, basis):
    return F.normalize(
        x - (x @ basis) @ basis.T,
        p=2,
        dim=-1
    )

def rank_of(scores, index):
    value = scores[index]
    greater = int(np.sum(scores > value))
    equal = int(np.sum(scores == value))
    return [greater + 1, greater + equal]

def describe_top(scores, response_rows):
    indices = np.argsort(-scores, kind="stable")[:TOP_K]
    output = []

    for rank, idx in enumerate(indices, 1):
        idx = int(idx)

        if idx == len(response_rows):
            text = "EXTERNAL_GOLD"
            uid = "EXTERNAL_GOLD"
        else:
            text = response_rows[idx].get(
                "response_text", ""
            )
            uid = response_rows[idx].get(
                "response_uid", ""
            )

        output.append({
            "rank": rank,
            "index": idx,
            "response_uid": uid,
            "score": float(scores[idx]),
            "text": str(text)[:250]
        })

    return output

# Load frozen modules and test data.
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

response_rows = test_base.read_jsonl(test_base.RESP)

q_np = data["q"][QUERY_INDEX:QUERY_INDEX+1]
gold_np = data["gold"][QUERY_INDEX:QUERY_INDEX+1]
responses_np = data["responses"]

assert len(response_rows) == 61502
assert responses_np.shape == (61502, 1024)
assert int(data["match"][QUERY_INDEX]) == -1
assert response_rows[EMPTY_ROW]["response_text"] == ""

query_row = data["rows"][QUERY_INDEX]

print("\n========== CASE INFORMATION ==========")
print("SAMPLE_ID:", query_row["sample_id"])
print("DIALOGUE_ID:", query_row["dialogue_id"])
print("QUERY:", query_row["query"])
print("GOLD:", query_row["gold_answer"])

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

q = torch.from_numpy(q_np.copy()).to(DEVICE)
gold = torch.from_numpy(gold_np.copy()).to(DEVICE)
responses = torch.from_numpy(
    responses_np.copy()
).to(DEVICE)

manifest = json.loads(
    (ROOT / "data/qr_future/manifest.json").read_text()
)

results = []

def record(mode, scores):
    gold_idx = len(responses_np)

    all_scores = np.concatenate([
        scores[:len(responses_np)],
        np.asarray([scores[gold_idx]], dtype=np.float32)
    ])

    item = {
        "mode": mode,
        "gold_rank": rank_of(all_scores, gold_idx),
        "empty_rank": rank_of(all_scores, EMPTY_ROW),
        "gold_score": float(all_scores[gold_idx]),
        "empty_score": float(all_scores[EMPTY_ROW]),
        "gold_empty_margin": float(
            all_scores[gold_idx] - all_scores[EMPTY_ROW]
        ),
        "top10": describe_top(all_scores, response_rows)
    }

    results.append(item)

    print("\n", "=" * 55)
    print("MODE:", mode)
    print("GOLD_RANK:", item["gold_rank"])
    print("EMPTY_RANK:", item["empty_rank"])
    print("GOLD_EMPTY_MARGIN:", item["gold_empty_margin"])

    for x in item["top10"][:5]:
        print(
            f"TOP{x['rank']} index={x['index']} "
            f"score={x['score']:.6f} "
            f"uid={x['response_uid']}"
        )
        print("TEXT:", x["text"])

# --------------------------------------------------
# 1. Dense BGE
# --------------------------------------------------

print("\nDENSE BGE")

with torch.inference_mode():
    dense_scores = (
        q @ responses.T
    ).squeeze(0).float().cpu().numpy()

    dense_gold = (
        q @ gold.T
    ).item()

dense_all = np.concatenate([
    dense_scores,
    np.asarray([dense_gold], dtype=np.float32)
])

record("DENSE_BGE", dense_all)

# --------------------------------------------------
# 2. ORCA and ORCA-FSAR
# --------------------------------------------------

for fold in range(5):

    print("\nPROCESSING FOLD", fold)

    info = manifest["folds"][str(fold)]

    basis_path = ROOT / "data/qr_future" / info["basis_file"]
    checkpoint_path = ROOT / "data/qr_future" / info["checkpoint_file"]

    basis_np = np.load(
        basis_path, allow_pickle=False
    )

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

    assert checkpoint["orca_basis_sha256"] == (
        info["basis_array_sha256"]
    )

    with torch.inference_mode():

        q_proj = project(q, basis)
        gold_proj = project(gold, basis)

        # ORCA cosine scores
        orca_scores = np.empty(
            len(responses_np), dtype=np.float32
        )

        # QR-Future ranking scores
        model = QRFutureFullMatchedModel().to(DEVICE)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True
        )
        model.eval()

        fsar_scores = np.empty(
            len(responses_np), dtype=np.float32
        )

        for start in range(0, len(responses_np), CHUNK):

            end = min(start + CHUNK, len(responses_np))

            r_proj = project(
                responses[start:end], basis
            )

            sim = q_proj @ r_proj.T

            orca_scores[start:end] = (
                sim.squeeze(0).cpu().numpy()
            )

            pred = model.rank_scores(
                q_proj,
                r_proj.unsqueeze(0)
            )

            fsar_scores[start:end] = (
                pred.squeeze(0).float().cpu().numpy()
            )

        orca_gold = (
            q_proj @ gold_proj.T
        ).item()

        fsar_gold = model.rank_scores(
            q_proj,
            gold_proj.unsqueeze(1)
        ).item()

    orca_all = np.concatenate([
        orca_scores,
        np.asarray([orca_gold], dtype=np.float32)
    ])

    fsar_all = np.concatenate([
        fsar_scores,
        np.asarray([fsar_gold], dtype=np.float32)
    ])

    record(f"ORCA_FOLD_{fold}", orca_all)
    record(f"ORCA_FSAR_FOLD_{fold}", fsar_all)

    del model, basis, q_proj, gold_proj
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

# --------------------------------------------------
# Save
# --------------------------------------------------

output = ROOT / "logs/qr_future_three_way_case_3148.json"

output.write_text(
    json.dumps({
        "query_index": QUERY_INDEX,
        "sample_id": str(data["sample_ids"][QUERY_INDEX]),
        "results": results
    }, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\n========== COMPARISON SUMMARY ==========")

for x in results:
    print(
        x["mode"],
        "GOLD=", x["gold_rank"],
        "EMPTY=", x["empty_rank"],
        "MARGIN=", round(x["gold_empty_margin"], 6)
    )

print("\nOUTPUT:", output)
print("THREE_WAY_CASE_AUDIT_COMPLETE")
