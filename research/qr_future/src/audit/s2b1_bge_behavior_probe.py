import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[4]

MODEL_PATH = ROOT / "models/BAAI/bge-large-zh-v1___5"

OUT = ROOT / "results/audit/s2b1_bge_behavior_probe_v1.json"


print("=" * 72)
print("FSAR S2-B1 BGE BEHAVIOR PROBE")
print("=" * 72)

assert torch.cuda.is_available()

model = SentenceTransformer(
    str(MODEL_PATH),
    device="cuda"
)

model.eval()

for p in model.parameters():
    p.requires_grad = False


print()
print("===== 1. MODEL STRUCTURE =====")

print("max_seq_length =", model.max_seq_length)

dim = model.get_sentence_embedding_dimension()

print("sentence_embedding_dimension =", dim)

print("modules:")

for name, module in model._modules.items():
    print(
        name,
        module.__class__.__name__
    )


assert model.max_seq_length == 512
assert dim == 1024


# ------------------------------------------------------------
# Pooling configuration
# ------------------------------------------------------------

pooling = None
normalize_found = False

for module in model._modules.values():

    cls = module.__class__.__name__

    if cls == "Pooling":
        pooling = module

    if cls == "Normalize":
        normalize_found = True


assert pooling is not None
assert normalize_found


pooling_config = {
    "pooling_mode_cls_token":
        bool(pooling.pooling_mode_cls_token),

    "pooling_mode_mean_tokens":
        bool(pooling.pooling_mode_mean_tokens),

    "pooling_mode_max_tokens":
        bool(pooling.pooling_mode_max_tokens),

    "pooling_mode_mean_sqrt_len_tokens":
        bool(
            pooling.pooling_mode_mean_sqrt_len_tokens
        )
}


print("pooling_config =", pooling_config)
print("normalize_module =", normalize_found)


assert (
    pooling_config["pooling_mode_cls_token"]
    is True
)

assert (
    pooling_config["pooling_mode_mean_tokens"]
    is False
)

assert (
    pooling_config["pooling_mode_max_tokens"]
    is False
)


print("MODEL_STRUCTURE = PASS")


print()
print("===== 2. SMALL TEXT PROBE =====")


texts = [
    "最近我总是睡不好，晚上会想很多事情。",
    "最近我总是睡不好，晚上会想很多事情。",
    "听起来这段时间的压力让你很难真正放松下来。",
    "我不知道自己到底应该怎么办。",
    "你愿意多说说这种感觉通常什么时候最明显吗？"
]


with torch.inference_mode():

    emb_bs1 = model.encode(
        texts,
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    emb_bs3 = model.encode(
        texts,
        batch_size=3,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


emb_bs1 = np.asarray(
    emb_bs1,
    dtype=np.float32
)

emb_bs3 = np.asarray(
    emb_bs3,
    dtype=np.float32
)


print("shape_bs1 =", emb_bs1.shape)
print("shape_bs3 =", emb_bs3.shape)

assert emb_bs1.shape == (5, 1024)
assert emb_bs3.shape == (5, 1024)


norm1 = np.linalg.norm(
    emb_bs1,
    axis=1
)

norm3 = np.linalg.norm(
    emb_bs3,
    axis=1
)


print(
    "norm_bs1_min_mean_max =",
    float(norm1.min()),
    float(norm1.mean()),
    float(norm1.max())
)

print(
    "norm_bs3_min_mean_max =",
    float(norm3.min()),
    float(norm3.mean()),
    float(norm3.max())
)


assert np.allclose(
    norm1,
    1.0,
    atol=1e-5
)

assert np.allclose(
    norm3,
    1.0,
    atol=1e-5
)


# identical text must be identical
duplicate_max_abs_diff = float(
    np.max(
        np.abs(
            emb_bs1[0]
            -
            emb_bs1[1]
        )
    )
)


print(
    "duplicate_text_max_abs_diff =",
    duplicate_max_abs_diff
)


assert duplicate_max_abs_diff <= 1e-6


# batch-size invariance
batch_max_abs_diff = float(
    np.max(
        np.abs(
            emb_bs1
            -
            emb_bs3
        )
    )
)


batch_mean_abs_diff = float(
    np.mean(
        np.abs(
            emb_bs1
            -
            emb_bs3
        )
    )
)


print(
    "batch_size_max_abs_diff =",
    batch_max_abs_diff
)

print(
    "batch_size_mean_abs_diff =",
    batch_mean_abs_diff
)


# GPU kernels can differ by tiny numerical noise.
assert batch_max_abs_diff < 1e-4


# ------------------------------------------------------------
# No NaN / Inf
# ------------------------------------------------------------

assert np.isfinite(emb_bs1).all()
assert np.isfinite(emb_bs3).all()

print("NUMERICAL_FINITE = PASS")
print("DUPLICATE_TEXT_STABILITY = PASS")
print("BATCH_SIZE_INVARIANCE = PASS")


report = {
    "artifact":
        "FSAR S2-B1 BGE Behavior Probe V1",

    "status":
        "PASS",

    "model_path":
        str(MODEL_PATH),

    "max_seq_length":
        model.max_seq_length,

    "embedding_dimension":
        dim,

    "pooling":
        pooling_config,

    "normalize_module":
        normalize_found,

    "probe_shape":
        list(emb_bs1.shape),

    "norm_bs1": {
        "min": float(norm1.min()),
        "mean": float(norm1.mean()),
        "max": float(norm1.max())
    },

    "norm_bs3": {
        "min": float(norm3.min()),
        "mean": float(norm3.mean()),
        "max": float(norm3.max())
    },

    "duplicate_text_max_abs_diff":
        duplicate_max_abs_diff,

    "batch_size_max_abs_diff":
        batch_max_abs_diff,

    "batch_size_mean_abs_diff":
        batch_mean_abs_diff,

    "cuda":
        True,

    "gpu":
        torch.cuda.get_device_name(0)
}


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("PHASE_S2B1_BGE_BEHAVIOR_PROBE = PASS")
print("REPORT =", OUT)
