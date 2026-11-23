import json
import hashlib
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[4]

MASTER_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3a_ranking_future_master_index_v1.jsonl"
)

RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/"
      "fsar_train_dense_top100_reservoir_v1.npz"
)

SPLIT_FILE = (
    ROOT
    / "data/splits/psydial_dialogue_split.json"
)

OUT_DIR = ROOT / "data/supervision/fsar_s3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_JSONL = (
    OUT_DIR
    / "fsar_s3b_internal_fold_assignment_v1.jsonl"
)

FOLD_NPY = (
    OUT_DIR
    / "fsar_s3b_internal_fold_assignment_v1.npy"
)

MANIFEST = (
    OUT_DIR
    / "fsar_s3b_internal_fold_manifest_v1.json"
)

FOLD_JSONL_TMP = Path(str(FOLD_JSONL) + ".tmp")
FOLD_NPY_TMP = (
    OUT_DIR
    / "fsar_s3b_internal_fold_assignment_v1.tmp.npy"
)
MANIFEST_TMP = Path(str(MANIFEST) + ".tmp")


EXPECTED_MASTER_SHA = (
    "d9196a5ea11a9433c0550db6ebcfce76"
    "a6965ebec14ae9e3fb6660cc1e6ab3be"
)

EXPECTED_RESERVOIR_SHA = (
    "e122dfe8f2c4eb8902068090d5852c79"
    "317641ea850720d58e4d4f4133474ab0"
)

EXPECTED_SPLIT_SHA = (
    "c688695f423dd489a11ad0b145bd1997"
    "08d0910f09bfba0081ff8ef39c19529c"
)

N = 62889
N_FOLDS = 5
SEED = 42
DEPTH = 100

EXPECTED_VALUE_COUNTS = {
    -2: 13,
    -1: 1914,
    0: 3041,
    1: 8798,
    2: 3692
}


def sha256_file(path, block=1024 * 1024):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            x = f.read(block)
            if not x:
                break
            h.update(x)

    return h.hexdigest()


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


print("=" * 80)
print("FSAR PHASE S3-B INTERNAL DIALOGUE-LEVEL FOLD CONSTRUCTION")
print("=" * 80)


# ============================================================
# 1. Output protection
# ============================================================

print()
print("===== 1. OUTPUT PROTECTION =====")

formal = [
    FOLD_JSONL,
    FOLD_NPY,
    MANIFEST
]

existing = [
    str(p)
    for p in formal
    if p.exists()
]

if existing:
    raise RuntimeError(
        "Formal S3-B output already exists; refusing overwrite:\n"
        + "\n".join(existing)
    )

for p in [
    FOLD_JSONL_TMP,
    FOLD_NPY_TMP,
    MANIFEST_TMP
]:
    if p.exists():
        p.unlink()

print("OUTPUT_PROTECTION = PASS")


# ============================================================
# 2. Frozen input SHA
# ============================================================

print()
print("===== 2. FROZEN INPUT SHA =====")

master_sha = sha256_file(MASTER_FILE)
reservoir_sha = sha256_file(RESERVOIR_FILE)
split_sha = sha256_file(SPLIT_FILE)

print("master =", master_sha)
print("reservoir =", reservoir_sha)
print("dialogue_split =", split_sha)

assert master_sha == EXPECTED_MASTER_SHA
assert reservoir_sha == EXPECTED_RESERVOIR_SHA
assert split_sha == EXPECTED_SPLIT_SHA

print("FROZEN_INPUT_SHA_GATE = PASS")


# ============================================================
# 3. Load and verify TRAIN-only universe
# ============================================================

print()
print("===== 3. TRAIN UNIVERSE =====")

master = load_jsonl(MASTER_FILE)

assert len(master) == N

with SPLIT_FILE.open("r", encoding="utf-8") as f:
    split = json.load(f)

train_ids = {
    str(x)
    for x in split["train_dialogues"]
}

dev_ids = {
    str(x)
    for x in split["dev_dialogues"]
}

test_ids = {
    str(x)
    for x in split["test_dialogues"]
}

master_dialogues = {
    str(x["dialogue_id"])
    for x in master
}

assert master_dialogues <= train_ids
assert not (master_dialogues & dev_ids)
assert not (master_dialogues & test_ids)

assert len(master_dialogues) == 1666

reservoir = np.load(
    RESERVOIR_FILE,
    allow_pickle=False
)

candidate_rows = np.asarray(
    reservoir["candidate_response_row"],
    dtype=np.int32
)

gold_rows = np.asarray(
    reservoir["gold_response_row"],
    dtype=np.int32
)

assert candidate_rows.shape == (N, DEPTH)
assert gold_rows.shape == (N,)

assert np.all(
    candidate_rows
    != gold_rows[:, None]
)

print("ranking_rows =", N)
print("train_dialogues =", len(master_dialogues))
print("reservoir_shape =", candidate_rows.shape)
print("gold_in_negative_reservoir = 0")
print("DEV_rows = 0")
print("TEST_rows = 0")
print("TRAIN_UNIVERSE = PASS")


# ============================================================
# 4. Construct stratification target
# ============================================================

print()
print("===== 4. STRATIFICATION TARGET =====")

groups = np.asarray(
    [
        str(x["dialogue_id"])
        for x in master
    ],
    dtype=object
)

strata = []

future_labels = []

for row in master:

    if row["future_value_available"]:
        label = int(row["future_value_label"])
        strata.append(f"V_{label}")
        future_labels.append(label)

    else:
        strata.append("NO_FUTURE")

strata = np.asarray(
    strata,
    dtype=object
)

future_counts = Counter(future_labels)

assert dict(sorted(future_counts.items())) == EXPECTED_VALUE_COUNTS

strata_counts = Counter(strata.tolist())

print("strata_counts =", dict(sorted(strata_counts.items())))
print("future_value_counts =", dict(sorted(future_counts.items())))
print("STRATIFICATION_TARGET = PASS")


# ============================================================
# 5. One-shot 5-fold SGKF
# ============================================================

print()
print("===== 5. STRATIFIED GROUP K-FOLD =====")

sgkf = StratifiedGroupKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

X_dummy = np.zeros(
    (N, 1),
    dtype=np.uint8
)

fold_id = np.full(
    N,
    -1,
    dtype=np.int8
)

split_records = []

for fold, (train_idx, valid_idx) in enumerate(
    sgkf.split(
        X_dummy,
        strata,
        groups
    )
):

    assert np.all(
        fold_id[valid_idx] == -1
    )

    fold_id[valid_idx] = fold

    train_groups = set(
        groups[train_idx].tolist()
    )

    valid_groups = set(
        groups[valid_idx].tolist()
    )

    overlap = (
        train_groups
        &
        valid_groups
    )

    assert len(overlap) == 0

    split_records.append(
        (train_idx, valid_idx)
    )

assert np.all(fold_id >= 0)
assert set(fold_id.tolist()) == {0, 1, 2, 3, 4}

# Every dialogue must map to exactly one fold.
dialogue_to_folds = defaultdict(set)

for i, did in enumerate(groups):
    dialogue_to_folds[did].add(
        int(fold_id[i])
    )

bad_dialogues = {
    did: values
    for did, values in dialogue_to_folds.items()
    if len(values) != 1
}

assert not bad_dialogues

print("n_splits =", N_FOLDS)
print("random_state =", SEED)
print("seed_search = NO")
print("dialogue_cross_fold_overlap = 0")
print("unassigned_rows = 0")
print("FOLD_CONSTRUCTION = PASS")


# ============================================================
# 6. Fold audit
# ============================================================

print()
print("===== 6. FOLD AUDIT =====")

fold_summary = {}

thresholds = [-2, -1, 0, 1]

for fold, (train_idx, valid_idx) in enumerate(split_records):

    valid_rows = [
        master[i]
        for i in valid_idx
    ]

    train_rows = [
        master[i]
        for i in train_idx
    ]

    valid_dialogues = {
        str(x["dialogue_id"])
        for x in valid_rows
    }

    train_dialogues = {
        str(x["dialogue_id"])
        for x in train_rows
    }

    assert not (
        valid_dialogues
        &
        train_dialogues
    )

    valid_future = [
        int(x["future_value_label"])
        for x in valid_rows
        if x["future_value_available"]
    ]

    train_future = [
        int(x["future_value_label"])
        for x in train_rows
        if x["future_value_available"]
    ]

    valid_counts = Counter(valid_future)
    train_counts = Counter(train_future)

    threshold_feasibility = {}

    train_future_np = np.asarray(
        train_future,
        dtype=np.int8
    )

    for tau in thresholds:

        binary = (
            train_future_np > tau
        ).astype(np.int8)

        neg_n = int(
            np.sum(binary == 0)
        )

        pos_n = int(
            np.sum(binary == 1)
        )

        threshold_feasibility[str(tau)] = {
            "negative": neg_n,
            "positive": pos_n
        }

        assert neg_n > 0
        assert pos_n > 0

    info = {
        "fold": fold,

        "validation_rows":
            len(valid_idx),

        "validation_dialogues":
            len(valid_dialogues),

        "training_rows":
            len(train_idx),

        "training_dialogues":
            len(train_dialogues),

        "validation_future_rows":
            len(valid_future),

        "training_future_rows":
            len(train_future),

        "validation_future_label_counts": {
            str(k): int(v)
            for k, v in sorted(
                valid_counts.items()
            )
        },

        "training_future_label_counts": {
            str(k): int(v)
            for k, v in sorted(
                train_counts.items()
            )
        },

        "training_threshold_feasibility":
            threshold_feasibility,

        "validation_listwise_candidate_instances":
            int(
                len(valid_idx)
                * 101
            ),

        "training_listwise_candidate_instances":
            int(
                len(train_idx)
                * 101
            )
    }

    fold_summary[str(fold)] = info

    print()
    print(f"--- FOLD {fold} ---")

    print(
        "validation_rows =",
        info["validation_rows"]
    )

    print(
        "validation_dialogues =",
        info["validation_dialogues"]
    )

    print(
        "validation_future_rows =",
        info["validation_future_rows"]
    )

    print(
        "validation_future_label_counts =",
        info["validation_future_label_counts"]
    )

    print(
        "training_future_rows =",
        info["training_future_rows"]
    )

    print(
        "training_threshold_feasibility =",
        threshold_feasibility
    )


print()
print("ALL_TRAINING_FOLDS_HAVE_BINARY_CLASSES = PASS")
print("FOLD_AUDIT = PASS")


# ============================================================
# 7. Write fold assignment
# ============================================================

print()
print("===== 7. WRITE FOLD ASSIGNMENT =====")

with FOLD_JSONL_TMP.open(
    "w",
    encoding="utf-8"
) as f:

    for i, row in enumerate(master):

        out = {
            "ranking_row": i,
            "query_id": row["query_id"],
            "dialogue_id": row["dialogue_id"],
            "fold_id": int(fold_id[i])
        }

        f.write(
            json.dumps(
                out,
                ensure_ascii=False
            )
            + "\n"
        )


np.save(
    FOLD_NPY_TMP,
    fold_id,
    allow_pickle=False
)

reload_fold = np.load(
    FOLD_NPY_TMP,
    allow_pickle=False
)

assert np.array_equal(
    reload_fold,
    fold_id
)

written = load_jsonl(
    FOLD_JSONL_TMP
)

assert len(written) == N

for i, row in enumerate(written):
    assert row["ranking_row"] == i
    assert row["query_id"] == master[i]["query_id"]
    assert int(row["fold_id"]) == int(fold_id[i])

print("fold_jsonl_rows =", len(written))
print("fold_npy_shape =", fold_id.shape)
print("FOLD_ASSIGNMENT_WRITE = PASS")


# ============================================================
# 8. Manifest
# ============================================================

manifest = {
    "artifact":
        "FSAR S3-B TRAIN Internal Dialogue Folds V1",

    "status":
        "PASS",

    "scope":
        "TRAIN_ONLY",

    "ranking_rows":
        N,

    "train_dialogues":
        1666,

    "n_splits":
        N_FOLDS,

    "splitter":
        "StratifiedGroupKFold",

    "group":
        "dialogue_id",

    "shuffle":
        True,

    "random_state":
        SEED,

    "seed_search":
        False,

    "stratification":
        "NO_FUTURE_OR_FUTURE_VALUE_LABEL",

    "ranking_supervision": {
        "gold_per_query": 1,
        "negative_per_query": 100,
        "negative_sampling": "NONE",
        "all_top100_negatives_used": True,
        "candidates_per_query": 101,
        "full_pass_candidate_instances":
            N * 101
    },

    "future_supervision": {
        "rows": 17458,
        "gold_only": True,
        "negative_future_labels": False,
        "oversampling": False
    },

    "fold_summary":
        fold_summary,

    "dialogue_cross_fold_overlap":
        0,

    "all_training_folds_have_all_ordinal_binary_classes":
        True,

    "model_architecture":
        "NOT_YET_FROZEN",

    "future_loss_weight_beta":
        "NOT_YET_FROZEN",

    "reranker_training_performed":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256": {
        "master":
            master_sha,
        "reservoir":
            reservoir_sha,
        "dialogue_split":
            split_sha
    }
}


MANIFEST_TMP.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


os.replace(
    FOLD_JSONL_TMP,
    FOLD_JSONL
)

os.replace(
    FOLD_NPY_TMP,
    FOLD_NPY
)

os.replace(
    MANIFEST_TMP,
    MANIFEST
)


print()
print("===== 8. FINAL SHA256 =====")

print(
    sha256_file(FOLD_JSONL),
    FOLD_JSONL
)

print(
    sha256_file(FOLD_NPY),
    FOLD_NPY
)

print(
    sha256_file(MANIFEST),
    MANIFEST
)

print()
print("NEGATIVE_SAMPLING = NONE")
print("NEGATIVES_PER_QUERY = 100")
print("CANDIDATES_PER_QUERY = 101")
print("SEED_SEARCH = NO")
print("RERANKER_TRAINING_PERFORMED = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S3B_INTERNAL_FOLDS = PASS")
