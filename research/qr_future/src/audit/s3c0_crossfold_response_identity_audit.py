import json
import hashlib
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

MASTER_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3a_ranking_future_master_index_v1.jsonl"
)

FOLD_FILE = (
    ROOT
    / "data/supervision/fsar_s3/"
      "fsar_s3b_internal_fold_assignment_v1.npy"
)

RESERVOIR_FILE = (
    ROOT
    / "cache/hard_negatives/"
      "fsar_train_dense_top100_reservoir_v1.npz"
)

OUT_DIR = ROOT / "data/supervision/fsar_s3"

MASK_OUT = (
    OUT_DIR
    / "fsar_s3c0_future_loss_eligibility_by_fold_v1.npz"
)

SUMMARY_OUT = (
    ROOT
    / "results/phase_s3/"
      "s3c0_crossfold_response_identity_summary_v1.json"
)

MASK_TMP = (
    OUT_DIR
    / "fsar_s3c0_future_loss_eligibility_by_fold_v1.tmp.npz"
)

SUMMARY_TMP = Path(str(SUMMARY_OUT) + ".tmp")


EXPECTED_SHA = {
    "master":
        "d9196a5ea11a9433c0550db6ebcfce76"
        "a6965ebec14ae9e3fb6660cc1e6ab3be",

    "fold":
        "a4f129fb6b5e5d23aad4215d261d33e4"
        "be2dc1ff34f8dcb7e4f0c92d430bcdf9",

    "reservoir":
        "e122dfe8f2c4eb8902068090d5852c79"
        "317641ea850720d58e4d4f4133474ab0"
}

N = 62889
NR = 61502
N_FOLDS = 5
DEPTH = 100

THRESHOLDS = [-2, -1, 0, 1]


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
print("FSAR PHASE S3-C0 CROSS-FOLD RESPONSE IDENTITY AUDIT")
print("=" * 80)


print()
print("===== 1. OUTPUT PROTECTION =====")

if MASK_OUT.exists() or SUMMARY_OUT.exists():
    raise RuntimeError(
        "Formal S3-C0 output already exists; refusing overwrite."
    )

for p in [MASK_TMP, SUMMARY_TMP]:
    if p.exists():
        p.unlink()

print("OUTPUT_PROTECTION = PASS")


print()
print("===== 2. FROZEN INPUT SHA =====")

actual_sha = {
    "master": sha256_file(MASTER_FILE),
    "fold": sha256_file(FOLD_FILE),
    "reservoir": sha256_file(RESERVOIR_FILE)
}

for k, v in actual_sha.items():
    print(k, "=", v)
    assert v == EXPECTED_SHA[k]

print("FROZEN_INPUT_SHA_GATE = PASS")


print()
print("===== 3. LOAD STRUCTURE =====")

master = load_jsonl(MASTER_FILE)

fold_id = np.load(
    FOLD_FILE,
    allow_pickle=False
)

reservoir = np.load(
    RESERVOIR_FILE,
    allow_pickle=False
)

candidate_rows = np.asarray(
    reservoir["candidate_response_row"],
    dtype=np.int32
)

assert len(master) == N
assert fold_id.shape == (N,)
assert candidate_rows.shape == (N, DEPTH)

gold_rows = np.asarray(
    [int(x["gold_response_row"]) for x in master],
    dtype=np.int32
)

future_available = np.asarray(
    [bool(x["future_value_available"]) for x in master],
    dtype=np.bool_
)

future_label = np.full(
    N,
    -99,
    dtype=np.int8
)

for i, row in enumerate(master):
    if row["future_value_available"]:
        future_label[i] = int(
            row["future_value_label"]
        )

assert future_available.sum() == 17458
assert gold_rows.min() >= 0
assert gold_rows.max() < NR

print("ranking_rows =", N)
print("future_rows =", int(future_available.sum()))
print("reservoir_shape =", candidate_rows.shape)
print("STRUCTURE = PASS")


print()
print("===== 4. RESPONSE IDENTITY FOLD DISTRIBUTION =====")

response_fold_bits = np.zeros(
    NR,
    dtype=np.uint8
)

response_occurrence_rows = np.zeros(
    NR,
    dtype=np.int32
)

for i in range(N):
    r = int(gold_rows[i])
    f = int(fold_id[i])

    response_fold_bits[r] |= np.uint8(
        1 << f
    )

    response_occurrence_rows[r] += 1


def bit_count_5(x):
    return sum(
        1
        for f in range(N_FOLDS)
        if x & (1 << f)
    )


response_fold_count = np.asarray(
    [
        bit_count_5(int(x))
        for x in response_fold_bits
    ],
    dtype=np.int8
)

assert np.all(response_fold_count >= 1)

crossfold_response_ids = int(
    np.sum(
        response_fold_count > 1
    )
)

crossfold_response_occurrences = int(
    response_occurrence_rows[
        response_fold_count > 1
    ].sum()
)

print(
    "response_ids_spanning_multiple_folds =",
    crossfold_response_ids
)

print(
    "response_occurrences_on_crossfold_ids =",
    crossfold_response_occurrences
)

print(
    "crossfold_response_identity_rate =",
    crossfold_response_ids / NR
)


future_response_labels = defaultdict(list)

for i in np.where(future_available)[0]:
    future_response_labels[
        int(gold_rows[i])
    ].append(
        (
            int(fold_id[i]),
            int(future_label[i])
        )
    )


future_identity_crossfold = 0
future_identity_label_conflict = 0

for r, values in future_response_labels.items():

    folds = {
        f
        for f, _ in values
    }

    labels = {
        y
        for _, y in values
    }

    if len(folds) > 1:
        future_identity_crossfold += 1

        if len(labels) > 1:
            future_identity_label_conflict += 1


print(
    "future_response_ids_spanning_multiple_folds =",
    future_identity_crossfold
)

print(
    "future_response_ids_with_crossfold_label_conflict =",
    future_identity_label_conflict
)

print("RESPONSE_IDENTITY_DISTRIBUTION = COMPLETE")


print()
print("===== 5. BUILD FOLD-SPECIFIC FUTURE LOSS MASK =====")

eligible = np.zeros(
    (N_FOLDS, N),
    dtype=np.bool_
)

fold_summary = {}


for f in range(N_FOLDS):

    validation_mask = (
        fold_id == f
    )

    training_mask = (
        fold_id != f
    )

    validation_gold_lookup = np.zeros(
        NR,
        dtype=np.bool_
    )

    validation_gold_lookup[
        np.unique(
            gold_rows[validation_mask]
        )
    ] = True


    training_future_mask = (
        training_mask
        &
        future_available
    )

    identity_overlap_mask = (
        training_future_mask
        &
        validation_gold_lookup[
            gold_rows
        ]
    )

    eligible_f = (
        training_future_mask
        &
        ~identity_overlap_mask
    )

    eligible[f] = eligible_f


    before_labels = future_label[
        training_future_mask
    ]

    after_labels = future_label[
        eligible_f
    ]


    before_counts = Counter(
        int(x)
        for x in before_labels
    )

    after_counts = Counter(
        int(x)
        for x in after_labels
    )


    threshold_feasibility = {}

    for tau in THRESHOLDS:

        binary = (
            after_labels > tau
        )

        neg_n = int(
            np.sum(~binary)
        )

        pos_n = int(
            np.sum(binary)
        )

        threshold_feasibility[
            str(tau)
        ] = {
            "negative": neg_n,
            "positive": pos_n
        }

        assert neg_n > 0
        assert pos_n > 0


    # Candidate corpus diagnostic only.
    train_candidate_rows = candidate_rows[
        training_mask
    ]

    heldout_candidate_exposure = (
        validation_gold_lookup[
            train_candidate_rows
        ]
    )

    candidate_exposure_n = int(
        heldout_candidate_exposure.sum()
    )

    candidate_query_n = int(
        np.sum(
            heldout_candidate_exposure.any(
                axis=1
            )
        )
    )


    # Validation future rows whose exact response identity
    # has a future-supervised occurrence in training folds.
    training_future_response_lookup = np.zeros(
        NR,
        dtype=np.bool_
    )

    training_future_response_lookup[
        gold_rows[
            training_future_mask
        ]
    ] = True

    validation_future_mask = (
        validation_mask
        &
        future_available
    )

    validation_future_exact_identity_seen = int(
        np.sum(
            training_future_response_lookup[
                gold_rows[
                    validation_future_mask
                ]
            ]
        )
    )


    info = {
        "fold": f,

        "training_rows":
            int(training_mask.sum()),

        "validation_rows":
            int(validation_mask.sum()),

        "training_future_rows_before_mask":
            int(training_future_mask.sum()),

        "future_rows_masked_for_response_identity":
            int(identity_overlap_mask.sum()),

        "training_future_rows_after_mask":
            int(eligible_f.sum()),

        "future_mask_rate_within_training_future":
            float(
                identity_overlap_mask.sum()
                /
                training_future_mask.sum()
            ),

        "future_label_counts_before_mask": {
            str(k): int(v)
            for k, v in sorted(
                before_counts.items()
            )
        },

        "future_label_counts_after_mask": {
            str(k): int(v)
            for k, v in sorted(
                after_counts.items()
            )
        },

        "post_mask_threshold_feasibility":
            threshold_feasibility,

        "training_candidate_exposures_from_validation_response_identities":
            candidate_exposure_n,

        "training_candidate_exposure_rate_from_validation_response_identities":
            float(
                candidate_exposure_n
                /
                train_candidate_rows.size
            ),

        "training_queries_with_validation_response_identity_candidate":
            candidate_query_n,

        "training_query_rate_with_validation_response_identity_candidate":
            float(
                candidate_query_n
                /
                training_mask.sum()
            ),

        "validation_future_rows":
            int(
                validation_future_mask.sum()
            ),

        "validation_future_rows_with_exact_response_future_supervision_in_training":
            validation_future_exact_identity_seen
    }


    fold_summary[str(f)] = info


    print()
    print(f"--- FOLD {f} ---")

    print(
        "training_future_rows_before_mask =",
        info[
            "training_future_rows_before_mask"
        ]
    )

    print(
        "future_rows_masked_for_response_identity =",
        info[
            "future_rows_masked_for_response_identity"
        ]
    )

    print(
        "training_future_rows_after_mask =",
        info[
            "training_future_rows_after_mask"
        ]
    )

    print(
        "future_label_counts_after_mask =",
        info[
            "future_label_counts_after_mask"
        ]
    )

    print(
        "post_mask_threshold_feasibility =",
        threshold_feasibility
    )

    print(
        "training_candidate_exposure_rate_from_validation_response_identities =",
        info[
            "training_candidate_exposure_rate_from_validation_response_identities"
        ]
    )

    print(
        "validation_future_rows_with_exact_response_future_supervision_in_training =",
        validation_future_exact_identity_seen
    )


print()
print(
    "ALL_POST_MASK_TRAINING_FOLDS_HAVE_BINARY_CLASSES = PASS"
)


print()
print("===== 6. SAVE MASK + SUMMARY =====")

np.savez_compressed(
    MASK_TMP,
    future_loss_eligible_by_fold=eligible,
    fold_id=fold_id,
    future_available=future_available,
    future_label=future_label,
    gold_response_row=gold_rows
)


summary = {
    "artifact":
        "FSAR S3-C0 Cross-Fold Response Identity Audit V1",

    "status":
        "PASS",

    "scope":
        "TRAIN_ONLY",

    "response_ids_spanning_multiple_folds":
        crossfold_response_ids,

    "response_occurrences_on_crossfold_ids":
        crossfold_response_occurrences,

    "future_response_ids_spanning_multiple_folds":
        future_identity_crossfold,

    "future_response_ids_with_crossfold_label_conflict":
        future_identity_label_conflict,

    "future_loss_identity_control":
        "MASK_TRAINING_FUTURE_LABEL_IF_RESPONSE_IDENTITY_OCCURS_IN_VALIDATION_FOLD",

    "ranking_candidate_corpus_policy":
        "GLOBAL_TRAIN_RESPONSE_CORPUS_RETAINED",

    "fold_summary":
        fold_summary,

    "all_post_mask_training_folds_have_all_binary_classes":
        True,

    "reranker_training_performed":
        False,

    "model_architecture_frozen":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256":
        actual_sha
}


SUMMARY_TMP.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


os.replace(
    MASK_TMP,
    MASK_OUT
)

os.replace(
    SUMMARY_TMP,
    SUMMARY_OUT
)

print("MASK_SHAPE =", eligible.shape)
print("MASK_SAVE = PASS")


print()
print("===== 7. FINAL SHA256 =====")

print(
    sha256_file(MASK_OUT),
    MASK_OUT
)

print(
    sha256_file(SUMMARY_OUT),
    SUMMARY_OUT
)

print()
print("RANKING_CORPUS_POLICY = GLOBAL_TRAIN_RESPONSE_CORPUS")
print("FUTURE_EXACT_RESPONSE_IDENTITY_CONTROL = MASKED")
print("RERANKER_TRAINING_PERFORMED = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print(
    "PHASE_S3C0_CROSSFOLD_RESPONSE_IDENTITY_AUDIT = PASS"
)
