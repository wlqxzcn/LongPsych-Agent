import json
import hashlib
import os
from pathlib import Path
from collections import Counter

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

QUERY_INDEX = (
    ROOT
    / "cache/embeddings/fsar_train_queries_bge_index_v1.jsonl"
)

RESERVOIR = (
    ROOT
    / "cache/hard_negatives/fsar_train_dense_top100_reservoir_v1.npz"
)

RESERVOIR_MANIFEST = (
    ROOT
    / "cache/hard_negatives/fsar_train_dense_top100_reservoir_manifest_v1.json"
)

OBS_FILE = (
    ROOT
    / "data/supervision/psydial_phase5b0_observability_train_v1.jsonl"
)

VALUE_FILE = (
    ROOT
    / "data/supervision/psydial_phase5b0_value_train_v1.jsonl"
)

OUT_DIR = ROOT / "data/supervision/fsar_s3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_OUT = (
    OUT_DIR
    / "fsar_s3a_ranking_future_master_index_v1.jsonl"
)

MANIFEST_OUT = (
    OUT_DIR
    / "fsar_s3a_ranking_future_master_manifest_v1.json"
)

MASTER_TMP = Path(str(MASTER_OUT) + ".tmp")
MANIFEST_TMP = Path(str(MANIFEST_OUT) + ".tmp")


EXPECTED_SHA = {
    "query_index":
        "7d3659d38668f653289fd649fef186cd2f0a1003338af88ff6c805c23960de75",

    "reservoir":
        "e122dfe8f2c4eb8902068090d5852c79317641ea850720d58e4d4f4133474ab0",

    "reservoir_manifest":
        "191fbd12f235c1bb7c059267f8e7bc0d0b75332da2014f7cfa3aca97b6a226f5",

    "observability":
        "1ada4725a82fd51e3e3c43d4aad7e003542b9d2366f1ea71c1e530bfa7f86808",

    "value":
        "696fec90539938ef41836d2b813dbf8ee1b58cdf2fce950cdb4b3d6c76576be0"
}

N_QUERY = 62889
N_RESPONSE = 61502
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


print("=" * 78)
print("FSAR PHASE S3-A RANKING-FUTURE SUPERVISION JOIN")
print("=" * 78)


print()
print("===== 1. OUTPUT PROTECTION =====")

if MASTER_OUT.exists() or MANIFEST_OUT.exists():
    raise RuntimeError(
        "Formal S3-A outputs already exist; refusing overwrite."
    )

for p in [MASTER_TMP, MANIFEST_TMP]:
    if p.exists():
        p.unlink()

print("OUTPUT_PROTECTION = PASS")


print()
print("===== 2. FROZEN INPUT SHA =====")

paths = {
    "query_index": QUERY_INDEX,
    "reservoir": RESERVOIR,
    "reservoir_manifest": RESERVOIR_MANIFEST,
    "observability": OBS_FILE,
    "value": VALUE_FILE
}

actual_sha = {}

for key, path in paths.items():
    actual_sha[key] = sha256_file(path)
    print(key, "=", actual_sha[key])

    assert actual_sha[key] == EXPECTED_SHA[key]

print("FROZEN_INPUT_SHA_GATE = PASS")


print()
print("===== 3. LOAD RANKING UNIVERSE =====")

query_rows = load_jsonl(QUERY_INDEX)

assert len(query_rows) == N_QUERY

query_ids = [
    x["query_id"]
    for x in query_rows
]

assert len(query_ids) == len(set(query_ids))

query_id_to_row = {
    qid: i
    for i, qid in enumerate(query_ids)
}

reservoir = np.load(
    RESERVOIR,
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

assert candidate_rows.shape == (
    N_QUERY,
    DEPTH
)

assert gold_rows.shape == (N_QUERY,)

assert candidate_rows.min() >= 0
assert candidate_rows.max() < N_RESPONSE

assert np.all(
    candidate_rows
    != gold_rows[:, None]
)

print("ranking_queries =", len(query_rows))
print("reservoir_shape =", candidate_rows.shape)
print("gold_in_negative_reservoir = 0")
print("RANKING_UNIVERSE = PASS")


print()
print("===== 4. LOAD FUTURE SUPERVISION =====")

obs_rows = load_jsonl(OBS_FILE)
value_rows = load_jsonl(VALUE_FILE)

assert len(obs_rows) == 46091
assert len(value_rows) == 17458

obs_by_sample = {
    x["sample_id"]: x
    for x in obs_rows
}

value_by_sample = {
    x["sample_id"]: x
    for x in value_rows
}

assert len(obs_by_sample) == len(obs_rows)
assert len(value_by_sample) == len(value_rows)

value_counts = Counter(
    int(x["value_label"])
    for x in value_rows
)

assert dict(sorted(value_counts.items())) == EXPECTED_VALUE_COUNTS

print("observability_rows =", len(obs_rows))
print("future_value_rows =", len(value_rows))
print(
    "future_value_label_counts =",
    dict(sorted(value_counts.items()))
)

print("FUTURE_SUPERVISION_COUNTS = PASS")


print()
print("===== 5. FUTURE -> RANKING TRACEABILITY =====")

missing_from_ranking = 0
not_observability_positive = 0
identity_mismatch = 0

for sample_id, value_row in value_by_sample.items():

    if sample_id not in query_id_to_row:
        missing_from_ranking += 1
        continue

    obs = obs_by_sample.get(sample_id)

    if obs is None or int(obs["observability_label"]) != 1:
        not_observability_positive += 1

    rank_row = query_rows[
        query_id_to_row[sample_id]
    ]

    if (
        str(rank_row["dialogue_id"])
        != str(value_row["dialogue_id"])
        or
        int(rank_row["target_pair_index"])
        != int(value_row["target_pair_index"])
    ):
        identity_mismatch += 1


assert missing_from_ranking == 0
assert not_observability_positive == 0
assert identity_mismatch == 0

print(
    "future_rows_missing_from_ranking =",
    missing_from_ranking
)

print(
    "future_rows_not_observability_positive =",
    not_observability_positive
)

print(
    "future_identity_mismatch =",
    identity_mismatch
)

print("FUTURE_RANKING_TRACEABILITY = PASS")


print()
print("===== 6. CONSTRUCT MASTER INDEX =====")

future_dialogues = set()
future_rows_written = 0

with MASTER_TMP.open(
    "w",
    encoding="utf-8"
) as f:

    for i, q in enumerate(query_rows):

        qid = q["query_id"]

        value = value_by_sample.get(qid)

        future_available = (
            value is not None
        )

        if future_available:
            label = int(
                value["value_label"]
            )

            future_rows_written += 1
            future_dialogues.add(
                str(q["dialogue_id"])
            )

        else:
            label = None

        out = {
            "ranking_row": i,
            "query_id": qid,
            "dialogue_id": q["dialogue_id"],
            "target_pair_index":
                q["target_pair_index"],

            "gold_response_uid":
                q["gold_response_uid"],

            "gold_response_row":
                int(gold_rows[i]),

            "reservoir_row":
                i,

            "reservoir_depth":
                DEPTH,

            "future_value_available":
                future_available,

            "future_value_label":
                label
        }

        f.write(
            json.dumps(
                out,
                ensure_ascii=False
            )
            + "\n"
        )


assert future_rows_written == 17458

print("master_rows =", N_QUERY)
print(
    "future_value_available_rows =",
    future_rows_written
)

print(
    "future_value_unavailable_rows =",
    N_QUERY - future_rows_written
)

print(
    "future_value_dialogues =",
    len(future_dialogues)
)

print(
    "future_value_row_coverage =",
    future_rows_written / N_QUERY
)

print("MASTER_INDEX_CONSTRUCTION = PASS")


print()
print("===== 7. OUTPUT AUDIT =====")

master_reload = load_jsonl(
    MASTER_TMP
)

assert len(master_reload) == N_QUERY

assert [
    x["ranking_row"]
    for x in master_reload
] == list(range(N_QUERY))

assert len({
    x["query_id"]
    for x in master_reload
}) == N_QUERY

output_value_counts = Counter(
    int(x["future_value_label"])
    for x in master_reload
    if x["future_value_available"]
)

assert (
    dict(sorted(output_value_counts.items()))
    ==
    EXPECTED_VALUE_COUNTS
)

# Explicitly forbid text/future-user payload.
for row in master_reload:
    forbidden = {
        k
        for k in row.keys()
        if (
            "future_user" in k.lower()
            or
            "next_user" in k.lower()
            or
            k in {
                "current_user",
                "counselor_response",
                "response_text"
            }
        )
    }

    assert not forbidden


print("duplicate_query_id = 0")
print("text_payload_fields = 0")
print("future_user_fields = 0")
print("OUTPUT_AUDIT = PASS")


manifest = {
    "artifact":
        "FSAR S3-A Ranking-Future Master Index V1",

    "status":
        "PASS",

    "scope":
        "TRAIN_ONLY",

    "ranking_rows":
        N_QUERY,

    "reservoir_depth":
        DEPTH,

    "future_value_rows":
        future_rows_written,

    "future_value_dialogues":
        len(future_dialogues),

    "future_value_row_coverage":
        future_rows_written / N_QUERY,

    "future_value_label_counts":
        {
            str(k): v
            for k, v in sorted(
                output_value_counts.items()
            )
        },

    "future_loss_attachment":
        "FACTUAL_GOLD_PAIR_ONLY",

    "negative_future_state_labels_created":
        False,

    "output_contains_text":
        False,

    "formal_training_negative_count":
        "NOT_YET_FROZEN",

    "negative_sampling_strategy":
        "NOT_YET_FROZEN",

    "internal_train_validation_split":
        "NOT_YET_FROZEN",

    "reranker_training_performed":
        False,

    "dev_payload_used":
        False,

    "test_payload_used":
        False,

    "input_sha256":
        actual_sha
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
    MASTER_TMP,
    MASTER_OUT
)

os.replace(
    MANIFEST_TMP,
    MANIFEST_OUT
)


print()
print("===== 8. FINAL SHA256 =====")

print(
    sha256_file(MASTER_OUT),
    MASTER_OUT
)

print(
    sha256_file(MANIFEST_OUT),
    MANIFEST_OUT
)

print()
print("NEGATIVE_FUTURE_STATE_LABELS_CREATED = NO")
print("RERANKER_TRAINING_PERFORMED = NO")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S3A_RANKING_FUTURE_JOIN = PASS")
