import json
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[4]

CANONICAL = ROOT / "data/canonical/psydial_dialogues.jsonl"
SPLIT = ROOT / "data/splits/psydial_dialogue_split.json"

OUT_DIR = ROOT / "data/canonical/fsar_s2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_OUT = OUT_DIR / "train_unique_responses_v1.jsonl"
QUERIES_OUT = OUT_DIR / "train_query_gold_map_v1.jsonl"
MANIFEST_OUT = OUT_DIR / "train_candidate_universe_manifest_v1.json"


def normalize_text(x):
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x.strip())
    return x


def make_response_uid(norm_text):
    h = hashlib.sha256(
        norm_text.encode("utf-8")
    ).hexdigest()
    return "resp_" + h[:24]


with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

train_ids = set(split["train_dialogues"])

dialogues = {}

with CANONICAL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        d = json.loads(line)

        if d["dialogue_id"] in train_ids:
            dialogues[d["dialogue_id"]] = d


assert len(dialogues) == 1666


# ============================================================
# Build normalized unique response universe
# ============================================================

response_occurrences = defaultdict(list)
response_original_forms = defaultdict(Counter)

query_rows = []

total_pairs = 0

for did in sorted(dialogues):

    d = dialogues[did]

    for pos, pair in enumerate(d["pairs"]):

        assert pair["pair_index"] == pos

        user = pair["user"]
        response = pair["assistant"]

        norm_response = normalize_text(response)
        response_uid = make_response_uid(norm_response)

        response_occurrences[response_uid].append({
            "dialogue_id": did,
            "pair_index": pos
        })

        response_original_forms[response_uid][response] += 1

        query_rows.append({
            "query_id": f"psydial_{did}_{pos}",
            "dialogue_id": did,
            "target_pair_index": pos,
            "current_user": user,
            "gold_response_uid": response_uid
        })

        total_pairs += 1


assert total_pairs == 62889
assert len(query_rows) == 62889
assert len(response_occurrences) == 61502


# ============================================================
# Write unique response catalog
# ============================================================

response_rows = []

for response_uid in sorted(response_occurrences):

    forms = response_original_forms[response_uid]

    # Canonical representative:
    # most frequent exact form;
    # lexical order only breaks ties deterministically.
    representative = sorted(
        forms.items(),
        key=lambda x: (-x[1], x[0])
    )[0][0]

    norm_response = normalize_text(representative)

    occurrences = response_occurrences[response_uid]

    row = {
        "response_uid": response_uid,
        "response_text": representative,
        "normalized_response_text": norm_response,
        "occurrence_count": len(occurrences),
        "distinct_exact_forms": len(forms),
        "first_dialogue_id": occurrences[0]["dialogue_id"],
        "first_pair_index": occurrences[0]["pair_index"]
    }

    response_rows.append(row)


with RESPONSES_OUT.open("w", encoding="utf-8") as f:
    for row in response_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )


with QUERIES_OUT.open("w", encoding="utf-8") as f:
    for row in query_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# Structural audit
# ============================================================

response_uid_set = {
    r["response_uid"]
    for r in response_rows
}

assert len(response_uid_set) == 61502


query_ids = [
    r["query_id"]
    for r in query_rows
]

assert len(query_ids) == len(set(query_ids))


missing_gold = [
    r["query_id"]
    for r in query_rows
    if r["gold_response_uid"] not in response_uid_set
]

assert not missing_gold


# Every UID must be deterministic from normalized text
uid_reconstruction_failures = 0

for r in response_rows:

    expected = make_response_uid(
        r["normalized_response_text"]
    )

    if expected != r["response_uid"]:
        uid_reconstruction_failures += 1

assert uid_reconstruction_failures == 0


# Check normalized response uniqueness
normalized_texts = [
    r["normalized_response_text"]
    for r in response_rows
]

assert len(normalized_texts) == len(set(normalized_texts))


# Occurrence count must sum back to original pair count
occurrence_sum = sum(
    r["occurrence_count"]
    for r in response_rows
)

assert occurrence_sum == 62889


frequency_counter = Counter(
    r["occurrence_count"]
    for r in response_rows
)

multi_occurrence_candidates = sum(
    1
    for r in response_rows
    if r["occurrence_count"] > 1
)

max_occurrence = max(
    r["occurrence_count"]
    for r in response_rows
)


manifest = {
    "artifact":
        "FSAR S2-A TRAIN Candidate Universe V1",

    "status":
        "FROZEN_COMPLETE",

    "source":
        "TRAIN_ONLY",

    "train_dialogues":
        len(dialogues),

    "train_query_rows":
        len(query_rows),

    "response_occurrences":
        total_pairs,

    "unique_normalized_responses":
        len(response_rows),

    "multi_occurrence_unique_responses":
        multi_occurrence_candidates,

    "max_response_occurrence_count":
        max_occurrence,

    "occurrence_count_distribution":
        {
            str(k): v
            for k, v in sorted(
                frequency_counter.items()
            )
        },

    "normalization":
        "NFKC + strip + collapse whitespace",

    "candidate_unit":
        "UNIQUE_NORMALIZED_RESPONSE",

    "gold_mapping_complete":
        True,

    "gold_mapping_missing":
        0,

    "duplicate_response_uid":
        0,

    "duplicate_normalized_response_text":
        0,

    "duplicate_query_id":
        0,

    "response_uid_reconstruction_failures":
        0,

    "dev_payload_used":
        False,

    "test_payload_used":
        False
}


MANIFEST_OUT.write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print("=" * 70)
print("FSAR PHASE S2-A TRAIN CANDIDATE UNIVERSE")
print("=" * 70)

print("train_dialogues =", len(dialogues))
print("train_query_rows =", len(query_rows))
print("response_occurrences =", total_pairs)

print(
    "unique_normalized_responses =",
    len(response_rows)
)

print(
    "multi_occurrence_unique_responses =",
    multi_occurrence_candidates
)

print(
    "max_response_occurrence_count =",
    max_occurrence
)

print("duplicate_response_uid = 0")
print("duplicate_normalized_response_text = 0")
print("duplicate_query_id = 0")
print("gold_mapping_missing = 0")
print("uid_reconstruction_failures = 0")

print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S2A_CANDIDATE_UNIVERSE = PASS")

print("RESPONSES =", RESPONSES_OUT)
print("QUERIES =", QUERIES_OUT)
print("MANIFEST =", MANIFEST_OUT)
