import json
import re
import statistics
import unicodedata
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]

CANONICAL = ROOT / "data/canonical/psydial_dialogues.jsonl"
SPLIT = ROOT / "data/splits/psydial_dialogue_split.json"

OBS = ROOT / (
    "data/supervision/"
    "psydial_phase5b0_observability_train_v1.jsonl"
)

VAL = ROOT / (
    "data/supervision/"
    "psydial_phase5b0_value_train_v1.jsonl"
)

MANIFEST = ROOT / (
    "data/supervision/"
    "psydial_phase5b0_supervised_dataset_manifest_v1.json"
)

OUT = ROOT / (
    "results/audit/"
    "s1b3_final_train_integrity_audit_v1.json"
)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def norm_text(x):
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x.strip())
    return x


def duplicate_stats(values):
    c = Counter(values)
    n = len(values)
    unique = len(c)

    return {
        "rows": n,
        "unique": unique,
        "duplicate_rows": n - unique,
        "duplicate_row_rate":
            (n - unique) / n if n else None,
        "repeated_values":
            sum(v > 1 for v in c.values()),
        "max_frequency":
            max(c.values()) if c else 0
    }


def dist_summary(values):
    if not values:
        return {}

    values = sorted(values)

    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values)
    }


print("=" * 72)
print("FSAR PHASE S1-B3 FINAL TRAIN INTEGRITY AUDIT")
print("=" * 72)


# ============================================================
# Load
# ============================================================

with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

with MANIFEST.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

obs = load_jsonl(OBS)
val = load_jsonl(VAL)

train_ids = set(split["train_dialogues"])
dev_ids = set(split["dev_dialogues"])
test_ids = set(split["test_dialogues"])


# ============================================================
# 1. Split integrity
# ============================================================

print()
print("===== 1. SPLIT INTEGRITY =====")

assert split["seed"] == 42

assert len(train_ids) == 1666
assert len(dev_ids) == 239
assert len(test_ids) == 477

assert not (train_ids & dev_ids)
assert not (train_ids & test_ids)
assert not (dev_ids & test_ids)

print("train_dialogues =", len(train_ids))
print("dev_dialogues =", len(dev_ids))
print("test_dialogues =", len(test_ids))
print("dialogue_overlap = 0")
print("SPLIT_INTEGRITY = PASS")


# ============================================================
# 2. Canonical coverage
# ============================================================

print()
print("===== 2. CANONICAL COVERAGE =====")

canonical_ids = set()
train_dialogues = {}

with CANONICAL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        d = json.loads(line)
        did = d["dialogue_id"]

        assert did not in canonical_ids
        canonical_ids.add(did)

        if did in train_ids:
            train_dialogues[did] = d

assert canonical_ids == (
    train_ids | dev_ids | test_ids
)

assert len(canonical_ids) == 2382
assert len(train_dialogues) == 1666

print("canonical_dialogues =", len(canonical_ids))
print("canonical_train_dialogues =", len(train_dialogues))
print("CANONICAL_COVERAGE = PASS")


# ============================================================
# 3. Canonical pair schema / pair_index
# ============================================================

print()
print("===== 3. PAIR INDEX INTEGRITY =====")

total_train_pairs = 0
pair_index_position_mismatch = 0
duplicate_pair_index_dialogues = 0

for did, d in train_dialogues.items():
    seen = set()

    for pos, pair in enumerate(d["pairs"]):
        assert set(
            ["user", "assistant", "pair_index"]
        ).issubset(pair.keys())

        pi = pair["pair_index"]

        if pi in seen:
            duplicate_pair_index_dialogues += 1

        seen.add(pi)

        if pi != pos:
            pair_index_position_mismatch += 1

        total_train_pairs += 1

assert duplicate_pair_index_dialogues == 0
assert pair_index_position_mismatch == 0

print("train_pairs =", total_train_pairs)
print(
    "pair_index_position_mismatch =",
    pair_index_position_mismatch
)
print(
    "duplicate_pair_index_dialogues =",
    duplicate_pair_index_dialogues
)
print("PAIR_INDEX_INTEGRITY = PASS")


# ============================================================
# 4. Frozen supervision counts
# ============================================================

print()
print("===== 4. FROZEN SUPERVISION COUNTS =====")

assert len(obs) == 46091
assert len(val) == 17458

obs_counts = Counter(
    r["observability_label"]
    for r in obs
)

val_counts = Counter(
    r["value_label"]
    for r in val
)

assert obs_counts == Counter({
    0: 19705,
    1: 26386
})

assert val_counts == Counter({
    -2: 13,
    -1: 1914,
    0: 3041,
    1: 8798,
    2: 3692
})

print("observability_rows =", len(obs))
print(
    "observability_label_counts =",
    dict(sorted(obs_counts.items()))
)

print("value_rows =", len(val))
print(
    "value_label_counts =",
    dict(sorted(val_counts.items()))
)

print("FROZEN_SUPERVISION_COUNTS = PASS")


# ============================================================
# 5. Supervision belongs only to TRAIN
# ============================================================

print()
print("===== 5. SUPERVISION SPLIT ISOLATION =====")

obs_dialogues = {
    r["dialogue_id"] for r in obs
}

val_dialogues = {
    r["dialogue_id"] for r in val
}

assert obs_dialogues <= train_ids
assert val_dialogues <= train_ids

assert not (obs_dialogues & dev_ids)
assert not (obs_dialogues & test_ids)

assert not (val_dialogues & dev_ids)
assert not (val_dialogues & test_ids)

print("observability_dialogues =", len(obs_dialogues))
print("value_dialogues =", len(val_dialogues))
print("DEV_supervision_rows = 0")
print("TEST_supervision_rows = 0")
print("SUPERVISION_SPLIT_ISOLATION = PASS")


# ============================================================
# 6. No future fields
# ============================================================

print()
print("===== 6. FUTURE FIELD LEAKAGE =====")

keys = set()

for r in obs:
    keys.update(r.keys())

for r in val:
    keys.update(r.keys())

bad = [
    k for k in keys
    if any(
        x in k.lower()
        for x in [
            "next_user",
            "future_user",
            "next_utterance",
            "future_utterance"
        ]
    )
]

assert bad == []

assert (
    manifest["future_user_in_model_input"]
    is False
)

assert (
    manifest["next_user_written_to_output"]
    is False
)

print("forbidden_future_fields =", bad)
print("FUTURE_FIELD_LEAKAGE = PASS")


# ============================================================
# 7. Identifier uniqueness
# ============================================================

print()
print("===== 7. IDENTIFIER UNIQUENESS =====")

for name, rows in [
    ("OBS", obs),
    ("VALUE", val)
]:
    for field in [
        "item_id",
        "sample_id",
        "source_row_index",
        "presentation_index"
    ]:
        vals = [
            r[field]
            for r in rows
        ]

        dup = (
            len(vals)
            - len(set(vals))
        )

        print(
            name,
            field,
            "duplicates =",
            dup
        )

        assert dup == 0

print("IDENTIFIER_UNIQUENESS = PASS")


# ============================================================
# 8. Sample-id consistency
# ============================================================

print()
print("===== 8. SAMPLE ID CONSISTENCY =====")

sample_id_mismatch = 0

for r in obs:
    expected = (
        f"psydial_{r['dialogue_id']}_"
        f"{r['target_pair_index']}"
    )

    if r["sample_id"] != expected:
        sample_id_mismatch += 1

assert sample_id_mismatch == 0

print("sample_id_mismatches = 0")
print("SAMPLE_ID_CONSISTENCY = PASS")


# ============================================================
# 9. VALUE subset of OBS-positive
# ============================================================

print()
print("===== 9. VALUE SUBSET =====")

obs_by_item = {
    r["item_id"]: r
    for r in obs
}

assert len(obs_by_item) == len(obs)

value_missing = 0
value_not_positive = 0
metadata_mismatch = 0

fields = [
    "item_id",
    "presentation_index",
    "source_row_index",
    "dialogue_id",
    "sample_id",
    "target_pair_index"
]

for vr in val:
    oid = vr["item_id"]

    if oid not in obs_by_item:
        value_missing += 1
        continue

    orow = obs_by_item[oid]

    if (
        orow["observability_label"]
        != 1
    ):
        value_not_positive += 1

    for f in fields:
        if vr[f] != orow[f]:
            metadata_mismatch += 1

assert value_missing == 0
assert value_not_positive == 0
assert metadata_mismatch == 0

print("value_missing_from_obs = 0")
print("value_not_obs_positive = 0")
print("metadata_mismatch = 0")
print("VALUE_SUBSET = PASS")


# ============================================================
# 10. Canonical provenance traceability
#     Report exact mismatch, gate on normalized mismatch.
# ============================================================

print()
print("===== 10. CANONICAL PROVENANCE TRACEABILITY =====")

exact_user_mismatch = 0
exact_response_mismatch = 0
exact_full_pair_mismatch = 0

normalized_user_mismatch = 0
normalized_response_mismatch = 0
normalized_full_pair_mismatch = 0

out_of_range = 0

exact_mismatch_dialogues = set()

value_item_ids = {
    r["item_id"]
    for r in val
}

exact_mismatch_value_rows = 0

for r in obs:
    did = r["dialogue_id"]
    idx = r["target_pair_index"]

    pairs = train_dialogues[did]["pairs"]

    if not (
        0 <= idx < len(pairs)
    ):
        out_of_range += 1
        continue

    p = pairs[idx]

    user_exact = (
        r["current_user"]
        ==
        p["user"]
    )

    response_exact = (
        r["counselor_response"]
        ==
        p["assistant"]
    )

    user_norm = (
        norm_text(
            r["current_user"]
        )
        ==
        norm_text(
            p["user"]
        )
    )

    response_norm = (
        norm_text(
            r["counselor_response"]
        )
        ==
        norm_text(
            p["assistant"]
        )
    )

    if not user_exact:
        exact_user_mismatch += 1

    if not response_exact:
        exact_response_mismatch += 1

    if not (
        user_exact
        and response_exact
    ):
        exact_full_pair_mismatch += 1
        exact_mismatch_dialogues.add(did)

        if (
            r["item_id"]
            in value_item_ids
        ):
            exact_mismatch_value_rows += 1

    if not user_norm:
        normalized_user_mismatch += 1

    if not response_norm:
        normalized_response_mismatch += 1

    if not (
        user_norm
        and response_norm
    ):
        normalized_full_pair_mismatch += 1


assert out_of_range == 0

# Frozen result from S1-B2
assert exact_full_pair_mismatch == 263
assert len(exact_mismatch_dialogues) == 125
assert exact_mismatch_value_rows == 90

# Hard provenance gate
assert normalized_user_mismatch == 0
assert normalized_response_mismatch == 0
assert normalized_full_pair_mismatch == 0


print(
    "exact_user_mismatch =",
    exact_user_mismatch
)

print(
    "exact_response_mismatch =",
    exact_response_mismatch
)

print(
    "exact_full_pair_mismatch =",
    exact_full_pair_mismatch
)

print(
    "exact_mismatch_dialogues =",
    len(exact_mismatch_dialogues)
)

print(
    "exact_mismatch_value_rows =",
    exact_mismatch_value_rows
)

print(
    "normalized_user_mismatch =",
    normalized_user_mismatch
)

print(
    "normalized_response_mismatch =",
    normalized_response_mismatch
)

print(
    "normalized_full_pair_mismatch =",
    normalized_full_pair_mismatch
)

print("CANONICAL_PROVENANCE_TRACEABILITY = PASS")


# ============================================================
# 11. TRAIN response / pair duplicate diagnostics
# ============================================================

print()
print("===== 11. TRAIN DUPLICATE DIAGNOSTICS =====")

users = []
responses = []
pairs_exact = []
pairs_norm = []

for did in sorted(train_ids):
    for p in train_dialogues[did]["pairs"]:
        u = p["user"]
        r = p["assistant"]

        users.append(u)
        responses.append(r)

        pairs_exact.append(
            (u, r)
        )

        pairs_norm.append(
            (
                norm_text(u),
                norm_text(r)
            )
        )


user_exact_stats = duplicate_stats(users)

user_norm_stats = duplicate_stats(
    [
        norm_text(x)
        for x in users
    ]
)

response_exact_stats = duplicate_stats(
    responses
)

response_norm_values = [
    norm_text(x)
    for x in responses
]

response_norm_stats = duplicate_stats(
    response_norm_values
)

pair_exact_stats = duplicate_stats(
    pairs_exact
)

pair_norm_stats = duplicate_stats(
    pairs_norm
)

response_freq = Counter(
    response_norm_values
)

top20_response_frequencies = sorted(
    response_freq.values(),
    reverse=True
)[:20]


print(
    "current_user_exact =",
    user_exact_stats
)

print(
    "current_user_normalized =",
    user_norm_stats
)

print(
    "response_exact =",
    response_exact_stats
)

print(
    "response_normalized =",
    response_norm_stats
)

print(
    "query_response_pair_exact =",
    pair_exact_stats
)

print(
    "query_response_pair_normalized =",
    pair_norm_stats
)

print(
    "top20_normalized_response_frequencies =",
    top20_response_frequencies
)

print("TRAIN_DUPLICATE_DIAGNOSTIC = COMPLETE")


# ============================================================
# 12. Future-State coverage
# ============================================================

print()
print("===== 12. FUTURE-STATE COVERAGE =====")

obs_per_dialogue = Counter(
    r["dialogue_id"]
    for r in obs
)

val_per_dialogue = Counter(
    r["dialogue_id"]
    for r in val
)

obs_dist = dist_summary(
    list(
        obs_per_dialogue.values()
    )
)

val_dist = dist_summary(
    list(
        val_per_dialogue.values()
    )
)

print(
    "observability_dialogue_coverage =",
    f"{len(obs_per_dialogue)}/{len(train_ids)}"
)

print(
    "value_dialogue_coverage =",
    f"{len(val_per_dialogue)}/{len(train_ids)}"
)

print(
    "observability_rows_per_dialogue =",
    obs_dist
)

print(
    "value_rows_per_dialogue =",
    val_dist
)

print("FUTURE_STATE_COVERAGE = COMPLETE")


# ============================================================
# 13. TRAIN-only candidate-pool feasibility
# ============================================================

print()
print("===== 13. TRAIN CANDIDATE POOL =====")

unique_responses_exact = len(
    set(responses)
)

unique_responses_norm = len(
    set(response_norm_values)
)

assert unique_responses_exact > 1
assert unique_responses_norm > 1

print(
    "train_response_rows =",
    len(responses)
)

print(
    "train_unique_responses_exact =",
    unique_responses_exact
)

print(
    "train_unique_responses_normalized =",
    unique_responses_norm
)

print("candidate_pool_source = TRAIN_ONLY")
print("DEV_RESPONSE_POOL_USED = NO")
print("TEST_RESPONSE_POOL_USED = NO")
print("TRAIN_CANDIDATE_POOL = PASS")


# ============================================================
# 14. Final report
# ============================================================

report = {
    "artifact":
        "FSAR Phase S1-B3 Final TRAIN Integrity Audit V1",

    "status":
        "PASS",

    "split": {
        "train_dialogues":
            len(train_ids),
        "dev_dialogues":
            len(dev_ids),
        "test_dialogues":
            len(test_ids),
        "dialogue_overlap":
            0
    },

    "canonical": {
        "dialogues":
            len(canonical_ids),
        "train_dialogues":
            len(train_dialogues),
        "train_pairs":
            total_train_pairs,
        "pair_index_position_mismatch":
            pair_index_position_mismatch
    },

    "supervision": {
        "observability_rows":
            len(obs),
        "value_rows":
            len(val),
        "observability_dialogues":
            len(obs_dialogues),
        "value_dialogues":
            len(val_dialogues)
    },

    "provenance": {
        "authoritative_text_source":
            "canonical",

        "exact_user_mismatch":
            exact_user_mismatch,

        "exact_response_mismatch":
            exact_response_mismatch,

        "exact_full_pair_mismatch":
            exact_full_pair_mismatch,

        "exact_mismatch_dialogues":
            len(exact_mismatch_dialogues),

        "exact_mismatch_value_rows":
            exact_mismatch_value_rows,

        "normalized_user_mismatch":
            normalized_user_mismatch,

        "normalized_response_mismatch":
            normalized_response_mismatch,

        "normalized_full_pair_mismatch":
            normalized_full_pair_mismatch,

        "normalization":
            "NFKC + strip + collapse whitespace"
    },

    "duplicates": {
        "current_user_exact":
            user_exact_stats,

        "current_user_normalized":
            user_norm_stats,

        "response_exact":
            response_exact_stats,

        "response_normalized":
            response_norm_stats,

        "query_response_pair_exact":
            pair_exact_stats,

        "query_response_pair_normalized":
            pair_norm_stats,

        "top20_normalized_response_frequencies":
            top20_response_frequencies
    },

    "future_state_coverage": {
        "observability_dialogues":
            len(obs_per_dialogue),

        "value_dialogues":
            len(val_per_dialogue),

        "observability_rows_per_dialogue":
            obs_dist,

        "value_rows_per_dialogue":
            val_dist
    },

    "candidate_pool": {
        "source":
            "TRAIN_ONLY",

        "response_rows":
            len(responses),

        "unique_responses_exact":
            unique_responses_exact,

        "unique_responses_normalized":
            unique_responses_norm,

        "dev_used":
            False,

        "test_used":
            False
    },

    "gate": {
        "split_integrity":
            "PASS",

        "supervision_integrity":
            "PASS",

        "future_field_leakage":
            "PASS",

        "normalized_canonical_traceability":
            "PASS",

        "train_candidate_pool_feasibility":
            "PASS"
    }
}


OUT.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("===== FINAL PHASE S1 GATE =====")

print("SPLIT_INTEGRITY = PASS")
print("SUPERVISION_INTEGRITY = PASS")
print("FUTURE_FIELD_LEAKAGE = PASS")
print("NORMALIZED_CANONICAL_TRACEABILITY = PASS")
print("TRAIN_CANDIDATE_POOL_FEASIBILITY = PASS")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")

print()
print("PHASE_S1_FINAL_GATE = PASS")
print("REPORT =", OUT)
