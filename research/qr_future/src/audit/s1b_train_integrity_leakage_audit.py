import json
import re
import unicodedata
import hashlib
import statistics
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[4]

CANONICAL = ROOT / "data/canonical/psydial_dialogues.jsonl"
SPLIT = ROOT / "data/splits/psydial_dialogue_split.json"
OBS = ROOT / "data/supervision/psydial_phase5b0_observability_train_v1.jsonl"
VAL = ROOT / "data/supervision/psydial_phase5b0_value_train_v1.jsonl"
MANIFEST = ROOT / "data/supervision/psydial_phase5b0_supervised_dataset_manifest_v1.json"

OUT_JSON = ROOT / "results/audit/s1b_train_integrity_leakage_audit_v1.json"


# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"JSONL parse failure: {path} line={ln}: {e}"
                )
    return rows


def norm_text(x):
    """
    Conservative normalization for duplicate diagnostics only.
    Does NOT remove punctuation or semantic material.
    """
    if not isinstance(x, str):
        return x
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x.strip())
    return x


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def counter_to_json(c):
    return {str(k): int(v) for k, v in sorted(c.items(), key=lambda x: str(x[0]))}


def detect_text_field(pair, candidates, role):
    if not isinstance(pair, dict):
        raise RuntimeError(
            f"Canonical pair is not dict: {type(pair).__name__}"
        )

    lower_map = {str(k).lower(): k for k in pair.keys()}

    found = []
    for c in candidates:
        if c.lower() in lower_map:
            found.append(lower_map[c.lower()])

    # remove duplicates while preserving order
    unique = []
    for x in found:
        if x not in unique:
            unique.append(x)

    if len(unique) != 1:
        raise RuntimeError(
            f"Cannot uniquely detect canonical {role} field. "
            f"pair_keys={list(pair.keys())}, candidates_found={unique}"
        )

    return unique[0]


def duplicate_stats(values):
    c = Counter(values)
    n = len(values)
    unique = len(c)
    duplicate_rows = n - unique
    repeated_values = sum(1 for v in c.values() if v > 1)
    max_frequency = max(c.values()) if c else 0

    return {
        "rows": n,
        "unique": unique,
        "duplicate_rows": duplicate_rows,
        "duplicate_row_rate": duplicate_rows / n if n else None,
        "values_repeated_at_least_twice": repeated_values,
        "max_frequency": max_frequency,
    }


def assert_equal(a, b, name):
    if a != b:
        raise AssertionError(f"{name}: expected={b!r}, observed={a!r}")


# ============================================================
# Load frozen artifacts
# ============================================================

print("=" * 68)
print("FSAR PHASE S1-B TRAIN INTEGRITY / LEAKAGE AUDIT")
print("=" * 68)

for p in [CANONICAL, SPLIT, OBS, VAL, MANIFEST]:
    assert p.exists(), f"MISSING FILE: {p}"

with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

with MANIFEST.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

obs_rows = load_jsonl(OBS)
val_rows = load_jsonl(VAL)

train_ids = set(split["train_dialogues"])
dev_ids = set(split["dev_dialogues"])
test_ids = set(split["test_dialogues"])


# ============================================================
# 1. Frozen split integrity
# ============================================================

print()
print("===== 1. FROZEN SPLIT INTEGRITY =====")

assert_equal(split["seed"], 42, "split seed")

assert len(train_ids & dev_ids) == 0, "TRAIN/DEV dialogue overlap"
assert len(train_ids & test_ids) == 0, "TRAIN/TEST dialogue overlap"
assert len(dev_ids & test_ids) == 0, "DEV/TEST dialogue overlap"

assert_equal(len(train_ids), 1666, "TRAIN dialogue count")
assert_equal(len(dev_ids), 239, "DEV dialogue count")
assert_equal(len(test_ids), 477, "TEST dialogue count")

print("split_seed =", split["seed"])
print("train_dialogues =", len(train_ids))
print("dev_dialogues =", len(dev_ids))
print("test_dialogues =", len(test_ids))
print("train_dev_overlap = 0")
print("train_test_overlap = 0")
print("dev_test_overlap = 0")
print("FROZEN_SPLIT_DISJOINT = PASS")


# ============================================================
# 2. Load canonical dialogues
#
# IMPORTANT:
# Pair-level payload is retained only for TRAIN dialogues.
# DEV/TEST pair content is NOT used for overlap or model analysis here.
# ============================================================

print()
print("===== 2. CANONICAL COVERAGE =====")

canonical_ids = set()
train_dialogues = {}

canonical_rows = 0

with CANONICAL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        obj = json.loads(line)
        did = obj["dialogue_id"]

        canonical_rows += 1

        if did in canonical_ids:
            raise AssertionError(
                f"Duplicate canonical dialogue_id: {did}"
            )

        canonical_ids.add(did)

        if did in train_ids:
            train_dialogues[did] = obj

expected_all_ids = train_ids | dev_ids | test_ids

assert canonical_ids == expected_all_ids, (
    f"Canonical/split coverage mismatch: "
    f"canonical_only={sorted(canonical_ids - expected_all_ids)[:20]}, "
    f"split_only={sorted(expected_all_ids - canonical_ids)[:20]}"
)

assert_equal(canonical_rows, 2382, "canonical dialogue rows")
assert_equal(len(train_dialogues), 1666, "canonical TRAIN dialogues")

print("canonical_dialogues =", canonical_rows)
print("canonical_train_dialogues =", len(train_dialogues))
print("split_covers_all_canonical_dialogues = PASS")
print("canonical_covers_all_split_dialogues = PASS")
print("DEV_PAIR_PAYLOAD_ANALYZED = NO")
print("TEST_PAIR_PAYLOAD_ANALYZED = NO")


# ============================================================
# 3. Detect canonical pair fields
# ============================================================

print()
print("===== 3. CANONICAL PAIR FIELD DETECTION =====")

first_train_obj = next(iter(train_dialogues.values()))

assert isinstance(first_train_obj.get("pairs"), list)
assert len(first_train_obj["pairs"]) > 0

first_pair = first_train_obj["pairs"][0]

USER_CANDIDATES = [
    "user",
    "user_text",
    "current_user",
    "client",
    "human",
    "user_utterance",
    "question",
    "input"
]

RESPONSE_CANDIDATES = [
    "assistant",
    "assistant_text",
    "counselor",
    "counselor_response",
    "response",
    "answer",
    "therapist",
    "assistant_utterance"
]

user_field = detect_text_field(
    first_pair,
    USER_CANDIDATES,
    "user"
)

response_field = detect_text_field(
    first_pair,
    RESPONSE_CANDIDATES,
    "response"
)

print("canonical_pair_keys =", sorted(first_pair.keys()))
print("detected_user_field =", user_field)
print("detected_response_field =", response_field)

for did, d in train_dialogues.items():
    pairs = d["pairs"]

    for i, pair in enumerate(pairs):
        if user_field not in pair:
            raise AssertionError(
                f"Missing user field: dialogue={did}, pair={i}"
            )
        if response_field not in pair:
            raise AssertionError(
                f"Missing response field: dialogue={did}, pair={i}"
            )
        if not isinstance(pair[user_field], str):
            raise AssertionError(
                f"Non-string user: dialogue={did}, pair={i}"
            )
        if not isinstance(pair[response_field], str):
            raise AssertionError(
                f"Non-string response: dialogue={did}, pair={i}"
            )

print("CANONICAL_PAIR_SCHEMA = PASS")


# ============================================================
# 4. Supervision frozen counts / labels
# ============================================================

print()
print("===== 4. SUPERVISION FROZEN COUNTS =====")

assert_equal(len(obs_rows), 46091, "observability rows")
assert_equal(len(val_rows), 17458, "value rows")

obs_label_counts = Counter(
    r["observability_label"] for r in obs_rows
)

val_label_counts = Counter(
    r["value_label"] for r in val_rows
)

expected_obs = {0: 19705, 1: 26386}
expected_val = {-2: 13, -1: 1914, 0: 3041, 1: 8798, 2: 3692}

assert dict(obs_label_counts) == expected_obs, (
    f"Observability label mismatch: {obs_label_counts}"
)

assert dict(val_label_counts) == expected_val, (
    f"Value label mismatch: {val_label_counts}"
)

assert manifest["stage1_observability_rows"] == 46091
assert manifest["stage2_value_rows"] == 17458
assert manifest["stage2_subset_of_stage1_positive"] is True
assert manifest["future_user_in_model_input"] is False
assert manifest["next_user_written_to_output"] is False
assert manifest["dev_used"] is False
assert manifest["test_used"] is False

print("observability_rows =", len(obs_rows))
print("observability_label_counts =", dict(sorted(obs_label_counts.items())))
print("value_rows =", len(val_rows))
print("value_label_counts =", dict(sorted(val_label_counts.items())))
print("FROZEN_COUNTS_AND_LABELS = PASS")


# ============================================================
# 5. Supervision split leakage
# ============================================================

print()
print("===== 5. SUPERVISION SPLIT LEAKAGE =====")

obs_dialogues = {r["dialogue_id"] for r in obs_rows}
val_dialogues = {r["dialogue_id"] for r in val_rows}

obs_outside_train = obs_dialogues - train_ids
val_outside_train = val_dialogues - train_ids

assert not obs_outside_train, (
    f"Observability rows outside TRAIN: "
    f"{sorted(obs_outside_train)[:20]}"
)

assert not val_outside_train, (
    f"Value rows outside TRAIN: "
    f"{sorted(val_outside_train)[:20]}"
)

assert not (obs_dialogues & dev_ids)
assert not (obs_dialogues & test_ids)
assert not (val_dialogues & dev_ids)
assert not (val_dialogues & test_ids)

print("observability_dialogues =", len(obs_dialogues))
print("value_dialogues =", len(val_dialogues))
print("observability_DEV_dialogues = 0")
print("observability_TEST_dialogues = 0")
print("value_DEV_dialogues = 0")
print("value_TEST_dialogues = 0")
print("SUPERVISION_SPLIT_LEAKAGE = PASS")


# ============================================================
# 6. Forbidden future fields
# ============================================================

print()
print("===== 6. FUTURE FIELD LEAKAGE =====")

prohibited_fragments = [
    "next_user",
    "future_user",
    "next_utterance",
    "future_utterance"
]

all_supervision_keys = set()

for r in obs_rows + val_rows:
    all_supervision_keys.update(r.keys())

bad_fields = [
    k for k in all_supervision_keys
    if any(x in k.lower() for x in prohibited_fragments)
]

assert not bad_fields, (
    f"Forbidden future fields found: {bad_fields}"
)

print("all_supervision_fields =", sorted(all_supervision_keys))
print("forbidden_future_fields =", bad_fields)
print("FUTURE_FIELD_LEAKAGE = PASS")


# ============================================================
# 7. Identifier uniqueness
# ============================================================

print()
print("===== 7. IDENTIFIER UNIQUENESS =====")

for name, rows in [
    ("OBS", obs_rows),
    ("VALUE", val_rows),
]:
    for key in [
        "item_id",
        "sample_id",
        "source_row_index",
        "presentation_index"
    ]:
        vals = [r[key] for r in rows]
        dup = len(vals) - len(set(vals))

        print(
            f"{name}_{key}_rows={len(vals)} "
            f"unique={len(set(vals))} "
            f"duplicates={dup}"
        )

        assert dup == 0, (
            f"{name} duplicate {key}: {dup}"
        )

print("SUPERVISION_IDENTIFIER_UNIQUENESS = PASS")


# ============================================================
# 8. sample_id structural consistency
# ============================================================

print()
print("===== 8. SAMPLE ID CONSISTENCY =====")

bad_sample_ids = []

for r in obs_rows:
    expected = (
        f"psydial_{r['dialogue_id']}_"
        f"{r['target_pair_index']}"
    )

    if r["sample_id"] != expected:
        bad_sample_ids.append({
            "observed": r["sample_id"],
            "expected": expected,
            "item_id": r["item_id"]
        })

assert not bad_sample_ids, (
    f"sample_id mismatch count={len(bad_sample_ids)}, "
    f"examples={bad_sample_ids[:5]}"
)

print("sample_id_mismatches = 0")
print("SAMPLE_ID_CONSISTENCY = PASS")


# ============================================================
# 9. VALUE must be exact subset of OBSERVABILITY positive
# ============================================================

print()
print("===== 9. VALUE SUBSET OF OBS POSITIVE =====")

obs_by_item = {}

for r in obs_rows:
    if r["item_id"] in obs_by_item:
        raise AssertionError(
            f"Duplicate OBS item_id: {r['item_id']}"
        )
    obs_by_item[r["item_id"]] = r

metadata_fields = [
    "item_id",
    "presentation_index",
    "source_row_index",
    "dialogue_id",
    "sample_id",
    "target_pair_index",
    "current_user",
    "counselor_response"
]

missing_from_obs = []
not_positive = []
field_mismatches = []

for vr in val_rows:
    oid = vr["item_id"]

    if oid not in obs_by_item:
        missing_from_obs.append(oid)
        continue

    orow = obs_by_item[oid]

    if orow["observability_label"] != 1:
        not_positive.append(oid)

    for k in metadata_fields:
        if vr[k] != orow[k]:
            field_mismatches.append({
                "item_id": oid,
                "field": k
            })

assert not missing_from_obs, (
    f"VALUE rows absent from OBS: {len(missing_from_obs)}"
)

assert not not_positive, (
    f"VALUE rows not OBS-positive: {len(not_positive)}"
)

assert not field_mismatches, (
    f"VALUE/OBS mismatched fields: "
    f"{len(field_mismatches)}"
)

print("value_missing_from_observability = 0")
print("value_not_observability_positive = 0")
print("value_observability_field_mismatches = 0")
print("VALUE_SUBSET_OF_OBSERVABILITY_POSITIVE = PASS")


# ============================================================
# 10. Detect target_pair_index convention and trace supervision
#     exactly back to canonical TRAIN data
# ============================================================

print()
print("===== 10. CANONICAL TRACEABILITY =====")

def pair_matches(row, pair):
    return (
        row["current_user"] == pair[user_field]
        and
        row["counselor_response"] == pair[response_field]
    )


probe_rows = obs_rows[:min(5000, len(obs_rows))]

zero_based_matches = 0
one_based_matches = 0
zero_based_eligible = 0
one_based_eligible = 0

for r in probe_rows:
    pairs = train_dialogues[r["dialogue_id"]]["pairs"]
    idx = r["target_pair_index"]

    if 0 <= idx < len(pairs):
        zero_based_eligible += 1
        if pair_matches(r, pairs[idx]):
            zero_based_matches += 1

    idx1 = idx - 1
    if 0 <= idx1 < len(pairs):
        one_based_eligible += 1
        if pair_matches(r, pairs[idx1]):
            one_based_matches += 1

print(
    "probe_zero_based_matches =",
    f"{zero_based_matches}/{zero_based_eligible}"
)
print(
    "probe_one_based_matches =",
    f"{one_based_matches}/{one_based_eligible}"
)

zero_perfect = (
    zero_based_eligible == len(probe_rows)
    and zero_based_matches == len(probe_rows)
)

one_perfect = (
    one_based_eligible == len(probe_rows)
    and one_based_matches == len(probe_rows)
)

if zero_perfect and not one_perfect:
    index_base = 0
elif one_perfect and not zero_perfect:
    index_base = 1
else:
    raise AssertionError(
        "Could not uniquely determine target_pair_index convention. "
        f"zero={zero_based_matches}/{zero_based_eligible}, "
        f"one={one_based_matches}/{one_based_eligible}"
    )

print("detected_target_pair_index_base =", index_base)

trace_failures = []
target_out_of_range = 0

for r in obs_rows:
    pairs = train_dialogues[r["dialogue_id"]]["pairs"]

    idx = r["target_pair_index"] - index_base

    if not (0 <= idx < len(pairs)):
        target_out_of_range += 1
        if len(trace_failures) < 10:
            trace_failures.append({
                "item_id": r["item_id"],
                "reason": "target_index_out_of_range",
                "dialogue_id": r["dialogue_id"],
                "target_pair_index": r["target_pair_index"],
                "pairs": len(pairs)
            })
        continue

    pair = pairs[idx]

    if not pair_matches(r, pair):
        if len(trace_failures) < 10:
            trace_failures.append({
                "item_id": r["item_id"],
                "reason": "text_mismatch",
                "dialogue_id": r["dialogue_id"],
                "target_pair_index": r["target_pair_index"]
            })

assert target_out_of_range == 0, (
    f"Out-of-range target indices: {target_out_of_range}"
)

assert not trace_failures, (
    f"Canonical traceability failures: {trace_failures}"
)

print("observability_rows_traced =", len(obs_rows))
print("target_index_out_of_range = 0")
print("text_traceability_failures = 0")
print("CANONICAL_TRACEABILITY = PASS")


# ============================================================
# 11. TRAIN-only response shortcut diagnostics
# ============================================================

print()
print("===== 11. TRAIN RESPONSE DUPLICATION DIAGNOSTICS =====")

train_users = []
train_responses = []
train_pairs_exact = []
train_pairs_norm = []

train_pair_counts_by_dialogue = []

for did in sorted(train_ids):
    pairs = train_dialogues[did]["pairs"]

    train_pair_counts_by_dialogue.append(len(pairs))

    for pair in pairs:
        u = pair[user_field]
        r = pair[response_field]

        train_users.append(u)
        train_responses.append(r)

        train_pairs_exact.append((u, r))
        train_pairs_norm.append(
            (norm_text(u), norm_text(r))
        )

response_exact_stats = duplicate_stats(train_responses)

response_norm = [norm_text(x) for x in train_responses]
response_norm_stats = duplicate_stats(response_norm)

pair_exact_stats = duplicate_stats(train_pairs_exact)
pair_norm_stats = duplicate_stats(train_pairs_norm)

user_exact_stats = duplicate_stats(train_users)
user_norm_stats = duplicate_stats(
    [norm_text(x) for x in train_users]
)

print("TRAIN_total_pairs =", len(train_responses))

print()
print("response_exact =", response_exact_stats)
print("response_normalized =", response_norm_stats)

print()
print("query_response_pair_exact =", pair_exact_stats)
print("query_response_pair_normalized =", pair_norm_stats)

print()
print("current_user_exact =", user_exact_stats)
print("current_user_normalized =", user_norm_stats)

response_freq = Counter(response_norm)

top_response_freqs = sorted(
    response_freq.values(),
    reverse=True
)[:20]

print()
print("top20_normalized_response_frequencies =", top_response_freqs)

print("TRAIN_RESPONSE_DUPLICATION_DIAGNOSTIC = COMPLETE")


# ============================================================
# 12. Future-State coverage
# ============================================================

print()
print("===== 12. FUTURE-STATE COVERAGE =====")

obs_per_dialogue = Counter(
    r["dialogue_id"] for r in obs_rows
)

val_per_dialogue = Counter(
    r["dialogue_id"] for r in val_rows
)

obs_counts = list(obs_per_dialogue.values())
val_counts = list(val_per_dialogue.values())

def distribution_summary(values):
    if not values:
        return {}
    return {
        "n_dialogues": len(values),
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p75": percentile(values, 0.75),
        "max": max(values),
    }

obs_cov = distribution_summary(obs_counts)
val_cov = distribution_summary(val_counts)

print(
    "observability_dialogue_coverage = "
    f"{len(obs_per_dialogue)}/{len(train_ids)} "
    f"({len(obs_per_dialogue)/len(train_ids):.6f})"
)

print(
    "value_dialogue_coverage = "
    f"{len(val_per_dialogue)}/{len(train_ids)} "
    f"({len(val_per_dialogue)/len(train_ids):.6f})"
)

print("observability_rows_per_dialogue =", obs_cov)
print("value_rows_per_dialogue =", val_cov)

print("FUTURE_STATE_COVERAGE = COMPLETE")


# ============================================================
# 13. TRAIN-only candidate pool feasibility
# ============================================================

print()
print("===== 13. TRAIN CANDIDATE POOL FEASIBILITY =====")

train_unique_responses_exact = len(set(train_responses))
train_unique_responses_norm = len(set(response_norm))

assert train_unique_responses_exact > 1
assert train_unique_responses_norm > 1

print("train_response_rows =", len(train_responses))
print("train_unique_responses_exact =", train_unique_responses_exact)
print("train_unique_responses_normalized =", train_unique_responses_norm)
print("candidate_pool_source = TRAIN_ONLY")
print("DEV_RESPONSE_POOL_USED = NO")
print("TEST_RESPONSE_POOL_USED = NO")
print("TRAIN_CANDIDATE_POOL_FEASIBILITY = PASS")


# ============================================================
# 14. Build machine-readable audit report
# ============================================================

report = {
    "artifact": "FSAR Phase S1-B TRAIN Integrity and Leakage Audit V1",
    "status": "PASS",

    "scope": {
        "train_payload_audited": True,
        "dev_pair_payload_analyzed": False,
        "test_pair_payload_analyzed": False,
        "dev_performance_used": False,
        "test_performance_used": False,
    },

    "split": {
        "seed": split["seed"],
        "train_dialogues": len(train_ids),
        "dev_dialogues": len(dev_ids),
        "test_dialogues": len(test_ids),
        "train_dev_dialogue_overlap": 0,
        "train_test_dialogue_overlap": 0,
        "dev_test_dialogue_overlap": 0,
        "canonical_dialogues": len(canonical_ids),
    },

    "canonical_schema": {
        "user_field": user_field,
        "response_field": response_field,
        "target_pair_index_base": index_base,
    },

    "supervision": {
        "observability_rows": len(obs_rows),
        "observability_label_counts": counter_to_json(obs_label_counts),
        "value_rows": len(val_rows),
        "value_label_counts": counter_to_json(val_label_counts),
        "observability_dialogues": len(obs_dialogues),
        "value_dialogues": len(val_dialogues),
        "future_field_leakage": False,
        "value_subset_of_observability_positive": True,
        "canonical_traceability_pass": True,
    },

    "train_response_diagnostics": {
        "response_exact": response_exact_stats,
        "response_normalized": response_norm_stats,
        "query_response_pair_exact": pair_exact_stats,
        "query_response_pair_normalized": pair_norm_stats,
        "current_user_exact": user_exact_stats,
        "current_user_normalized": user_norm_stats,
        "top20_normalized_response_frequencies": top_response_freqs,
    },

    "future_state_coverage": {
        "observability": obs_cov,
        "value": val_cov,
        "observability_dialogue_fraction":
            len(obs_per_dialogue) / len(train_ids),
        "value_dialogue_fraction":
            len(val_per_dialogue) / len(train_ids),
    },

    "candidate_pool": {
        "source": "TRAIN_ONLY",
        "response_rows": len(train_responses),
        "unique_responses_exact": train_unique_responses_exact,
        "unique_responses_normalized": train_unique_responses_norm,
        "dev_response_pool_used": False,
        "test_response_pool_used": False,
    },

    "gate": {
        "structural_integrity": "PASS",
        "split_leakage": "PASS",
        "future_field_leakage": "PASS",
        "supervision_provenance": "PASS",
        "candidate_pool_feasibility": "PASS",
        "response_duplicate_rate_is_diagnostic_not_a_posthoc_gate": True,
    }
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

OUT_JSON.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("===== FINAL S1-B GATE =====")
print("STRUCTURAL_INTEGRITY = PASS")
print("DIALOGUE_SPLIT_LEAKAGE = PASS")
print("FUTURE_FIELD_LEAKAGE = PASS")
print("SUPERVISION_PROVENANCE = PASS")
print("TRAIN_CANDIDATE_POOL_FEASIBILITY = PASS")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")
print()
print("PHASE_S1B_TRAIN_INTEGRITY_AUDIT = PASS")
print("REPORT =", OUT_JSON)
