import json
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict

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

OUT_JSON = ROOT / (
    "results/audit/"
    "s1b2_trace_mismatch_diagnostic_v1.json"
)

OUT_JSONL = ROOT / (
    "results/audit/"
    "s1b2_trace_mismatch_rows_v1.jsonl"
)


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_basic(x):
    x = unicodedata.normalize("NFKC", x)
    x = re.sub(r"\s+", " ", x.strip())
    return x


def find_positions_exact(pairs, field, text):
    return [
        i for i, p in enumerate(pairs)
        if p[field] == text
    ]


def find_positions_norm(pairs, field, text):
    target = normalize_basic(text)
    return [
        i for i, p in enumerate(pairs)
        if normalize_basic(p[field]) == target
    ]


print("=" * 72)
print("FSAR S1-B2 TRACE MISMATCH PROVENANCE DIAGNOSTIC")
print("=" * 72)

with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

train_ids = set(split["train_dialogues"])

canonical = {}

with CANONICAL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        obj = json.loads(line)

        if obj["dialogue_id"] in train_ids:
            canonical[obj["dialogue_id"]] = obj

obs = load_jsonl(OBS)
val = load_jsonl(VAL)

value_item_ids = {
    r["item_id"] for r in val
}

print("TRAIN dialogues =", len(canonical))
print("OBS rows =", len(obs))
print("VALUE rows =", len(val))


# ============================================================
# 1. Exact mismatch classification
# ============================================================

print()
print("===== 1. MISMATCH CLASSIFICATION =====")

mismatch_rows = []

kind_counts = Counter()
normalized_resolution = Counter()

dialogue_counts = Counter()

for r in obs:
    did = r["dialogue_id"]
    idx = r["target_pair_index"]

    pair = canonical[did]["pairs"][idx]

    user_equal = (
        r["current_user"] == pair["user"]
    )

    response_equal = (
        r["counselor_response"] == pair["assistant"]
    )

    if user_equal and response_equal:
        continue

    if (not user_equal) and response_equal:
        kind = "USER_ONLY_MISMATCH"
    elif user_equal and (not response_equal):
        kind = "RESPONSE_ONLY_MISMATCH"
    else:
        kind = "BOTH_MISMATCH"

    kind_counts[kind] += 1
    dialogue_counts[did] += 1

    user_norm_equal = (
        normalize_basic(r["current_user"])
        ==
        normalize_basic(pair["user"])
    )

    response_norm_equal = (
        normalize_basic(r["counselor_response"])
        ==
        normalize_basic(pair["assistant"])
    )

    if user_norm_equal and response_norm_equal:
        normalized_resolution[
            "FULLY_RESOLVED_BY_BASIC_NORMALIZATION"
        ] += 1
    elif user_norm_equal or response_norm_equal:
        normalized_resolution[
            "PARTIALLY_RESOLVED_BY_BASIC_NORMALIZATION"
        ] += 1
    else:
        normalized_resolution[
            "NOT_RESOLVED_BY_BASIC_NORMALIZATION"
        ] += 1

    pairs = canonical[did]["pairs"]

    user_exact_positions = find_positions_exact(
        pairs,
        "user",
        r["current_user"]
    )

    response_exact_positions = find_positions_exact(
        pairs,
        "assistant",
        r["counselor_response"]
    )

    user_norm_positions = find_positions_norm(
        pairs,
        "user",
        r["current_user"]
    )

    response_norm_positions = find_positions_norm(
        pairs,
        "assistant",
        r["counselor_response"]
    )

    same_pair_exact_positions = [
        i for i, p in enumerate(pairs)
        if (
            p["user"] == r["current_user"]
            and
            p["assistant"] == r["counselor_response"]
        )
    ]

    same_pair_norm_positions = [
        i for i, p in enumerate(pairs)
        if (
            normalize_basic(p["user"])
            ==
            normalize_basic(r["current_user"])
            and
            normalize_basic(p["assistant"])
            ==
            normalize_basic(r["counselor_response"])
        )
    ]

    mismatch_rows.append({
        "item_id": r["item_id"],
        "sample_id": r["sample_id"],
        "dialogue_id": did,
        "target_pair_index": idx,
        "source_row_index": r["source_row_index"],

        "mismatch_kind": kind,

        "in_value_dataset":
            r["item_id"] in value_item_ids,

        "user_exact_equal": user_equal,
        "response_exact_equal": response_equal,

        "user_norm_equal": user_norm_equal,
        "response_norm_equal": response_norm_equal,

        "supervision_user_len":
            len(r["current_user"]),
        "canonical_user_len":
            len(pair["user"]),

        "supervision_response_len":
            len(r["counselor_response"]),
        "canonical_response_len":
            len(pair["assistant"]),

        "user_exact_positions_in_same_dialogue":
            user_exact_positions,

        "response_exact_positions_in_same_dialogue":
            response_exact_positions,

        "user_norm_positions_in_same_dialogue":
            user_norm_positions,

        "response_norm_positions_in_same_dialogue":
            response_norm_positions,

        "full_pair_exact_positions_in_same_dialogue":
            same_pair_exact_positions,

        "full_pair_norm_positions_in_same_dialogue":
            same_pair_norm_positions,
    })


print("total_mismatches =", len(mismatch_rows))
print("mismatch_kind_counts =", dict(kind_counts))

print(
    "normalization_resolution =",
    dict(normalized_resolution)
)

print(
    "affected_dialogues =",
    len(dialogue_counts)
)

print(
    "max_mismatches_in_one_dialogue =",
    max(dialogue_counts.values())
    if dialogue_counts else 0
)


# ============================================================
# 2. Value supervision exposure
# ============================================================

print()
print("===== 2. VALUE SUPERVISION EXPOSURE =====")

value_mismatch_rows = [
    r for r in mismatch_rows
    if r["in_value_dataset"]
]

value_kind_counts = Counter(
    r["mismatch_kind"]
    for r in value_mismatch_rows
)

print(
    "mismatches_in_value_dataset =",
    len(value_mismatch_rows)
)

print(
    "value_mismatch_kind_counts =",
    dict(value_kind_counts)
)

print(
    "value_mismatch_rate =",
    (
        len(value_mismatch_rows)
        / len(val)
        if val else None
    )
)


# ============================================================
# 3. Same-dialogue relocation analysis
# ============================================================

print()
print("===== 3. SAME-DIALOGUE RELOCATION ANALYSIS =====")

relocation = Counter()

distance_counts = Counter()

for r in mismatch_rows:
    target = r["target_pair_index"]

    exact_pair_positions = (
        r[
            "full_pair_exact_positions_in_same_dialogue"
        ]
    )

    norm_pair_positions = (
        r[
            "full_pair_norm_positions_in_same_dialogue"
        ]
    )

    if exact_pair_positions:
        relocation[
            "FULL_PAIR_FOUND_EXACT_ELSEWHERE"
        ] += 1

        for p in exact_pair_positions:
            distance_counts[p - target] += 1

    elif norm_pair_positions:
        relocation[
            "FULL_PAIR_FOUND_NORMALIZED_ELSEWHERE"
        ] += 1

        for p in norm_pair_positions:
            distance_counts[p - target] += 1

    else:
        user_positions = set(
            r["user_exact_positions_in_same_dialogue"]
        )

        response_positions = set(
            r[
                "response_exact_positions_in_same_dialogue"
            ]
        )

        if user_positions and response_positions:
            relocation[
                "USER_AND_RESPONSE_FOUND_SEPARATELY"
            ] += 1

        elif user_positions:
            relocation[
                "ONLY_USER_FOUND_ELSEWHERE"
            ] += 1

        elif response_positions:
            relocation[
                "ONLY_RESPONSE_FOUND_ELSEWHERE"
            ] += 1

        else:
            relocation[
                "NEITHER_FOUND_ELSEWHERE"
            ] += 1


print("relocation_counts =", dict(relocation))

print(
    "relative_position_differences =",
    dict(sorted(distance_counts.items()))
)


# ============================================================
# 4. Adjacent-position diagnostic
# ============================================================

print()
print("===== 4. ADJACENT POSITION DIAGNOSTIC =====")

adjacent = Counter()

for r in mismatch_rows:
    target = r["target_pair_index"]

    positions = (
        r[
            "full_pair_exact_positions_in_same_dialogue"
        ]
    )

    if target - 1 in positions:
        adjacent["FOUND_AT_TARGET_MINUS_1"] += 1

    if target + 1 in positions:
        adjacent["FOUND_AT_TARGET_PLUS_1"] += 1

print("adjacent_counts =", dict(adjacent))


# ============================================================
# 5. Write mismatch metadata only
#    Do not dump full dialogue texts into report.
# ============================================================

OUT_JSONL.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT_JSONL.open(
    "w",
    encoding="utf-8"
) as f:
    for r in mismatch_rows:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# 6. Summary report
# ============================================================

report = {
    "artifact":
        "FSAR S1-B2 Trace Mismatch Diagnostic V1",

    "status":
        "DIAGNOSTIC_COMPLETE",

    "observability_rows":
        len(obs),

    "total_mismatches":
        len(mismatch_rows),

    "mismatch_rate":
        len(mismatch_rows) / len(obs),

    "mismatch_kind_counts":
        dict(kind_counts),

    "normalization_resolution":
        dict(normalized_resolution),

    "affected_dialogues":
        len(dialogue_counts),

    "max_mismatches_in_one_dialogue":
        (
            max(dialogue_counts.values())
            if dialogue_counts else 0
        ),

    "value_dataset": {
        "value_rows":
            len(val),

        "mismatches_in_value_dataset":
            len(value_mismatch_rows),

        "mismatch_rate":
            (
                len(value_mismatch_rows)
                / len(val)
                if val else None
            ),

        "mismatch_kind_counts":
            dict(value_kind_counts),
    },

    "relocation_counts":
        dict(relocation),

    "relative_position_differences":
        {
            str(k): v
            for k, v
            in sorted(distance_counts.items())
        },

    "adjacent_counts":
        dict(adjacent),

    "decision":
        "DO_NOT_REPAIR_OR_DELETE_YET"
}

OUT_JSON.write_text(
    json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    ),
    encoding="utf-8"
)

print()
print("===== S1-B2 STATUS =====")
print("DIAGNOSTIC_ONLY = YES")
print("ROWS_MODIFIED = 0")
print("ROWS_DELETED = 0")
print("DEV_PAYLOAD_USED = NO")
print("TEST_PAYLOAD_USED = NO")
print("PHASE_S1B2_TRACE_MISMATCH_DIAGNOSTIC = COMPLETE")

print("SUMMARY =", OUT_JSON)
print("ROWS =", OUT_JSONL)
