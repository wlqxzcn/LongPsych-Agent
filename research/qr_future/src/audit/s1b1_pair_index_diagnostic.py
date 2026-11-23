import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]

CANONICAL = ROOT / "data/canonical/psydial_dialogues.jsonl"
SPLIT = ROOT / "data/splits/psydial_dialogue_split.json"
OBS = ROOT / "data/supervision/psydial_phase5b0_observability_train_v1.jsonl"

OUT = ROOT / "results/audit/s1b1_pair_index_diagnostic_v1.json"


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


print("=" * 70)
print("FSAR S1-B1 PAIR INDEX DIAGNOSTIC")
print("=" * 70)

with SPLIT.open("r", encoding="utf-8") as f:
    split = json.load(f)

train_ids = set(split["train_dialogues"])

canonical = {}

with CANONICAL.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        did = obj["dialogue_id"]
        if did in train_ids:
            canonical[did] = obj

obs = load_jsonl(OBS)

print("train_dialogues =", len(canonical))
print("observability_rows =", len(obs))


# ============================================================
# 1. Inspect canonical pair_index itself
# ============================================================

print()
print("===== 1. CANONICAL pair_index STRUCTURE =====")

missing_pair_index = 0
duplicate_pair_index_dialogues = []
non_integer_pair_index = []
position_not_equal_pair_index = 0
position_not_equal_examples = []

all_pair_indices = []

for did, d in canonical.items():
    seen = set()

    for pos, pair in enumerate(d["pairs"]):
        if "pair_index" not in pair:
            missing_pair_index += 1
            continue

        pi = pair["pair_index"]
        all_pair_indices.append(pi)

        if not isinstance(pi, int):
            if len(non_integer_pair_index) < 10:
                non_integer_pair_index.append({
                    "dialogue_id": did,
                    "position": pos,
                    "pair_index": repr(pi)
                })

        if pi in seen:
            duplicate_pair_index_dialogues.append(did)
        seen.add(pi)

        if pi != pos:
            position_not_equal_pair_index += 1
            if len(position_not_equal_examples) < 20:
                position_not_equal_examples.append({
                    "dialogue_id": did,
                    "list_position": pos,
                    "pair_index": pi,
                    "total_pairs": len(d["pairs"])
                })

print("missing_pair_index =", missing_pair_index)
print("non_integer_pair_index_examples =", non_integer_pair_index)
print(
    "duplicate_pair_index_dialogues =",
    len(set(duplicate_pair_index_dialogues))
)
print(
    "pairs_where_list_position_differs_from_pair_index =",
    position_not_equal_pair_index
)

print("examples:")
for x in position_not_equal_examples:
    print(x)

if all_pair_indices:
    print("pair_index_min =", min(all_pair_indices))
    print("pair_index_max =", max(all_pair_indices))


# ============================================================
# 2. Build exact pair_index mapping
# ============================================================

print()
print("===== 2. BUILD pair_index MAP =====")

pair_maps = {}

for did, d in canonical.items():
    m = {}

    for pos, pair in enumerate(d["pairs"]):
        pi = pair["pair_index"]

        if pi in m:
            raise AssertionError(
                f"Duplicate pair_index in dialogue {did}: {pi}"
            )

        m[pi] = {
            "position": pos,
            "pair": pair
        }

    pair_maps[did] = m

print("PAIR_INDEX_MAP_BUILD = PASS")


# ============================================================
# 3. Compare three lookup strategies over ALL 46,091 rows
# ============================================================

print()
print("===== 3. FULL OBSERVABILITY TRACE COMPARISON =====")

stats = Counter()
examples = {
    "direct_position_fail_pairindex_success": [],
    "pairindex_fail": [],
    "text_mismatch_after_pairindex_lookup": [],
}

for r in obs:
    did = r["dialogue_id"]
    target = r["target_pair_index"]
    pairs = canonical[did]["pairs"]

    # A: target as direct Python list position
    if 0 <= target < len(pairs):
        p = pairs[target]
        if (
            p["user"] == r["current_user"]
            and
            p["assistant"] == r["counselor_response"]
        ):
            stats["direct_position_exact_match"] += 1
        else:
            stats["direct_position_text_mismatch"] += 1
    else:
        stats["direct_position_out_of_range"] += 1

    # B: target-1 as one-based list position
    pos1 = target - 1
    if 0 <= pos1 < len(pairs):
        p = pairs[pos1]
        if (
            p["user"] == r["current_user"]
            and
            p["assistant"] == r["counselor_response"]
        ):
            stats["one_based_position_exact_match"] += 1
        else:
            stats["one_based_position_text_mismatch"] += 1
    else:
        stats["one_based_position_out_of_range"] += 1

    # C: canonical pair["pair_index"] exact lookup
    lookup = pair_maps[did].get(target)

    if lookup is None:
        stats["pair_index_lookup_missing"] += 1

        if len(examples["pairindex_fail"]) < 20:
            examples["pairindex_fail"].append({
                "item_id": r["item_id"],
                "dialogue_id": did,
                "target_pair_index": target,
                "available_pair_indices": sorted(pair_maps[did].keys())[:100]
            })
        continue

    pair = lookup["pair"]

    if (
        pair["user"] == r["current_user"]
        and
        pair["assistant"] == r["counselor_response"]
    ):
        stats["pair_index_exact_match"] += 1

        # capture cases where list-position method failed
        direct_ok = False
        if 0 <= target < len(pairs):
            direct_pair = pairs[target]
            direct_ok = (
                direct_pair["user"] == r["current_user"]
                and
                direct_pair["assistant"] == r["counselor_response"]
            )

        if not direct_ok:
            if len(
                examples["direct_position_fail_pairindex_success"]
            ) < 30:
                examples[
                    "direct_position_fail_pairindex_success"
                ].append({
                    "item_id": r["item_id"],
                    "dialogue_id": did,
                    "target_pair_index": target,
                    "actual_list_position": lookup["position"],
                    "dialogue_total_pairs": len(pairs)
                })

    else:
        stats["pair_index_text_mismatch"] += 1

        if len(
            examples["text_mismatch_after_pairindex_lookup"]
        ) < 20:
            examples[
                "text_mismatch_after_pairindex_lookup"
            ].append({
                "item_id": r["item_id"],
                "dialogue_id": did,
                "target_pair_index": target,
                "actual_list_position": lookup["position"],
                "user_equal": (
                    pair["user"] == r["current_user"]
                ),
                "response_equal": (
                    pair["assistant"] == r["counselor_response"]
                )
            })


for k in sorted(stats):
    print(f"{k} = {stats[k]}")

print()
print(
    "direct_position_fail_pairindex_success_examples ="
)
for x in examples["direct_position_fail_pairindex_success"]:
    print(x)

print()
print("pair_index_lookup_failure_examples =")
for x in examples["pairindex_fail"]:
    print(x)

print()
print("pair_index_text_mismatch_examples =")
for x in examples["text_mismatch_after_pairindex_lookup"]:
    print(x)


# ============================================================
# 4. Gate
# ============================================================

n = len(obs)

pair_index_perfect = (
    stats["pair_index_exact_match"] == n
    and stats["pair_index_lookup_missing"] == 0
    and stats["pair_index_text_mismatch"] == 0
)

report = {
    "artifact": "FSAR S1-B1 Pair Index Diagnostic V1",
    "observability_rows": n,
    "canonical_train_dialogues": len(canonical),
    "canonical_pair_index": {
        "missing_pair_index": missing_pair_index,
        "duplicate_pair_index_dialogues":
            len(set(duplicate_pair_index_dialogues)),
        "non_integer_pair_index_examples":
            non_integer_pair_index,
        "position_not_equal_pair_index":
            position_not_equal_pair_index,
        "examples":
            position_not_equal_examples,
    },
    "trace_stats": dict(stats),
    "examples": examples,
    "pair_index_exact_trace_perfect": pair_index_perfect
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
print("===== S1-B1 GATE =====")

if pair_index_perfect:
    print("PAIR_INDEX_EXACT_TRACE = PASS")
    print("CORRECT_CANONICAL_LOOKUP = pair['pair_index']")
    print("S1B_CAN_BE_REPAIRED = YES")
    print("PHASE_S1B1_PAIR_INDEX_DIAGNOSTIC = PASS")
else:
    print("PAIR_INDEX_EXACT_TRACE = FAIL")
    print("S1B_CAN_BE_REPAIRED = NO")
    print("PHASE_S1B1_PAIR_INDEX_DIAGNOSTIC = STOP")
    raise SystemExit(2)

print("REPORT =", OUT)
