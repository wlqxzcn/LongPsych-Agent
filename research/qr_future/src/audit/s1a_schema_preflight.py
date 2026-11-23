import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]

FILES = {
    "canonical": ROOT / "data/canonical/psydial_dialogues.jsonl",
    "observability": ROOT / "data/supervision/psydial_phase5b0_observability_train_v1.jsonl",
    "value": ROOT / "data/supervision/psydial_phase5b0_value_train_v1.jsonl",
    "split": ROOT / "data/splits/psydial_dialogue_split.json",
    "manifest": ROOT / "data/supervision/psydial_phase5b0_supervised_dataset_manifest_v1.json",
}

def count_jsonl(path):
    n = 0
    first = None
    second = None
    key_counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            if first is None:
                first = obj
            elif second is None:
                second = obj

            if isinstance(obj, dict):
                key_counter.update(obj.keys())

            n += 1

    return n, first, second, key_counter


def describe_value(v, depth=0):
    if depth >= 2:
        return type(v).__name__

    if isinstance(v, dict):
        return {
            str(k): describe_value(val, depth + 1)
            for k, val in list(v.items())[:20]
        }

    if isinstance(v, list):
        if not v:
            return "list[empty]"
        return {
            "type": "list",
            "length": len(v),
            "first_item_type": type(v[0]).__name__,
            "first_item": describe_value(v[0], depth + 1),
        }

    if isinstance(v, str):
        return f"str(len={len(v)})"

    return type(v).__name__


def print_record_schema(name, obj):
    print()
    print(f"===== {name} RECORD SCHEMA =====")

    if obj is None:
        print("EMPTY")
        return

    print("record_type =", type(obj).__name__)

    if isinstance(obj, dict):
        print("keys =", sorted(obj.keys()))

        for k in sorted(obj.keys()):
            v = obj[k]

            if isinstance(v, str):
                # 只展示标识符类字段的值，文本内容不打印
                kl = k.lower()
                if any(x in kl for x in [
                    "id", "index", "split", "label",
                    "target", "pair", "source"
                ]) and len(v) <= 120:
                    print(f"{k}: str value={v!r}")
                else:
                    print(f"{k}: str len={len(v)}")

            elif isinstance(v, (int, float, bool)) or v is None:
                print(f"{k}: {type(v).__name__} value={v!r}")

            elif isinstance(v, list):
                print(
                    f"{k}: list len={len(v)} "
                    f"first_type="
                    f"{type(v[0]).__name__ if v else 'EMPTY'}"
                )

            elif isinstance(v, dict):
                print(
                    f"{k}: dict keys="
                    f"{list(v.keys())[:20]}"
                )

            else:
                print(f"{k}: {type(v).__name__}")


print("============================================================")
print("FSAR PHASE S1-A SCHEMA PREFLIGHT")
print("============================================================")

print()
print("===== FILE EXISTENCE =====")
for name, path in FILES.items():
    assert path.exists(), f"MISSING: {path}"
    print(f"{name}: PASS | {path}")

# ------------------------------------------------------------
# JSONL files
# ------------------------------------------------------------

for name in ["canonical", "observability", "value"]:
    path = FILES[name]

    n, first, second, key_counter = count_jsonl(path)

    print()
    print(f"===== {name.upper()} =====")
    print("rows =", n)
    print("all_record_keys =", sorted(key_counter.keys()))

    print_record_schema(
        f"{name.upper()} FIRST",
        first
    )

    if second is not None:
        print_record_schema(
            f"{name.upper()} SECOND",
            second
        )

# ------------------------------------------------------------
# Split
# ------------------------------------------------------------

print()
print("===== SPLIT FILE =====")

with FILES["split"].open("r", encoding="utf-8") as f:
    split_obj = json.load(f)

print("root_type =", type(split_obj).__name__)

if isinstance(split_obj, dict):
    print("root_keys =", list(split_obj.keys()))

    for k, v in split_obj.items():
        print()
        print(f"[split field] {k}")
        print("type =", type(v).__name__)

        if isinstance(v, list):
            print("length =", len(v))
            if v:
                print("first_item_type =", type(v[0]).__name__)
                if isinstance(v[0], str):
                    print("first_item =", v[0])
                elif isinstance(v[0], dict):
                    print("first_item_keys =", list(v[0].keys()))

        elif isinstance(v, dict):
            print("keys =", list(v.keys())[:30])

            for kk, vv in list(v.items())[:10]:
                if isinstance(vv, list):
                    print(
                        f"  {kk}: list length={len(vv)} "
                        f"first_type="
                        f"{type(vv[0]).__name__ if vv else 'EMPTY'}"
                    )
                    if vv and isinstance(vv[0], str):
                        print(f"    first_item={vv[0]}")
                else:
                    print(
                        f"  {kk}: "
                        f"{type(vv).__name__}"
                    )

        elif isinstance(v, (str, int, float, bool)) or v is None:
            print("value =", v)

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

print()
print("===== SUPERVISION MANIFEST =====")

with FILES["manifest"].open("r", encoding="utf-8") as f:
    manifest = json.load(f)

print(json.dumps(
    manifest,
    ensure_ascii=False,
    indent=2
))

print()
print("PHASE_S1A_SCHEMA_PREFLIGHT = PASS")
