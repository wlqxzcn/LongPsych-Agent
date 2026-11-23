import json
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

A4 = PHASE / (
    "step_2f18ak_u3_d14_a4_final_evidence_synthesis_v1"
)

PROTOCOL = A4 / "protocol_v1.json"

PREP_SCRIPT = ROOT / (
    "src/evaluation/"
    "s12c_step_2f18ak_u3_d14_a4_final_evidence_synthesis.py"
)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schema(obj, depth=0):
    if depth >= 3:
        return type(obj).__name__

    if isinstance(obj, dict):
        return {
            k: schema(v, depth + 1)
            for k, v in list(obj.items())[:35]
        }

    if isinstance(obj, list):
        return {
            "type": "list",
            "length": len(obj),
            "first_item": (
                schema(obj[0], depth + 1)
                if obj else None
            )
        }

    return type(obj).__name__


def main():
    p = json.loads(PROTOCOL.read_text())

    print("========== A4 PROTOCOL ==========")

    assert p["status"] == (
        "FROZEN_BEFORE_FINAL_EVIDENCE_SYNTHESIS"
    )

    assert sha(PREP_SCRIPT) == (
        p["evaluator_script_sha256"]
    )

    print("PROTOCOL_STATUS_PASS")
    print("PROTOCOL_SHA256:", sha(PROTOCOL))
    print("SOURCE_COUNT:", p["source_file_count"])

    total = 0

    for group, files in p["source_group_files"].items():

        print("\n" + "=" * 75)
        print("GROUP:", group)
        print("=" * 75)

        for relative in files:

            path = ROOT / relative
            expected = p["source_sha256"][relative]

            if not path.is_file():
                raise RuntimeError("MISSING: " + relative)

            actual = sha(path)

            if actual != expected:
                raise RuntimeError(
                    "SOURCE_HASH_MISMATCH: " + relative
                )

            total += 1
            suffix = path.suffix

            print("\nFILE:", path.name)
            print("SHA256: PASS")

            if suffix == ".json":

                obj = json.loads(
                    path.read_text(encoding="utf-8")
                )

                print("SCHEMA:")
                print(json.dumps(
                    schema(obj),
                    ensure_ascii=False,
                    indent=2
                ))

                if path.name.startswith("run_manifest"):
                    print("MANIFEST_STATUS:", obj.get("status"))
                    print("MANIFEST_KEYS:", list(obj.keys()))

            elif suffix == ".jsonl":

                count = 0
                first = None

                with path.open(
                    encoding="utf-8"
                ) as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if first is None:
                                first = item
                            count += 1

                print("JSONL_RECORDS:", count)
                print(
                    "FIRST_RECORD_KEYS:",
                    list(first.keys()) if first else None
                )

            elif suffix == ".csv":

                with path.open(
                    encoding="utf-8-sig",
                    newline=""
                ) as f:
                    reader = csv.DictReader(f)
                    rows = sum(1 for _ in reader)

                print("CSV_COLUMNS:", reader.fieldnames)
                print("CSV_ROWS:", rows)

    if total != 48:
        raise RuntimeError(
            f"SOURCE_COUNT_MISMATCH: {total}"
        )

    print("\n" + "=" * 75)
    print("U3_D14_A4_SOURCE_SCHEMA_AUDIT_PASS")
    print("TOTAL_SOURCE_FILES:", total)
    print("SOURCE_HASH_VERIFICATION: PASS")
    print("HISTORICAL_FILES_MODIFIED: 0")
    print("=" * 75)


if __name__ == "__main__":
    main()
