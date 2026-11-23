import json
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"
A4 = PHASE / "step_2f18ak_u3_d14_a4_final_evidence_synthesis_v1"

PROTOCOL = A4 / "protocol_v1.json"
EXPECTED_PROTOCOL = (
    "5fe914d1aab0a7e1c1efee25c564c4336b3526743e7fa9df677887f2b7e93a20"
)

GROUPS = {
    "CONSOLIDATION": "step_2f18aa_final_results_consolidation_v1",
    "D10": "step_2f18ab_u3_d10_case_mechanism_audit_v1",
    "D11": "step_2f18ac_u3_d11_paired_interaction_audit_v2",
    "D12": "step_2f18ad_u3_d12_product_interaction_mechanism_v1",
    "D13": "step_2f18af_u3_d13_final_evidence_consistency_audit_v2",
    "D14_A0": "step_2f18ag_u3_d14_a0_index_rank_audit_v1",
    "D14_A1": "step_2f18ah_u3_d14_a1_gold_negative_margin_v1",
    "D14_A2": "step_2f18ai_u3_d14_a2_score_decomposition_v2",
    "D14_A3": "step_2f18aj_u3_d14_a3_component_interaction_v1",
}

REPORTS = {
    "CONSOLIDATION": "final_results_report_v1.txt",
    "D10": "final_report_v1.txt",
    "D11": "interaction_report_v2.txt",
    "D12": "d12_report_v1.txt",
    "D13": "audit_report_v1.txt",
    "D14_A0": "audit_report_v1.txt",
    "D14_A1": "margin_report_v1.txt",
    "D14_A2": "decomposition_report_v1.txt",
    "D14_A3": "a3_report_v1.txt",
}

JSONS = {
    "CONSOLIDATION": "final_results_v1.json",
    "D10": "case_statistics_v1.json",
    "D11": "interaction_results_v2.json",
    "D12": "d12_results_v1.json",
    "D13": "audit_results_v1.json",
    "D14_A0": "audit_results_v1.json",
    "D14_A1": "margin_results_v1.json",
    "D14_A2": "decomposition_results_v1.json",
    "D14_A3": "a3_results_v1.json",
}

CSVS = {
    "CONSOLIDATION": [
        "model_performance_v1.csv",
        "paired_contrasts_v1.csv",
    ],
    "D10": ["case_statistics_v1.csv"],
    "D11": ["interaction_results_v2.csv"],
    "D12": ["d12_results_v1.csv"],
    "D14_A1": [
        "margin_descriptive_v1.csv",
        "margin_contrasts_v1.csv",
    ],
    "D14_A2": ["decomposition_results_v1.csv"],
    "D14_A3": ["a3_results_v1.csv"],
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def preview(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list):
                out[k] = {
                    "type": "list",
                    "length": len(v),
                    "records": v[:3],
                }
            elif isinstance(v, dict):
                out[k] = {
                    "keys": list(v.keys()),
                    "values": v,
                }
            else:
                out[k] = v
        return out
    return obj


def main():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    assert sha(PROTOCOL) == EXPECTED_PROTOCOL
    assert p["status"] == (
        "FROZEN_BEFORE_FINAL_EVIDENCE_SYNTHESIS"
    )

    print("A4_PROTOCOL_PASS")

    checked = 0

    for group, files in p["source_group_files"].items():
        for relative in files:
            path = ROOT / relative
            assert path.is_file(), relative
            assert sha(path) == p["source_sha256"][relative], relative
            checked += 1

    assert checked == 48
    print("SOURCE_HASH_PASS:", checked)

    print("\n========== JSON EVIDENCE ==========")

    for group, filename in JSONS.items():
        path = PHASE / GROUPS[group] / filename
        obj = json.loads(path.read_text(encoding="utf-8"))

        print("\nGROUP:", group)
        print(json.dumps(
            preview(obj),
            ensure_ascii=False,
            indent=2
        ))

        if group == "D10":
            extra = PHASE / GROUPS[group] / (
                "long_vs_nonlong_statistics_v1.json"
            )
            print("\nD10 LONG VS NONLONG")
            print(extra.read_text(encoding="utf-8"))

    print("\n========== CSV EVIDENCE ==========")

    for group, filenames in CSVS.items():
        for filename in filenames:
            path = PHASE / GROUPS[group] / filename

            with path.open(
                encoding="utf-8-sig",
                newline=""
            ) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            print("\nGROUP:", group)
            print("FILE:", filename)
            print("ROWS:", len(rows))
            print("COLUMNS:", reader.fieldnames)
            print("RECORDS:")
            print(json.dumps(
                rows if len(rows) <= 20 else rows[:5],
                ensure_ascii=False,
                indent=2
            ))

    print("\n========== FINAL REPORTS ==========")

    for group, filename in REPORTS.items():
        path = PHASE / GROUPS[group] / filename

        print("\n" + "=" * 75)
        print("REPORT:", group)
        print("=" * 75)
        print(path.read_text(encoding="utf-8"))

    print("\n" + "=" * 75)
    print("U3_D14_A4_S2_EVIDENCE_INSPECTION_COMPLETE")
    print("SOURCE_FILES:", checked)
    print("HISTORICAL_FILES_MODIFIED: 0")
    print("TRAINING: NO")
    print("NEW_RANKING_INFERENCE: NO")
    print("TEST_TIME_FITTING: NO")
    print("=" * 75)


if __name__ == "__main__":
    main()
