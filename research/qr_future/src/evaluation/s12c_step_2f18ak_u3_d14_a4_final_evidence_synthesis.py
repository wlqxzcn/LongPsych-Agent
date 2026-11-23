import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
PHASE = ROOT / "results/phase_s12c"

OUT = PHASE / (
    "step_2f18ak_u3_d14_a4_final_evidence_synthesis_v1"
)

SCRIPT = Path(__file__).resolve()
PROTOCOL = OUT / "protocol_v1.json"

GROUPS = {
    "CONSOLIDATION": (
        "step_2f18aa_final_results_consolidation_v1",
        [
            "final_results_v1.json",
            "model_performance_v1.csv",
            "paired_contrasts_v1.csv",
            "final_results_report_v1.txt",
            "run_manifest.json",
        ],
    ),
    "D10": (
        "step_2f18ab_u3_d10_case_mechanism_audit_v1",
        [
            "protocol_v1.json",
            "success_cases_v1.jsonl",
            "regression_cases_v1.jsonl",
            "qr_specific_cases_v1.jsonl",
            "case_statistics_v1.json",
            "case_statistics_v1.csv",
            "long_vs_nonlong_statistics_v1.json",
            "final_report_v1.txt",
            "run_manifest.json",
        ],
    ),
    "D11": (
        "step_2f18ac_u3_d11_paired_interaction_audit_v2",
        [
            "protocol_v1.json",
            "interaction_results_v2.json",
            "interaction_results_v2.csv",
            "interaction_report_v2.txt",
            "run_manifest_v2.json",
        ],
    ),
    "D12": (
        "step_2f18ad_u3_d12_product_interaction_mechanism_v1",
        [
            "protocol_v1.json",
            "d12_results_v1.json",
            "d12_results_v1.csv",
            "d12_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
    "D13": (
        "step_2f18af_u3_d13_final_evidence_consistency_audit_v2",
        [
            "protocol_v1.json",
            "audit_results_v1.json",
            "audit_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
    "D14_A0": (
        "step_2f18ag_u3_d14_a0_index_rank_audit_v1",
        [
            "protocol_v1.json",
            "audit_results_v1.json",
            "audit_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
    "D14_A1": (
        "step_2f18ah_u3_d14_a1_gold_negative_margin_v1",
        [
            "protocol_v1.json",
            "margin_results_v1.json",
            "margin_descriptive_v1.csv",
            "margin_contrasts_v1.csv",
            "margin_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
    "D14_A2": (
        "step_2f18ai_u3_d14_a2_score_decomposition_v2",
        [
            "protocol_v1.json",
            "decomposition_results_v1.json",
            "decomposition_results_v1.csv",
            "decomposition_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
    "D14_A3": (
        "step_2f18aj_u3_d14_a3_component_interaction_v1",
        [
            "protocol_v1.json",
            "a3_results_v1.json",
            "a3_results_v1.csv",
            "a3_report_v1.txt",
            "run_manifest_v1.json",
        ],
    ),
}

CONSOLIDATION_ALIAS_MAP = {
    "json": "final_results_v1.json",
    "model_csv": "model_performance_v1.csv",
    "contrast_csv": "paired_contrasts_v1.csv",
    "report": "final_results_report_v1.txt",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    if OUT.exists():
        raise RuntimeError(
            f"OUTPUT_COLLISION: {OUT}"
        )

    print("========== A4 SOURCE PREFLIGHT ==========")

    sources = {}
    group_files = {}

    for group, (dirname, filenames) in GROUPS.items():
        folder = PHASE / dirname
        group_files[group] = []

        print("\nGROUP:", group)

        for filename in filenames:
            path = folder / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"MISSING_SOURCE: {path}"
                )

            relative = str(path.relative_to(ROOT))
            digest = sha(path)

            if relative in sources:
                raise RuntimeError(
                    f"DUPLICATE_SOURCE: {relative}"
                )

            sources[relative] = digest
            group_files[group].append(relative)

            print("SOURCE_PASS", filename, digest)

    # Verify consolidation alias-to-filename mapping.
    consolidation_dir = PHASE / GROUPS["CONSOLIDATION"][0]
    manifest = json.loads(
        (consolidation_dir / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    for alias, filename in CONSOLIDATION_ALIAS_MAP.items():
        path = consolidation_dir / filename
        expected = manifest["outputs"][alias]

        if isinstance(expected, dict):
            expected = (
                expected.get("sha256")
                or expected.get("hash")
            )

        actual = sha(path)

        if expected != actual:
            raise RuntimeError(
                f"CONSOLIDATION_HASH_FAILURE: {alias}"
            )

        print("CONSOLIDATION_HASH_PASS", alias)

    protocol = {
        "artifact": (
            "S12C U3 D14-A4 Final Evidence "
            "Synthesis Protocol V1"
        ),
        "status": "FROZEN_BEFORE_FINAL_EVIDENCE_SYNTHESIS",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluator_script": str(SCRIPT),
        "evaluator_script_sha256": sha(SCRIPT),
        "source_directories": list(GROUPS.keys()),
        "source_group_files": group_files,
        "source_file_count": len(sources),
        "source_sha256": sources,
        "consolidation_alias_map": CONSOLIDATION_ALIAS_MAP,
        "analysis_scope": {
            "training": False,
            "new_ranking_inference": False,
            "test_time_fitting": False,
            "checkpoint_reselection": False,
            "statistical_recomputation": False,
            "historical_artifact_modification": False,
            "purpose": "FINAL_EVIDENCE_SYNTHESIS_ONLY"
        },
        "evidence_policy": {
            "ranking_evidence": [
                "CONSOLIDATION",
                "D10",
                "D11"
            ],
            "product_interaction_evidence": [
                "D12"
            ],
            "data_integrity_evidence": [
                "D13",
                "D14_A0"
            ],
            "gold_margin_evidence": [
                "D14_A1"
            ],
            "score_component_evidence": [
                "D14_A2"
            ],
            "component_interaction_evidence": [
                "D14_A3"
            ]
        },
        "statistical_interpretation_rules": [
            "Distinguish within-subset effects from Long20-specific interactions.",
            "Distinguish raw p-values from multiplicity-adjusted p-values.",
            "Do not describe exploratory post-hoc results as confirmatory.",
            "Do not infer product interaction significance from positive point estimates.",
            "Preserve historical Holm artifacts unchanged.",
            "Document the historical Holm 200 versus actual Holm 400 label discrepancy."
        ],
        "planned_outputs": [
            "evidence_matrix_v1.json",
            "evidence_matrix_v1.csv",
            "final_evidence_report_v1.txt",
            "run_manifest_v1.json"
        ],
        "overwrite_existing": False
    }

    OUT.mkdir(parents=True)

    tmp = PROTOCOL.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            protocol,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )
    tmp.replace(PROTOCOL)

    print("\n========== A4 PROTOCOL FREEZE ==========")
    print("SOURCE_DIRECTORIES:", len(GROUPS))
    print("SOURCE_FILES:", len(sources))
    print("CONSOLIDATION_HASH: PASS")
    print("SCRIPT_HASH:", protocol["evaluator_script_sha256"])
    print("PROTOCOL_SHA256:", sha(PROTOCOL))
    print("STATUS:", protocol["status"])
    print("OUTPUT:", OUT)
    print("U3_D14_A4_PROTOCOL_FREEZE_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare",
        action="store_true"
    )
    args = parser.parse_args()

    if not args.prepare:
        raise RuntimeError(
            "Only --prepare is implemented in this stage."
        )

    prepare()
