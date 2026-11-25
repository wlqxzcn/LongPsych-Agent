from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from longpsych_agent.config import settings
from longpsych_agent.reranking.qr_future_pipeline import (
    QRFutureRetrievalPipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(settings.root))
    parser.add_argument(
        "--model",
        default=str(settings.model_path),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    pipeline = QRFutureRetrievalPipeline.from_pretrained(
        model_name_or_path=args.model,
        index_dir=root / "data" / "indexes",
        qr_future_project_root=root,
        default_candidate_k=args.candidate_k,
        device=args.device,
        batch_size=16,
        show_progress_bar=False,
        verify_hashes=True,
    )

    query = "最近总是感到焦虑，不知道该怎么办。"

    print("\n========== DENSE BASELINE ==========")
    dense = pipeline.search(query, top_k=args.top_k, mode="dense")
    assert len(dense["results"]) == args.top_k
    print(dense)

    print("\n========== ORCA BASELINE ==========")
    orca = pipeline.search(query, top_k=args.top_k, mode="orca")
    assert len(orca["results"]) == args.top_k
    print(orca)

    print("\n========== DENSE -> QR-FUTURE ==========")
    result = pipeline.search(
        query,
        top_k=args.top_k,
        mode="qr_future",
        candidate_k=args.candidate_k,
    )
    assert result["mode"] == "qr_future"
    assert result["first_stage_mode"] == "dense"
    assert result["candidate_k"] >= args.top_k
    assert len(result["results"]) == args.top_k
    assert all(
        item["qr_future_score"] == item["qr_future_score"]
        for item in result["results"]
    )
    assert [item["rank"] for item in result["results"]] == list(
        range(1, args.top_k + 1)
    )

    print("QR_FUTURE_PIPELINE_SMOKE = PASS")
    print("HEALTH =", pipeline.health())
    print("CANDIDATES =", result["candidate_k"])
    print("TOP_RESULTS:")
    for item in result["results"]:
        print(item)


if __name__ == "__main__":
    main()
