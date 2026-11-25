from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from longpsych_agent.config import settings
from longpsych_agent.reranking.qr_future_scorer import QRFutureScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(settings.root),
        help="LongPsychAgent project root",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidates", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scorer = QRFutureScorer(
        project_root=root,
        device=args.device,
        verify_hashes=True,
        load_response_assets=True,
    )

    assert scorer.response_embeddings is not None
    assert scorer.faiss_to_original is not None

    n = min(max(args.candidates, 2), len(scorer.faiss_to_original))
    faiss_rows = list(range(n))
    query = np.asarray(scorer.response_embeddings[0], dtype=np.float32)
    candidate_embeddings = scorer.response_embeddings_for_faiss_rows(faiss_rows)

    detailed = scorer.score_candidates_detailed(query, candidate_embeddings)
    scores = detailed["ensemble_scores"]
    fold_scores = detailed["fold_scores"]

    assert scores.shape == (n,)
    assert fold_scores.shape == (5, n)
    assert np.isfinite(scores).all()
    assert np.isfinite(fold_scores).all()

    # Exercise the retrieval-result adapter and verify that text/UID metadata
    # is preserved while ranking follows the QR-Future ensemble score.
    fake_results = [
        {
            "id": f"smoke-response-{row}",
            "row": row,
            "text": f"smoke-response-{row}",
            "score": 0.0,
        }
        for row in faiss_rows
    ]
    reranked = scorer.rerank_results(query, fake_results)
    assert len(reranked) == n
    assert all("text" in item and "id" in item for item in reranked)
    assert [item["rerank_rank"] for item in reranked] == list(range(1, n + 1))

    print("QR_FUTURE_SCORER_SMOKE = PASS")
    print("HEALTH =", scorer.health())
    print("FOLD_SCORE_SHAPE =", fold_scores.shape)
    print("ENSEMBLE_SCORE_RANGE =", float(scores.min()), float(scores.max()))
    print("TOP_RESULT =", reranked[0])


if __name__ == "__main__":
    main()
