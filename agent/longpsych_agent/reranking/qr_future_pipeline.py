from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np


class QRFutureRetrievalPipeline:
    """
    Non-invasive integration wrapper around the existing LongPsychRetriever
    and QRFutureScorer.

    Modes:
      - dense: existing Dense BGE retrieval
      - orca: existing global ORCA k=2 retrieval
      - qr_future: Dense BGE candidate recall -> five-fold QR-Future reranking

    QR-Future uses Dense top-k candidate recall by default because the frozen
    S12C ranking models were trained with a Dense top-100 candidate reservoir.
    The global ORCA index remains available as its own retrieval mode. An
    ORCA-recall + QR-Future cascade is deliberately not silently substituted
    for the trained candidate protocol.
    """

    VALID_MODES = {"dense", "orca", "qr_future"}
    QR_FUTURE_MODE = "qr_future"
    QR_FUTURE_FIRST_STAGE = "dense"
    QR_FUTURE_RERANKER = "QR_FUTURE_FULL_5FOLD"

    def __init__(
        self,
        retriever: Any,
        qr_future_scorer: Any,
        default_candidate_k: int = 100,
    ):
        if not isinstance(default_candidate_k, int) or default_candidate_k <= 0:
            raise ValueError("default_candidate_k must be a positive integer.")

        self.retriever = retriever
        self.qr_future_scorer = qr_future_scorer
        self.default_candidate_k = default_candidate_k

        self._validate_components()

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        index_dir: str | Path,
        qr_future_project_root: str | Path,
        default_candidate_k: int = 100,
        device: Optional[str] = None,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        verify_hashes: bool = True,
    ) -> "QRFutureRetrievalPipeline":
        """Load the existing dual retriever and the frozen QR-Future scorer."""
        from ..retrieval.unified_retriever import LongPsychRetriever
        from .qr_future_scorer import QRFutureScorer

        retriever = LongPsychRetriever.from_pretrained(
            model_name_or_path=model_name_or_path,
            index_dir=index_dir,
            default_mode="dense",
            device=device,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        scorer = QRFutureScorer(
            project_root=qr_future_project_root,
            device=device or "auto",
            verify_hashes=verify_hashes,
            load_response_assets=True,
        )

        return cls(
            retriever=retriever,
            qr_future_scorer=scorer,
            default_candidate_k=default_candidate_k,
        )

    def _validate_components(self) -> None:
        dimension = int(self.retriever.dimension)
        if dimension != int(self.qr_future_scorer.EXPECTED_DIMENSION):
            raise ValueError(
                f"Retriever/QR-Future dimension mismatch: "
                f"{dimension} != {self.qr_future_scorer.EXPECTED_DIMENSION}."
            )

        dense = self.retriever.dense_retriever
        if dense.index is None:
            raise ValueError("Dense FAISS index is not initialized.")

        if self.qr_future_scorer.response_embeddings is None:
            raise ValueError("QR-Future response embeddings are not loaded.")
        if self.qr_future_scorer.faiss_to_original is None:
            raise ValueError("QR-Future FAISS row mapping is not loaded.")
        if self.qr_future_scorer.row_mapping is None:
            raise ValueError("QR-Future row mapping records are not loaded.")

        dense_count = len(dense.document_ids)
        mapping_count = len(self.qr_future_scorer.row_mapping)
        if dense_count != mapping_count:
            raise ValueError(
                f"Dense index / row mapping count mismatch: "
                f"{dense_count} != {mapping_count}."
            )

        # Critical cross-artifact alignment check:
        # FAISS local row -> row_mapping response_uid -> Dense index UID.
        for item in self.qr_future_scorer.row_mapping:
            faiss_row = int(item["faiss_row"])
            if faiss_row < 0 or faiss_row >= dense_count:
                raise ValueError(
                    f"Row mapping contains out-of-range FAISS row {faiss_row}."
                )
            mapped_uid = str(item["response_uid"])
            indexed_uid = str(dense.document_ids[faiss_row])
            if mapped_uid != indexed_uid:
                raise ValueError(
                    "Dense FAISS / QR-Future row mapping UID mismatch at "
                    f"faiss_row={faiss_row}: {indexed_uid!r} != {mapped_uid!r}."
                )

        if len(self.qr_future_scorer.folds) != 5:
            raise ValueError("QR-Future scorer must have exactly five folds.")

        if self.qr_future_scorer.EXPECTED_CONDITION != "QR_FUTURE_FULL":
            raise ValueError("QR-Future scorer condition must be QR_FUTURE_FULL.")

        print("[QRFutureRetrievalPipeline] COMPONENT ALIGNMENT PASS")
        print("  corpus_size:", dense_count)
        print("  embedding_dimension:", dimension)
        print("  qr_future_folds:", len(self.qr_future_scorer.folds))
        print("  default_candidate_k:", self.default_candidate_k)

    @staticmethod
    def _validate_query(query: Any) -> None:
        if not isinstance(query, str):
            raise TypeError("Each query must be a string.")
        if not query.strip():
            raise ValueError("Query cannot be empty.")

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

    def _search_qr_future(
        self,
        query: str,
        top_k: int,
        candidate_k: Optional[int] = None,
    ) -> dict[str, Any]:
        self._validate_query(query)
        self._validate_top_k(top_k)

        requested_candidate_k = (
            self.default_candidate_k
            if candidate_k is None
            else candidate_k
        )
        if not isinstance(requested_candidate_k, int) or requested_candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer.")
        if requested_candidate_k < top_k:
            raise ValueError(
                f"candidate_k ({requested_candidate_k}) must be >= top_k ({top_k})."
            )

        corpus_size = int(self.retriever.corpus_size)
        effective_candidate_k = min(requested_candidate_k, corpus_size)
        if effective_candidate_k < top_k:
            raise ValueError(
                f"top_k ({top_k}) exceeds retrieval corpus size ({corpus_size})."
            )

        # Encode once. The same normalized original BGE query vector is used
        # both for Dense candidate retrieval and as QR-Future's raw input.
        query_embedding = self.retriever.encoder.encode(
            [query],
            normalize=True,
        )[0]
        query_embedding = np.asarray(query_embedding, dtype=np.float32)

        candidate_result = self.retriever.search_embeddings(
            embeddings=query_embedding,
            top_k=effective_candidate_k,
            mode=self.QR_FUTURE_FIRST_STAGE,
            query_texts=[query],
        )
        candidates = candidate_result["results"]
        if not candidates:
            return {
                "query": query,
                "mode": self.QR_FUTURE_MODE,
                "first_stage_mode": self.QR_FUTURE_FIRST_STAGE,
                "reranker": self.QR_FUTURE_RERANKER,
                "top_k": top_k,
                "candidate_k": 0,
                "results": [],
            }

        # Adapt LongPsychRetriever's public result schema to QRFutureScorer's
        # row/text/id/score schema. `faiss_row` is not an original embedding row.
        scorer_inputs = []
        for candidate in candidates:
            scorer_inputs.append(
                {
                    "id": candidate["response_uid"],
                    "row": int(candidate["faiss_row"]),
                    "text": candidate["response_text"],
                    "score": float(candidate["retrieval_score"]),
                    "candidate_rank": int(candidate["rank"]),
                }
            )

        reranked = self.qr_future_scorer.rerank_results(
            query_embedding=query_embedding,
            retrieval_results=scorer_inputs,
        )

        final_results = []
        for item in reranked[:top_k]:
            final_results.append(
                {
                    "rank": int(item["rerank_rank"]),
                    "response_uid": str(item["id"]),
                    "response_text": str(item["text"]),
                    "retrieval_score": float(item["retrieval_score"]),
                    "qr_future_score": float(item["qr_future_score"]),
                    "candidate_rank": int(item["candidate_rank"]),
                    "faiss_row": int(item["row"]),
                }
            )

        return {
            "query": query,
            "mode": self.QR_FUTURE_MODE,
            "first_stage_mode": self.QR_FUTURE_FIRST_STAGE,
            "reranker": self.QR_FUTURE_RERANKER,
            "score_type": "raw_ranking_score_not_probability",
            "top_k": int(top_k),
            "candidate_k": len(candidates),
            "results": final_results,
        }

    def search(
        self,
        query: str | Sequence[str],
        top_k: int = 5,
        mode: str = "qr_future",
        candidate_k: Optional[int] = None,
    ):
        """
        Search one query or a batch.

        `qr_future` performs Dense candidate recall followed by QR-Future
        reranking. `dense` and `orca` preserve the existing retriever behavior.
        """
        self._validate_top_k(top_k)
        selected_mode = str(mode).lower().strip()
        if selected_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported mode {mode!r}; choose from {sorted(self.VALID_MODES)}."
            )

        if isinstance(query, str):
            self._validate_query(query)
            if selected_mode == self.QR_FUTURE_MODE:
                return self._search_qr_future(
                    query=query,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )
            return self.retriever.search(
                query=query,
                top_k=top_k,
                mode=selected_mode,
            )

        if not isinstance(query, Sequence):
            raise TypeError("query must be a string or a sequence of strings.")

        queries = list(query)
        if not queries:
            raise ValueError("Query batch cannot be empty.")
        for item in queries:
            self._validate_query(item)

        if selected_mode == self.QR_FUTURE_MODE:
            return [
                self._search_qr_future(
                    query=item,
                    top_k=top_k,
                    candidate_k=candidate_k,
                )
                for item in queries
            ]

        return self.retriever.search(
            query=queries,
            top_k=top_k,
            mode=selected_mode,
        )

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "retrieval_modes": ["dense", "orca", "qr_future"],
            "corpus_size": int(self.retriever.corpus_size),
            "dimension": int(self.retriever.dimension),
            "default_candidate_k": self.default_candidate_k,
            "qr_future": self.qr_future_scorer.health(),
        }
