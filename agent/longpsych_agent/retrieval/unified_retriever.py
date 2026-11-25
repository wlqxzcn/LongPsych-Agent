from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..encoders.bge_encoder import BGEEncoder
from .faiss_retriever import FAISSRetriever


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


class FrozenORCABasis:
    """
    Inference-only ORCA projector.

    Loads the original raw-SVD basis directly from NPY.
    Projection expression matches the historical ORCA implementation.
    """

    def __init__(self, basis: np.ndarray):
        basis = np.asarray(basis)

        if basis.dtype != np.float32:
            raise ValueError("ORCA basis must be float32.")

        if basis.ndim != 2:
            raise ValueError("ORCA basis must be 2D.")

        if not np.isfinite(basis).all():
            raise ValueError("ORCA basis contains NaN or Inf.")

        gram = basis.T @ basis
        identity = np.eye(basis.shape[1], dtype=np.float32)

        if not np.allclose(gram, identity, atol=1e-4):
            raise ValueError("ORCA basis is not orthonormal.")

        self.basis = np.ascontiguousarray(basis)
        self.input_dim = int(basis.shape[0])
        self.k = int(basis.shape[1])

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_sha256: Optional[str] = None,
    ) -> "FrozenORCABasis":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(path)

        actual_sha = sha256_file(path)

        if expected_sha256 is not None:
            if actual_sha.lower() != expected_sha256.lower():
                raise ValueError(
                    "ORCA basis file SHA256 mismatch: "
                    f"{actual_sha}"
                )

        basis = np.load(
            path,
            allow_pickle=False,
        )

        if not isinstance(basis, np.ndarray):
            raise ValueError("Expected a raw NPY array.")

        obj = cls(basis)

        print(
            "[FrozenORCABasis] LOAD PASS "
            f"shape={obj.basis.shape} "
            f"k={obj.k}"
        )

        print("[FrozenORCABasis] SHA256:", actual_sha)

        return obj

    def transform_queries(self, embeddings):
        x = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        was_1d = x.ndim == 1

        if was_1d:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError("Expected 1D or 2D embeddings.")

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Dimension mismatch: {x.shape[1]} "
                f"!= {self.input_dim}"
            )

        if not np.isfinite(x).all():
            raise ValueError("Query contains NaN or Inf.")

        # Exact historical ORCA projection.
        projected = (
            x - (x @ self.basis) @ self.basis.T
        ).astype(np.float32)

        # Historical ORCA L2 normalization.
        norms = np.linalg.norm(
            projected,
            axis=1,
            keepdims=True,
        )

        projected = (
            projected / np.maximum(norms, 1e-12)
        ).astype(np.float32)

        return projected[0] if was_1d else projected


class LongPsychRetriever:
    """
    Unified Dense / ORCA psychological dialogue retriever.
    """

    VALID_MODES = {
        "dense": "dense",
        "dense_bge": "dense",
        "orca": "orca",
        "orca_k2": "orca",
    }

    def __init__(
        self,
        encoder,
        dense_retriever,
        orca_retriever,
        orca_projector,
        default_mode: str = "orca",
    ):
        self.encoder = encoder
        self.dense_retriever = dense_retriever
        self.orca_retriever = orca_retriever
        self.orca_projector = orca_projector

        self.default_mode = self._resolve_mode(default_mode)

        self._validate_components()

        self.dimension = int(encoder.dimension)
        self.corpus_size = len(
            dense_retriever.document_ids
        )

        print("[LongPsychRetriever] INITIALIZED")
        print("  dimension:", self.dimension)
        print("  corpus_size:", self.corpus_size)
        print("  default_mode:", self.default_mode)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        index_dir: str | Path,
        default_mode: str = "orca",
        device: Optional[str] = None,
        batch_size: int = 64,
        show_progress_bar: bool = True,
    ) -> "LongPsychRetriever":

        index_dir = Path(index_dir)

        dense_dir = index_dir / "dense_bge_v1"
        orca_dir = index_dir / "orca_k2_v1"

        print("=" * 60)
        print("LOADING DENSE INDEX")
        print("=" * 60)

        dense = FAISSRetriever.load(dense_dir)

        print("=" * 60)
        print("LOADING ORCA INDEX")
        print("=" * 60)

        orca = FAISSRetriever.load(orca_dir)

        dense_meta = dense.metadata
        orca_meta = orca.metadata

        if dense_meta.get("retrieval_mode") != "dense_bge":
            raise ValueError("Invalid Dense index metadata.")

        if orca_meta.get("retrieval_mode") != "orca_k2":
            raise ValueError("Invalid ORCA index metadata.")

        if orca_meta.get("orca_algorithm") != "raw_svd":
            raise ValueError("Unsupported ORCA algorithm.")

        if int(orca_meta.get("orca_k", -1)) != 2:
            raise ValueError("Expected frozen ORCA k=2.")

        if dense_meta.get("encoder") != orca_meta.get("encoder"):
            raise ValueError("Encoder metadata mismatch.")

        if dense_meta.get("dimension") != orca_meta.get("dimension"):
            raise ValueError("Index dimension metadata mismatch.")

        if dense_meta.get("documents_sha256") != (
            orca_meta.get("documents_sha256")
        ):
            raise ValueError("Dense/ORCA document hashes differ.")

        basis_path = (
            orca_dir / "orca_basis_k2_v1.npy"
        )

        projector = FrozenORCABasis.load(
            basis_path,
            expected_sha256=orca_meta.get(
                "orca_basis_sha256"
            ),
        )

        if projector.input_dim != int(orca_meta["dimension"]):
            raise ValueError("ORCA basis dimension mismatch.")

        if projector.k != 2:
            raise ValueError("ORCA basis k mismatch.")

        print("=" * 60)
        print("LOADING BGE ENCODER")
        print("=" * 60)

        encoder = BGEEncoder(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            device=device,
            expected_dimension=int(dense_meta["dimension"]),
            show_progress_bar=show_progress_bar,
        )

        return cls(
            encoder=encoder,
            dense_retriever=dense,
            orca_retriever=orca,
            orca_projector=projector,
            default_mode=default_mode,
        )

    @classmethod
    def _resolve_mode(cls, mode: str) -> str:
        mode = str(mode).lower().strip()

        if mode not in cls.VALID_MODES:
            raise ValueError(
                f"Unsupported retrieval mode: {mode}. "
                f"Choose from {sorted(cls.VALID_MODES)}."
            )

        return cls.VALID_MODES[mode]

    def _validate_components(self):

        dimension = int(self.encoder.dimension)

        if self.dense_retriever.dimension != dimension:
            raise ValueError("Dense encoder dimension mismatch.")

        if self.orca_retriever.dimension != dimension:
            raise ValueError("ORCA index dimension mismatch.")

        if self.dense_retriever.document_ids != (
            self.orca_retriever.document_ids
        ):
            raise ValueError("Dense/ORCA UID alignment failed.")

        if self.dense_retriever.documents != (
            self.orca_retriever.documents
        ):
            raise ValueError("Dense/ORCA document alignment failed.")

        basis = self.orca_projector.basis

        if basis.shape[0] != dimension:
            raise ValueError("ORCA projector dimension mismatch.")

        if len(set(self.dense_retriever.document_ids)) != (
            len(self.dense_retriever.document_ids)
        ):
            raise ValueError("Duplicate response UIDs.")

    def search(
        self,
        query: str | Sequence[str],
        top_k: int = 5,
        mode: Optional[str] = None,
    ):
        """
        Search using text queries.

        Single query:
            dict

        Batch queries:
            list[dict]
        """

        single = isinstance(query, str)

        if single:
            queries = [query]
        else:
            if not isinstance(query, Sequence):
                raise TypeError("Query must be a string or sequence.")

            queries = list(query)

        if not queries:
            return []

        if any(
            not isinstance(q, str) or not q.strip()
            for q in queries
        ):
            raise ValueError("Queries cannot be empty.")

        embeddings = self.encoder.encode(
            queries,
            normalize=True,
        )

        outputs = self.search_embeddings(
            embeddings=embeddings,
            top_k=top_k,
            mode=mode,
            query_texts=queries,
        )

        return outputs[0] if single else outputs

    def search_embeddings(
        self,
        embeddings,
        top_k: int = 5,
        mode: Optional[str] = None,
        query_texts: Optional[Sequence[str]] = None,
    ):
        """
        Search using precomputed query embeddings.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        selected_mode = self._resolve_mode(
            mode or self.default_mode
        )

        x = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        was_1d = x.ndim == 1

        if was_1d:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError("Expected 1D or 2D embeddings.")

        if x.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"{x.shape[1]} != {self.dimension}"
            )

        if not np.isfinite(x).all():
            raise ValueError("Embeddings contain NaN or Inf.")

        norms = np.linalg.norm(
            x,
            axis=1,
            keepdims=True,
        )

        if np.any(norms <= 1e-12):
            raise ValueError("Zero query embedding.")

        # Ensure Dense queries follow the frozen L2 protocol.
        x = (x / norms).astype(np.float32)

        if selected_mode == "orca":
            query_vectors = (
                self.orca_projector.transform_queries(x)
            )
            retriever = self.orca_retriever
        else:
            query_vectors = x
            retriever = self.dense_retriever

        raw_results = retriever.search(
            query_vectors,
            top_k=top_k,
        )

        if len(raw_results) != len(x):
            raise RuntimeError("Unexpected batch search output.")

        if query_texts is None:
            query_texts = [None] * len(x)
        else:
            query_texts = list(query_texts)

            if len(query_texts) != len(x):
                raise ValueError("Query text count mismatch.")

        outputs = []

        for query_text, items in zip(query_texts, raw_results):

            results = []

            for rank, item in enumerate(items, start=1):
                results.append({
                    "rank": rank,
                    "response_uid": item["id"],
                    "response_text": item["text"],
                    "retrieval_score": float(item["score"]),
                    "faiss_row": int(item["row"]),
                })

            outputs.append({
                "query": query_text,
                "mode": selected_mode,
                "top_k": int(top_k),
                "results": results,
            })

        return outputs[0] if was_1d else outputs
