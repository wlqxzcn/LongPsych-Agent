from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class FAISSRetriever:
    """
    Exact FAISS retrieval using IndexFlatIP.

    Supports:
      - Dense BGE embeddings
      - ORCA-projected embeddings
      - Batch queries
      - Persistent indexes
      - Metadata and SHA256 verification

    Embeddings are L2-normalized by default,
    making inner product equivalent to cosine similarity.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        normalize: bool = True,
        eps: float = 1e-12,
    ):
        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.normalize = bool(normalize)
        self.eps = float(eps)

        self.index = None
        self.documents = []
        self.document_ids = []

        self.dimension: Optional[int] = None
        self.metadata = {}

    @staticmethod
    def _sha256_array(arr):
        arr = np.ascontiguousarray(arr)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def _sha256_file(path):
        h = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(
                lambda: f.read(1024 * 1024), b""
            ):
                h.update(chunk)

        return h.hexdigest()

    def _prepare_vectors(self, embeddings, name):
        x = np.asarray(embeddings, dtype=np.float32)

        if x.ndim == 1:
            x = x[None, :]

        if x.ndim != 2:
            raise ValueError(
                f"{name} must be a 2D matrix."
            )

        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(
                f"{name} cannot be empty."
            )

        if not np.isfinite(x).all():
            raise ValueError(
                f"{name} contains NaN or Inf."
            )

        x = np.ascontiguousarray(x.copy())

        if self.normalize:
            norms = np.linalg.norm(
                x, axis=1, keepdims=True
            )

            x /= np.maximum(norms, self.eps)

        return x

    def build(
        self,
        embeddings,
        documents,
        document_ids=None,
    ):
        """
        Build an exact inner-product index.
        """

        vectors = self._prepare_vectors(
            embeddings,
            "embeddings",
        )

        n, dim = vectors.shape

        if len(documents) != n:
            raise ValueError(
                "Document count must match embedding count."
            )

        documents = [str(x) for x in documents]

        if document_ids is None:
            document_ids = [
                str(i) for i in range(n)
            ]
        else:
            document_ids = [
                str(x) for x in document_ids
            ]

        if len(document_ids) != n:
            raise ValueError(
                "Document ID count mismatch."
            )

        if len(set(document_ids)) != n:
            raise ValueError(
                "Document IDs must be unique."
            )

        base_index = faiss.IndexFlatIP(dim)

        index = faiss.IndexIDMap2(base_index)

        internal_ids = np.arange(
            n, dtype=np.int64
        )

        index.add_with_ids(
            vectors,
            internal_ids,
        )

        self.index = index
        self.documents = documents
        self.document_ids = document_ids
        self.dimension = int(dim)

        self.metadata = {
            "format_version": self.FORMAT_VERSION,
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "metric": "inner_product",
            "normalized": self.normalize,
            "dimension": dim,
            "corpus_size": n,
            "embedding_sha256": self._sha256_array(
                vectors
            ),
        }

        print(
            "[FAISSRetriever] BUILD PASS "
            f"corpus={n} dim={dim} "
            f"normalize={self.normalize}"
        )

        return self

    def _check_built(self):
        if self.index is None:
            raise RuntimeError(
                "FAISS index not initialized. "
                "Call build() or load() first."
            )

    def search(
        self,
        query_embeddings,
        top_k=100,
    ):
        """
        Search one or multiple query embeddings.

        Single query:
            list[dict]

        Batch queries:
            list[list[dict]]
        """

        self._check_built()

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        queries = np.asarray(
            query_embeddings,
            dtype=np.float32,
        )

        was_1d = queries.ndim == 1

        queries = self._prepare_vectors(
            queries,
            "query_embeddings",
        )

        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: "
                f"got {queries.shape[1]}, "
                f"expected {self.dimension}."
            )

        k = min(int(top_k), self.index.ntotal)

        scores, rows = self.index.search(
            queries,
            k,
        )

        results = []

        for query_scores, query_rows in zip(
            scores, rows
        ):
            items = []

            for score, row in zip(
                query_scores, query_rows
            ):
                if row < 0:
                    continue

                row = int(row)

                items.append({
                    "id": self.document_ids[row],
                    "row": row,
                    "text": self.documents[row],
                    "score": float(score),
                })

            results.append(items)

        return results[0] if was_1d else results

    def save(self, directory):
        """
        Save index, documents and metadata.
        """

        self._check_built()

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = directory / "index.faiss"
        docs_path = directory / "documents.jsonl"
        metadata_path = directory / "metadata.json"

        index_tmp = directory / "index.faiss.tmp"
        docs_tmp = directory / "documents.jsonl.tmp"
        metadata_tmp = directory / "metadata.json.tmp"

        faiss.write_index(
            self.index,
            str(index_tmp),
        )

        index_tmp.replace(index_path)

        with docs_tmp.open(
            "w", encoding="utf-8"
        ) as f:
            for row, (doc_id, text) in enumerate(
                zip(
                    self.document_ids,
                    self.documents,
                )
            ):
                f.write(
                    json.dumps(
                        {
                            "row": row,
                            "id": doc_id,
                            "text": text,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

        docs_tmp.replace(docs_path)

        metadata = dict(self.metadata)

        metadata["index_sha256"] = (
            self._sha256_file(index_path)
        )

        metadata["documents_sha256"] = (
            self._sha256_file(docs_path)
        )

        with metadata_tmp.open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                metadata,
                f,
                ensure_ascii=False,
                indent=2,
            )

        metadata_tmp.replace(metadata_path)

        print(
            "[FAISSRetriever] SAVE PASS:",
            directory,
        )

    @classmethod
    def load(cls, directory):
        """
        Load and verify a persisted FAISS index.
        """

        directory = Path(directory)

        index_path = directory / "index.faiss"
        docs_path = directory / "documents.jsonl"
        metadata_path = directory / "metadata.json"

        for path in (
            index_path,
            docs_path,
            metadata_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)

        with metadata_path.open(
            "r", encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        if metadata["format_version"] != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported index format."
            )

        if cls._sha256_file(index_path) != metadata["index_sha256"]:
            raise ValueError(
                "FAISS index SHA256 mismatch."
            )

        if cls._sha256_file(docs_path) != metadata["documents_sha256"]:
            raise ValueError(
                "Document file SHA256 mismatch."
            )

        index = faiss.read_index(str(index_path))

        if index.ntotal != metadata["corpus_size"]:
            raise ValueError(
                "Index corpus size mismatch."
            )

        if index.d != metadata["dimension"]:
            raise ValueError(
                "Index dimension mismatch."
            )

        records = []

        with docs_path.open(
            "r", encoding="utf-8"
        ) as f:
            for line in f:
                records.append(json.loads(line))

        if len(records) != index.ntotal:
            raise ValueError(
                "Document count does not match index."
            )

        records.sort(key=lambda x: x["row"])

        if [x["row"] for x in records] != list(
            range(len(records))
        ):
            raise ValueError(
                "Invalid document row mapping."
            )

        obj = cls(
            normalize=metadata["normalized"]
        )

        obj.index = index
        obj.documents = [
            x["text"] for x in records
        ]
        obj.document_ids = [
            x["id"] for x in records
        ]
        obj.dimension = int(index.d)
        obj.metadata = metadata

        if len(set(obj.document_ids)) != len(records):
            raise ValueError(
                "Duplicate document IDs."
            )

        print(
            "[FAISSRetriever] LOAD PASS "
            f"corpus={index.ntotal} "
            f"dim={index.d}"
        )

        return obj
