from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np


class ORCAProjector:
    """
    ORCA source-subspace projection.

    Historical experiment parity:
      1. Raw SVD, without centering.
      2. float32 input.
      3. basis = Vt[:k].T
      4. projected = x - (x @ basis) @ basis.T
      5. L2 normalization after projection.

    The source embeddings must come from TRAIN user utterances.
    No source normalization or centering is performed internally.
    """

    FORMAT_VERSION = 1
    ALGORITHM = "raw_svd"

    def __init__(
        self,
        k: int = 2,
        normalize_output: bool = True,
        eps: float = 1e-12,
    ):
        if k <= 0:
            raise ValueError("k must be positive.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.k = int(k)
        self.normalize_output = bool(normalize_output)
        self.eps = float(eps)

        self.basis: Optional[np.ndarray] = None
        self.singular_values: Optional[np.ndarray] = None

        self.input_dim: Optional[int] = None
        self.source_count: Optional[int] = None

        self.fit_metadata = {}

    @staticmethod
    def _as_2d(
        embeddings,
        name: str,
        allow_1d: bool = True,
    ):
        arr = np.asarray(embeddings)

        was_1d = arr.ndim == 1

        if was_1d:
            if not allow_1d:
                raise ValueError(f"{name} must be 2D.")
            arr = arr[None, :]

        if arr.ndim != 2:
            raise ValueError(
                f"{name} must have shape (N, D), "
                f"got {arr.shape}."
            )

        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError(f"{name} cannot be empty.")

        arr = np.asarray(arr, dtype=np.float32)

        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} contains NaN or Inf."
            )

        return arr, was_1d

    @staticmethod
    def _sha256(arr: np.ndarray) -> str:
        arr = np.ascontiguousarray(arr)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _check_fitted(self):
        if self.basis is None:
            raise RuntimeError(
                "ORCAProjector has not been fitted. "
                "Call fit() or load() first."
            )

    def fit(self, source_embeddings):
        """
        Estimate the removable basis from TRAIN source embeddings.

        This intentionally reproduces the historical NumPy
        float32 raw-SVD implementation.
        """

        source, _ = self._as_2d(
            source_embeddings,
            "source_embeddings",
            allow_1d=False,
        )

        n, dim = source.shape

        if n < 2:
            raise ValueError(
                "At least two source embeddings are required."
            )

        if self.k > min(n, dim):
            raise ValueError(
                f"k={self.k} exceeds maximum rank "
                f"{min(n, dim)}."
            )

        # Historical ORCA implementation.
        # No centering, no additional normalization,
        # no QR reorthogonalization.
        _, singular_values, vt = np.linalg.svd(
            source.astype("float32"),
            full_matrices=False,
        )

        basis = vt[:self.k].T.astype("float32")

        total_energy = float(
            np.sum(singular_values ** 2)
        )

        top_energy = float(
            np.sum(singular_values[:self.k] ** 2)
        )

        orthogonality_error = float(
            np.linalg.norm(
                basis.T @ basis
                - np.eye(self.k, dtype=np.float32),
                ord="fro",
            )
        )

        self.basis = np.ascontiguousarray(basis)
        self.singular_values = singular_values[
            :self.k
        ].astype(np.float32)

        self.input_dim = int(dim)
        self.source_count = int(n)

        self.fit_metadata = {
            "format_version": self.FORMAT_VERSION,
            "algorithm": self.ALGORITHM,
            "k": self.k,
            "input_dim": dim,
            "source_count": n,
            "dtype": "float32",
            "centered": False,
            "normalize_for_fit": False,
            "normalize_output": self.normalize_output,
            "source_sha256": self._sha256(source),
            "basis_sha256": self._sha256(self.basis),
            "singular_values": [
                float(v) for v in self.singular_values
            ],
            "explained_energy_ratio": (
                top_energy / total_energy
                if total_energy > self.eps
                else 0.0
            ),
            "orthogonality_error": orthogonality_error,
        }

        return self

    def project(self, embeddings):
        """
        Remove the fitted source subspace.

        Returns unnormalized projected embeddings.
        """

        self._check_fitted()

        x, was_1d = self._as_2d(
            embeddings,
            "embeddings",
        )

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Dimension mismatch: got {x.shape[1]}, "
                f"expected {self.input_dim}."
            )

        # Exact historical projection expression.
        projected = (
            x - (x @ self.basis) @ self.basis.T
        ).astype(np.float32)

        return projected[0] if was_1d else projected

    def transform(
        self,
        embeddings,
        normalize: Optional[bool] = None,
    ):
        """
        Project and optionally L2-normalize.

        Default behavior matches historical ORCA retrieval.
        """

        projected = self.project(embeddings)

        if normalize is None:
            normalize = self.normalize_output

        if not normalize:
            return projected

        was_1d = projected.ndim == 1

        if was_1d:
            projected = projected[None, :]

        norms = np.linalg.norm(
            projected,
            axis=1,
            keepdims=True,
        )

        projected = (
            projected / np.maximum(norms, self.eps)
        ).astype(np.float32)

        return projected[0] if was_1d else projected

    def transform_queries(self, query_embeddings):
        return self.transform(query_embeddings)

    def transform_responses(self, response_embeddings):
        return self.transform(response_embeddings)

    def save(self, path):
        """
        Save basis and metadata without saving training data.
        """

        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = dict(self.fit_metadata)

        tmp_path = path.with_name(
            path.name + ".tmp"
        )

        with tmp_path.open("wb") as f:
            np.savez_compressed(
                f,
                basis=self.basis,
                singular_values=self.singular_values,
                metadata=np.asarray(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    )
                ),
            )

        tmp_path.replace(path)

    @classmethod
    def load(cls, path):
        """
        Load a frozen ORCA projection basis.
        """

        path = Path(path)

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            basis = np.asarray(
                data["basis"],
                dtype=np.float32,
            )

            singular_values = np.asarray(
                data["singular_values"],
                dtype=np.float32,
            )

            metadata = json.loads(
                str(data["metadata"].item())
            )

        if metadata.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported ORCA basis format."
            )

        if metadata.get("algorithm") != cls.ALGORITHM:
            raise ValueError(
                "Unsupported ORCA algorithm."
            )

        k = int(metadata["k"])

        if basis.ndim != 2 or basis.shape[1] != k:
            raise ValueError("Invalid basis shape.")

        if not np.isfinite(basis).all():
            raise ValueError("Basis contains NaN or Inf.")

        if cls._sha256(basis) != metadata["basis_sha256"]:
            raise ValueError(
                "ORCA basis SHA256 verification failed."
            )

        obj = cls(
            k=k,
            normalize_output=metadata["normalize_output"],
        )

        obj.basis = np.ascontiguousarray(basis)
        obj.singular_values = singular_values
        obj.input_dim = int(basis.shape[0])
        obj.source_count = int(metadata["source_count"])
        obj.fit_metadata = metadata

        return obj
