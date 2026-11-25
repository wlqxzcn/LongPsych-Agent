from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _as_2d_float32(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array; got {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return np.ascontiguousarray(array)


@dataclass
class FoldRuntime:
    fold: int
    basis: np.ndarray
    model: torch.nn.Module
    basis_path: Path
    checkpoint_path: Path

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply this fold's frozen raw-SVD ORCA k=2 projection."""
        x = _as_2d_float32(embeddings, "embeddings")
        if x.shape[1] != self.basis.shape[0]:
            raise ValueError(
                f"Fold {self.fold}: embedding dimension {x.shape[1]} "
                f"does not match basis dimension {self.basis.shape[0]}."
            )

        # Keep the historical expression and float32 behavior:
        # x - (x @ basis) @ basis.T, followed by L2 normalization.
        projected = (x - (x @ self.basis) @ self.basis.T).astype(np.float32)
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        projected = projected / np.maximum(norms, 1e-12)
        return np.ascontiguousarray(projected, dtype=np.float32)


class QRFutureScorer:
    """
    Frozen five-fold QR-Future FULL inference scorer.

    Input:
        Original BGE embeddings (N, 1024), not pre-projected vectors.

    Per fold:
        ORCA k=2 fold-specific projection -> L2 normalization
        -> QRFutureFullMatchedModel.rank_scores()

    Ensemble:
        Arithmetic mean of the five raw ranking scores.
        The auxiliary future_head is not used for online ranking.
    """

    EXPECTED_CONDITION = "QR_FUTURE_FULL"
    EXPECTED_INPUT_SPACE = "ORCA_K2_FOLD_SPECIFIC"
    EXPECTED_FOLDS = 5
    EXPECTED_DIMENSION = 1024
    EXPECTED_RESPONSE_ROWS = 61502

    def __init__(
        self,
        project_root: str | Path,
        manifest_path: str | Path | None = None,
        device: str = "auto",
        candidate_chunk_size: int = 128,
        verify_hashes: bool = True,
        load_response_assets: bool = True,
    ):
        self.project_root = Path(project_root).resolve()
        self.qr_root = self.project_root / "data" / "qr_future"
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.qr_root / "manifest.json"
        )
        self.verify_hashes = bool(verify_hashes)
        self.candidate_chunk_size = int(candidate_chunk_size)

        if self.candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive.")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)

        self.device = self._resolve_device(device)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()

        # The research model implementation is reused directly. No model
        # architecture is copied or reimplemented in this package.
        root_text = str(self.project_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from src.models.s12c_matched_ablation_models import QRFutureFullMatchedModel

        self.folds: list[FoldRuntime] = []
        for fold in range(self.EXPECTED_FOLDS):
            item = self.manifest["folds"][str(fold)]
            basis_path = self.qr_root / item["basis_file"]
            checkpoint_path = self.qr_root / item["checkpoint_file"]

            self._check_file(basis_path, item["basis_file_sha256"], f"fold{fold} basis")
            self._check_file(
                checkpoint_path,
                item["checkpoint_sha256"],
                f"fold{fold} checkpoint",
            )

            basis = np.load(basis_path, allow_pickle=False)
            if not isinstance(basis, np.ndarray):
                raise ValueError(f"Fold {fold}: expected a raw NPY basis array.")
            if basis.dtype != np.float32:
                raise ValueError(f"Fold {fold}: basis dtype must be float32.")
            if basis.shape != (self.EXPECTED_DIMENSION, 2):
                raise ValueError(
                    f"Fold {fold}: expected basis shape "
                    f"({self.EXPECTED_DIMENSION}, 2), got {basis.shape}."
                )
            if not np.isfinite(basis).all():
                raise ValueError(f"Fold {fold}: basis contains NaN or Inf.")
            if sha256_array(basis) != item["basis_array_sha256"]:
                raise ValueError(f"Fold {fold}: basis array SHA256 mismatch.")

            gram = basis.T @ basis
            identity = np.eye(2, dtype=np.float32)
            if not np.allclose(gram, identity, atol=1e-4, rtol=0.0):
                raise ValueError(f"Fold {fold}: ORCA basis is not orthonormal.")

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            self._validate_checkpoint_metadata(
                checkpoint=checkpoint,
                fold=fold,
                expected_basis_sha=item["basis_array_sha256"],
            )

            model = QRFutureFullMatchedModel()
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            model.requires_grad_(False)

            self.folds.append(
                FoldRuntime(
                    fold=fold,
                    basis=np.ascontiguousarray(basis, dtype=np.float32),
                    model=model,
                    basis_path=basis_path,
                    checkpoint_path=checkpoint_path,
                )
            )
            del checkpoint

            print(
                f"[QRFutureScorer] FOLD {fold} LOAD PASS "
                f"basis={basis.shape} device={self.device}"
            )

        self.response_embeddings = None
        self.faiss_to_original: np.ndarray | None = None
        self.row_mapping: list[dict[str, Any]] | None = None
        if load_response_assets:
            self._load_response_assets()

        print("[QRFutureScorer] INITIALIZED")
        print("  condition:", self.EXPECTED_CONDITION)
        print("  folds:", len(self.folds))
        print("  input_space:", self.EXPECTED_INPUT_SPACE)
        print("  device:", self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        value = str(device).lower().strip()
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(value)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {value}")
        return resolved

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("status") != "PASS":
            raise ValueError("QR-Future deployment manifest status is not PASS.")
        if manifest.get("condition") != self.EXPECTED_CONDITION:
            raise ValueError(
                f"Expected condition {self.EXPECTED_CONDITION}, "
                f"got {manifest.get('condition')}."
            )
        if int(manifest.get("fold_count", -1)) != self.EXPECTED_FOLDS:
            raise ValueError("QR-Future manifest must declare exactly five folds.")

        folds = manifest.get("folds", {})
        expected_keys = {str(i) for i in range(self.EXPECTED_FOLDS)}
        if set(folds) != expected_keys:
            raise ValueError(
                f"Expected fold keys {sorted(expected_keys)}, got {sorted(folds)}."
            )
        for fold in range(self.EXPECTED_FOLDS):
            item = folds[str(fold)]
            if item.get("condition") != self.EXPECTED_CONDITION:
                raise ValueError(f"Fold {fold}: condition mismatch in manifest.")
            if item.get("input_space") != self.EXPECTED_INPUT_SPACE:
                raise ValueError(f"Fold {fold}: input-space mismatch in manifest.")

        response_meta = manifest.get("response_embeddings", {})
        if response_meta.get("shape") != [
            self.EXPECTED_RESPONSE_ROWS,
            self.EXPECTED_DIMENSION,
        ]:
            raise ValueError(
                "Response embedding shape in manifest does not match "
                f"({self.EXPECTED_RESPONSE_ROWS}, {self.EXPECTED_DIMENSION})."
            )
        if response_meta.get("dtype") != "float32":
            raise ValueError("Response embedding dtype in manifest must be float32.")

    def _check_file(self, path: Path, expected_sha: str, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        if self.verify_hashes:
            actual = sha256_file(path)
            if actual.lower() != str(expected_sha).lower():
                raise ValueError(
                    f"{label} SHA256 mismatch: expected={expected_sha}, actual={actual}"
                )
            print(f"[QRFutureScorer] SHA256 PASS: {label}")

    @staticmethod
    def _validate_checkpoint_metadata(
        checkpoint: dict[str, Any],
        fold: int,
        expected_basis_sha: str,
    ) -> None:
        if checkpoint.get("condition") != QRFutureScorer.EXPECTED_CONDITION:
            raise ValueError(f"Fold {fold}: checkpoint condition mismatch.")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"Fold {fold}: checkpoint identity mismatch.")
        if checkpoint.get("input_space") != QRFutureScorer.EXPECTED_INPUT_SPACE:
            raise ValueError(f"Fold {fold}: checkpoint input-space mismatch.")
        if checkpoint.get("orca_fold") != fold:
            raise ValueError(f"Fold {fold}: checkpoint ORCA fold mismatch.")
        if checkpoint.get("orca_basis_sha256") != expected_basis_sha:
            raise ValueError(f"Fold {fold}: checkpoint/Basis SHA pairing mismatch.")
        if not isinstance(checkpoint.get("model_state_dict"), dict):
            raise ValueError(f"Fold {fold}: model_state_dict is missing or invalid.")

    def _load_response_assets(self) -> None:
        response_meta = self.manifest["response_embeddings"]
        response_path = self.qr_root / response_meta["file"]
        self._check_file(
            response_path,
            response_meta["sha256"],
            "response embeddings",
        )
        response_embeddings = np.load(
            response_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        if response_embeddings.shape != tuple(response_meta["shape"]):
            raise ValueError(
                f"Response embedding shape mismatch: {response_embeddings.shape}."
            )
        if response_embeddings.dtype != np.float32:
            raise ValueError("Response embeddings must be float32.")
        self.response_embeddings = response_embeddings

        row_meta = self.manifest.get("row_mapping", {})
        preferred_path = (
            self.project_root / "data" / "indexes" / "dense_bge_v1"
            / "row_mapping.jsonl"
        )
        manifest_path = Path(row_meta.get("file", ""))
        if preferred_path.is_file():
            row_path = preferred_path
        elif manifest_path.is_file():
            row_path = manifest_path
        else:
            raise FileNotFoundError(
                "Cannot locate row_mapping.jsonl. Checked "
                f"{preferred_path} and {manifest_path}."
            )

        self._check_file(row_path, row_meta["sha256"], "FAISS row mapping")
        rows = []
        with row_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))

        expected_count = int(row_meta["count"])
        if len(rows) != expected_count:
            raise ValueError(
                f"Row mapping count mismatch: {len(rows)} != {expected_count}."
            )

        faiss_to_original = np.full(len(rows), -1, dtype=np.int64)
        seen_uids: set[str] = set()
        for item in rows:
            faiss_row = int(item["faiss_row"])
            original_row = int(item["original_embedding_row"])
            uid = str(item["response_uid"])
            if faiss_row < 0 or faiss_row >= len(rows):
                raise ValueError(f"Invalid FAISS row in mapping: {faiss_row}.")
            if original_row < 0 or original_row >= response_embeddings.shape[0]:
                raise ValueError(f"Invalid original embedding row: {original_row}.")
            if faiss_to_original[faiss_row] != -1:
                raise ValueError(f"Duplicate FAISS row in mapping: {faiss_row}.")
            if uid in seen_uids:
                raise ValueError(f"Duplicate response UID in row mapping: {uid}.")
            faiss_to_original[faiss_row] = original_row
            seen_uids.add(uid)

        if np.any(faiss_to_original < 0):
            raise ValueError("Row mapping contains missing FAISS row positions.")
        if len(np.unique(faiss_to_original)) != len(faiss_to_original):
            raise ValueError("Row mapping contains duplicate original embedding rows.")

        excluded = row_meta.get("excluded_original_row")
        if excluded is not None and int(excluded) in set(faiss_to_original.tolist()):
            raise ValueError("Excluded original embedding row appears in row mapping.")

        self.row_mapping = rows
        self.faiss_to_original = faiss_to_original
        print(
            "[QRFutureScorer] RESPONSE ASSETS PASS "
            f"embeddings={response_embeddings.shape} "
            f"faiss_rows={len(rows)}"
        )

    def response_embeddings_for_faiss_rows(
        self,
        faiss_rows: Sequence[int],
    ) -> np.ndarray:
        if self.response_embeddings is None or self.faiss_to_original is None:
            raise RuntimeError(
                "Response assets were not loaded. Initialize with "
                "load_response_assets=True."
            )
        rows = np.asarray(faiss_rows, dtype=np.int64).reshape(-1)
        if rows.size == 0:
            raise ValueError("faiss_rows cannot be empty.")
        if np.any(rows < 0) or np.any(rows >= len(self.faiss_to_original)):
            raise IndexError("One or more FAISS rows are out of range.")

        original_rows = self.faiss_to_original[rows]
        return np.asarray(
            self.response_embeddings[original_rows],
            dtype=np.float32,
        )

    def score_candidates_detailed(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_chunk_size: int | None = None,
    ) -> dict[str, np.ndarray]:
        query = _as_2d_float32(query_embedding, "query_embedding")
        candidates = _as_2d_float32(candidate_embeddings, "candidate_embeddings")
        if query.shape[0] != 1:
            raise ValueError("This initial scorer API accepts one query at a time.")
        if query.shape[1] != self.EXPECTED_DIMENSION:
            raise ValueError(
                f"Query dimension must be {self.EXPECTED_DIMENSION}, got {query.shape[1]}."
            )
        if candidates.shape[1] != self.EXPECTED_DIMENSION:
            raise ValueError(
                "Candidate dimension must be "
                f"{self.EXPECTED_DIMENSION}, got {candidates.shape[1]}."
            )

        chunk_size = (
            self.candidate_chunk_size
            if candidate_chunk_size is None
            else int(candidate_chunk_size)
        )
        if chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive.")

        fold_scores: list[np.ndarray] = []
        with torch.inference_mode():
            for runtime in self.folds:
                q_projected = runtime.project(query)
                q_tensor = torch.from_numpy(q_projected).to(self.device)

                fold_parts: list[np.ndarray] = []
                for start in range(0, len(candidates), chunk_size):
                    chunk = candidates[start : start + chunk_size]
                    r_projected = runtime.project(chunk)
                    r_tensor = torch.from_numpy(r_projected[None, :, :]).to(
                        self.device
                    )
                    scores = runtime.model.rank_scores(q_tensor, r_tensor)
                    fold_parts.append(
                        scores.reshape(-1).detach().to("cpu").numpy().astype(
                            np.float32, copy=False
                        )
                    )
                fold_scores.append(np.concatenate(fold_parts, axis=0))

        matrix = np.stack(fold_scores, axis=0).astype(np.float32, copy=False)
        ensemble = np.mean(matrix, axis=0, dtype=np.float32)
        return {
            "ensemble_scores": np.asarray(ensemble, dtype=np.float32),
            "fold_scores": matrix,
        }

    def score_candidates(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_chunk_size: int | None = None,
    ) -> np.ndarray:
        return self.score_candidates_detailed(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_chunk_size=candidate_chunk_size,
        )["ensemble_scores"]

    def score_faiss_rows(
        self,
        query_embedding: np.ndarray,
        faiss_rows: Sequence[int],
        candidate_chunk_size: int | None = None,
    ) -> np.ndarray:
        candidates = self.response_embeddings_for_faiss_rows(faiss_rows)
        return self.score_candidates(
            query_embedding,
            candidates,
            candidate_chunk_size=candidate_chunk_size,
        )

    def rerank_results(
        self,
        query_embedding: np.ndarray,
        retrieval_results: Sequence[dict[str, Any]],
        candidate_chunk_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank existing FAISS results while preserving each result's text and UID.

        Each retrieval result must contain the FAISS-local `row` field.
        """
        if not retrieval_results:
            return []
        faiss_rows = [int(item["row"]) for item in retrieval_results]
        scores = self.score_faiss_rows(
            query_embedding,
            faiss_rows,
            candidate_chunk_size=candidate_chunk_size,
        )
        reranked = []
        for item, score in zip(retrieval_results, scores):
            updated = dict(item)
            updated["retrieval_score"] = float(item.get("score", 0.0))
            updated["qr_future_score"] = float(score)
            reranked.append(updated)

        reranked.sort(key=lambda item: item["qr_future_score"], reverse=True)
        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank
        return reranked

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "condition": self.EXPECTED_CONDITION,
            "fold_count": len(self.folds),
            "input_space": self.EXPECTED_INPUT_SPACE,
            "device": str(self.device),
            "response_assets_loaded": self.response_embeddings is not None,
            "row_mapping_loaded": self.faiss_to_original is not None,
        }
