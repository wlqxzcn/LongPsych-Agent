from typing import Any, Optional, Sequence, List, Union

import numpy as np


def normalize_text(x: Any) -> str:
    """Keep preprocessing identical to the original ORCA implementation."""
    import re

    if x is None:
        return ""

    text = str(x).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def l2_normalize(
    x: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


class BGEEncoder:
    """
    BGE-large-zh-v1.5 encoder.

    The SentenceTransformer implementation follows the original
    ORCA Encoder implementation.

    Transformers mean pooling is retained as a fallback.
    """

    def __init__(
        self,
        model_name_or_path: Union[str, Any],
        batch_size: int = 64,
        device: Optional[str] = None,
        expected_dimension: int = 1024,
        show_progress_bar: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = str(model_name_or_path)
        self.batch_size = batch_size
        self.device = device
        self.expected_dimension = expected_dimension
        self.show_progress_bar = show_progress_bar

        self.backend = None
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.dimension = None

        st_error = None

        # Primary backend: SentenceTransformer
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name,
                device=device,
            )

            self.backend = "sentence_transformers"
            self.dimension = int(
                self.model.get_sentence_embedding_dimension()
            )

            self.device = str(self.model.device)

        except Exception as exc:
            st_error = exc
            self.model = None

        # Fallback backend: Transformers + mean pooling
        if self.model is None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self.torch = torch

                if device is None:
                    device = (
                        "cuda"
                        if torch.cuda.is_available()
                        else "cpu"
                    )

                self.device = device

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                )

                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                ).to(device)

                self.model.eval()

                self.dimension = int(
                    self.model.config.hidden_size
                )

                self.backend = "transformers_mean_pooling"

            except Exception as exc:
                raise RuntimeError(
                    "Failed to initialize BGE encoder.\n"
                    f"SentenceTransformer error: {st_error}\n"
                    f"Transformers error: {exc}"
                ) from exc

        if (
            expected_dimension is not None
            and self.dimension != expected_dimension
        ):
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected={expected_dimension}, "
                f"actual={self.dimension}"
            )

        print("[BGEEncoder] INITIALIZED")
        print("  model:", self.model_name)
        print("  backend:", self.backend)
        print("  dimension:", self.dimension)
        print("  device:", self.device)
        print("  batch_size:", self.batch_size)

    def encode(
        self,
        texts: Sequence[str],
        normalize: bool = True,
    ) -> np.ndarray:

        texts = [normalize_text(t) for t in texts]

        if len(texts) == 0:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        # SentenceTransformer: identical to original ORCA.
        if self.backend == "sentence_transformers":

            arr = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=self.show_progress_bar,
            )

            arr = np.asarray(arr, dtype=np.float32)

        # Transformers fallback.
        elif self.backend == "transformers_mean_pooling":

            arrs: List[np.ndarray] = []
            torch = self.torch

            for start in range(
                0, len(texts), self.batch_size
            ):
                batch = texts[
                    start:start + self.batch_size
                ]

                with torch.no_grad():

                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )

                    enc = {
                        k: v.to(self.device)
                        for k, v in enc.items()
                    }

                    out = self.model(**enc)

                    last = out.last_hidden_state

                    mask = (
                        enc["attention_mask"]
                        .unsqueeze(-1)
                        .float()
                    )

                    emb = (
                        (last * mask).sum(dim=1)
                        / mask.sum(dim=1).clamp(min=1e-6)
                    )

                    emb = (
                        emb.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    arrs.append(emb)

            arr = np.vstack(arrs).astype(np.float32)

            if normalize:
                arr = l2_normalize(arr)

        else:
            raise RuntimeError(
                "Encoder backend not initialized."
            )

        arr = np.asarray(arr, dtype=np.float32)

        if arr.ndim != 2:
            raise RuntimeError(
                f"Expected 2D embeddings, got {arr.shape}"
            )

        if arr.shape != (len(texts), self.dimension):
            raise RuntimeError(
                f"Unexpected embedding shape: {arr.shape}"
            )

        if not np.isfinite(arr).all():
            raise RuntimeError(
                "Embedding contains NaN or Inf values."
            )

        return arr

    def encode_one(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:

        return self.encode(
            [text],
            normalize=normalize,
        )[0]
