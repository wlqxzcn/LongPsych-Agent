import torch
import torch.nn as nn

from src.models.s12c_matched_ablation_models import (
    SharedRankerFullMatched,
)

DIM = 1024
FUTURE_DIM = 2048
N_THRESHOLDS = 4


class _FutureInputSourceBase(nn.Module):

    SOURCE_MODE = None

    def __init__(self):
        super().__init__()

        # Preserve initialization order:
        # Future head first, shared ranker second.
        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True,
        )

        self.ranker = SharedRankerFullMatched()

    def rank_scores(
        self,
        query_embeddings,
        candidate_embeddings,
    ):

        if query_embeddings.ndim != 2:
            raise ValueError("Query must have shape [B, D].")

        if candidate_embeddings.ndim != 3:
            raise ValueError("Candidates must have shape [B, C, D].")

        if query_embeddings.shape[0] != candidate_embeddings.shape[0]:
            raise ValueError("Query/candidate batch mismatch.")

        z_q = self.ranker.adapt_query(
            query_embeddings
        )

        z_r = self.ranker.adapt_response(
            candidate_embeddings
        )

        z_q_expanded = z_q[:, None, :].expand_as(z_r)

        return self.ranker.score_from_adapted(
            z_q_expanded,
            z_r,
        )

    def future_features(
        self,
        query_embeddings,
        gold_response_embeddings,
    ):

        if query_embeddings.ndim != 2:
            raise ValueError("Query must have shape [B, D].")

        if gold_response_embeddings.ndim != 2:
            raise ValueError("Gold response must have shape [B, D].")

        if query_embeddings.shape != gold_response_embeddings.shape:
            raise ValueError("Query/gold shape mismatch.")

        if query_embeddings.shape[-1] != DIM:
            raise ValueError("Unexpected embedding dimension.")

        if self.SOURCE_MODE == "R_ONLY":

            z_r = self.ranker.adapt_response(
                gold_response_embeddings
            )

            zeros = torch.zeros_like(z_r)

            return torch.cat(
                [z_r, zeros],
                dim=-1,
            )

        if self.SOURCE_MODE == "Q_ONLY":

            z_q = self.ranker.adapt_query(
                query_embeddings
            )

            zeros = torch.zeros_like(z_q)

            return torch.cat(
                [zeros, z_q],
                dim=-1,
            )

        if self.SOURCE_MODE == "QR_ADD":

            z_r = self.ranker.adapt_response(
                gold_response_embeddings
            )

            z_q = self.ranker.adapt_query(
                query_embeddings
            )

            return torch.cat(
                [z_r, z_q],
                dim=-1,
            )

        raise RuntimeError(
            f"Unknown source mode: {self.SOURCE_MODE}"
        )

    def future_logits(
        self,
        query_embeddings,
        gold_response_embeddings,
    ):

        features = self.future_features(
            query_embeddings,
            gold_response_embeddings,
        )

        return self.future_head(features)


class ROnlyFullMatchedModel(_FutureInputSourceBase):

    SOURCE_MODE = "R_ONLY"


class QOnlyFullMatchedModel(_FutureInputSourceBase):

    SOURCE_MODE = "Q_ONLY"


class QRAddFullMatchedModel(_FutureInputSourceBase):

    SOURCE_MODE = "QR_ADD"


CONDITIONS = {
    "R_ONLY": ROnlyFullMatchedModel,
    "Q_ONLY": QOnlyFullMatchedModel,
    "QR_ADD": QRAddFullMatchedModel,
}
