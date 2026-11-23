import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 1024
FUTURE_DIM = 2048
N_THRESHOLDS = 4


class SharedRankerNoProduct(nn.Module):
    """
    S12B ranking architecture.

    Ranking features:
        [z_r, abs(z_q - z_r)]

    Explicitly excluded:
        z_q * z_r

    Initialization:
        q_adapter = 0
        r_adapter = 0
        response weights = 0
        abs-difference weights = -1/sqrt(DIM)

    Initial ranking therefore corresponds to negative scaled
    L1 distance, allowing gradients to reach both adapters.
    """

    def __init__(self):
        super().__init__()

        self.q_adapter = nn.Linear(
            DIM, DIM, bias=False
        )

        self.r_adapter = nn.Linear(
            DIM, DIM, bias=False
        )

        self.rank_head = nn.Linear(
            2 * DIM, 1, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.zeros_(self.q_adapter.weight)
        nn.init.zeros_(self.r_adapter.weight)
        nn.init.zeros_(self.rank_head.weight)

        with torch.no_grad():
            self.rank_head.weight[
                0, DIM:2 * DIM
            ] = -1.0 / math.sqrt(DIM)

    def adapt_query(self, e_q):

        z_q = e_q + self.q_adapter(e_q)

        return F.normalize(
            z_q, p=2, dim=-1
        )

    def adapt_response(self, e_r):

        z_r = e_r + self.r_adapter(e_r)

        return F.normalize(
            z_r, p=2, dim=-1
        )

    def interaction(self, z_q, z_r):

        return torch.cat(
            [
                z_r,
                torch.abs(z_q - z_r),
            ],
            dim=-1,
        )

    def score_from_adapted(self, z_q, z_r):

        features = self.interaction(z_q, z_r)

        return self.rank_head(
            features
        ).squeeze(-1)

    def score(self, e_q, e_r):

        z_q = self.adapt_query(e_q)
        z_r = self.adapt_response(e_r)

        return self.score_from_adapted(
            z_q, z_r
        )


class RankOnlyNoProductModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.ranker = SharedRankerNoProduct()

    def rank_scores(
        self,
        query_embeddings,
        candidate_embeddings,
    ):

        z_q = self.ranker.adapt_query(
            query_embeddings
        )

        z_r = self.ranker.adapt_response(
            candidate_embeddings
        )

        z_q_expanded = z_q[:, None, :].expand_as(
            z_r
        )

        return self.ranker.score_from_adapted(
            z_q_expanded,
            z_r,
        )


class RFutureNoRankProductModel(
    RankOnlyNoProductModel
):

    def __init__(self):
        super().__init__()

        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True,
        )

    def future_logits(
        self,
        query_embeddings,
        gold_response_embeddings,
    ):

        del query_embeddings

        z_r = self.ranker.adapt_response(
            gold_response_embeddings
        )

        zeros = torch.zeros_like(z_r)

        future_feature = torch.cat(
            [z_r, zeros],
            dim=-1,
        )

        return self.future_head(future_feature)


class QRFutureAuxNoRankProductModel(
    RankOnlyNoProductModel
):

    def __init__(self):
        super().__init__()

        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True,
        )

    def future_logits(
        self,
        query_embeddings,
        gold_response_embeddings,
    ):

        z_q = self.ranker.adapt_query(
            query_embeddings
        )

        z_r = self.ranker.adapt_response(
            gold_response_embeddings
        )

        # Product interaction is retained ONLY
        # in the auxiliary future prediction branch.
        qr_interaction = z_q * z_r

        future_feature = torch.cat(
            [
                z_r,
                qr_interaction,
            ],
            dim=-1,
        )

        return self.future_head(future_feature)


def ordinal_targets(value_labels):

    thresholds = torch.tensor(
        [-2, -1, 0, 1],
        device=value_labels.device,
        dtype=value_labels.dtype,
    )

    return (
        value_labels[:, None]
        > thresholds[None, :]
    ).float()


def compute_listwise_rank_loss(
    scores,
    gold_position=0,
):

    target = torch.full(
        (scores.shape[0],),
        int(gold_position),
        dtype=torch.long,
        device=scores.device,
    )

    return F.cross_entropy(
        scores,
        target,
        reduction="mean",
    )


def compute_masked_future_loss(
    future_logits,
    value_labels,
    eligibility_mask,
):

    targets = ordinal_targets(value_labels)

    per_threshold = F.binary_cross_entropy_with_logits(
        future_logits,
        targets,
        reduction="none",
    )

    per_query = per_threshold.mean(dim=1)

    mask = eligibility_mask.float()

    # Preserve the original S7 denominator policy.
    return (per_query * mask).mean()
