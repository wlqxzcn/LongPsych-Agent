import math

import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 1024
FUTURE_DIM = 2048
N_THRESHOLDS = 4


# ============================================================
# Shared ranking architecture
# ============================================================

class SharedRankerMatched(nn.Module):

    def __init__(self, include_product: bool):

        super().__init__()

        self.include_product = bool(include_product)

        self.q_adapter = nn.Linear(
            DIM, DIM, bias=False
        )

        self.r_adapter = nn.Linear(
            DIM, DIM, bias=False
        )

        rank_dim = (
            4 * DIM
            if self.include_product
            else 3 * DIM
        )

        self.rank_head = nn.Linear(
            rank_dim, 1, bias=False
        )

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.zeros_(self.q_adapter.weight)
        nn.init.zeros_(self.r_adapter.weight)
        nn.init.zeros_(self.rank_head.weight)

        # Both architectures share exactly the same
        # initial ranking function:
        #
        # -||z_q - z_r||_1 / sqrt(DIM)

        with torch.no_grad():

            self.rank_head.weight[
                0, 2 * DIM:3 * DIM
            ] = -1.0 / math.sqrt(DIM)

        # In FULL architecture:
        # Product block is initialized to zero,
        # but remains trainable.

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

        features = [
            z_q,
            z_r,
            torch.abs(z_q - z_r),
        ]

        if self.include_product:

            features.append(
                z_q * z_r
            )

        return torch.cat(
            features,
            dim=-1
        )

    def score_from_adapted(self, z_q, z_r):

        features = self.interaction(
            z_q, z_r
        )

        return self.rank_head(
            features
        ).squeeze(-1)


# ============================================================
# Ranking model base
# ============================================================

class _RankOnlyBase(nn.Module):

    RANKER_CLASS = None

    def __init__(self):

        super().__init__()

        self.ranker = self.RANKER_CLASS()

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
            z_r
        )


# ============================================================
# FULL architecture
# ============================================================

class SharedRankerFullMatched(SharedRankerMatched):

    def __init__(self):

        super().__init__(
            include_product=True
        )


class RankOnlyFullMatchedModel(_RankOnlyBase):

    RANKER_CLASS = SharedRankerFullMatched


# ============================================================
# NO-PRODUCT architecture
# ============================================================

class SharedRankerNoProductMatched(SharedRankerMatched):

    def __init__(self):

        super().__init__(
            include_product=False
        )


class RankOnlyNoProductMatchedModel(_RankOnlyBase):

    RANKER_CLASS = SharedRankerNoProductMatched


# ============================================================
# Future prediction base
#
# Future head is created BEFORE the ranker.
# This ensures identical future-head initialization
# between FULL and NO-PRODUCT models under the same seed.
# ============================================================

class _FutureBase(nn.Module):

    RANKER_CLASS = None

    def __init__(self):

        super().__init__()

        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True
        )

        self.ranker = self.RANKER_CLASS()

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
            z_r
        )


# ============================================================
# R-Future
# ============================================================

class RFutureFullMatchedModel(_FutureBase):

    RANKER_CLASS = SharedRankerFullMatched

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
            dim=-1
        )

        return self.future_head(
            future_feature
        )


class RFutureNoProductMatchedModel(_FutureBase):

    RANKER_CLASS = SharedRankerNoProductMatched

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
            dim=-1
        )

        return self.future_head(
            future_feature
        )


# ============================================================
# QR-Future
#
# Product interaction is retained in the auxiliary
# future-prediction branch in BOTH architectures.
# It is removed ONLY from the ranking branch.
# ============================================================

class QRFutureFullMatchedModel(_FutureBase):

    RANKER_CLASS = SharedRankerFullMatched

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

        qr_interaction = z_q * z_r

        future_feature = torch.cat(
            [z_r, qr_interaction],
            dim=-1
        )

        return self.future_head(
            future_feature
        )


class QRFutureNoProductMatchedModel(_FutureBase):

    RANKER_CLASS = SharedRankerNoProductMatched

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

        qr_interaction = z_q * z_r

        future_feature = torch.cat(
            [z_r, qr_interaction],
            dim=-1
        )

        return self.future_head(
            future_feature
        )


# ============================================================
# Ordinal targets
# ============================================================

def ordinal_targets(value_labels):

    thresholds = torch.tensor(
        [-2, -1, 0, 1],
        device=value_labels.device,
        dtype=value_labels.dtype
    )

    return (
        value_labels[:, None]
        > thresholds[None, :]
    ).float()


# ============================================================
# Ranking loss
# ============================================================

def compute_listwise_rank_loss(
    scores,
    gold_position=0
):

    target = torch.full(
        (scores.shape[0],),
        int(gold_position),
        dtype=torch.long,
        device=scores.device
    )

    return F.cross_entropy(
        scores,
        target,
        reduction="mean"
    )


# ============================================================
# Masked future loss
# ============================================================

def compute_masked_future_loss(
    future_logits,
    value_labels,
    eligibility_mask
):

    targets = ordinal_targets(
        value_labels
    )

    per_threshold = F.binary_cross_entropy_with_logits(
        future_logits,
        targets,
        reduction="none"
    )

    per_query = per_threshold.mean(
        dim=1
    )

    mask = eligibility_mask.float()

    # Preserve S7 denominator policy.

    return (
        per_query * mask
    ).mean()
