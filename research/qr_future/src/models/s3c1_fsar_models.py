import torch
import torch.nn as nn
import torch.nn.functional as F


DIM = 1024
INTERACTION_DIM = 4096
FUTURE_DIM = 2048
N_THRESHOLDS = 4


class SharedRanker(nn.Module):

    def __init__(self):
        super().__init__()

        self.q_adapter = nn.Linear(
            DIM,
            DIM,
            bias=False
        )

        self.r_adapter = nn.Linear(
            DIM,
            DIM,
            bias=False
        )

        self.rank_head = nn.Linear(
            INTERACTION_DIM,
            1,
            bias=False
        )

        self.reset_parameters()


    def reset_parameters(self):

        nn.init.zeros_(
            self.q_adapter.weight
        )

        nn.init.zeros_(
            self.r_adapter.weight
        )

        nn.init.zeros_(
            self.rank_head.weight
        )

        # h = [zq, zr, |zq-zr|, zq*zr]
        # Initialize ranking score exactly as dot product.
        with torch.no_grad():

            self.rank_head.weight[
                0,
                3 * DIM : 4 * DIM
            ] = 1.0


    def adapt_query(self, e_q):

        z_q = (
            e_q
            +
            self.q_adapter(e_q)
        )

        return F.normalize(
            z_q,
            p=2,
            dim=-1
        )


    def adapt_response(self, e_r):

        z_r = (
            e_r
            +
            self.r_adapter(e_r)
        )

        return F.normalize(
            z_r,
            p=2,
            dim=-1
        )


    def interaction(self, z_q, z_r):

        return torch.cat(
            [
                z_q,
                z_r,
                torch.abs(z_q - z_r),
                z_q * z_r
            ],
            dim=-1
        )


    def score_from_adapted(
        self,
        z_q,
        z_r
    ):

        h = self.interaction(
            z_q,
            z_r
        )

        return self.rank_head(
            h
        ).squeeze(-1)


    def score(self, e_q, e_r):

        z_q = self.adapt_query(e_q)
        z_r = self.adapt_response(e_r)

        return self.score_from_adapted(
            z_q,
            z_r
        )


class RankOnlyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.ranker = SharedRanker()


    def rank_scores(
        self,
        query_embeddings,
        candidate_embeddings
    ):
        # query_embeddings: [B, D]
        # candidate_embeddings: [B, C, D]

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


class RFutureModel(RankOnlyModel):

    def __init__(self):
        super().__init__()

        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True
        )


    def future_logits(
        self,
        query_embeddings,
        gold_response_embeddings
    ):

        # q intentionally unused by future feature,
        # retained in signature for interface parity.
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


class QRFutureModel(RankOnlyModel):

    def __init__(self):
        super().__init__()

        self.future_head = nn.Linear(
            FUTURE_DIM,
            N_THRESHOLDS,
            bias=True
        )


    def future_logits(
        self,
        query_embeddings,
        gold_response_embeddings
    ):

        z_q = self.ranker.adapt_query(
            query_embeddings
        )

        z_r = self.ranker.adapt_response(
            gold_response_embeddings
        )

        qr_interaction = (
            z_q * z_r
        )

        future_feature = torch.cat(
            [
                z_r,
                qr_interaction
            ],
            dim=-1
        )

        return self.future_head(
            future_feature
        )


def ordinal_targets(
    value_labels
):
    thresholds = torch.tensor(
        [-2, -1, 0, 1],
        device=value_labels.device,
        dtype=value_labels.dtype
    )

    return (
        value_labels[:, None]
        >
        thresholds[None, :]
    ).float()


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

    # Deliberately divide by full query batch,
    # not by the number of labeled examples.
    return (
        per_query * mask
    ).mean()
