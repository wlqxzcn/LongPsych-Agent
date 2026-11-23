import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.s12c_future_input_source_models import (
    CONDITIONS,
    DIM,
    FUTURE_DIM,
    N_THRESHOLDS,
)

from src.models.s12c_matched_ablation_models import (
    compute_masked_future_loss,
    ordinal_targets,
)

BASE_MODEL = ROOT / "src/models/s12c_matched_ablation_models.py"

EXPECTED_BASE_SHA = (
    "be4db64f5203830e742d9bc078900d106769b4b6380a1de95ca77c542521f99c"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():

    print("=" * 76)
    print("S12C STEP 2F-8B ARCHITECTURE AUDIT")
    print("=" * 76)

    torch.set_num_threads(1)

    # --------------------------------------------------------
    # 1. Existing source protection
    # --------------------------------------------------------

    base_sha = sha256(BASE_MODEL)

    require(
        base_sha == EXPECTED_BASE_SHA,
        f"BASE MODEL HASH MISMATCH: {base_sha}",
    )

    print("BASE_MODEL_HASH = PASS")
    print("BASE_MODEL_SHA256 =", base_sha)

    # --------------------------------------------------------
    # 2. Identical initialization
    # --------------------------------------------------------

    models = {}

    for name, model_class in CONDITIONS.items():

        torch.manual_seed(42)

        model = model_class()
        model.eval()

        models[name] = model

        require(
            model.future_head.weight.shape == (4, 2048),
            f"{name}: Future head weight shape error",
        )

        require(
            model.future_head.bias.shape == (4,),
            f"{name}: Future head bias shape error",
        )

        print(
            name,
            "PARAMETERS =",
            sum(p.numel() for p in model.parameters()),
        )

    state_reference = models["R_ONLY"].state_dict()

    for name, model in models.items():

        state = model.state_dict()

        require(
            set(state) == set(state_reference),
            f"{name}: state dict key mismatch",
        )

        for key in state_reference:

            require(
                torch.equal(state[key], state_reference[key]),
                f"{name}: initialization mismatch: {key}",
            )

    print("IDENTICAL_INITIALIZATION = PASS")
    print("IDENTICAL_PARAMETER_COUNT = PASS")

    # --------------------------------------------------------
    # 3. Synthetic input and feature routing
    # --------------------------------------------------------

    torch.manual_seed(2026)

    batch = 8

    q = torch.randn(batch, DIM)
    r = torch.randn(batch, DIM)
    candidates = torch.randn(batch, 5, DIM)

    with torch.no_grad():

        for name, model in models.items():

            features = model.future_features(q, r)
            logits = model.future_logits(q, r)
            scores = model.rank_scores(q, candidates)

            require(
                features.shape == (batch, FUTURE_DIM),
                f"{name}: feature shape mismatch",
            )

            require(
                logits.shape == (batch, N_THRESHOLDS),
                f"{name}: logits shape mismatch",
            )

            require(
                scores.shape == (batch, 5),
                f"{name}: ranking shape mismatch",
            )

            z_q = model.ranker.adapt_query(q)
            z_r = model.ranker.adapt_response(r)

            zeros = torch.zeros_like(z_q)

            if name == "R_ONLY":

                require(
                    torch.allclose(features[:, :DIM], z_r),
                    "R_ONLY response block mismatch",
                )

                require(
                    torch.equal(features[:, DIM:], zeros),
                    "R_ONLY unused block is not zero",
                )

            elif name == "Q_ONLY":

                require(
                    torch.equal(features[:, :DIM], zeros),
                    "Q_ONLY unused block is not zero",
                )

                require(
                    torch.allclose(features[:, DIM:], z_q),
                    "Q_ONLY query block mismatch",
                )

            elif name == "QR_ADD":

                require(
                    torch.allclose(features[:, :DIM], z_r),
                    "QR_ADD response block mismatch",
                )

                require(
                    torch.allclose(features[:, DIM:], z_q),
                    "QR_ADD query block mismatch",
                )

            print(name, "INPUT_ROUTING = PASS")

    # --------------------------------------------------------
    # 4. Ranking initialization equivalence
    # --------------------------------------------------------

    ranking_outputs = {}

    with torch.no_grad():

        for name, model in models.items():

            ranking_outputs[name] = model.rank_scores(
                q,
                candidates,
            )

    reference = ranking_outputs["R_ONLY"]

    for name, scores in ranking_outputs.items():

        require(
            torch.allclose(
                scores,
                reference,
                atol=1e-7,
                rtol=0,
            ),
            f"{name}: initial ranking mismatch",
        )

    print("INITIAL_RANKING_FUNCTION_MATCH = PASS")

    # --------------------------------------------------------
    # 5. Future gradient routing
    # --------------------------------------------------------

    labels = torch.tensor(
        [-2, -1, 0, 1, 2, -2, 0, 1],
        dtype=torch.float32,
    )

    mask = torch.tensor(
        [1, 1, 0, 1, 0, 1, 1, 1],
        dtype=torch.float32,
    )

    for name, model in models.items():

        model.zero_grad(set_to_none=True)

        logits = model.future_logits(q, r)

        loss = compute_masked_future_loss(
            logits,
            labels,
            mask,
        )

        require(
            torch.isfinite(loss).item(),
            f"{name}: non-finite Future loss",
        )

        loss.backward()

        q_grad = model.ranker.q_adapter.weight.grad
        r_grad = model.ranker.r_adapter.weight.grad

        q_present = (
            q_grad is not None
            and torch.linalg.vector_norm(q_grad).item() > 0
        )

        r_present = (
            r_grad is not None
            and torch.linalg.vector_norm(r_grad).item() > 0
        )

        if name == "R_ONLY":
            require(
                not q_present and r_present,
                "R_ONLY gradient routing incorrect",
            )

        elif name == "Q_ONLY":
            require(
                q_present and not r_present,
                "Q_ONLY gradient routing incorrect",
            )

        elif name == "QR_ADD":
            require(
                q_present and r_present,
                "QR_ADD gradient routing incorrect",
            )

        require(
            model.future_head.weight.grad is not None,
            f"{name}: Future head has no gradient",
        )

        print(
            name,
            "FUTURE_GRADIENT_ROUTING = PASS",
            f"q_grad={q_present}",
            f"r_grad={r_present}",
        )

    # --------------------------------------------------------
    # 6. Frozen loss denominator verification
    # --------------------------------------------------------

    torch.manual_seed(2027)

    test_logits = torch.randn(batch, N_THRESHOLDS)

    actual_loss = compute_masked_future_loss(
        test_logits,
        labels,
        mask,
    )

    targets = ordinal_targets(labels)

    per_threshold = F.binary_cross_entropy_with_logits(
        test_logits,
        targets,
        reduction="none",
    )

    per_query = per_threshold.mean(dim=1)

    expected_loss = (
        per_query * mask.float()
    ).mean()

    require(
        torch.allclose(
            actual_loss,
            expected_loss,
            atol=1e-8,
            rtol=0,
        ),
        "Future loss denominator mismatch",
    )

    print("FUTURE_LOSS_DENOMINATOR = PASS")

    # --------------------------------------------------------
    # 7. Old source unchanged
    # --------------------------------------------------------

    require(
        sha256(BASE_MODEL) == base_sha,
        "Existing source file was modified",
    )

    print("OLD_MODEL_UNMODIFIED = PASS")
    print("DEV_PAYLOAD_USED = NO")
    print("TEST_PAYLOAD_USED = NO")
    print("TRAINING = NO")
    print("CHECKPOINTS_MODIFIED = NO")

    print("=" * 76)
    print("STEP_2F8B_ARCHITECTURE_AUDIT_COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
