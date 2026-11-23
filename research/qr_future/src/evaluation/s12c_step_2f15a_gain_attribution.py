import numpy as np
from pathlib import Path

ROOT = Path(".")
P = ROOT / "results/phase_s12c/dev_top100_matched_v1/predictions.npz"

z = np.load(P, allow_pickle=False)

N_BOOT = 5000
SEED = 20261109

conditions = [
    "DENSE_BGE",
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

ranks = {
    c: z[f"ranks__{c}"].astype(np.int32)
    for c in conditions
}

dids = z["dialogue_ids"].astype(str)
long_mask = z["dev_long20_mask"].astype(bool)
matched = z["gold_match_idx"] >= 0

assert len(dids) == 9070
assert len(np.unique(dids)) == 239

# Verify Dense ranks independently.
dense_recomputed = 1 + (
    z["candidate_dense_scores"] >=
    z["gold_dense_scores"][:, None]
).sum(axis=1)

assert np.array_equal(
    dense_recomputed, ranks["DENSE_BGE"]
)

print("DENSE_RANK_RECOMPUTATION=PASS")


def metric_matrix(r):
    return np.column_stack([
        1.0 / r,
        r <= 1,
        r <= 5,
        r <= 10,
        r <= 20,
    ]).astype(np.float64)


METRICS = ["MRR", "R@1", "R@5", "R@10", "R@20"]


def aggregate_by_dialogue(values, ids):
    unique, inverse = np.unique(ids, return_inverse=True)

    sums = np.zeros(
        (len(unique), values.shape[1]),
        dtype=np.float64
    )

    counts = np.bincount(
        inverse, minlength=len(unique)
    )

    np.add.at(sums, inverse, values)

    return unique, sums / counts[:, None]


def bootstrap_ci(delta, seed):
    rng = np.random.default_rng(seed)

    n = len(delta)

    draws = rng.integers(
        0, n, size=(N_BOOT, n)
    )

    boot = delta[draws].mean(axis=1)

    estimate = delta.mean(axis=0)

    lower, upper = np.quantile(
        boot, [0.025, 0.975], axis=0
    )

    return estimate, lower, upper


COMPARISONS = [
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
]

INTERACTIONS = [
    (
        "R_FUTURE_PRODUCT_INTERACTION",
        "R_FUTURE_FULL",
        "RANK_ONLY_FULL_MATCHED",
        "R_FUTURE_NO_PRODUCT",
        "RANK_ONLY_NO_PRODUCT_MATCHED",
    ),
    (
        "QR_FUTURE_PRODUCT_INTERACTION",
        "QR_FUTURE_FULL",
        "RANK_ONLY_FULL_MATCHED",
        "QR_FUTURE_NO_PRODUCT",
        "RANK_ONLY_NO_PRODUCT_MATCHED",
    ),
]

SUBSETS = {
    "DEV_ALL": np.ones(9070, dtype=bool),
    "DEV_LONG20": long_mask,
    "DEV_GOLD_MATCHED": matched,
    "DEV_GOLD_UNMATCHED": ~matched,
}


def show_delta(name, delta, seed):
    estimate, lower, upper = bootstrap_ci(delta, seed)

    print("\n", name)

    for j, metric in enumerate(METRICS):
        print(
            f"{metric}: "
            f"{estimate[j]:+.6f} "
            f"[{lower[j]:+.6f}, {upper[j]:+.6f}]"
        )


for subset_no, (subset, mask) in enumerate(SUBSETS.items()):

    indices = np.flatnonzero(mask)
    ids = dids[indices]

    print("\n" + "=" * 90)
    print(subset)
    print("queries =", len(indices))
    print("dialogues =", len(np.unique(ids)))

    grouped = {}

    for c in conditions:
        _, grouped[c] = aggregate_by_dialogue(
            metric_matrix(ranks[c][indices]),
            ids
        )

    seed = SEED + subset_no * 100

    # Paired model comparisons.
    for j, (a, b) in enumerate(COMPARISONS):
        delta = grouped[a] - grouped[b]

        show_delta(
            f"{a} MINUS {b}",
            delta,
            seed + j
        )

    # Difference-in-differences.
    for j, (name, a, b, c, d) in enumerate(INTERACTIONS):

        delta_full = grouped[a] - grouped[b]
        delta_no_product = grouped[c] - grouped[d]

        interaction = delta_full - delta_no_product

        show_delta(
            name,
            interaction,
            seed + 50 + j
        )

    # Query-level win/tie/loss.
    print("\nQUERY_LEVEL_WIN_TIE_LOSS")

    for a, b in COMPARISONS:
        ra = ranks[a][indices]
        rb = ranks[b][indices]

        win = int(np.sum(ra < rb))
        tie = int(np.sum(ra == rb))
        loss = int(np.sum(ra > rb))

        print(
            f"{a} vs {b}: "
            f"win={win} tie={tie} loss={loss}"
        )

print("\nSTEP_2F15A_GAIN_ATTRIBUTION_COMPLETE")
