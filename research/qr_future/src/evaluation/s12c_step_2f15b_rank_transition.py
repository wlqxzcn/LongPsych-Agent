import numpy as np
from pathlib import Path

P = Path(
    "results/phase_s12c/dev_top100_matched_v1/predictions.npz"
)

OUT = Path(
    "results/phase_s12c/step_2f15b_rank_transition_v1"
)

OUT.mkdir(parents=True, exist_ok=False)

z = np.load(P, allow_pickle=False)

N_BOOT = 5000
SEED = 20261110

CONDITIONS = [
    "DENSE_BGE",
    "RANK_ONLY_FULL_MATCHED",
    "RANK_ONLY_NO_PRODUCT_MATCHED",
    "R_FUTURE_FULL",
    "R_FUTURE_NO_PRODUCT",
    "QR_FUTURE_FULL",
    "QR_FUTURE_NO_PRODUCT",
]

PAIRS = [
    ("R_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "RANK_ONLY_FULL_MATCHED"),
    ("QR_FUTURE_FULL", "R_FUTURE_FULL"),
    ("R_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "RANK_ONLY_NO_PRODUCT_MATCHED"),
    ("QR_FUTURE_NO_PRODUCT", "R_FUTURE_NO_PRODUCT"),
]

SUBSETS = {
    "DEV_ALL": np.ones(9070, dtype=bool),
    "DEV_LONG20": z["dev_long20_mask"].astype(bool),
    "DEV_GOLD_MATCHED": z["gold_match_idx"] >= 0,
    "DEV_GOLD_UNMATCHED": z["gold_match_idx"] < 0,
}

BUCKETS = [
    "RANK_1",
    "RANK_2_5",
    "RANK_6_10",
    "RANK_11_20",
    "RANK_21_101",
]


def bucketize(ranks):
    b = np.full(len(ranks), 4, dtype=np.int8)
    b[ranks <= 20] = 3
    b[ranks <= 10] = 2
    b[ranks <= 5] = 1
    b[ranks == 1] = 0
    return b


def aggregate_dialogues(values, ids):
    unique, inverse = np.unique(
        ids, return_inverse=True
    )

    sums = np.zeros(
        (len(unique), values.shape[1]),
        dtype=np.float64
    )

    counts = np.bincount(
        inverse, minlength=len(unique)
    )

    np.add.at(sums, inverse, values)

    return unique, sums / counts[:, None]


def bootstrap_ci(delta, boot_idx):
    boot = delta[boot_idx].mean(axis=1)

    estimate = delta.mean(axis=0)

    lower, upper = np.quantile(
        boot, [0.025, 0.975], axis=0
    )

    return estimate, lower, upper


ranks = {}

for condition in CONDITIONS:
    key = f"ranks__{condition}"
    ranks[condition] = z[key].astype(np.int32)

    assert ranks[condition].shape == (9070,)
    assert np.all(
        (ranks[condition] >= 1) &
        (ranks[condition] <= 101)
    )

dialogue_ids = z["dialogue_ids"].astype(str)

report = []

def emit(s=""):
    print(s)
    report.append(str(s))


for subset_no, (subset, mask) in enumerate(SUBSETS.items()):

    indices = np.flatnonzero(mask)

    ids = dialogue_ids[indices]

    unique = np.unique(ids)

    rng = np.random.default_rng(SEED + subset_no)

    boot_idx = rng.integers(
        0,
        len(unique),
        size=(N_BOOT, len(unique))
    )

    emit("\n" + "=" * 95)
    emit(subset)
    emit(f"queries={len(indices)} dialogues={len(unique)}")

    grouped = {}
    bucket_indices = {}

    for condition in CONDITIONS:

        b = bucketize(ranks[condition][indices])
        bucket_indices[condition] = b

        onehot = np.eye(5)[b]

        _, grouped[condition] = aggregate_dialogues(
            onehot, ids
        )

        estimate = grouped[condition].mean(axis=0)

        emit("\n" + condition)

        for j, name in enumerate(BUCKETS):
            emit(f"{name}: {estimate[j]:.6f}")

    for pair_no, (new, base) in enumerate(PAIRS):

        emit("\n" + "-" * 85)
        emit(f"TRANSITION: {base} -> {new}")

        old_b = bucket_indices[base]
        new_b = bucket_indices[new]

        # Rows: baseline bucket.
        # Columns: new model bucket.
        matrix = np.zeros((5, 5), dtype=np.int64)

        np.add.at(matrix, (old_b, new_b), 1)

        emit("TRANSITION COUNTS")

        emit(
            "BASE\\NEW       " +
            " ".join(f"{x:>12}" for x in BUCKETS)
        )

        for i, name in enumerate(BUCKETS):
            emit(
                f"{name:<14}" +
                " ".join(
                    f"{matrix[i,j]:>12d}"
                    for j in range(5)
                )
            )

        emit("ROW NORMALIZED (%)")

        row_sums = matrix.sum(axis=1, keepdims=True)

        percentages = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums > 0
        ) * 100

        for i, name in enumerate(BUCKETS):
            emit(
                f"{name:<14}" +
                " ".join(
                    f"{percentages[i,j]:>11.2f}%"
                    for j in range(5)
                )
            )

        improved = int(np.sum(new_b < old_b))
        unchanged = int(np.sum(new_b == old_b))
        deteriorated = int(np.sum(new_b > old_b))

        emit(
            f"BUCKET_TRANSITION: "
            f"improved={improved} "
            f"unchanged={unchanged} "
            f"deteriorated={deteriorated}"
        )

        # Per-query reciprocal-rank changes.
        old_r = ranks[base][indices]
        new_r = ranks[new][indices]

        rr_delta = 1.0 / new_r - 1.0 / old_r

        emit(
            f"RR_QUERY_DELTA: "
            f"mean={rr_delta.mean():+.6f} "
            f"median={np.median(rr_delta):+.6f} "
            f"positive={(rr_delta > 0).sum()} "
            f"zero={(rr_delta == 0).sum()} "
            f"negative={(rr_delta < 0).sum()}"
        )

        # Dialogue-level paired bucket changes.
        delta = grouped[new] - grouped[base]

        estimate, lower, upper = bootstrap_ci(
            delta, boot_idx
        )

        emit("DIALOGUE_BOOTSTRAP_BUCKET_DELTA")

        for j, name in enumerate(BUCKETS):
            emit(
                f"{name}: "
                f"{estimate[j]:+.6f} "
                f"[{lower[j]:+.6f}, {upper[j]:+.6f}]"
            )

OUT_FILE = OUT / "rank_transition_report.txt"

OUT_FILE.write_text(
    "\n".join(report) + "\n",
    encoding="utf-8"
)

print("\nSTEP_2F15B_RANK_TRANSITION_COMPLETE")
print("REPORT =", OUT_FILE)
