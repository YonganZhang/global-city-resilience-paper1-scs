"""Run-to-run score, rank and mode stability statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score


def compare_runs(
    baseline: pd.DataFrame,
    repeat: pd.DataFrame,
    *,
    key: str = "city",
    score: str = "gciri",
    mode: str = "cluster",
) -> dict[str, float | int]:
    """Compare paired city scores, ranks and P1--P4 assignments."""
    for label, frame in (("baseline", baseline), ("repeat", repeat)):
        missing = {key, score, mode} - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        if frame[key].duplicated().any():
            raise ValueError(f"{label} has duplicate {key!r} values")
    paired = baseline[[key, score, mode]].merge(
        repeat[[key, score, mode]],
        on=key,
        suffixes=("_baseline", "_repeat"),
        validate="one_to_one",
    ).dropna()
    if len(paired) < 2:
        raise ValueError("at least two paired cities are required")
    delta = paired[f"{score}_repeat"] - paired[f"{score}_baseline"]
    rank_a = paired[f"{score}_baseline"].rank(ascending=False, method="average")
    rank_b = paired[f"{score}_repeat"].rank(ascending=False, method="average")
    rank_delta = (rank_b - rank_a).abs()
    return {
        "n": int(len(paired)),
        "pearson_r": float(stats.pearsonr(paired[f"{score}_baseline"], paired[f"{score}_repeat"]).statistic),
        "spearman_rho": float(stats.spearmanr(paired[f"{score}_baseline"], paired[f"{score}_repeat"]).statistic),
        "mean_absolute_difference": float(delta.abs().mean()),
        "median_absolute_rank_change": float(rank_delta.median()),
        "maximum_absolute_rank_change": float(rank_delta.max()),
        "mode_changes": int((paired[f"{mode}_baseline"] != paired[f"{mode}_repeat"]).sum()),
        "adjusted_rand_index": float(adjusted_rand_score(paired[f"{mode}_baseline"], paired[f"{mode}_repeat"])),
    }
