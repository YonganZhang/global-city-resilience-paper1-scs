"""Canonical 27-point local sensitivity grid for HEV exponents."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats


CANONICAL = (0.8, 0.3, 0.3)
LOCAL_GRID = tuple(
    itertools.product((0.6, 0.8, 1.0), (0.2, 0.3, 0.4), (0.2, 0.3, 0.4))
)


def downscaled_risk(
    features: pd.DataFrame,
    exponents: tuple[float, float, float],
) -> pd.Series:
    """Recompute soft-capped physical risk for one exponent triple."""
    required = {"r_ph_national", "H", "E", "V"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features missing columns: {sorted(missing)}")
    alpha_h, alpha_e, alpha_v = exponents
    raw = (
        features["r_ph_national"]
        * features["H"].pow(alpha_h)
        * features["E"].pow(alpha_e)
        * features["V"].pow(alpha_v)
    )
    return 1.0 - np.exp(-raw)


def summarise_grid(
    scores: pd.DataFrame,
    *,
    city_column: str = "city",
    score_column: str = "gciri",
) -> pd.DataFrame:
    """Summarise each exponent arm against the canonical arm.

    ``scores`` must contain ``alpha_H``, ``alpha_E`` and ``alpha_V`` columns.
    """
    keys = ["alpha_H", "alpha_E", "alpha_V"]
    missing = set(keys + [city_column, score_column]) - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns: {sorted(missing)}")
    canonical = scores[
        (scores["alpha_H"] == CANONICAL[0])
        & (scores["alpha_E"] == CANONICAL[1])
        & (scores["alpha_V"] == CANONICAL[2])
    ].set_index(city_column)[score_column]
    if canonical.empty:
        raise ValueError("canonical (0.8, 0.3, 0.3) arm is missing")
    rows = []
    for exponents, group in scores.groupby(keys, sort=True):
        paired = pd.concat(
            [canonical.rename("canonical"), group.set_index(city_column)[score_column].rename("candidate")],
            axis=1,
            join="inner",
        ).dropna()
        delta = paired["candidate"] - paired["canonical"]
        rows.append(
            {
                **dict(zip(keys, exponents)),
                "mean_absolute_change": float(delta.abs().mean()),
                "maximum_absolute_change": float(delta.abs().max()),
                "spearman_rho": float(stats.spearmanr(paired["canonical"], paired["candidate"]).statistic),
            }
        )
    return pd.DataFrame(rows)
