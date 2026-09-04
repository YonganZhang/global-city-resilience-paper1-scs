"""Rank-correlation, bootstrap and pairwise-concordance utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rank_residual(values: pd.Series, controls: pd.DataFrame) -> np.ndarray:
    """Residualise ranks on an intercept and ranked controls."""
    y = values.rank(method="average").to_numpy(dtype=float)
    ranked = controls.rank(method="average").to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(ranked)), ranked])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def partial_spearman(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    subset = frame[[x, y, *controls]].dropna()
    if len(subset) < len(controls) + 3:
        raise ValueError("not enough complete observations for partial correlation")
    rx = rank_residual(subset[x], subset[controls])
    ry = rank_residual(subset[y], subset[controls])
    return float(stats.pearsonr(rx, ry).statistic)


def pairwise_order_concordance(x: pd.Series, y: pd.Series) -> float:
    """Fraction of untied city pairs ordered in the same direction."""
    paired = pd.concat([x, y], axis=1, join="inner").dropna().to_numpy(dtype=float)
    concordant = 0
    eligible = 0
    for i in range(len(paired)):
        dx = paired[i + 1 :, 0] - paired[i, 0]
        dy = paired[i + 1 :, 1] - paired[i, 1]
        valid = (dx != 0) & (dy != 0)
        concordant += int(np.sum(np.sign(dx[valid]) == np.sign(dy[valid])))
        eligible += int(np.sum(valid))
    if eligible == 0:
        raise ValueError("no untied city pairs")
    return float(concordant / eligible)


def bootstrap_ci(
    frame: pd.DataFrame,
    statistic,
    *,
    n_boot: int = 10_000,
    seed: int = 20260717,
) -> tuple[float, float]:
    """Percentile bootstrap interval using row resampling."""
    if frame.empty:
        raise ValueError("cannot bootstrap an empty frame")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        value = statistic(sample)
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        raise ValueError("all bootstrap statistics were non-finite")
    return tuple(map(float, np.quantile(values, [0.025, 0.975])))
