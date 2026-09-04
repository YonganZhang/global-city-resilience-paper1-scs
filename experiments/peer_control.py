"""Real-peer versus scrambled-peer negative-control statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


KEY = ["city", "indicator", "model"]
SCRAMBLE_SEED = 20260717


def scramble_peer_scores(
    r1: pd.DataFrame,
    *,
    city_order: list[str],
    batch_size: int = 10,
    seed: int = SCRAMBLE_SEED,
) -> pd.DataFrame:
    """Rotate displayed peer scores within each indicator/model batch.

    The target city's evidence is unchanged; only the peer-score signal shown
    during R2 is reassigned. Rotation preserves each batch's tier distribution
    and never maps a city's score back to itself.
    """
    required = set(KEY + ["tier"])
    missing = required - set(r1.columns)
    if missing:
        raise ValueError(f"R1 scores missing columns: {sorted(missing)}")
    order = {city: position for position, city in enumerate(city_order)}
    if set(r1["city"]) - set(order):
        raise ValueError("city_order does not cover all R1 cities")
    indicators = {value: i for i, value in enumerate(sorted(r1["indicator"].unique()))}
    models = {value: i for i, value in enumerate(sorted(r1["model"].unique()))}
    frame = r1.copy()
    frame["_position"] = frame["city"].map(order)
    frame["_batch"] = frame["_position"] // batch_size
    pieces = []
    for (indicator, model, batch), group in frame.groupby(
        ["indicator", "model", "_batch"], sort=False
    ):
        group = group.sort_values("_position").copy()
        n = len(group)
        if n < 2:
            raise ValueError("each scrambled batch must contain at least two cities")
        offset = 1 + (
            seed + 7 * indicators[indicator] + 5 * models[model] + 3 * int(batch)
        ) % (n - 1)
        source = group["city"].to_numpy()
        tiers = group["tier"].to_numpy()
        group["peer_source_city"] = np.roll(source, -offset)
        group["displayed_peer_tier"] = np.roll(tiers, -offset)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).drop(columns=["_position", "_batch"])


def pair_peer_arms(r1: pd.DataFrame, real: pd.DataFrame, scrambled: pd.DataFrame) -> pd.DataFrame:
    """Inner-pair identical city-indicator-model rows across all three arms."""
    required = set(KEY + ["score"])
    for label, frame in (("R1", r1), ("real", real), ("scrambled", scrambled)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} missing columns: {sorted(missing)}")
        if frame.duplicated(KEY).any():
            raise ValueError(f"{label} has duplicate {KEY} rows")
    return (
        r1[KEY + ["score"]].rename(columns={"score": "score_r1"})
        .merge(
            real[KEY + ["score"]].rename(columns={"score": "score_real"}),
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
        .merge(
            scrambled[KEY + ["score"]].rename(columns={"score": "score_scrambled"}),
            on=KEY,
            how="inner",
            validate="one_to_one",
        )
        .dropna(subset=["score_r1", "score_real", "score_scrambled"])
    )


def paired_cell_metrics(paired_agents: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-agent SD for each paired city-indicator cell."""
    grouped = paired_agents.groupby(["city", "indicator"], as_index=False)
    return grouped.agg(
        n_paired_models=("model", "size"),
        sd_r1=("score_r1", "std"),
        sd_real=("score_real", "std"),
        sd_scrambled=("score_scrambled", "std"),
    )


def dispersion_reductions(cell_metrics: pd.DataFrame) -> dict[str, float]:
    """Return the manuscript's mean-SD reduction definition for both arms."""
    means = cell_metrics[["sd_r1", "sd_real", "sd_scrambled"]].dropna().mean()
    if means["sd_r1"] <= 0:
        raise ValueError("mean R1 dispersion must be positive")
    return {
        "real_reduction_vs_r1": float(1.0 - means["sd_real"] / means["sd_r1"]),
        "scrambled_reduction_vs_r1": float(
            1.0 - means["sd_scrambled"] / means["sd_r1"]
        ),
    }
