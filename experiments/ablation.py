"""Ablation summaries used for full/no-Consul/no-R2/no-fusion comparisons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gciri import downscale, fusion


def scores_to_council(scores: pd.DataFrame, model: str | None = None) -> pd.DataFrame:
    """Convert R1/R2 tier rows to an equally weighted pseudo-council."""
    required = {"city", "indicator", "model", "tier"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing columns: {sorted(missing)}")
    frame = scores if model is None else scores[scores["model"] == model]
    frame = frame.copy()
    frame["resilience"] = frame["tier"].map(fusion._tier_to_resilience)
    frame = frame.dropna(subset=["resilience"])
    return (
        frame.groupby(["city", "indicator"], as_index=False)["resilience"]
        .mean()
        .rename(columns={"resilience": "s_parl"})
    )


def compute_configuration(
    council: pd.DataFrame,
    sigma2_llm: pd.Series,
    national_resilience: pd.DataFrame,
    sigma2_national: pd.Series,
    metadata: pd.DataFrame,
    ucdb_resilience: pd.DataFrame,
    benchmark_physical_features: pd.DataFrame,
    *,
    use_fusion: bool,
) -> pd.DataFrame:
    """Compute one internally consistent ablation configuration."""
    if use_fusion:
        scores, _ = fusion.fuse_scores(
            council,
            sigma2_llm,
            national_resilience,
            sigma2_national,
            metadata,
            ucdb_resilience,
        )
    else:
        scores = council[["city", "indicator", "s_parl"]].rename(
            columns={"s_parl": "s_fused"}
        )
    city_f = fusion.compute_city_absorption_deficit(scores)
    national_f = fusion.compute_national_absorption_deficit(national_resilience)
    risk, _ = downscale.finalize_benchmark_physical_risk(
        benchmark_physical_features,
        city_f,
        national_f,
    )
    _, _, output = fusion.compute_downstream(scores, risk, expected_cities=None)
    return output


def run_configurations(
    full_council: pd.DataFrame,
    r1_scores: pd.DataFrame,
    r2_scores: pd.DataFrame,
    sigma2_llm: pd.Series,
    national_resilience: pd.DataFrame,
    sigma2_national: pd.Series,
    metadata: pd.DataFrame,
    ucdb_resilience: pd.DataFrame,
    benchmark_physical_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build full, no-Consul, no-R2, no-fusion and one-agent arms."""
    shared = (
        sigma2_llm,
        national_resilience,
        sigma2_national,
        metadata,
        ucdb_resilience,
        benchmark_physical_features,
    )
    arms = {
        "Full": (full_council, True),
        "No Consul": (scores_to_council(r2_scores), True),
        "No R2 deliberation": (scores_to_council(r1_scores), True),
        "No Bayesian fusion": (full_council, False),
    }
    outputs = []
    for label, (council, use_fusion) in arms.items():
        output = compute_configuration(council, *shared, use_fusion=use_fusion)
        output["configuration"] = label
        outputs.append(output)
    for model in sorted(r1_scores["model"].dropna().unique()):
        council = scores_to_council(r1_scores, model=model)
        output = compute_configuration(council, *shared, use_fusion=True)
        output["configuration"] = "One agent"
        output["model"] = model
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def deviation(candidate: pd.Series, reference: pd.Series) -> dict[str, float]:
    """Return paired absolute-deviation statistics after index alignment."""
    paired = pd.concat([candidate, reference], axis=1, join="inner").dropna()
    if paired.empty:
        raise ValueError("candidate and reference have no paired scores")
    delta = paired.iloc[:, 0].sub(paired.iloc[:, 1])
    return {
        "mean_abs_delta": float(delta.abs().mean()),
        "max_abs_delta": float(delta.abs().max()),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
    }


def normalised_improvement(
    metric: float,
    *,
    full_metric: float,
    one_agent_metric: float,
) -> float:
    """Map one-agent to 0% and the full framework to 100%."""
    span = one_agent_metric - full_metric
    if span <= 0:
        raise ValueError("one-agent metric must exceed the full-framework metric")
    return float(100.0 * (one_agent_metric - metric) / span)


def summarise_configurations(
    scores: pd.DataFrame,
    *,
    city_column: str = "city",
    configuration_column: str = "configuration",
    score_column: str = "gciri",
    full_label: str = "Full",
) -> pd.DataFrame:
    """Summarise each ablation against the paired full-framework scores."""
    required = {city_column, configuration_column, score_column}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    full = scores[scores[configuration_column] == full_label].set_index(city_column)[score_column]
    if full.empty:
        raise ValueError(f"missing full configuration {full_label!r}")
    rows = []
    for label, group in scores.groupby(configuration_column, sort=False):
        stats = deviation(group.set_index(city_column)[score_column], full)
        rows.append({"configuration": label, **stats})
    return pd.DataFrame(rows)
