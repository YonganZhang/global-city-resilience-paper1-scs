"""Ordered in-memory GCIRI workflow.

The order is part of the model contract: Bayesian indicator fusion precedes
the construction of city vulnerability and physical-risk downscaling.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import downscale, fusion


def run_gciri_workflow(
    council: pd.DataFrame,
    sigma2_llm: pd.Series,
    national_resilience: pd.DataFrame,
    sigma2_national: pd.Series,
    metadata: pd.DataFrame,
    ucdb_resilience: pd.DataFrame,
    benchmark_physical_features: pd.DataFrame,
    *,
    expected_cities: int | None = None,
    assign_four_modes: bool = True,
) -> dict[str, Any]:
    """Run fusion, vulnerability/downscaling, phase quantities and GCIRI.

    Inputs contain no assumptions about local file layout. Indicator values in
    ``council``, ``national_resilience`` and ``ucdb_resilience`` must already
    use resilience polarity (larger is better).
    """
    fused_scores, lambda_triggered = fusion.fuse_scores(
        council,
        sigma2_llm,
        national_resilience,
        sigma2_national,
        metadata,
        ucdb_resilience,
    )
    city_f = fusion.compute_city_absorption_deficit(fused_scores)
    national_f = fusion.compute_national_absorption_deficit(national_resilience)
    physical_risk, physical_audit = downscale.finalize_benchmark_physical_risk(
        benchmark_physical_features,
        city_f,
        national_f,
    )
    absorption, phase_quantities, gciri = fusion.compute_downstream(
        fused_scores,
        physical_risk,
        expected_cities=expected_cities,
    )
    modes = None
    if assign_four_modes:
        if len(gciri) < 4:
            raise ValueError("four-mode assignment requires at least four cities")
        modes = fusion.assign_modes(gciri, physical_audit)
    return {
        "fused_scores": fused_scores,
        "physical_risk": physical_risk,
        "phase_quantities": phase_quantities,
        "gciri": gciri,
        "modes": modes,
        "absorption_deficit": absorption,
        "lambda_triggered": lambda_triggered,
    }
