from __future__ import annotations

import numpy as np
import pandas as pd

from gciri.downscale import finalize_benchmark_physical_risk
from gciri.fusion import ABS_6, ALL_18
from gciri.trapezoid import compute_three_capabilities
from gciri.workflow import run_gciri_workflow


def _physical_features(cities: list[str]) -> pd.DataFrame:
    n = len(cities)
    return pd.DataFrame(
        {
            "city": cities,
            "source_iso": ["AAA"] * n,
            "iso_ind": ["AAA"] * n,
            "R_Ph_national_effective": np.linspace(0.2, 0.5, n),
            "denom_src": ["AAA"] * n,
            "n_country_cities": [4] * n,
            "H": np.linspace(0.8, 1.4, n),
            "E": np.linspace(0.9, 1.2, n),
            "hazard_composite_6haz": np.linspace(0.2, 0.5, n),
            "hazard_ref": [0.4] * n,
            "pop_ref": [1_000_000.0] * n,
            "ntl_ref": [10.0] * n,
            "ts_raw": [0.0] * n,
            "ts_w": [0.0] * n,
            "national_risk_override": [""] * n,
        }
    )


def test_phase_quantity_directions() -> None:
    result = compute_three_capabilities(0.4, 0.2, 0.7, 0.5)
    assert result == {
        "F_city": 0.4,
        "R_Ph_city": 0.5,
        "cap_abs": 35.0,
        "cap_resp": 20.0,
        "cap_rec": 70.0,
    }


def test_vulnerability_floor_and_softcap() -> None:
    features = _physical_features(["A"])
    city_f = pd.Series({"A": 0.6})
    national_f = pd.Series({"AAA": 0.1})
    step07, _ = finalize_benchmark_physical_risk(features, city_f, national_f)
    assert step07.loc[0, "V"] == 2.0
    raw = 0.2 * 0.8**0.8 * 0.9**0.3 * 2.0**0.3
    assert np.isclose(step07.loc[0, "r_ph_city"], 1.0 - np.exp(-raw))


def test_workflow_fuses_before_vulnerability_and_uses_cap_rec() -> None:
    cities = ["A", "B", "C", "D"]
    council_rows = []
    for city_index, city in enumerate(cities):
        for indicator in ALL_18:
            council_rows.append(
                {
                    "city": city,
                    "indicator": indicator,
                    "s_parl": 0.55 + 0.05 * city_index,
                }
            )
    council = pd.DataFrame(council_rows)
    metadata = pd.DataFrame({"city": cities, "iso": ["AAA"] * 4})
    national = pd.DataFrame(
        {indicator: [0.8] for indicator in ALL_18}, index=pd.Index(["AAA"], name="iso3cd")
    )
    variance = pd.Series({indicator: 0.05 for indicator in ALL_18})
    result = run_gciri_workflow(
        council,
        variance,
        national,
        variance,
        metadata,
        pd.DataFrame(index=pd.Index([], name="city")),
        _physical_features(cities),
        expected_cities=4,
        assign_four_modes=False,
    )
    fused_abs = result["fused_scores"]
    fused_abs = fused_abs[fused_abs["indicator"].isin(ABS_6)].groupby("city")["s_fused"].mean()
    expected_f = 1.0 - fused_abs
    actual_f = result["physical_risk"].set_index("city")["f_city"]
    pd.testing.assert_series_equal(actual_f.sort_index(), expected_f.sort_index(), check_names=False)
    assert "cap_rec" in result["gciri"].columns
    assert "cap_rest" not in result["gciri"].columns
