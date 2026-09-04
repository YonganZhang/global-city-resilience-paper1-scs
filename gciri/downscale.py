"""Deterministic Paper-1 city physical-risk builder.

This module replaces the historical city-name-only handoff used by the
50-city branch.  It selects benchmark records by the stable ``(city, ISO)``
key, samples tsunami at the corrected benchmark coordinates, constructs a
six-hazard full-city background, and applies the documented country-reference
rule:

* country reference when at least three background cities are available;
* otherwise the full-background global reference.

The physical-feature background supplies only hazard and exposure reference
information.  City vulnerability is derived after the current benchmark
fusion has completed, using the current city absorption deficit relative to
the corresponding national GIRI absorption baseline.

The exact HEV exponents are study settings, not published coefficients.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


HAZARD_WEIGHT_COLUMNS = ["EQ_w", "FL_w", "TC_w", "DR_w"]
LANDSLIDE_COLUMNS = ["LS_EQ_w", "LS_rain_w"]
TSUNAMI_P99 = 15.0
TSUNAMI_MAX_DISTANCE_DEG = 0.5
ALPHA_H = 0.8
ALPHA_E = 0.3
ALPHA_V = 0.3
EXPOSURE_POP_WEIGHT = 0.6
EXPOSURE_NTL_WEIGHT = 0.4
MIN_COUNTRY_CITIES = 3
MALE_R_PH_OVERRIDE = 0.95
F_NATIONAL_FLOOR = 0.3
V_MIN = 0.3
V_MAX = 10.0
LEGACY_VULNERABILITY_COLUMNS = [
    "F_nat_raw",
    "F_nat_floored",
    "F_nat_floor_trig",
    "F_city",
    "V",
]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {keys}: {examples}")


def smoothstep(values: pd.Series | np.ndarray) -> np.ndarray:
    """GIRI piecewise-quadratic smoothstep on [0, 1]."""
    x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.5, 2.0 * x * x, 1.0 - 2.0 * (1.0 - x) ** 2)


def select_benchmark_background(
    metadata: pd.DataFrame,
    background: pd.DataFrame,
) -> pd.DataFrame:
    """Select one full-background record per benchmark using ``city + ISO``."""
    _require_columns(metadata, {"city", "iso", "lat", "lon"}, "benchmark metadata")
    _require_columns(background, {"city", "iso"}, "full city background")
    _assert_unique(metadata, ["city", "iso"], "benchmark metadata")
    _assert_unique(background, ["city", "iso"], "full city background")

    identity = metadata[["city", "iso", "lat", "lon"] + [
        c for c in ("ucdb_id",) if c in metadata.columns
    ]].copy()
    source = background.rename(columns={"city": "source_city", "iso": "source_iso"}).copy()
    selected = identity.merge(
        source,
        left_on=["city", "iso"],
        right_on=["source_city", "source_iso"],
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("_benchmark", ""),
    )
    unmatched = selected[selected["_merge"] != "both"][["city", "iso"]]
    if not unmatched.empty:
        raise ValueError(
            "benchmark city+ISO keys missing from full background: "
            f"{unmatched.to_dict('records')}"
        )
    selected = selected.drop(columns="_merge")
    if len(selected) != len(metadata):
        raise ValueError(
            f"benchmark selection changed row count: {len(metadata)} -> {len(selected)}"
        )
    return selected


def apply_benchmark_coordinate_overrides(
    background: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Replace background coordinates only for exact benchmark city+ISO keys."""
    _require_columns(background, {"city", "iso", "lat", "lon"}, "full city background")
    _assert_unique(background, ["city", "iso"], "full city background")
    _assert_unique(metadata, ["city", "iso"], "benchmark metadata")

    corrected = background.copy()
    coordinates = metadata.set_index(["city", "iso"])[["lat", "lon"]]
    index = pd.MultiIndex.from_frame(corrected[["city", "iso"]])
    matched = index.isin(coordinates.index)
    if matched.any():
        replacement = coordinates.reindex(index[matched])
        corrected.loc[matched, "lat"] = replacement["lat"].to_numpy()
        corrected.loc[matched, "lon"] = replacement["lon"].to_numpy()
    return corrected


def add_tsunami_weights(
    background: pd.DataFrame,
    tsunami_path: Path,
) -> pd.DataFrame:
    """Sample nearest tsunami run-up for all background cities."""
    import geopandas as gpd
    from scipy.spatial import cKDTree

    if not tsunami_path.exists():
        raise FileNotFoundError(tsunami_path)
    tsunami = gpd.read_file(tsunami_path)
    if tsunami.crs is not None and not tsunami.crs.is_geographic:
        tsunami = tsunami.to_crs(4326)
    value_column = next(
        (c for c in ("Field3", "value", "run_up", "runup", "VALUE") if c in tsunami.columns),
        None,
    )
    if value_column is None:
        numeric = [
            c for c in tsunami.columns
            if c != "geometry" and pd.api.types.is_numeric_dtype(tsunami[c])
        ]
        if not numeric:
            raise ValueError("tsunami layer has no numeric run-up column")
        value_column = numeric[0]

    points = np.column_stack(
        [tsunami.geometry.y.to_numpy(), tsunami.geometry.x.to_numpy()]
    )
    values = pd.to_numeric(tsunami[value_column], errors="coerce").fillna(0.0).to_numpy()
    tree = cKDTree(points)

    result = background.copy()
    valid = result["lat"].notna() & result["lon"].notna()
    raw = np.zeros(len(result), dtype=float)
    distance = np.full(len(result), np.nan, dtype=float)
    if valid.any():
        dist_valid, index_valid = tree.query(
            np.column_stack(
                [result.loc[valid, "lat"].to_numpy(), result.loc[valid, "lon"].to_numpy()]
            )
        )
        positions = np.flatnonzero(valid.to_numpy())
        distance[positions] = dist_valid
        raw[positions] = np.where(
            dist_valid <= TSUNAMI_MAX_DISTANCE_DEG,
            values[index_valid],
            0.0,
        )
    result["ts_raw"] = raw
    result["ts_distance_deg"] = distance
    result["ts_w"] = smoothstep(raw / TSUNAMI_P99)
    result["ts_coordinate_missing"] = (~valid).astype(int)
    return result


def add_six_hazard_composite(background: pd.DataFrame) -> pd.DataFrame:
    """Add landslide maximum and equal-weight six-hazard composite."""
    required = set(HAZARD_WEIGHT_COLUMNS + LANDSLIDE_COLUMNS + ["ts_w"])
    _require_columns(background, required, "physical background")
    result = background.copy()
    result["LS_w"] = result[LANDSLIDE_COLUMNS].max(axis=1)
    six = HAZARD_WEIGHT_COLUMNS + ["LS_w", "ts_w"]
    result["hazard_composite_6haz"] = result[six].mean(axis=1)
    return result


def _population_weighted_hazard(frame: pd.DataFrame) -> float:
    weights = pd.to_numeric(frame["pop_total_LS"], errors="coerce").fillna(0.0).clip(lower=0)
    values = pd.to_numeric(
        frame["hazard_composite_6haz"], errors="coerce"
    ).fillna(0.0)
    if weights.sum() > 0:
        return float(np.average(values, weights=weights))
    return float(values.mean())


def compute_country_references(
    background: pd.DataFrame,
    min_country_cities: int = MIN_COUNTRY_CITIES,
) -> pd.DataFrame:
    """Compute country references and explicit global fallback records."""
    required = {
        "iso_rph",
        "city",
        "pop_total_LS",
        "ntl_mean",
        "hazard_composite_6haz",
    }
    _require_columns(background, required, "physical background")

    global_hazard = _population_weighted_hazard(background)
    global_population = float(
        pd.to_numeric(background["pop_total_LS"], errors="coerce").mean()
    )
    global_ntl = float(pd.to_numeric(background["ntl_mean"], errors="coerce").mean())

    rows: list[dict] = []
    for iso, group in background.groupby("iso_rph", dropna=False):
        n_cities = int(len(group))
        use_global = pd.isna(iso) or n_cities < min_country_cities
        rows.append(
            {
                "iso_rph": iso,
                "n_country_cities": n_cities,
                "denom_src": "global" if use_global else str(iso),
                "hazard_ref": (
                    global_hazard if use_global else _population_weighted_hazard(group)
                ),
                "pop_ref": (
                    global_population
                    if use_global
                    else float(pd.to_numeric(group["pop_total_LS"], errors="coerce").mean())
                ),
                "ntl_ref": (
                    global_ntl
                    if use_global
                    else float(pd.to_numeric(group["ntl_mean"], errors="coerce").mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def _national_risk_for_iso(background: pd.DataFrame, iso: str) -> float:
    values = pd.to_numeric(
        background.loc[background["iso_rph"] == iso, "R_Ph_national"],
        errors="coerce",
    ).dropna()
    if values.empty:
        raise ValueError(f"no R_Ph_national available for parent ISO {iso}")
    if float(values.max() - values.min()) > 1e-8:
        raise ValueError(f"R_Ph_national is not constant for parent ISO {iso}")
    return float(values.iloc[0])


def build_benchmark_physical_features(
    metadata_path: Path,
    background_path: Path,
    tsunami_path: Path,
    city_state_exceptions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build benchmark hazard/exposure features and their reference table.

    Historical vulnerability fields are removed deliberately.  They were
    produced by an earlier fusion generation and must not enter the current
    benchmark's physical-risk calculation.
    """
    metadata = pd.read_csv(metadata_path)
    background = pd.read_csv(background_path, low_memory=False)
    # The legacy 7,273-city file already contains a four/five-hazard
    # ``denom_src`` column.  It is not the six-hazard reference computed here
    # and must not collide with the new audit field during the merge.
    background = background.drop(columns=["denom_src"], errors="ignore")
    background = apply_benchmark_coordinate_overrides(background, metadata)
    background = add_tsunami_weights(background, tsunami_path)
    background = add_six_hazard_composite(background)
    references = compute_country_references(background)

    benchmark = select_benchmark_background(metadata, background)
    benchmark = benchmark.drop(
        columns=LEGACY_VULNERABILITY_COLUMNS,
        errors="ignore",
    )
    benchmark = benchmark.merge(
        references,
        on="iso_rph",
        how="left",
        validate="many_to_one",
    )
    if benchmark[["hazard_ref", "pop_ref", "ntl_ref"]].isna().any().any():
        raise ValueError("country reference merge produced missing denominators")

    benchmark["H"] = (
        benchmark["hazard_composite_6haz"]
        / benchmark["hazard_ref"].clip(lower=1e-3)
    )
    benchmark["pop_ratio"] = (
        benchmark["pop_total_LS"].clip(lower=1.0)
        / benchmark["pop_ref"].clip(lower=1.0)
    )
    benchmark["ntl_ratio"] = (
        benchmark["ntl_mean"].clip(lower=0.01)
        / benchmark["ntl_ref"].clip(lower=0.01)
    )
    benchmark["E"] = (
        benchmark["pop_ratio"] ** EXPOSURE_POP_WEIGHT
        * benchmark["ntl_ratio"] ** EXPOSURE_NTL_WEIGHT
    )
    benchmark["R_Ph_national_effective"] = benchmark["R_Ph_national"]
    benchmark["national_risk_override"] = ""

    exceptions = yaml.safe_load(
        city_state_exceptions_path.read_text(encoding="utf-8")
    ).get("exceptions", [])
    for exception in exceptions:
        city = exception["city"]
        if "R_Ph_national" not in exception.get("override", []):
            continue
        mask = benchmark["city"] == city
        if not mask.any():
            continue
        parent_iso = exception["parent_iso"]
        parent_value = _national_risk_for_iso(background, parent_iso)
        benchmark.loc[mask, "R_Ph_national_effective"] = parent_value
        benchmark.loc[mask, "national_risk_override"] = f"parent:{parent_iso}"

    if len(benchmark) != len(metadata) or benchmark["city"].nunique() != len(metadata):
        raise ValueError("benchmark physical features are not one row per city")
    return benchmark, references


def finalize_benchmark_physical_risk(
    benchmark_features: pd.DataFrame,
    city_f_deficit: pd.Series,
    national_f_deficit: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive current vulnerability and finalise the benchmark physical risk.

    ``city_f_deficit`` must come from the current 50-city fused indicator
    scores. ``national_f_deficit`` must use the same normalized national GIRI
    indicator basis.  The benchmark's ``iso_ind`` field identifies the
    corresponding national baseline, including the registered Macau-to-HKG
    baseline used by the historical study design.
    """
    required = {
        "city",
        "source_iso",
        "iso_ind",
        "R_Ph_national_effective",
        "denom_src",
        "n_country_cities",
        "H",
        "E",
        "hazard_composite_6haz",
        "hazard_ref",
        "pop_ref",
        "ntl_ref",
        "ts_raw",
        "ts_w",
        "national_risk_override",
    }
    _require_columns(benchmark_features, required, "benchmark physical features")
    _assert_unique(benchmark_features, ["city"], "benchmark physical features")
    if not city_f_deficit.index.is_unique:
        raise ValueError("current city F has duplicate city keys")
    if not national_f_deficit.index.is_unique:
        raise ValueError("national F has duplicate ISO keys")

    benchmark = benchmark_features.copy()
    city_f = pd.to_numeric(
        city_f_deficit.reindex(benchmark["city"]).set_axis(benchmark.index),
        errors="coerce",
    )
    national_f = pd.to_numeric(
        benchmark["iso_ind"].map(national_f_deficit),
        errors="coerce",
    )
    if city_f.isna().any():
        missing = benchmark.loc[city_f.isna(), "city"].tolist()
        raise ValueError(f"current city F missing benchmark cities: {missing}")
    if national_f.isna().any():
        missing = benchmark.loc[
            national_f.isna(),
            ["city", "iso_ind"],
        ].to_dict("records")
        raise ValueError(f"national F missing benchmark baselines: {missing}")
    if not city_f.between(0.0, 1.0).all():
        raise ValueError("current city F contains values outside [0, 1]")
    if not national_f.between(0.0, 1.0).all():
        raise ValueError("national F contains values outside [0, 1]")

    benchmark["f_city"] = city_f
    benchmark["vulnerability_baseline_iso"] = benchmark["iso_ind"]
    benchmark["f_national_raw"] = national_f
    benchmark["f_national_floored"] = national_f.clip(lower=F_NATIONAL_FLOOR)
    benchmark["f_national_floor_triggered"] = (
        national_f < F_NATIONAL_FLOOR
    ).astype(int)
    benchmark["V_raw"] = (
        benchmark["f_city"] / benchmark["f_national_floored"]
    )
    benchmark["V"] = benchmark["V_raw"].clip(lower=V_MIN, upper=V_MAX)

    benchmark["raw_before_softcap"] = (
        benchmark["R_Ph_national_effective"]
        * benchmark["H"] ** ALPHA_H
        * benchmark["E"] ** ALPHA_E
        * benchmark["V"] ** ALPHA_V
    )
    benchmark["r_ph_city"] = 1.0 - np.exp(-benchmark["raw_before_softcap"])
    benchmark["manual_risk_override"] = ""
    male = benchmark["city"] == "Male"
    benchmark.loc[male, "r_ph_city"] = MALE_R_PH_OVERRIDE
    benchmark.loc[male, "manual_risk_override"] = (
        "Male=0.95 frozen study exception"
    )
    benchmark["hardcap_saturated"] = (
        benchmark["raw_before_softcap"] >= 1.0
    ).astype(int)

    if (
        benchmark["r_ph_city"].isna().any()
        or not benchmark["r_ph_city"].between(0, 1).all()
    ):
        raise ValueError("invalid city physical-risk values")
    if benchmark["city"].nunique() != len(benchmark):
        raise ValueError("benchmark physical-risk output is not one row per city")

    step07 = benchmark[
        [
            "city",
            "source_iso",
            "r_ph_city",
            "R_Ph_national_effective",
            "denom_src",
            "n_country_cities",
            "H",
            "E",
            "f_national_raw",
            "f_national_floored",
            "f_national_floor_triggered",
            "V_raw",
            "V",
            "hazard_composite_6haz",
            "hazard_ref",
            "pop_ref",
            "ntl_ref",
            "ts_raw",
            "ts_w",
            "raw_before_softcap",
            "hardcap_saturated",
            "national_risk_override",
            "manual_risk_override",
        ]
    ].rename(
        columns={
            "source_iso": "iso",
            "R_Ph_national_effective": "r_ph_national",
            "raw_before_softcap": "raw_before_cap",
        }
    )
    step07.insert(2, "f_city", benchmark["f_city"].to_numpy())
    step07.insert(
        5,
        "source",
        "full_background_city_iso_mean_current_fusion_v_softcap",
    )
    return step07, benchmark


__all__ = [
    "ALPHA_E",
    "ALPHA_H",
    "ALPHA_V",
    "F_NATIONAL_FLOOR",
    "MIN_COUNTRY_CITIES",
    "V_MAX",
    "V_MIN",
    "add_six_hazard_composite",
    "add_tsunami_weights",
    "apply_benchmark_coordinate_overrides",
    "build_benchmark_physical_features",
    "compute_country_references",
    "finalize_benchmark_physical_risk",
    "select_benchmark_background",
    "smoothstep",
]
