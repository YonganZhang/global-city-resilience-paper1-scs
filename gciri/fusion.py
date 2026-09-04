"""Deterministic Paper-1 indicator fusion and downstream computation.

The implementation intentionally separates three concepts that had become
mixed in historical experiment scripts:

* Consul removes anomalous R2 members; retained members are aggregated equally.
* The LLM dispersion proxy is measured before deliberation from R1.
* UCDB benchmark rows are selected by the stable ``(city, country_iso3)`` key,
  after normalization on the full 7,273-city background.

All indicator values passed to the fusion are in resilience polarity.  The
absorption and response aggregates are converted once to deficit polarity
before the trapezoid calculation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from .tiers import TIER_TO_MIDPOINT, compute_resilience_from_tier
from . import trapezoid as tz


ABS_6 = [
    "Building_safety_governance",
    "Urban_ecosystem_health",
    "Urban_income_equity",
    "Urban_safety_and_social_cohesion",
    "Housing_habitability",
    "Infrastructure_quality",
]
RESP_6 = [
    "Urban_mobile_connectivity",
    "Municipal_public_integrity",
    "Local_financial_buffer_capacity",
    "Logistics_performance_index",
    "Local_economic_stability",
    "Local_governance_stability",
]
REC_6 = [
    "Access_to_quality_education",
    "Economic_complexity_index",
    "Government_Effectiveness_Index",
    "Human_Development_Index",
    "Research_Development",
    "Technology_achievement_index",
]
ALL_18 = ABS_6 + RESP_6 + REC_6
DEFICIT_INDICATORS = set(ABS_6) | set(RESP_6)

CITY_TO_NAT_COL = {
    "Infrastructure_quality": "ind__cap_abs__Infrastructure_quality",
    "Building_safety_governance": "ind__cap_abs__Building_quality_control_index",
    "Urban_ecosystem_health": "ind__cap_abs__Ecosystem_vitality",
    "Urban_income_equity": "ind__cap_abs__GINI_Index",
    "Housing_habitability": "ind__cap_abs__Housing_deprivation",
    "Urban_safety_and_social_cohesion": "ind__cap_abs__Global_Peace_Index",
    "Urban_mobile_connectivity": "ind__cap_resp__2G_3G_and_4G_network_coverage",
    "Municipal_public_integrity": "ind__cap_resp__Control_of_corruption",
    "Local_financial_buffer_capacity": "ind__cap_resp__Gross_tiol_Savings",
    "Logistics_performance_index": "ind__cap_resp__Logistics_performance_index",
    "Local_economic_stability": "ind__cap_resp__Macroeconomic_stability",
    "Local_governance_stability": "ind__cap_resp__Political_stability",
    "Access_to_quality_education": "ind__cap_rec__Access_to_quality_education",
    "Economic_complexity_index": "ind__cap_rec__Economic_complexity_index",
    "Government_Effectiveness_Index": "ind__cap_rec__Government_Effectiveness_Index",
    "Human_Development_Index": "ind__cap_rec__Human_Development_Index",
    "Research_Development": "ind__cap_rec__Research_Development",
    "Technology_achievement_index": "ind__cap_rec__Technology_achievement_index",
}

CITY_TO_UCDB_COL = {
    "Infrastructure_quality": "abs_1_infra",
    "Urban_ecosystem_health": "abs_3_eco",
    "Housing_habitability": "abs_5_housing",
    "Urban_mobile_connectivity": "resp_1_network",
    "Logistics_performance_index": "resp_4_logistics",
    "Local_governance_stability": "resp_6_political",
    "Access_to_quality_education": "rest_1_edu",
    "Economic_complexity_index": "rest_2_eci",
    "Human_Development_Index": "rest_4_hdi",
}

R1_VARIANCE_FLOOR = 1e-4
SIGMA2_UCDB = 0.04
LAMBDA_MAX = 3.0


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"{label} has duplicate keys {keys}: {examples}")


def _tier_to_resilience(tier: str) -> float:
    if tier not in TIER_TO_MIDPOINT:
        return np.nan
    return float(compute_resilience_from_tier(tier))


def load_council(consul_path: Path) -> pd.DataFrame:
    """Load the frozen council decisions in resilience polarity."""
    council = pd.read_csv(consul_path)
    _require_columns(council, {"indicator", "city", "tier"}, "Consul output")
    _assert_unique(council, ["indicator", "city"], "Consul output")
    council["s_parl"] = council["tier"].map(_tier_to_resilience)
    if council["s_parl"].isna().any():
        bad = council.loc[council["s_parl"].isna(), "tier"].unique().tolist()
        raise ValueError(f"Consul output contains invalid tiers: {bad}")
    expected = set(ALL_18)
    actual = set(council["indicator"])
    if actual != expected:
        raise ValueError(
            f"Consul indicator set differs from canonical 18: "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return council


def compute_r1_indicator_dispersion(
    r1_path: Path,
    floor: float = R1_VARIANCE_FLOOR,
) -> tuple[pd.Series, pd.DataFrame]:
    """Return the pre-deliberation per-indicator dispersion proxy.

    Sample variance is first calculated across R1 agents for each
    ``(indicator, city)`` cell, then averaged across cities.
    """
    r1 = pd.read_csv(r1_path)
    _require_columns(r1, {"indicator", "city", "model", "tier"}, "R1 output")
    r1 = r1.dropna(subset=["tier"]).copy()
    r1["resilience"] = r1["tier"].map(_tier_to_resilience)
    # Fourteen historical R1 responses from one model used non-canonical
    # labels G/G-/H/H-.  They have no registered midpoint and therefore cannot
    # be silently invented.  The frozen sensitivity analysis treated them as
    # missing; every affected cell still has seven valid independent agents.
    r1 = r1.dropna(subset=["resilience"]).copy()

    per_cell = (
        r1.groupby(["indicator", "city"], as_index=False)["resilience"]
        .agg(n_agents="count", var_city="var")
    )
    if (per_cell["n_agents"] < 2).any():
        bad = per_cell.loc[per_cell["n_agents"] < 2, ["indicator", "city"]]
        raise ValueError(f"R1 cells with fewer than two agents: {bad.to_dict('records')[:10]}")
    sigma2 = (
        per_cell.groupby("indicator")["var_city"]
        .mean()
        .reindex(ALL_18)
        .clip(lower=floor)
    )
    if sigma2.isna().any():
        raise ValueError(
            f"R1 dispersion missing indicators: {sigma2[sigma2.isna()].index.tolist()}"
        )
    detail = per_cell.merge(
        sigma2.rename("sigma2_r1"),
        left_on="indicator",
        right_index=True,
        validate="many_to_one",
    )
    return sigma2, detail


def rebuild_consul_equal_weight(
    r2_path: Path,
    frozen_consul_path: Path,
) -> pd.DataFrame:
    """Recompute the council tiers from frozen drop decisions using equal weights."""
    r2 = pd.read_csv(r2_path)
    frozen = pd.read_csv(frozen_consul_path, keep_default_na=False)
    _require_columns(r2, {"indicator", "city", "model", "tier"}, "R2 output")
    _require_columns(
        frozen,
        {"indicator", "city", "tier", "drop_models", "kept_models"},
        "frozen Consul output",
    )
    _assert_unique(frozen, ["indicator", "city"], "frozen Consul output")

    r2 = r2.dropna(subset=["tier"]).copy()
    r2["resilience"] = r2["tier"].map(_tier_to_resilience)
    if r2["resilience"].isna().any():
        raise ValueError("R2 output contains invalid tiers")

    rows: list[dict] = []
    for record in frozen.to_dict("records"):
        indicator, city = record["indicator"], record["city"]
        cell = r2[(r2["indicator"] == indicator) & (r2["city"] == city)].copy()
        if cell.empty:
            raise ValueError(f"missing R2 cell for {(indicator, city)}")
        dropped = {
            value.strip()
            for value in str(record.get("drop_models", "")).split(",")
            if value.strip()
        }
        kept = cell[~cell["model"].isin(dropped)].copy()
        if kept.empty:
            kept = cell.copy()
        mean_score = float(kept["resilience"].mean())
        mean_score_pct = mean_score * 100.0
        comparison_score = round(mean_score_pct, 10)
        final_tier = min(
            TIER_TO_MIDPOINT,
            key=lambda tier: abs(float(TIER_TO_MIDPOINT[tier]) - comparison_score),
        )
        row = dict(record)
        row["tier"] = final_tier
        row["kept_models"] = ",".join(sorted(kept["model"].astype(str)))
        row["mean_score"] = round(mean_score_pct, 2)
        row["mean_simple"] = round(mean_score_pct, 2)
        row["aggregation"] = "equal_weight_after_consul_drop"
        if "evidence_weights" in row:
            row["evidence_strengths_audit"] = row.pop("evidence_weights")
        rows.append(row)

    rebuilt = pd.DataFrame(rows)
    _assert_unique(rebuilt, ["indicator", "city"], "rebuilt Consul output")
    if len(rebuilt) != 900:
        raise ValueError(f"rebuilt Consul output must have 900 rows, got {len(rebuilt)}")
    return rebuilt


def load_national(
    national_path: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Normalize the full country background and align all indicators to resilience."""
    raw = pd.read_excel(national_path)
    _require_columns(raw, {"iso3cd", *CITY_TO_NAT_COL.values()}, "national prior")
    _assert_unique(raw, ["iso3cd"], "national prior")

    normalized = raw[["iso3cd"]].copy()
    variances: dict[str, float] = {}
    report: list[dict] = []
    for indicator, source_column in CITY_TO_NAT_COL.items():
        values = pd.to_numeric(raw[source_column], errors="coerce")
        lower, upper = float(values.min()), float(values.max())
        if upper - lower > 1e-12:
            unit = (values - lower) / (upper - lower)
        else:
            unit = values * 0.0
        flipped = indicator in DEFICIT_INDICATORS
        resilience = 1.0 - unit if flipped else unit
        normalized[indicator] = resilience
        variances[indicator] = float(resilience.var(ddof=1))
        report.append(
            {
                "indicator": indicator,
                "source_column": source_column,
                "n_countries": int(values.notna().sum()),
                "raw_min": lower,
                "raw_max": upper,
                "polarity_action": "deficit_to_resilience"
                if flipped
                else "keep_resilience",
                "sigma2_national": variances[indicator],
            }
        )
    return (
        normalized.set_index("iso3cd"),
        pd.Series(variances).reindex(ALL_18),
        pd.DataFrame(report),
    )


def load_ucdb_for_benchmark(
    metadata: pd.DataFrame,
    ucdb_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize full UCDB data, then select benchmark rows by city and country."""
    _require_columns(metadata, {"city", "iso"}, "benchmark metadata")
    _assert_unique(metadata, ["city", "iso"], "benchmark metadata")
    raw = pd.read_csv(ucdb_path)
    _require_columns(
        raw,
        {"city", "country_iso3", *CITY_TO_UCDB_COL.values()},
        "UCDB background",
    )
    _assert_unique(raw, ["city", "country_iso3"], "UCDB background")

    normalized = raw[["city", "country_iso3"]].copy()
    report: list[dict] = []
    for indicator, source_column in CITY_TO_UCDB_COL.items():
        values = pd.to_numeric(raw[source_column], errors="coerce")
        lower, upper = float(values.min()), float(values.max())
        if upper - lower > 1e-12:
            unit = (values - lower) / (upper - lower)
        else:
            unit = values * 0.0
        flipped = indicator in DEFICIT_INDICATORS
        normalized[indicator] = 1.0 - unit if flipped else unit
        report.append(
            {
                "indicator": indicator,
                "source_column": source_column,
                "n_cities": int(values.notna().sum()),
                "raw_min": lower,
                "raw_max": upper,
                "polarity_action": "deficit_to_resilience"
                if flipped
                else "keep_resilience",
            }
        )

    identity = metadata[["city", "iso"]].copy()
    identity["ucdb_iso"] = identity["iso"].replace({"HKG": "CHN", "MAC": "CHN"})
    selected = identity.merge(
        normalized.rename(
            columns={
                "city": "source_city",
                "country_iso3": "source_country_iso3",
            }
        ),
        left_on=["city", "ucdb_iso"],
        right_on=["source_city", "source_country_iso3"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = selected[selected["_merge"] != "both"][["city", "iso", "ucdb_iso"]]
    if not unmatched.empty:
        raise ValueError(
            "benchmark city+country keys missing from UCDB: "
            f"{unmatched.to_dict('records')}"
        )
    selected = selected.drop(columns="_merge").set_index("city")
    if len(selected) != len(metadata) or selected.index.nunique() != len(metadata):
        raise ValueError("UCDB benchmark selection is not exactly one row per city")
    return selected, pd.DataFrame(report)


def fuse_scores(
    council: pd.DataFrame,
    sigma2_llm: pd.Series,
    national: pd.DataFrame,
    sigma2_national: pd.Series,
    metadata: pd.DataFrame,
    ucdb: pd.DataFrame,
    sigma2_ucdb: float = SIGMA2_UCDB,
) -> tuple[pd.DataFrame, int]:
    """Fuse the council, national prior, and UCDB by bounded inverse variance."""
    _require_columns(council, {"city", "indicator", "s_parl"}, "council scores")
    city_iso = metadata.set_index("city")["iso"]
    rows: list[dict] = []
    lambda_triggered = 0
    for record in council.to_dict("records"):
        city = record["city"]
        indicator = record["indicator"]
        parliament = float(record["s_parl"])
        iso = city_iso.get(city)
        national_mean = (
            national.loc[iso, indicator]
            if iso in national.index and indicator in national.columns
            else np.nan
        )
        national_base_variance = float(sigma2_national[indicator])
        llm_variance = float(sigma2_llm[indicator])

        inflation = 1.0
        if pd.notna(national_mean) and national_base_variance > 0:
            threshold = 2.0 * np.sqrt(national_base_variance)
            difference = abs(parliament - national_mean)
            if threshold > 0 and difference > threshold:
                inflation = min(LAMBDA_MAX, max(1.0, difference / threshold))
                if inflation > 1.0:
                    lambda_triggered += 1
        national_variance = national_base_variance * inflation

        has_ucdb = (
            indicator in CITY_TO_UCDB_COL
            and city in ucdb.index
            and pd.notna(ucdb.loc[city, indicator])
        )
        ucdb_value = float(ucdb.loc[city, indicator]) if has_ucdb else np.nan
        weight_national = (
            1.0 / national_variance
            if pd.notna(national_mean) and national_variance > 0
            else 0.0
        )
        weight_llm = 1.0 / llm_variance if llm_variance > 0 else 0.0
        weight_ucdb = 1.0 / sigma2_ucdb if has_ucdb else 0.0
        weight_sum = weight_national + weight_llm + weight_ucdb
        if weight_sum <= 0:
            fused = parliament
        else:
            fused = (
                weight_national
                * (float(national_mean) if pd.notna(national_mean) else 0.0)
                + weight_llm * parliament
                + weight_ucdb * (ucdb_value if has_ucdb else 0.0)
            ) / weight_sum
        rows.append(
            {
                "city": city,
                "indicator": indicator,
                "s_parl": parliament,
                "mu_nat": national_mean,
                "u_ucdb": ucdb_value,
                "sigma2_nat_base": national_base_variance,
                "sigma2_nat": national_variance,
                "sigma2_llm": llm_variance,
                "sigma2_ucdb": sigma2_ucdb if has_ucdb else np.nan,
                "lambda": inflation,
                "w_nat_share": weight_national / weight_sum,
                "w_llm_share": weight_llm / weight_sum,
                "w_ucdb_share": weight_ucdb / weight_sum,
                "s_fused": float(fused),
            }
        )
    fused_scores_frame = pd.DataFrame(rows)
    _assert_unique(fused_scores_frame, ["indicator", "city"], "fused scores")
    return fused_scores_frame, lambda_triggered


def compute_downstream(
    fused_scores_frame: pd.DataFrame,
    step07: pd.DataFrame,
    expected_cities: int | None = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute F, the three capabilities, and GCIRI with aligned polarity."""
    _require_columns(
        fused_scores_frame,
        {"city", "indicator", "s_fused"},
        "fused scores",
    )
    _require_columns(step07, {"city", "r_ph_city"}, "Step 07")
    _assert_unique(step07, ["city"], "Step 07")
    pivot = fused_scores_frame.pivot(
        index="city",
        columns="indicator",
        values="s_fused",
    )
    risk = step07.set_index("city")["r_ph_city"]

    rows06: list[dict] = []
    rows08: list[dict] = []
    rows09: list[dict] = []
    for city in pivot.index:
        values = pivot.loc[city]
        if city not in risk.index:
            raise ValueError(f"physical risk missing for {city}")
        absorption_values = values[ABS_6].dropna()
        response_values = values[RESP_6].dropna()
        recovery_values = values[REC_6].dropna()
        if min(
            len(absorption_values),
            len(response_values),
            len(recovery_values),
        ) < 5:
            continue
        absorption_resilience = float(absorption_values.mean())
        response_resilience = float(response_values.mean())
        recovery_resilience = float(recovery_values.mean())
        f_deficit = 1.0 - absorption_resilience
        response_deficit = 1.0 - response_resilience
        r_ph_city = float(risk.loc[city])
        capabilities = tz.compute_three_capabilities(
            f_deficit,
            response_deficit,
            recovery_resilience,
            r_ph_city,
        )
        gciri = tz.compute_trapezoid_giri(
            f_deficit,
            capabilities["cap_resp"],
            capabilities["cap_rec"],
            r_ph_city,
        )
        rows06.append(
            {
                "city": city,
                "f_factor": f_deficit,
                "f_resilience": absorption_resilience,
                "n_indicators_used": len(absorption_values),
                "polarity": "deficit",
            }
        )
        rows08.append(
            {
                "city": city,
                "F": f_deficit,
                "F_resilience": absorption_resilience,
                "response_resilience": response_resilience,
                "recovery_resilience": recovery_resilience,
                "R_Ph_city": r_ph_city,
                "cap_abs": capabilities["cap_abs"],
                "cap_resp": capabilities["cap_resp"],
                "cap_rec": capabilities["cap_rec"],
            }
        )
        rows09.append(
            {
                "city": city,
                "cap_abs": capabilities["cap_abs"],
                "cap_resp": capabilities["cap_resp"],
                "cap_rec": capabilities["cap_rec"],
                "r_ph_city": r_ph_city,
                "f_city": f_deficit,
                "gciri": gciri,
                "source": "gciri_fused_city_model",
            }
        )
    step06 = pd.DataFrame(rows06)
    step08 = pd.DataFrame(rows08)
    step09 = pd.DataFrame(rows09)
    for label, frame in (("Step 06", step06), ("Step 08", step08), ("Step 09", step09)):
        _assert_unique(frame, ["city"], label)
        if expected_cities is not None and len(frame) != expected_cities:
            raise ValueError(
                f"{label} must have {expected_cities} rows, got {len(frame)}"
            )
    return step06, step08, step09


def compute_city_absorption_deficit(
    fused_scores_frame: pd.DataFrame,
    minimum_indicators: int = 5,
) -> pd.Series:
    """Return current city ``F`` from the available absorption indicators.

    The canonical fused run contains all six indicators.  Historical
    single-model ablations may contain five because non-canonical archived
    tier labels are treated as missing; this matches ``compute_downstream``.
    """
    _require_columns(
        fused_scores_frame,
        {"city", "indicator", "s_fused"},
        "fused scores",
    )
    absorption = fused_scores_frame[
        fused_scores_frame["indicator"].isin(ABS_6)
    ].copy()
    counts = absorption.groupby("city")["indicator"].nunique()
    incomplete = counts[counts < minimum_indicators]
    if not incomplete.empty:
        raise ValueError(
            "fused absorption indicators incomplete for cities: "
            f"{incomplete.to_dict()}"
        )
    resilience = absorption.groupby("city")["s_fused"].mean()
    deficit = 1.0 - pd.to_numeric(resilience, errors="coerce")
    if deficit.isna().any() or not deficit.between(0.0, 1.0).all():
        raise ValueError("current city absorption deficit is invalid")
    deficit.name = "f_city"
    return deficit


def compute_national_absorption_deficit(
    national_resilience: pd.DataFrame,
) -> pd.Series:
    """Return national ``F`` on the normalized basis used by the fusion."""
    _require_columns(
        national_resilience.reset_index(),
        set(ABS_6),
        "national resilience",
    )
    deficit = 1.0 - national_resilience[ABS_6].mean(axis=1)
    deficit = pd.to_numeric(deficit, errors="coerce")
    available = deficit.dropna()
    if available.empty or not available.between(0.0, 1.0).all():
        raise ValueError("national absorption deficit is invalid")
    deficit.name = "f_national"
    return deficit


def assign_modes(
    step09: pd.DataFrame,
    benchmark_physical: pd.DataFrame,
) -> pd.DataFrame:
    """Assign the four descriptive modes using the frozen k=4 procedure."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    _require_columns(
        benchmark_physical,
        {"city", "EQ_w", "FL_w", "TC_w", "DR_w", "LS_w", "ts_w"},
        "benchmark physical features",
    )
    _require_columns(
        step09,
        {"city", "f_city", "cap_resp", "cap_rec", "gciri"},
        "Step 09",
    )
    hazard = benchmark_physical.set_index("city")[
        ["EQ_w", "FL_w", "TC_w", "DR_w", "LS_w", "ts_w"]
    ].sum(axis=1)
    frame = step09.set_index("city").copy()
    frame["total_haz"] = hazard.reindex(frame.index)
    frame["total_cap"] = frame.apply(
        lambda row: tz.compute_trapezoid_giri(
            row["f_city"],
            row["cap_resp"],
            row["cap_rec"],
            0.5,
        ),
        axis=1,
    )
    features = frame[["total_haz", "total_cap"]].to_numpy()
    scaler = StandardScaler().fit(features)
    model = KMeans(n_clusters=4, n_init=50, random_state=0).fit(
        scaler.transform(features)
    )
    centers = scaler.inverse_transform(model.cluster_centers_)
    mapping: dict[int, str] = {}
    for label, (hazard_center, capacity_center) in enumerate(centers):
        high_hazard = hazard_center >= np.median(centers[:, 0])
        high_capacity = capacity_center >= np.median(centers[:, 1])
        mapping[label] = (
            "P1"
            if high_hazard and high_capacity
            else "P2"
            if high_hazard
            else "P3"
            if high_capacity
            else "P4"
        )
    frame["cluster"] = [mapping[label] for label in model.labels_]
    frame["cluster_label"] = model.labels_
    return frame.reset_index()


def model_mean_pseudo_council(scores_path: Path) -> pd.DataFrame:
    """Aggregate all available agents in one stage into a pseudo-council."""
    frame = pd.read_csv(scores_path).dropna(subset=["tier"]).copy()
    frame["resilience"] = frame["tier"].map(_tier_to_resilience)
    frame = frame.dropna(subset=["resilience"])
    output = (
        frame.groupby(["city", "indicator"], as_index=False)["resilience"]
        .mean()
        .rename(columns={"resilience": "s_parl"})
    )
    return output


def single_model_pseudo_council(scores_path: Path, model: str) -> pd.DataFrame:
    """Return one R1 agent as a pseudo-council for an ablation run."""
    frame = pd.read_csv(scores_path)
    frame = frame[(frame["model"] == model) & frame["tier"].notna()].copy()
    frame["s_parl"] = frame["tier"].map(_tier_to_resilience)
    output = frame.dropna(subset=["s_parl"])[["city", "indicator", "s_parl"]]
    return output


__all__ = [
    "ABS_6",
    "ALL_18",
    "CITY_TO_NAT_COL",
    "CITY_TO_UCDB_COL",
    "DEFICIT_INDICATORS",
    "REC_6",
    "RESP_6",
    "R1_VARIANCE_FLOOR",
    "SIGMA2_UCDB",
    "assign_modes",
    "compute_downstream",
    "compute_r1_indicator_dispersion",
    "fuse_scores",
    "load_council",
    "load_national",
    "load_ucdb_for_benchmark",
    "model_mean_pseudo_council",
    "rebuild_consul_equal_weight",
    "single_model_pseudo_council",
]
