from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hazards"
TABLE = DATA / "processed" / "paper_50city_physical_inputs.csv"
NORMALISED = (
    "EQ_w",
    "FL_w",
    "TC_w",
    "DR_w",
    "LS_EQ_w",
    "LS_rain_w",
    "LS_w",
    "ts_w",
    "hazard_composite_6haz",
)
FORBIDDEN = {
    "f_city",
    "V",
    "r_ph_city",
    "R_Ph_city",
    "cap_abs",
    "cap_resp",
    "cap_rest",
    "GIRI_city",
    "GIRI_off_nat",
    "s_fused",
    "manual_risk_override",
}


def _rows() -> list[dict[str, str]]:
    with TABLE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_public_table_has_50_city_input_grain() -> None:
    rows = _rows()
    assert len(rows) == 50
    assert len({(row["city"], row["iso"]) for row in rows}) == 50
    assert not (set(rows[0]) & FORBIDDEN)


def test_hazard_population_and_nightlight_inputs_are_complete() -> None:
    rows = _rows()
    required = (
        "pop_total_LS",
        "ntl_mean",
        "hazard_ref",
        "pop_ref",
        "ntl_ref",
        "H",
        "pop_ratio",
        "ntl_ratio",
        "E",
    )
    assert all(row[column] for row in rows for column in required)
    nonpositive_population = [
        row for row in rows if float(row["pop_total_LS"]) <= 0.0
    ]
    assert [(row["city"], row["iso"]) for row in nonpositive_population] == [
        ("Male", "MDV")
    ]
    assert nonpositive_population[0]["pop_ratio"] == "0.0000012666"
    assert all(float(row["ntl_mean"]) >= 0.0 for row in rows)
    assert all(float(row["H"]) > 0.0 and float(row["E"]) > 0.0 for row in rows)


def test_public_hazard_values_are_finite_and_in_range() -> None:
    rows = _rows()
    for column in NORMALISED:
        values = [float(row[column]) for row in rows]
        assert all(0.0 <= value <= 1.0 for value in values)
    assert all(row["ts_coordinate_missing"] == "0" for row in rows)


def test_processed_checksum_matches() -> None:
    expected, relative = (DATA / "PROCESSED_SHA256SUMS.txt").read_text().split()
    payload = (DATA / relative).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == expected
