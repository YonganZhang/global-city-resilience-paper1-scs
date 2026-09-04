# Open physical-risk inputs and global hazard maps

This directory has two deliberately separate release layers:

1. a processed input table for exactly the 50 cities evaluated in the GCIRI
   manuscript; and
2. manifests and retrieval tools for the global hazard probability, intensity
   and susceptibility maps used to construct the hazard inputs.

It does **not** contain a processed 7,273-city table. Landslide uses separate
earthquake-triggered and precipitation-triggered global source layers.

## Processed tables

### `paper_50city_physical_inputs.csv`

The single public data table has 50 rows, one per benchmark city. It contains:

- city identifiers, coordinates and GHS-UCDB identifiers where available;
- raw and normalised values for all six hazards;
- LandScan 2024 population and VIIRS 2025 nighttime-light values;
- hazard, population and nighttime-light reference values;
- the hazard ratio `H`, population ratio, nighttime-light ratio and exposure
  term `E` used by the downscaling calculation;
- the national physical-risk input and any declared study-specific override.

The table contains no LLM score, evidence excerpt, fused indicator,
vulnerability score, city physical-risk result, phase capacity, GCIRI value or
ranking. The 7,273-city background table used to construct reference values is
not published. `pop_total_LS` is the archived LandScan 2024 polygon sum;
`ntl_mean` is the archived VIIRS 2025 polygon mean over the workflow's valid
pixels.

## Hazard definitions

| Code | Hazard layer | Study setting |
|---|---|---|
| `EQ` | Peak ground acceleration | 250-year return period |
| `FL` | River-flood depth, existing climate | 100-year return period; 1979–2016 climate |
| `TC` | Tropical-cyclone wind with climate change | 100-year return period |
| `DR` | Mean SPI-6 drought-event duration | Historical/current climate, approximately 1980–2016 |
| `LS_EQ` | Earthquake-triggered landslide susceptibility | Current conditions |
| `LS_RAIN` | Precipitation-triggered landslide susceptibility | Current climate; rainfall from 1979–2016 |
| `TS` | Coastal tsunami run-up | 475-year return period |

All source URLs, data identifiers, file sizes, spatial specifications and
checksums are recorded in `dataset_manifest.csv` and
`dataset_manifest.json`.

## Transformation

The raster layers, except tsunami, were aggregated to urban centres using
population-weighted means. Each hazard was then transformed to a 0–1 scale:

```text
x = clip(city_value / threshold, 0, 1)
if x <= 0.5: w = 2 * x^2
if x > 0.5:  w = 1 - 2 * (1 - x)^2
```

The thresholds are EQ=400, FL=600, TC=300, DR=20, LS_EQ=4 and LS_RAIN=4.
Tsunami uses the nearest coastal point within 0.5 degrees and a threshold of
15. These are study normalisation settings, not properties of the source
datasets.

The two landslide layers and six hazards are combined as follows:

```text
LS_w = max(LS_EQ_w, LS_rain_w)
hazard_composite_6haz = mean(EQ_w, FL_w, TC_w, DR_w, LS_w, ts_w)
H = city_hazard_composite / country_or_global_hazard_reference
```

## Global source-map retrieval

The seven global files total approximately 7.5 GB. GitHub's ordinary Git
storage is not suitable for multi-gigabyte rasters, so the files remain on the
official GIRI host under CC BY 3.0 IGO. This repository openly provides their
stable URLs, dataset identifiers, spatial metadata, exact byte sizes and
SHA-256 hashes. To download and verify all seven files:

```bash
bash data/hazards/scripts/download_global_hazards.sh
python3 data/hazards/scripts/verify_downloads.py
```

Downloaded rasters are written to `data/hazards/raw/`, which is ignored by
Git.

## Quality checks and known limitations

- The public table has 50 unique city + ISO records and no exact duplicate
  rows.
- Numeric population, nighttime-light, reference and exposure fields are
  present for all 50 cities. Forty-nine population aggregates are positive.
  The archived LandScan polygon aggregation for Malé is zero because of a
  known coverage/matching gap; it is **not** a claim that Malé has zero
  population. The released value is retained unchanged to reproduce the
  frozen study input. The model uses `max(pop_total_LS, 1)` when constructing
  `pop_ratio`, and the frozen code records the study-specific Malé handling.
- All normalised hazard fields are finite and within 0–1. Hazard ratio `H` and
  exposure term `E` are finite and positive.
- All 50 cities have coordinates. No benchmark city carries the
  `ts_coordinate_missing` flag.
- The GIRI earthquake metadata page reports 30 arc-minutes, whereas the source
  GeoTIFF header is approximately 0.005 degrees. The manifest records the file
  header and should be treated as authoritative for computation.
- The flood metadata describes depth without an explicit local unit. No unit
  is inferred in these tables.
- The drought page contains conflicting scenario wording, while its title,
  dates and keywords identify the present/historical period. The public table
  labels it as historical/current climate.
- Tsunami uses nearest-coastal-point sampling; the other hazards use
  population-weighted urban-centre means. Raw values should therefore not be
  compared across hazard types.
- These inputs represent probabilistic hazard intensity or susceptibility.
  They are not observations of event losses, deaths or recovery duration.

The reference values were computed from the full background set, but only the
reference values attached to the 50 paper cities are released. See the
top-level `DATA_AVAILABILITY.md` for source licenses and required attribution.
