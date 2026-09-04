# Global City Infrastructure Resilience Index (GCIRI)

[![Tests](https://github.com/YonganZhang/global-city-resilience-paper1-scs/actions/workflows/tests.yml/badge.svg)](https://github.com/YonganZhang/global-city-resilience-paper1-scs/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)
[![Data licenses](https://img.shields.io/badge/data-source%20licenses-green.svg)](DATA_AVAILABILITY.md)

Reproducible model code and open physical-risk inputs for the manuscript
*GCIRI: A globally applicable method for city-scale infrastructure resilience
assessment using a multi-agent framework*, submitted to *Sustainable Cities
and Society*.

GCIRI combines physical-risk downscaling, multi-agent assessment and Bayesian
data fusion to estimate absorption loss, response deficit, recovery capacity
and a city-level infrastructure resilience index.

## What is public

| Material | Release status |
|---|---|
| Core model and experiment code | Public |
| Exact R1, R2 and Consul prompt builders and model specifications | Public |
| Global earthquake, flood, cyclone, drought, landslide and tsunami source maps | Public through official direct-download URLs, manifests and checksums |
| Hazard, LandScan population and VIIRS nighttime-light inputs for the paper cities | Public for exactly 50 cities |
| Processed 7,273-city background table | Not released |
| LLM scores/responses and qualitative evidence excerpts | Not released |
| Fused scores, phase capacities, GCIRI values and rankings | Not released |

The global hazard maps and the 50-city input table are different release
layers. The source maps remain downloadable from the official GIRI host; this
repository does not publish a derived 7,273-city dataset.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest -q
```

The tests use synthetic model inputs and validate the public 50-city table;
they do not require API keys or the 7.5 GB global rasters.

## Repository map

| Path | Contents |
|---|---|
| `gciri/` | Fusion, physical-risk downscaling, phase-capacity and GCIRI calculations |
| `experiments/` | Ablation, peer-control, repeatability, sensitivity and external-ranking utilities |
| `prompts/` | Executable R1, R2 and Consul prompt builders |
| `model_specifications.yml` | Model families, exact model identifiers and request settings |
| `data/hazards/processed/` | The 50-city hazard, population and nighttime-light input table |
| `data/hazards/dataset_manifest.*` | Provenance and direct-download metadata for seven global hazard layers |
| `data/hazards/scripts/` | Checksum-verifiable hazard download utilities |
| `tests/` | Directional, workflow, experiment and public-data checks |

## Open global hazard maps

The manifest covers seven GIRI files: earthquake PGA, river-flood depth,
tropical-cyclone wind, drought duration, earthquake-triggered landslide,
rainfall-triggered landslide and coastal tsunami run-up. Together they are
approximately 7.5 GB and remain under their source data licence.

```bash
bash data/hazards/scripts/download_global_hazards.sh
python data/hazards/scripts/verify_downloads.py
```

Downloads go to the Git-ignored `data/hazards/raw/` directory. See
[`data/hazards/README.md`](data/hazards/README.md) for transformation formulas,
known limitations and field definitions.

## Method contract

The workflow order is part of the released model contract:

```text
R1/R2/Consul assessments
          |
          v
Bayesian indicator fusion
          |
          v
city vulnerability + hazard/exposure downscaling
          |
          v
three phase capacities -> trapezoid GCIRI
```

Fusion is completed before fused absorption indicators are used to construct
vulnerability. The resulting physical risk then enters the phase-capacity and
final-index calculations.

## Data boundary and citation

The 50-city public table contains model inputs, not published resilience
results. One archived input needs special care: the LandScan polygon sum for
Malé is zero because of a coverage/matching gap and is not a real population
estimate. The unchanged value and its frozen computational handling are
documented in the data README.

Original source code is licensed under MIT. Third-party source data and
adapted inputs retain the source licences and attribution requirements listed
in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
