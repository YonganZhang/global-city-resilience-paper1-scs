# Data availability and licensing

## Public data in this repository

The `data/hazards/processed/` directory contains one 50-row model-input table,
`paper_50city_physical_inputs.csv`. It provides the hazard, LandScan 2024
population and VIIRS 2025 nighttime-light inputs used for the manuscript benchmark.
It also provides the country/global reference values, hazard ratio `H`,
population and nighttime-light ratios, and exposure term `E` used by the
physical-risk downscaling.

The table deliberately excludes the 7,273-city background records, LLM
assessments, evidence excerpts, fused indicator scores, vulnerability derived
from those scores, phase capacities, final GCIRI values and city rankings.

## Underlying sources

| Source | Role | Terms and attribution |
|---|---|---|
| CDRI/UNEP-GRID GIRI | Earthquake, flood, tropical-cyclone, drought, landslide and tsunami hazard layers; national physical-risk prior | [CC BY 3.0 IGO and required CDRI citation](https://giri.unepgrid.ch/form/subscribe-for-download-data) |
| European Commission GHSL / GHS-UCDB R2024A | Urban-centre definitions and selected city locations | [CC BY 4.0](https://human-settlement.emergency.copernicus.eu/GHSLhowToCite.php); [R2024A product page](https://human-settlement.emergency.copernicus.eu/ghs_ucdb_2024.php) |
| Oak Ridge National Laboratory LandScan | City population, population reference values and population weighting used to aggregate raster hazards | [Creative Commons public release](https://www.ornl.gov/news/public-release-ornl-global-population-distribution-data-aids-humanitarian-support) |
| Earth Observation Group VIIRS Nighttime Lights | City nighttime-light values and reference values used in the exposure term | [Product licensing and citation guidance](https://eogdata.mines.edu/products/vnl/) |

The repository's MIT License applies to original source code only. It does not
replace the licenses of third-party data. The 50-city input table is an
adaptation of the cited sources and must be reused with those source
attributions. Changes made by this study include city-level aggregation,
normalisation, landslide-layer combination and construction of hazard,
population and nighttime-light reference values.

Recommended GIRI citation:

> CDRI (2023). *Global Infrastructure Resilience: Capturing the resilience
> dividend—A Biennial Report from the Coalition for Disaster Resilient
> Infrastructure*. https://doi.org/10.59375/biennialreport.ed1

## Raw hazard data

The seven open global source maps total approximately 7.5 GB and are not
mirrored in ordinary Git storage. `data/hazards/dataset_manifest.csv` records
their official direct-download URLs, identifiers, spatial specifications,
exact byte sizes and SHA-256 checksums. The download and verification utilities
retrieve the files directly from the official GIRI host. This provides public,
checksum-verifiable access without publishing the derived 7,273-city table.

The official GIRI download page places the available metrics, including
tsunami, under CC BY 3.0 IGO. The locally downloaded tsunami metadata bundle
contained an incomplete license file, so this repository relies on the current
official GIRI license page and does not redistribute the raw GeoPackage.

The Malé record has an archived LandScan polygon sum of zero due to a known
coverage/matching gap. It must not be interpreted as a demographic estimate.
The value is published unchanged because it is the frozen computational input;
the repository documents and tests the special case explicitly.

## Data not released

Qualitative evidence excerpts and complete agent responses may contain
third-party text. External city-ranking datasets retain their publishers'
terms. These materials, intermediate model outputs and final city-level
resilience results are not included here and remain available from the
corresponding authors subject to their source conditions.
