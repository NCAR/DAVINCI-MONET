# ASIA-AQ Analysis

Model evaluation for the NASA ASIA-AQ (Airborne and Satellite Investigation of Asian Air Quality) campaign.

## Campaign Overview

ASIA-AQ was an international cooperative field study conducted from January-March 2024 to address local air quality challenges across eastern Asia. The campaign deployed multiple aircraft (including NASA's DC-8 and G-III) over four countries: South Korea, Philippines, Taiwan, and Thailand.

**Science Goals:**
- Satellite validation and interpretation (especially GEMS geostationary sensor)
- Emissions quantification and verification
- Model evaluation
- Aerosol and ozone chemistry

**Data Sources:**
- [NASA ESPO ASIA-AQ](https://espo.nasa.gov/asia-aq/)
- [NASA ASDC ASIA-AQ Data](https://asdc.larc.nasa.gov/project/ASIA-AQ)
- [CASEI Campaign Archive](https://impact.earthdata.nasa.gov/casei/campaign/ASIA-AQ)

## Model Data

**Model:** CESM/CAM with FC (full chemistry) configuration, nudged to meteorology

**Case:** `f.e3b06m.FCnudged.t6s.01x01.01`

**Resolution:** 0.1° x 0.1° (450 x 500 grid), 32 vertical levels

**Domain:** 0°-45°N, 90°-140°E (covers Southeast Asia, Korea, Taiwan, S. China, S. Japan)

**Period:** February 1-3, 2024 (hourly output)

**Location:** `~/Data/ASIA-AQ/`

**Key Variables:**
| Variable | Description |
|----------|-------------|
| O3 | Ozone concentration |
| NO | Nitric oxide |
| NO2 | Nitrogen dioxide |
| CO | Carbon monoxide |
| CH2O | Formaldehyde |
| PM25 | PM2.5 concentration |
| AODVISdn | Aerosol optical depth at 550 nm |

## Observation Data

| Type | Source | Status |
|------|--------|--------|
| Aircraft | ICARTT from DC-8/G-III | Pending download |
| Surface | OpenAQ (regional stations) | Pending |
| Satellite | TROPOMI NO2/CO | Pending |
| Satellite | GEMS NO2 (geostationary) | Pending |

## Directory Structure

```
asia-aq/
├── README.md           # This file
├── configs/            # YAML configuration files
├── scripts/            # Analysis Python scripts
└── output/             # Generated plots and statistics
```

## Usage

```bash
# Explore model data
python scripts/explore_model.py

# Run analysis (when configured)
davinci-monet run configs/cesm_aircraft.yaml
```
