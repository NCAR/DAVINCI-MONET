# DAVINCI-MONET Validation Plan

This document tracks the human validation of DAVINCI-MONET using real observational datasets. Model data is internal to our institutions; observational data sources are documented below.

## Validation Status

| Category | Reader | Status | Validated By | Date | Notes |
|----------|--------|--------|--------------|------|-------|
| **Surface** | AirNow | Not Started | | | |
| | AQS | Not Started | | | |
| | AERONET | Not Started | | | |
| | OpenAQ | Not Started | | | |
| **Sonde** | Ozonesonde | Not Started | | | |
| **Aircraft** | ICARTT | Not Started | | | |
| **Satellite L2** | TROPOMI | Not Started | | | |
| | TEMPO | Not Started | | | |
| | MODIS | Not Started | | | |
| **Satellite L3** | MOPITT | Not Started | | | |
| | OMPS | Not Started | | | |
| | GOES | Not Started | | | |

**Status Key:**
- Not Started
- In Progress
- Blocked (with reason)
- Complete

---

## Observational Data Sources

### Surface Observations

#### AirNow (EPA Real-Time Air Quality)
- **Variables:** O3, PM2.5, NO2, CO
- **Coverage:** United States, Canada, Mexico
- **Temporal:** Hourly (real-time), daily (historical)
- **Data Portal:** [AirNow API](https://docs.airnowapi.org/)
- **Alternative:** [EPA Outdoor Air Quality Data](https://www.epa.gov/outdoor-air-quality-data)
- **Notes:** Real-time data is preliminary; use AQS for verified historical data

#### AQS (EPA Air Quality System)
- **Variables:** O3, PM2.5, NO2, SO2, CO
- **Coverage:** United States
- **Temporal:** Hourly, daily, annual summaries (1990-present)
- **Data Portal:** [AQS Download Files](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
- **API:** [AQS Data Mart](https://aqs.epa.gov/aqsweb/documents/data_mart_welcome.html)
- **Notes:** Quality-assured data; 6+ month lag from collection

#### AERONET (Aerosol Robotic Network)
- **Variables:** AOD, Angstrom exponent, precipitable water
- **Coverage:** Global network (500+ stations)
- **Temporal:** ~15-minute intervals
- **Data Portal:** [AERONET Data Download Tool](https://aeronet.gsfc.nasa.gov/cgi-bin/webtool_aod_v3)
- **Main Site:** [AERONET Homepage](https://aeronet.gsfc.nasa.gov/)
- **Notes:** Level 2.0 is quality-assured (12+ month delay); Level 1.5 available in near real-time

#### OpenAQ (Global Air Quality Platform)
- **Variables:** O3, PM2.5, PM10, NO2, SO2, CO
- **Coverage:** Global (180+ countries)
- **Temporal:** Varies by source (typically hourly)
- **Data Portal:** [OpenAQ Explorer](https://explore.openaq.org/)
- **API:** [OpenAQ API v3](https://docs.openaq.org/)
- **Notes:** Aggregates data from government and research sources worldwide; API key required

---

### Sonde Observations

#### Ozonesonde (Balloon Profiles)
- **Variables:** O3 vertical profiles, temperature, humidity
- **Coverage:** Global network
- **Temporal:** Weekly launches (varies by station)

**Data Sources:**
- **WOUDC:** [World Ozone and UV Data Centre](https://woudc.org/data/explore.php?dataset=ozonesonde)
- **NOAA GML:** [Boulder Ozonesondes](https://gml.noaa.gov/ozwv/ozsondes/)
- **SHADOZ:** Southern Hemisphere ADditional OZonesondes

**Notes:** WOUDC is the WMO archive; NOAA GML provides U.S. stations

---

### Aircraft Observations

#### ICARTT (NASA/NOAA Flight Campaigns)
- **Variables:** Multiple trace gases (O3, CO, NO2, VOCs, aerosols)
- **Coverage:** Campaign-specific regions
- **Temporal:** Campaign periods

**Data Sources:**
- **ESPO Archive:** [NASA Earth Science Project Office](https://espoarchive.nasa.gov/)
- **Format Specification:** [ICARTT File Format](https://www.earthdata.nasa.gov/esdis/esco/standards-and-practices/icartt-file-format)

**Recent Campaigns:**
- ATom (Atmospheric Tomography)
- FIREX-AQ (Fire Influence on Regional to Global Environments)
- DISCOVER-AQ

**Notes:** Earthdata login required; data in ASCII ICARTT format (.ict files)

---

### Satellite L2 (Swath) Observations

#### TROPOMI (Sentinel-5P)
- **Variables:** NO2, O3, CO, HCHO, SO2, aerosol index
- **Coverage:** Global daily
- **Resolution:** ~5.5 km x 3.5 km (nadir)
- **Data Portal:** [Copernicus Data Space](https://dataspace.copernicus.eu/explore-data/data-collections/sentinel-data/sentinel-5p)
- **Alternative:** [S5P-PAL Data Portal](https://data-portal.s5p-pal.com/)
- **AWS:** [Sentinel-5P on AWS](https://registry.opendata.aws/sentinel5p/)
- **Notes:** Operational since 2018; use OFFL (offline) products for research

#### TEMPO (Tropospheric Emissions Monitoring of Pollution)
- **Variables:** NO2, O3, HCHO
- **Coverage:** North America (hourly daylight)
- **Resolution:** ~10 km² at center of field
- **Data Portal:** [NASA ASDC TEMPO](https://asdc.larc.nasa.gov/project/TEMPO)
- **L2 NO2:** [TEMPO_NO2_L2_V03](https://asdc.larc.nasa.gov/project/TEMPO/TEMPO_NO2_L2_V03)
- **L3 NO2:** [TEMPO_NO2_L3_V01](https://asdc.larc.nasa.gov/project/TEMPO/TEMPO_NO2_L3_V01)
- **Notes:** First geostationary air quality mission; operational since 2023

#### MODIS (Terra/Aqua AOD)
- **Variables:** AOD, Angstrom exponent
- **Coverage:** Global daily
- **Resolution:** 10 km (MOD04_L2), 3 km (MOD04_3K), 1 km (MCD19A2)
- **Data Portal:** [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/search/)
- **Product Info:** [MODIS Aerosol Product](https://modis.gsfc.nasa.gov/data/dataprod/mod04.php)
- **Notes:** Terra (morning), Aqua (afternoon); requires pyhdf for HDF4 files

---

### Satellite L3 (Gridded) Observations

#### MOPITT (Terra CO)
- **Variables:** CO total column and profiles
- **Coverage:** Global daily
- **Resolution:** 22 km x 22 km
- **Data Portal:** [NASA ASDC MOPITT](https://asdc.larc.nasa.gov/project/MOPITT)
- **L3 Daily:** [MOP03J_9](https://asdc.larc.nasa.gov/project/MOPITT/MOP03J_9)
- **Visualization:** [NASA Earth Observations](https://neo.gsfc.nasa.gov/view.php?datasetId=MOP_CO_M)
- **Notes:** Operational since 2000; Version 9 is current

#### OMPS (Suomi-NPP Total Ozone)
- **Variables:** Total column O3, UV aerosol index
- **Coverage:** Global daily
- **Resolution:** 50 km (nadir), 1° gridded
- **Data Portal:** [NASA Ozone Science Team](https://ozoneaq.gsfc.nasa.gov/data/omps/)
- **Earthdata:** [OMPS NRT Data](https://www.earthdata.nasa.gov/data/instruments/omps)
- **L3 Gridded:** [OMPS-NPP L3 Daily](https://catalog.data.gov/dataset/omps-npp-l3-nm-ozone-o3-total-column-1-0-deg-grid-daily-v2)
- **Notes:** Continues TOMS/OMI record; NRT available within 3 hours

#### GOES (GOES-R/S AOD)
- **Variables:** AOD at 550 nm
- **Coverage:** Western Hemisphere
- **Resolution:** 2 km
- **Temporal:** Full disk every 10 min, CONUS every 5 min
- **Data Portal:** [NOAA Data Catalog](https://data.noaa.gov/dataset/dataset/noaa-goes-r-series-advanced-baseline-imager-abi-level-2-aerosol-optical-depth-aod1)
- **Product Info:** [GOES-R AOD](https://www.goes-r.gov/products/baseline-aerosol-opt-depth.html)
- **Notes:** GOES-16 (East), GOES-18 (West); clear-sky only

---

## Validation Procedures

### For Each Reader

1. **Data Acquisition**
   - Download sample dataset from source
   - Document file format and size
   - Note any access requirements (accounts, API keys)

2. **Reader Testing**
   - Load data using DAVINCI-MONET reader
   - Verify coordinates (lat, lon, time)
   - Check variable names and units
   - Validate against source metadata

3. **Pairing Verification**
   - Pair with appropriate model output
   - Check spatial/temporal alignment
   - Verify paired dataset structure

4. **Statistics and Plots**
   - Generate basic statistics (MB, RMSE, R)
   - Create standard plots (scatter, time series)
   - Compare with published evaluation results if available

5. **Documentation**
   - Update status in table above
   - Note any issues or limitations
   - Record software versions used

---

## Known Limitations

### Reader Dependencies

| Reader | Required Package | Notes |
|--------|-----------------|-------|
| MODIS L2 | pyhdf | HDF4/HDF-EOS format |
| All satellite | monetio | Full functionality requires monetio.sat modules |
| TROPOMI | monetio.sat._tropomi_l2_no2_mm | Falls back to basic xarray without |

### Data Access Requirements

| Source | Authentication |
|--------|---------------|
| AirNow API | API key (free registration) |
| OpenAQ | API key (free registration) |
| NASA Earthdata | Earthdata login |
| Copernicus | Copernicus account |

---

## References

- EPA Air Quality Data: https://www.epa.gov/outdoor-air-quality-data
- NASA Earthdata: https://www.earthdata.nasa.gov/
- Copernicus Data Space: https://dataspace.copernicus.eu/
- WOUDC: https://woudc.org/
- NOAA GML: https://gml.noaa.gov/
