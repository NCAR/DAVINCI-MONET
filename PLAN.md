# DAVINCI-MONET Implementation Plan

## Overview

Complete rewrite of MELODIES-MONET addressing:
- `driver.py`: 3,116 lines → ~50 modules (<500 lines each)
- `plotting()`: 1,321-line method → individual plotter classes
- No type hints → Full type hints + mypy strict
- 211 print() → Structured logging
- 4 test files → Comprehensive test suite (>80% coverage)

### Key Design Principles

1. **Uniform Pairing Logic**: A single, consistent pairing engine that works across all observation types (surface, aircraft, satellite, sonde). The pairing strategy is determined by data geometry (point, profile, swath, grid) rather than data source.

2. **Synthetic Data for Testing**: Generate test data programmatically - no dependency on external datasets. Tests are self-contained and reproducible.

3. **xarray-Only Data Model**: All data flows through the system as `xr.Dataset`. No pandas in core pairing/analysis logic.

   | Component | Data Structure |
   |-----------|----------------|
   | Model data | `xr.Dataset` with dims (time, level, lat, lon) |
   | Point obs | `xr.Dataset` with dims (time, site) + lat/lon coords |
   | Track obs | `xr.Dataset` with dims (time,) + lat/lon/alt coords |
   | Profile obs | `xr.Dataset` with dims (time, level) + lat/lon coords |
   | Swath obs | `xr.Dataset` with dims (time, scanline, pixel) |
   | Grid obs | `xr.Dataset` with dims (time, lat, lon) |
   | Paired output | `xr.Dataset` |

   **pandas limited to:**
   - I/O adapters: Read CSV/tabular → convert to xarray
   - Statistics output: Final tables for export

---

## Phase 1: Foundation
**Status: COMPLETE**

- [x] Create package structure and `py.typed`
- [x] Create `pyproject.toml`
- [x] Implement `core/protocols.py` - Protocol definitions (15 protocols)
- [x] Implement `core/registry.py` - Plugin registry system
- [x] Implement `core/exceptions.py` - Custom exception hierarchy (20 exceptions)
- [x] Implement `core/types.py` - Type aliases
- [x] Implement `logging/config.py` - Structured logging

---

## Phase 2: Synthetic Data & Testing Infrastructure
**Status: COMPLETE**

Build test data generators early so all subsequent phases can be tested immediately.

- [x] Implement `tests/synthetic/generators.py` - Base data generators
  - Domain, TimeConfig, VariableSpec dataclasses
  - Coordinate grid, time axis, level axis creation
  - Random field generation with spatial correlation
  - Diurnal cycle and noise functions
- [x] Implement `tests/synthetic/models.py` - Synthetic model output
  - Gridded 3D/4D fields (lat, lon, time, level)
  - Configurable domain, resolution, variables
  - Realistic value ranges for common species (O3, PM2.5, NO2, etc.)
- [x] Implement `tests/synthetic/observations.py` - Synthetic observations
  - Point surface observations (station locations, time series)
  - Aircraft tracks (3D trajectories with measurements)
  - Satellite swaths (scan patterns, footprints)
  - Vertical profiles (sondes)
  - Gridded observations (L3)
- [x] Implement `tests/synthetic/scenarios.py` - Pre-built test scenarios
  - Perfect match (model = obs)
  - Known bias (model = obs + offset)
  - Spatial/temporal mismatch cases
- [x] Implement `tests/conftest.py` - Pytest fixtures using generators
- [x] Comprehensive tests (94 tests for synthetic module)

---

## Phase 3: Configuration
**Status: PENDING**

- [ ] Implement `config/schema.py` - Pydantic models for YAML validation
- [ ] Implement `config/parser.py` - YAML parsing with backward compat
- [ ] Implement `config/migration.py` - Config version migrations
- [ ] Write tests using synthetic data

---

## Phase 4: Core Data Classes
**Status: PENDING**

- [ ] Implement `core/base.py` - BaseModel, BaseObservation, Pair
- [ ] Implement `models/base.py` - Common model logic
- [ ] Implement `observations/base.py` - Common observation logic

---

## Phase 5: Unified Pairing Engine
**Status: PENDING**

A single pairing system based on data geometry, not data source.

### Data Geometries
| Geometry | Model Side | Observation Side | Examples |
|----------|------------|------------------|----------|
| Point-to-Grid | 3D/4D grid | Point locations | Surface stations, ground sites |
| Profile-to-Grid | 3D/4D grid | Vertical profiles | Sondes, aircraft profiles |
| Track-to-Grid | 3D/4D grid | 3D trajectory | Aircraft, mobile platforms |
| Swath-to-Grid | 3D/4D grid | 2D swath pixels | Satellite L2 products |
| Grid-to-Grid | 3D/4D grid | Gridded product | Satellite L3, reanalysis |

### Implementation
- [ ] Implement `pairing/engine.py` - Unified pairing orchestrator
- [ ] Implement `pairing/strategies/point.py` - Point-to-grid matching
- [ ] Implement `pairing/strategies/profile.py` - Profile-to-grid matching
- [ ] Implement `pairing/strategies/track.py` - Track-to-grid matching
- [ ] Implement `pairing/strategies/swath.py` - Swath-to-grid matching
- [ ] Implement `pairing/strategies/grid.py` - Grid-to-grid regridding
- [ ] Implement `pairing/interpolation.py` - Spatial/temporal interpolation
- [ ] Implement `pairing/averaging.py` - Averaging kernels, column integration

### Common Pairing Parameters
All strategies share:
- `radius_of_influence` - Spatial search radius
- `time_tolerance` - Temporal matching window
- `vertical_method` - Interpolation method for vertical
- `horizontal_method` - Nearest, bilinear, etc.

---

## Phase 6: Model Implementations
**Status: PENDING**

Each model reader produces standardized output that feeds into the unified pairing engine.

- [ ] Implement `models/cmaq.py`
- [ ] Implement `models/wrfchem.py`
- [ ] Implement `models/ufs.py`
- [ ] Implement `models/cesm.py`
- [ ] Implement `models/generic.py` - Fallback handler

---

## Phase 7: Observation Implementations
**Status: PENDING**

Each observation reader tags its data with geometry type for the pairing engine.

- [ ] Implement `observations/surface/point.py` - pt_sfc → geometry: point
- [ ] Implement `observations/surface/mobile.py` - mobile → geometry: track
- [ ] Implement `observations/surface/ground.py` - ground → geometry: point
- [ ] Implement `observations/aircraft/icartt.py` - aircraft → geometry: track
- [ ] Implement `observations/satellite/swath.py` - L2 → geometry: swath
- [ ] Implement `observations/satellite/gridded.py` - L3 → geometry: grid
- [ ] Implement `observations/sonde/ozonesonde.py` - sonde → geometry: profile

---

## Phase 8: Pipeline
**Status: PENDING**

- [ ] Implement `pipeline/runner.py` - PipelineRunner (replaces analysis class)
- [ ] Implement `pipeline/stages.py` - Stage definitions
- [ ] Implement `pipeline/parallel.py` - Parallel execution
- [ ] Implement `io/readers.py` - File readers
- [ ] Implement `io/writers.py` - File writers (netCDF, pickle)

---

## Phase 9: Plotting
**Status: PENDING**

- [ ] Implement `plots/base.py` - BasePlotter
- [ ] Implement `plots/registry.py` - Plot registry
- [ ] Implement `plots/renderers/timeseries.py`
- [ ] Implement `plots/renderers/diurnal.py`
- [ ] Implement `plots/renderers/taylor.py`
- [ ] Implement `plots/renderers/boxplot.py`
- [ ] Implement `plots/renderers/scatter.py`
- [ ] Implement `plots/renderers/spatial/bias.py`
- [ ] Implement `plots/renderers/spatial/overlay.py`
- [ ] Implement `plots/renderers/spatial/distribution.py`
- [ ] Implement `plots/renderers/curtain.py`
- [ ] Implement `plots/renderers/scorecard.py`

---

## Phase 10: Statistics
**Status: PENDING**

- [ ] Implement `stats/calculator.py` - Statistics calculator
- [ ] Implement `stats/metrics.py` - Individual metrics (MB, RMSE, etc.)
- [ ] Implement `stats/output.py` - Output formatters

---

## Phase 11: CLI
**Status: PENDING**

- [ ] Implement `cli/app.py` - Main Typer application
- [ ] Implement `cli/commands/run.py` - Run command
- [ ] Implement `cli/commands/get_data.py` - Data download commands
- [ ] Implement `cli/commands/validate.py` - Config validation

---

## Phase 12: Documentation & Polish
**Status: PENDING**

- [ ] Integration tests with full synthetic scenarios
- [ ] API documentation
- [ ] Migration guide from MELODIES-MONET
- [ ] Example notebooks

---

## Technical Decisions

| Aspect | Decision |
|--------|----------|
| Python version | 3.10+ |
| Type checking | mypy strict mode |
| Validation | Pydantic v2 |
| Logging | Python `logging` module |
| CLI | Typer |
| Testing | pytest + pytest-cov + synthetic data |
| Formatting | Black + isort |
| Data model | xarray-only (pandas for I/O adapters and stats tables only) |
| Dependencies | Continue using monet/monetio |

---

## Unified Pairing Architecture

**All data as xarray.Dataset throughout the pipeline.**

```
┌─────────────────┐     ┌─────────────────┐
│  Model Reader   │     │   Obs Reader    │
│  (NetCDF, etc)  │     │ (NetCDF, CSV→xr)│
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ xr.Dataset      │     │ xr.Dataset      │
│ dims: (time,    │     │ dims: varies by │
│   level, lat,   │     │   geometry type │
│   lon)          │     │ attrs: geometry │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   Pairing Engine      │
         │   (xarray operations) │
         │   ─────────────────   │
         │   Selects strategy    │
         │   based on geometry   │
         └───────────┬───────────┘
                     │
         ┌───────────┴────────────┐
         │   Strategy dispatch    │
         ├────────────────────────┤
         │ point   → PointPairer  │
         │ track   → TrackPairer  │
         │ profile → ProfilePairer│
         │ swath   → SwathPairer  │
         │ grid    → GridPairer   │
         └───────────┬────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   Paired xr.Dataset   │
         │   (model + obs vars   │
         │    aligned on coords) │
         └───────────────────────┘
```

**Observation Dimension Conventions:**
```
Point:   (time, site)         - coords: lat(site), lon(site)
Track:   (time,)              - coords: lat(time), lon(time), alt(time)
Profile: (time, level)        - coords: lat(time), lon(time)
Swath:   (time, scanline, pixel) - coords: lat(...), lon(...)
Grid:    (time, lat, lon)     - regular grid
```

---

## Files to Reference (MELODIES-MONET)

| File | Purpose |
|------|---------|
| `melodies_monet/driver.py` | Main logic to decompose (3,116 lines) |
| `melodies_monet/_cli.py` | CLI implementation (1,524 lines) |
| `melodies_monet/plots/` | Plotting modules |
| `melodies_monet/stats/` | Statistics modules |
| `examples/yaml/` | 31 example YAML configs for testing |
