# DAVINCI-MONET

A modern, type-safe rewrite of MELODIES-MONET for evaluating atmospheric chemistry and air quality models against observations.

## Quick Start

```bash
# Activate conda environment
conda activate davinci-monet

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy davinci_monet

# Format code
black davinci_monet && isort davinci_monet
```

## Conda Environment

**Name**: `davinci-monet`

**Created with**:
```bash
conda create --name davinci-monet -c conda-forge python=3.11 \
    melodies-monet pydantic mypy pytest pytest-cov black isort
```

**Key packages** (inherited from melodies-monet):
- monet, monetio - atmospheric data I/O
- xarray, numpy, pandas - data structures
- matplotlib, cartopy - plotting
- netCDF4 - file I/O

**Added for development**:
- pydantic - configuration validation
- mypy - static type checking
- pytest, pytest-cov - testing
- black, isort - formatting

## Reference Codebase

**MELODIES-MONET location**: `/Users/fillmore/EarthSystem/MELODIES-MONET`

Key files to reference:
- `melodies_monet/driver.py` - Main logic (3,116 lines) - decompose this
- `melodies_monet/_cli.py` - CLI implementation (1,524 lines)
- `melodies_monet/plots/` - Plotting modules
- `melodies_monet/stats/` - Statistics modules
- `examples/yaml/` - 31 example YAML configs for backward compat testing

## Project Goals

- **Maintainability**: Small, focused modules (<500 lines each)
- **Type Safety**: Full type hints, mypy strict mode
- **Performance**: Parallel processing, lazy loading
- **Extensibility**: Plugin architecture for models/observations/plotters

## Key Design Principles

1. **Uniform Pairing Logic**: Strategy based on data geometry (point, track, profile, swath, grid) not data source

2. **xarray-Only Data Model**: All data as `xr.Dataset` throughout pairing/analysis. Pandas only for I/O adapters and stats output tables.

3. **Synthetic Data for Testing**: Generate test data programmatically - no external dataset dependencies

## Architecture Overview

```
davinci_monet/
├── core/           # Protocols, registry, base classes, exceptions
├── config/         # Pydantic schemas, YAML parsing
├── models/         # Model implementations (CMAQ, WRF-Chem, etc.)
├── observations/   # Observation handlers (surface, aircraft, satellite)
├── pairing/        # Unified pairing engine + strategies
│   └── strategies/ # point, track, profile, swath, grid
├── plots/          # Modular plotting system
│   └── renderers/  # Individual plot types
├── stats/          # Statistics calculation
├── pipeline/       # Execution orchestration
├── io/             # File readers/writers
├── cli/            # Command-line interface
├── logging/        # Structured logging
├── util/           # Shared utilities
└── tests/
    └── synthetic/  # Test data generators
```

## Implementation Status

See `PLAN.md` for detailed 12-phase implementation plan.

**Current Phase**: 1 - Foundation (protocols, registry, exceptions, logging)

## Key Design Patterns

1. **Plugin Registry**: Components register via decorators
   ```python
   @model_registry.register('cmaq')
   class CMAQModel(BaseModel): ...
   ```

2. **Protocol-based Interfaces**: Python Protocols define contracts

3. **Pydantic Configuration**: Type-safe YAML parsing with validation

4. **Pipeline Architecture**: Composable stages replace monolithic methods

## Data Model (xarray-only)

```
Model:   xr.Dataset with dims (time, level, lat, lon)
Point:   xr.Dataset with dims (time, site) + lat/lon coords
Track:   xr.Dataset with dims (time,) + lat/lon/alt coords
Profile: xr.Dataset with dims (time, level) + lat/lon coords
Swath:   xr.Dataset with dims (time, scanline, pixel)
Grid:    xr.Dataset with dims (time, lat, lon)
Paired:  xr.Dataset with aligned model + obs variables
```

## Backward Compatibility

- Full compatibility with existing MELODIES-MONET YAML configuration files
- Continues using monet/monetio libraries for data I/O
