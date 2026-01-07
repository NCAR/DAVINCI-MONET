# DAVINCI-MONET

**Diagnostic Analysis and Validation Infrastructure for Numerical Chemistry Investigation - Model and ObservatioN Evaluation Toolkit**

A modern, type-safe Python toolkit for evaluating atmospheric chemistry and air quality models against observations. DAVINCI-MONET is a Claude Code (Opus 4.5) AI assisted full rewrite of [MELODIES-MONET](https://github.com/NOAA-CSL/MELODIES-MONET) with improved architecture, full type hints, and comprehensive testing.

## Features

- **Unified Pairing Engine** - Single pairing system based on data geometry (point, track, profile, swath, grid)
- **Multiple Model Support** - CMAQ, WRF-Chem, UFS, CESM, and generic NetCDF
- **Multiple Observation Types** - Surface (AirNow, AQS, AERONET, OpenAQ), aircraft (ICARTT), satellite (TROPOMI, GOES), sondes
- **27 Statistical Metrics** - Bias, error, correlation, and agreement metrics with groupby support
- **10 Plot Types** - Time series, scatter, Taylor diagrams, spatial maps, and more
- **Type-Safe Configuration** - Pydantic-validated YAML configs with backward compatibility
- **Full Test Coverage** - 732+ tests with synthetic data generation

## Quick Start

```bash
# Install
conda create -n davinci-monet python=3.11
conda activate davinci-monet
pip install davinci-monet

# Run analysis
davinci-monet run config.yaml

# Validate config
davinci-monet validate config.yaml
```

## Minimal Example

```python
from davinci_monet.config import load_config
from davinci_monet.pipeline import PipelineRunner, PipelineContext

# Load and run
config = load_config("config.yaml")
runner = PipelineRunner()
result = runner.run(PipelineContext(config=config))

print(f"Success: {result.success}")
```

## Documentation

See the [Wiki](../../wiki) for full documentation:

- [Installation](../../wiki/Installation) - Setup and dependencies
- [Configuration](../../wiki/Configuration) - YAML configuration guide
- [CLI Reference](../../wiki/CLI-Reference) - Command-line interface
- [API Reference](../../wiki/API-Reference) - Python API documentation
- [Migration Guide](../../wiki/Migration-Guide) - Migrating from MELODIES-MONET

## Architecture

```
davinci_monet/
├── config/       # Pydantic schemas, YAML parsing
├── models/       # Model readers (CMAQ, WRF-Chem, UFS, CESM)
├── observations/ # Observation handlers by type
├── pairing/      # Unified pairing engine + strategies
├── plots/        # Plotting system with registry
├── stats/        # Statistics calculation
├── pipeline/     # Execution orchestration
├── io/           # File readers/writers
└── cli/          # Command-line interface
```

## Data Flow

```
Model Files ──► Model Reader ──► xr.Dataset ──┐
                                              ├──► Pairing Engine ──► Paired Dataset
Obs Files ────► Obs Reader ───► xr.Dataset ──┘         │
                                                       ▼
                                              Statistics + Plots
```

## Requirements

- Python 3.10+
- Core: xarray, numpy, pandas, matplotlib, cartopy
- I/O: netCDF4, monet, monetio
- Config: pydantic, pyyaml
- CLI: typer

## License

Apache 2.0
