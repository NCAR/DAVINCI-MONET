# Installation

## Requirements

- Python 3.10+
- conda (recommended) or pip

## Quick Install with Conda

```bash
# Create environment with all dependencies
conda create -n davinci-monet -c conda-forge python=3.11 \
    melodies-monet pydantic mypy pytest pytest-cov black isort typer

# Activate environment
conda activate davinci-monet

# Install DAVINCI-MONET
pip install -e .
```

## Dependencies

### Core Dependencies
| Package | Purpose |
|---------|---------|
| xarray | N-dimensional data arrays |
| numpy | Numerical operations |
| pandas | Tabular data and I/O |
| matplotlib | Plotting |
| cartopy | Geographic projections |

### I/O Dependencies
| Package | Purpose |
|---------|---------|
| netCDF4 | NetCDF file I/O |
| monet | Atmospheric data utilities |
| monetio | Observation data readers |

### Configuration
| Package | Purpose |
|---------|---------|
| pydantic | Configuration validation |
| pyyaml | YAML parsing |

### CLI
| Package | Purpose |
|---------|---------|
| typer | Command-line interface |
| rich | Terminal formatting |

## Development Install

For development with testing and linting tools:

```bash
# Clone repository
git clone https://github.com/NCAR/DAVINCI-MONET.git
cd DAVINCI-MONET

# Create dev environment
conda create -n davinci-monet -c conda-forge python=3.11 \
    melodies-monet pydantic mypy pytest pytest-cov black isort typer

conda activate davinci-monet

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Check CLI
davinci-monet --help

# Run tests
pytest

# Type checking
mypy davinci_monet
```

## Troubleshooting

### Cartopy Issues
If cartopy fails to install via pip, use conda:
```bash
conda install -c conda-forge cartopy
```

### monetio Issues
monetio requires specific dependencies. Install via conda-forge:
```bash
conda install -c conda-forge monetio
```
