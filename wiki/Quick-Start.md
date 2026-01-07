# Quick Start

Get started with DAVINCI-MONET in 5 minutes.

## 1. Run an Example

The fastest way to see DAVINCI-MONET in action:

```bash
conda activate davinci-monet

# Run aircraft evaluation example
python examples/aircraft_evaluation.py

# View output
open examples/output/aircraft/
```

## 2. Basic Python Workflow

```python
import numpy as np
import xarray as xr
from davinci_monet.stats import compute_statistics

# Create paired data (normally from pairing engine)
paired = xr.Dataset({
    "obs_o3": (["time", "site"], np.random.rand(100, 10) * 50 + 30),
    "model_o3": (["time", "site"], np.random.rand(100, 10) * 50 + 32),
})

# Compute statistics
stats = compute_statistics(paired, "obs_o3", "model_o3")

print(f"Mean Bias: {stats['MB']:.2f} ppbv")
print(f"RMSE: {stats['RMSE']:.2f} ppbv")
print(f"Correlation: {stats['R']:.3f}")
```

## 3. CLI Usage

```bash
# Run full analysis from config file
davinci-monet run config.yaml

# Validate configuration
davinci-monet validate config.yaml

# Download observation data
davinci-monet get airnow --start 2024-07-01 --end 2024-07-07
```

## 4. Configuration File

Minimal `config.yaml`:

```yaml
analysis:
  start_time: 2024-07-01
  end_time: 2024-07-07
  output_dir: ./output

model:
  cmaq:
    files: /path/to/CMAQ/*.nc
    variables:
      O3: O3

obs:
  airnow:
    data_dir: /path/to/airnow/
    variables:
      - OZONE

plots:
  - type: timeseries
    variables: [O3]
  - type: scatter
    variables: [O3]
```

## 5. Pairing Data

```python
from davinci_monet.pairing.strategies.point import PointStrategy

# Load model and observation data
model = xr.open_dataset("model.nc")
obs = xr.open_dataset("obs.nc")

# Pair using point strategy (for surface stations)
strategy = PointStrategy()
paired = strategy.pair(model, obs)

# Now compute statistics or create plots
```

## Next Steps

- [[Examples]] - Complete working examples for all observation types
- [[Configuration]] - Full configuration reference
- [[Pairing Strategies]] - Understanding data geometry pairing
- [[Statistics Metrics]] - All 27 available metrics
