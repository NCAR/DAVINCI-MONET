# Examples

DAVINCI-MONET includes complete working examples demonstrating evaluation workflows for different observation types. All examples use synthetic data generation, so no external datasets are required.

## Running Examples

```bash
# Activate environment
conda activate davinci-monet

# Run any example
python examples/aircraft_evaluation.py

# Output is saved to examples/output/<name>/
```

All examples output both **PNG (300 DPI)** and **PDF** formats.

---

## Surface Evaluation

**File:** `examples/surface_evaluation.py`

Demonstrates evaluation of model output against surface monitoring stations (point observations).

### Workflow
1. Generate synthetic paired surface data (50 sites, 2 weeks)
2. Compute statistics for O₃, PM2.5, NO₂
3. Create evaluation plots

### Output Plots
| Plot | Description |
|------|-------------|
| `scatter_multi.png` | 3-panel scatter plots with regression statistics |
| `diurnal_o3.png` | Diurnal cycle comparison for ozone |
| `spatial_bias.png` | Map of mean bias at each station |

### Key Code
```python
from davinci_monet.stats import compute_statistics

# Compute comprehensive statistics
stats = compute_statistics(paired_ds, "obs_o3", "model_o3")
print(f"R = {stats['R']:.3f}, RMSE = {stats['RMSE']:.2f}")
```

---

## Aircraft Evaluation

**File:** `examples/aircraft_evaluation.py`

Demonstrates evaluation of model output against aircraft track observations (3D trajectory).

### Workflow
1. Generate synthetic 3D model (40×30 grid, 20 levels, 6 hours)
2. Generate synthetic aircraft track (200 points, spiral pattern)
3. Pair using `TrackStrategy` with vertical interpolation
4. Create evaluation plots

### Output Plots
| Plot | Description |
|------|-------------|
| `flight_path.png` | Map of flight track colored by altitude |
| `curtain_o3.png` | 3-panel curtain plot (obs, model, bias vs altitude/time) |
| `scatter.png` | Model vs obs scatter for O₃ and CO |
| `timeseries.png` | Time series of O₃, CO, and altitude along track |
| `vertical_profile.png` | Altitude-binned vertical profiles |

### Key Code
```python
from davinci_monet.pairing.strategies.track import TrackStrategy

# Pair aircraft with model
strategy = TrackStrategy()
paired = strategy.pair(model, aircraft, vertical_method="linear")
```

---

## Satellite Evaluation

**File:** `examples/satellite_evaluation.py`

Demonstrates evaluation of model output against satellite observations, including both L2 swath and L3 gridded products.

### Workflow
1. Generate synthetic surface model (60×50 grid)
2. Generate synthetic L2 swath (80 scanlines × 50 pixels)
3. Generate synthetic L3 gridded (30×25 grid, daily)
4. Pair using `SwathStrategy` and grid regridding
5. Create evaluation plots

### Output Plots
| Plot | Description |
|------|-------------|
| `swath_footprint.png` | L2 swath coverage colored by NO₂ |
| `swath_bias.png` | Spatial bias map for swath pixels |
| `gridded_comparison.png` | Side-by-side L3 satellite vs model vs bias |
| `scatter_density.png` | 2D histogram density scatter with statistics |
| `histograms.png` | Value distribution comparison for L2 and L3 |
| `qa_filtering.png` | QA flag filtering demonstration |

### Key Code
```python
from davinci_monet.pairing.strategies.swath import SwathStrategy

# Pair satellite swath with model
strategy = SwathStrategy()
paired = strategy.pair(model, swath, match_overpass=True)
```

---

## Sonde Evaluation

**File:** `examples/sonde_evaluation.py`

Demonstrates evaluation of model output against ozonesonde vertical profiles.

### Workflow
1. Generate synthetic 3D model (80×50 grid, 30 levels, 1 week)
2. Generate synthetic sonde profiles (12 soundings, 50 levels each)
3. Pair each sonde with nearest model column
4. Create evaluation plots

### Output Plots
| Plot | Description |
|------|-------------|
| `launch_locations.png` | Map of sonde launch sites |
| `profile_01.png` ... `profile_03.png` | Individual profile comparisons |
| `profiles_overlay.png` | All profiles overlaid (solid=obs, dashed=model) |
| `mean_bias_profiles.png` | Mean profiles with IQR + bias profiles |
| `layer_statistics.png` | MB, RMSE, R by altitude layer |
| `scatter_by_layer.png` | 6-panel scatter for different atmospheric layers |

### Key Code
```python
# Extract model column at sonde location
lat_idx = np.abs(model.lat.values - profile_lat).argmin()
lon_idx = np.abs(model.lon.values - profile_lon).argmin()
model_column = model.isel(lat=lat_idx, lon=lon_idx)

# Interpolate to sonde levels
model_interp = model_column.interp(z=sonde_levels)
```

---

## All Plot Types Demo

**File:** `examples/all_plot_types.py`

Demonstrates all 10 plot types available in DAVINCI-MONET.

### Output Plots
| # | Plot | Type |
|---|------|------|
| 1 | `01_timeseries.png` | Time series with IQR shading |
| 2 | `02_diurnal.png` | Diurnal cycle with error bars |
| 3 | `03_scatter.png` | Scatter with 1:1 line and regression |
| 4 | `04_taylor.png` | Taylor diagram (correlation vs std dev) |
| 5 | `05_boxplot.png` | Box plot comparison by variable |
| 6 | `06_spatial_bias.png` | Spatial bias map (requires cartopy) |
| 7 | `07_spatial_overlay.png` | Model contours + obs points |
| 8 | `08_spatial_distribution.png` | Spatial distribution of values |
| 9 | `09_curtain.png` | Vertical curtain plot |
| 10 | `10_scorecard.png` | Multi-metric performance scorecard |

---

## Additional Examples

### Basic Analysis
**File:** `examples/basic_analysis.py`

Minimal example showing the core workflow:
1. Create synthetic model and observation data
2. Pair data
3. Compute statistics
4. Create basic plots

### Custom Statistics
**File:** `examples/custom_statistics.py`

Demonstrates advanced statistics capabilities:
- Using individual metric functions
- Groupby operations (by site, time, region)
- Custom metric configurations
- Exporting statistics to CSV

---

## Output Directory Structure

```
examples/output/
├── aircraft/       # Aircraft evaluation plots
├── all_plots/      # All 10 plot types demo
├── basic/          # Basic analysis plots
├── custom_stats/   # Custom statistics CSV files
├── satellite/      # Satellite evaluation plots
├── sonde/          # Sonde evaluation plots
└── surface/        # Surface evaluation plots
```

Each directory contains both `.png` (300 DPI) and `.pdf` versions of all plots.
