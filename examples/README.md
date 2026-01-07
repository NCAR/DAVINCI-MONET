# DAVINCI-MONET Examples

Example scripts demonstrating DAVINCI-MONET usage.

## Scripts

### basic_analysis.py

A minimal example showing the core workflow:
- Creating synthetic model and observation data
- Simple point-to-grid pairing
- Computing basic statistics
- Creating time series and scatter plots

```bash
cd examples
python basic_analysis.py
```

### surface_evaluation.py

Surface observation evaluation example:
- Multi-variable comparison (O3, PM2.5, NO2)
- Statistics for all pollutants
- Diurnal cycle analysis
- Spatial bias mapping

```bash
python surface_evaluation.py
```

### all_plot_types.py

Complete demonstration of all 10 plot types:
1. Time Series
2. Diurnal Cycle
3. Scatter Plot
4. Taylor Diagram
5. Box Plot
6. Spatial Bias Map
7. Spatial Overlay
8. Spatial Distribution
9. Curtain Plot (vertical cross-section)
10. Scorecard

```bash
python all_plot_types.py
```

### custom_statistics.py

Statistics module deep dive:
- Quick statistics from arrays
- Individual metric calculation
- Grouped statistics (by site, by hour)
- Output formatting and export

```bash
python custom_statistics.py
```

## Configuration Files

### configs/cmaq_airnow.yaml

Example configuration for CMAQ evaluation against AirNow:
- Multiple variables (O3, PM2.5, NO2, CO)
- Time series, scatter, diurnal, spatial plots
- Standard statistics suite

Usage:
```bash
davinci-monet validate configs/cmaq_airnow.yaml
davinci-monet run configs/cmaq_airnow.yaml
```

## Output

All scripts write output to `./output/`:
- PNG plot files
- CSV statistics files

## Requirements

These examples require DAVINCI-MONET to be installed:

```bash
conda activate davinci-monet
pip install -e ..
```

For spatial plots, cartopy is required:
```bash
conda install -c conda-forge cartopy
```
