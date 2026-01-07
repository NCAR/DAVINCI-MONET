#!/usr/bin/env python
"""Basic DAVINCI-MONET analysis example.

This script demonstrates a simple model-observation comparison workflow
using synthetic data for testing and demonstration purposes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def create_synthetic_model_data() -> xr.Dataset:
    """Create synthetic model data for demonstration."""
    # Create coordinate arrays
    times = np.arange("2024-07-01", "2024-07-08", dtype="datetime64[h]")
    lats = np.linspace(25, 50, 50)
    lons = np.linspace(-125, -65, 100)

    # Create synthetic O3 field with diurnal cycle
    time_idx = np.arange(len(times))
    hour_of_day = time_idx % 24
    diurnal = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak at noon

    # Base field with spatial gradient
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    base_field = 40 + 10 * (lat_grid - 35) / 15  # Higher O3 at higher latitudes

    # Combine: (time, lat, lon)
    o3 = base_field[np.newaxis, :, :] + diurnal[:, np.newaxis, np.newaxis]
    o3 += np.random.randn(*o3.shape) * 5  # Add noise
    o3 = np.clip(o3, 0, 150)

    ds = xr.Dataset(
        {
            "O3": (["time", "lat", "lon"], o3.astype(np.float32)),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "Synthetic Model Data",
            "source": "DAVINCI-MONET Example",
        },
    )
    ds["O3"].attrs["units"] = "ppbv"
    ds["O3"].attrs["long_name"] = "Ozone"

    return ds


def create_synthetic_obs_data() -> xr.Dataset:
    """Create synthetic observation data for demonstration."""
    # Create station locations
    n_sites = 20
    np.random.seed(42)
    site_lats = np.random.uniform(28, 48, n_sites)
    site_lons = np.random.uniform(-120, -70, n_sites)
    site_ids = [f"SITE_{i:03d}" for i in range(n_sites)]

    # Create time series
    times = np.arange("2024-07-01", "2024-07-08", dtype="datetime64[h]")

    # Create O3 observations with diurnal cycle
    time_idx = np.arange(len(times))
    hour_of_day = time_idx % 24
    diurnal = 20 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)

    # Base values vary by latitude
    base_values = 40 + 10 * (site_lats - 35) / 15

    # Combine: (time, site)
    o3 = base_values[np.newaxis, :] + diurnal[:, np.newaxis]
    o3 += np.random.randn(*o3.shape) * 8  # Obs have more noise
    o3 = np.clip(o3, 0, 150)

    ds = xr.Dataset(
        {
            "o3": (["time", "site"], o3.astype(np.float32)),
        },
        coords={
            "time": times,
            "site": site_ids,
            "latitude": ("site", site_lats),
            "longitude": ("site", site_lons),
        },
        attrs={
            "title": "Synthetic Observation Data",
            "source": "DAVINCI-MONET Example",
            "geometry": "point",
        },
    )
    ds["o3"].attrs["units"] = "ppbv"
    ds["o3"].attrs["long_name"] = "Ozone"

    return ds


def pair_data(model_ds: xr.Dataset, obs_ds: xr.Dataset) -> xr.Dataset:
    """Simple point-to-grid pairing.

    Extracts model values at observation locations using nearest neighbor.
    """
    # Get observation coordinates
    obs_lats = obs_ds["latitude"].values
    obs_lons = obs_ds["longitude"].values

    # Find nearest model grid points
    model_lats = model_ds["lat"].values
    model_lons = model_ds["lon"].values

    lat_indices = np.abs(model_lats[:, np.newaxis] - obs_lats).argmin(axis=0)
    lon_indices = np.abs(model_lons[:, np.newaxis] - obs_lons).argmin(axis=0)

    # Extract model values at obs locations
    model_o3 = model_ds["O3"].values[:, lat_indices, lon_indices]

    # Create paired dataset
    paired = xr.Dataset(
        {
            "obs_o3": (["time", "site"], obs_ds["o3"].values),
            "model_o3": (["time", "site"], model_o3),
        },
        coords={
            "time": obs_ds["time"],
            "site": obs_ds["site"],
            "latitude": ("site", obs_lats),
            "longitude": ("site", obs_lons),
        },
    )

    return paired


def compute_statistics(paired_ds: xr.Dataset) -> dict:
    """Compute basic statistics."""
    obs = paired_ds["obs_o3"].values.flatten()
    model = paired_ds["model_o3"].values.flatten()

    # Remove NaN
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs = obs[mask]
    model = model[mask]

    stats = {
        "N": len(obs),
        "Mean Obs": np.mean(obs),
        "Mean Model": np.mean(model),
        "MB": np.mean(model - obs),
        "RMSE": np.sqrt(np.mean((model - obs) ** 2)),
        "R": np.corrcoef(obs, model)[0, 1],
        "NMB": 100 * np.sum(model - obs) / np.sum(obs),
    }

    return stats


def plot_timeseries(paired_ds: xr.Dataset, output_path: Path) -> None:
    """Create time series plot."""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Average over all sites
    obs_mean = paired_ds["obs_o3"].mean(dim="site")
    model_mean = paired_ds["model_o3"].mean(dim="site")

    ax.plot(paired_ds["time"], obs_mean, "b-", label="Observations", linewidth=1.5)
    ax.plot(paired_ds["time"], model_mean, "r-", label="Model", linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("O3 (ppbv)")
    ax.set_title("Time Series Comparison (All Sites Mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_scatter(paired_ds: xr.Dataset, output_path: Path) -> None:
    """Create scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 7))

    obs = paired_ds["obs_o3"].values.flatten()
    model = paired_ds["model_o3"].values.flatten()

    ax.scatter(obs, model, alpha=0.3, s=5)

    # 1:1 line
    lims = [0, max(obs.max(), model.max()) * 1.1]
    ax.plot(lims, lims, "k--", label="1:1", linewidth=1)

    # Linear regression
    mask = ~(np.isnan(obs) | np.isnan(model))
    coeffs = np.polyfit(obs[mask], model[mask], 1)
    ax.plot(lims, np.polyval(coeffs, lims), "r-", label=f"Fit: y={coeffs[0]:.2f}x+{coeffs[1]:.1f}")

    ax.set_xlabel("Observed O3 (ppbv)")
    ax.set_ylabel("Modeled O3 (ppbv)")
    ax.set_title("Scatter Plot")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    """Run basic analysis example."""
    print("DAVINCI-MONET Basic Analysis Example")
    print("=" * 40)

    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    print("\n1. Creating synthetic data...")
    model_ds = create_synthetic_model_data()
    obs_ds = create_synthetic_obs_data()
    print(f"   Model: {model_ds.dims}")
    print(f"   Obs: {obs_ds.dims}")

    # Pair data
    print("\n2. Pairing data...")
    paired_ds = pair_data(model_ds, obs_ds)
    print(f"   Paired: {paired_ds.dims}")

    # Compute statistics
    print("\n3. Computing statistics...")
    stats = compute_statistics(paired_ds)
    print("\n   Statistics:")
    for name, value in stats.items():
        if isinstance(value, float):
            print(f"   {name:12s}: {value:8.2f}")
        else:
            print(f"   {name:12s}: {value}")

    # Create plots
    print("\n4. Creating plots...")
    plot_timeseries(paired_ds, output_dir / "timeseries.png")
    plot_scatter(paired_ds, output_dir / "scatter.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
