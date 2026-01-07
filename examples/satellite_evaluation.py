#!/usr/bin/env python
"""Satellite evaluation example using synthetic data.

This script demonstrates how to:
1. Generate synthetic satellite swath (L2) observations
2. Generate synthetic satellite gridded (L3) observations
3. Generate synthetic model output
4. Pair satellite data with model data
5. Create various plots for satellite evaluation:
   - Swath footprint map
   - Spatial bias maps
   - Scatter plots
   - Histogram comparisons
   - Side-by-side grid comparison
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

# DAVINCI-MONET imports
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_surface_model
from davinci_monet.tests.synthetic.observations import (
    create_swath_observations,
    create_gridded_observations,
)
from davinci_monet.pairing.strategies.swath import SwathStrategy
from davinci_monet.pairing.strategies.grid import GridStrategy

# Set seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure as both PNG (300 DPI) and PDF."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def create_satellite_data() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Create synthetic satellite and model data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset, xr.Dataset]
        (model_data, swath_l2_data, gridded_l3_data)
    """
    # Define domain (Eastern US / Atlantic)
    domain = Domain(
        lon_min=-90, lon_max=-60,
        lat_min=25, lat_max=50,
        n_lon=60, n_lat=50,
    )

    # Time configuration - single day
    time_config = TimeConfig(
        start="2024-07-15T12:00:00",
        end="2024-07-15T18:00:00",
        freq="1h",
    )

    # Create surface model with NO2 (typical satellite product)
    model = create_surface_model(
        variables=["NO2"],
        domain=domain,
        time_config=time_config,
        seed=42,
    )

    # Create L2 swath data (simulating TROPOMI-like pass)
    swath = create_swath_observations(
        n_scans=80,
        n_pixels=50,
        variables=["NO2"],
        domain=domain,
        time_config=time_config,
        seed=123,
    )

    # Create L3 gridded data (daily average)
    l3_time_config = TimeConfig(
        start="2024-07-15T00:00:00",
        end="2024-07-15T23:59:59",
        freq="1D",
    )
    l3_domain = Domain(
        lon_min=-90, lon_max=-60,
        lat_min=25, lat_max=50,
        n_lon=30, n_lat=25,  # Coarser grid typical of L3
    )
    gridded = create_gridded_observations(
        variables=["NO2"],
        domain=l3_domain,
        time_config=l3_time_config,
        seed=456,
    )

    return model, swath, gridded


def pair_swath_with_model(
    model: xr.Dataset,
    swath: xr.Dataset,
) -> xr.Dataset:
    """Pair L2 swath observations with model output.

    Parameters
    ----------
    model
        Surface model dataset
    swath
        Satellite swath dataset

    Returns
    -------
    xr.Dataset
        Paired dataset with model values at swath pixels
    """
    strategy = SwathStrategy()
    paired = strategy.pair(
        model,
        swath,
        horizontal_method="nearest",
        match_overpass=True,
    )
    return paired


def pair_gridded_with_model(
    model: xr.Dataset,
    gridded: xr.Dataset,
) -> xr.Dataset:
    """Pair L3 gridded observations with model output.

    Parameters
    ----------
    model
        Surface model dataset
    gridded
        Gridded satellite dataset

    Returns
    -------
    xr.Dataset
        Paired dataset on common grid
    """
    # For L3 daily data, we average the model over time first
    # then regrid to the L3 grid (simpler than using GridStrategy
    # which has time alignment issues when times don't overlap)
    model_mean = model.mean(dim="time")

    # Regrid model to L3 grid
    model_regrid = model_mean.interp(
        lat=gridded.lat.values,
        lon=gridded.lon.values,
        method="nearest",
    )

    # Build paired dataset
    paired = xr.Dataset(
        {
            "NO2": gridded["NO2"].isel(time=0),
            "model_NO2": model_regrid["NO2"],
        },
        coords={
            "lat": gridded.lat,
            "lon": gridded.lon,
        },
    )

    return paired


def plot_swath_footprint(swath: xr.Dataset, output_dir: Path) -> None:
    """Plot the satellite swath footprint colored by NO2."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False
        print("Cartopy not available, creating simple swath plot")

    lons = swath["longitude"].values
    lats = swath["latitude"].values
    no2 = swath["NO2"].values

    if has_cartopy:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set extent with padding
        lon_pad = 5
        lat_pad = 3
        ax.set_extent([
            lons.min() - lon_pad, lons.max() + lon_pad,
            lats.min() - lat_pad, lats.max() + lat_pad
        ], crs=ccrs.PlateCarree())

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

        # Plot swath as pcolormesh
        pc = ax.pcolormesh(
            lons, lats, no2,
            cmap="YlOrRd",
            transform=ccrs.PlateCarree(),
            shading="auto",
        )

        cbar = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
        cbar.set_label("NO$_2$ (molec/cm$^2$)")
        ax.set_title("Satellite L2 Swath Footprint", fontsize=14, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        pc = ax.pcolormesh(lons, lats, no2, cmap="YlOrRd", shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Satellite L2 Swath Footprint")
        plt.colorbar(pc, label="NO$_2$")

    save_figure(fig, output_dir / "swath_footprint.png")
    plt.close(fig)


def plot_swath_bias(paired_swath: xr.Dataset, swath: xr.Dataset, output_dir: Path) -> None:
    """Plot spatial bias map for swath data."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    lons = swath["longitude"].values
    lats = swath["latitude"].values

    obs = paired_swath["NO2"].values
    # Model data needs reshaping to match swath dimensions
    model = paired_swath["model_NO2"].values
    if model.shape != obs.shape:
        model = model.reshape(obs.shape)

    bias = model - obs

    vmax = np.nanpercentile(np.abs(bias), 95)

    if has_cartopy:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        lon_pad, lat_pad = 5, 3
        ax.set_extent([
            lons.min() - lon_pad, lons.max() + lon_pad,
            lats.min() - lat_pad, lats.max() + lat_pad
        ], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

        pc = ax.pcolormesh(
            lons, lats, bias,
            cmap="RdBu_r",
            vmin=-vmax, vmax=vmax,
            transform=ccrs.PlateCarree(),
            shading="auto",
        )

        cbar = plt.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
        cbar.set_label("Bias (Model - Obs)")
        ax.set_title("L2 Swath Bias (Model - Satellite)", fontsize=14, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        pc = ax.pcolormesh(lons, lats, bias, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("L2 Swath Bias")
        plt.colorbar(pc, label="Bias")

    save_figure(fig, output_dir / "swath_bias.png")
    plt.close(fig)


def plot_gridded_comparison(
    paired_grid: xr.Dataset,
    output_dir: Path,
) -> None:
    """Plot side-by-side comparison of L3 gridded data."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    obs = paired_grid["NO2"].isel(time=0) if "time" in paired_grid["NO2"].dims else paired_grid["NO2"]
    model = paired_grid["model_NO2"].isel(time=0) if "time" in paired_grid["model_NO2"].dims else paired_grid["model_NO2"]
    bias = model - obs

    # Get coordinate names
    lat_name = "lat" if "lat" in paired_grid.coords else "latitude"
    lon_name = "lon" if "lon" in paired_grid.coords else "longitude"

    lats = paired_grid[lat_name].values
    lons = paired_grid[lon_name].values

    # Ensure 2D for plotting
    if lats.ndim == 1 and lons.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lon_grid, lat_grid = lons, lats

    vmin = min(float(obs.min()), float(model.min()))
    vmax = max(float(obs.max()), float(model.max()))
    bias_max = float(np.nanpercentile(np.abs(bias.values), 95))

    if has_cartopy:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                  subplot_kw={"projection": ccrs.PlateCarree()})

        for ax in axes:
            ax.set_extent([lons.min() - 2, lons.max() + 2,
                          lats.min() - 2, lats.max() + 2],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

        # Satellite L3
        pc1 = axes[0].pcolormesh(lon_grid, lat_grid, obs.values, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                                  shading="auto")
        axes[0].set_title("Satellite L3")
        plt.colorbar(pc1, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.8)

        # Model
        pc2 = axes[1].pcolormesh(lon_grid, lat_grid, model.values, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree(),
                                  shading="auto")
        axes[1].set_title("Model")
        plt.colorbar(pc2, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.8)

        # Bias
        pc3 = axes[2].pcolormesh(lon_grid, lat_grid, bias.values, cmap="RdBu_r",
                                  vmin=-bias_max, vmax=bias_max,
                                  transform=ccrs.PlateCarree(), shading="auto")
        axes[2].set_title("Bias (Model - Sat)")
        plt.colorbar(pc3, ax=axes[2], orientation="horizontal", pad=0.05, shrink=0.8)

    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        pc1 = axes[0].pcolormesh(lon_grid, lat_grid, obs.values, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, shading="auto")
        axes[0].set_title("Satellite L3")
        plt.colorbar(pc1, ax=axes[0])

        pc2 = axes[1].pcolormesh(lon_grid, lat_grid, model.values, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, shading="auto")
        axes[1].set_title("Model")
        plt.colorbar(pc2, ax=axes[1])

        pc3 = axes[2].pcolormesh(lon_grid, lat_grid, bias.values, cmap="RdBu_r",
                                  vmin=-bias_max, vmax=bias_max, shading="auto")
        axes[2].set_title("Bias")
        plt.colorbar(pc3, ax=axes[2])

    fig.suptitle("NO$_2$ L3 Gridded Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "gridded_comparison.png")
    plt.close(fig)


def plot_scatter_density(paired_swath: xr.Dataset, output_dir: Path) -> None:
    """Create scatter plot with density shading."""
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = sns.color_palette()

    obs = paired_swath["NO2"].values.flatten()
    model = paired_swath["model_NO2"].values.flatten()

    # Remove NaNs
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs_clean = obs[mask]
    model_clean = model[mask]

    # 2D histogram for density
    from matplotlib.colors import LogNorm

    h = ax.hist2d(obs_clean, model_clean, bins=50, cmap="Blues", norm=LogNorm())
    plt.colorbar(h[3], ax=ax, label="Count")

    # 1:1 line
    lims = [
        min(obs_clean.min(), model_clean.min()),
        max(obs_clean.max(), model_clean.max())
    ]
    ax.plot(lims, lims, "k--", label="1:1", linewidth=2)

    # Stats
    r = np.corrcoef(obs_clean, model_clean)[0, 1]
    mb = np.mean(model_clean - obs_clean)
    rmse = np.sqrt(np.mean((model_clean - obs_clean) ** 2))
    nmb = 100 * np.sum(model_clean - obs_clean) / np.sum(obs_clean)

    stats_text = (f"N = {len(obs_clean):,}\n"
                  f"R = {r:.3f}\n"
                  f"MB = {mb:.2e}\n"
                  f"RMSE = {rmse:.2e}\n"
                  f"NMB = {nmb:.1f}%")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment="top", fontsize=11,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_xlabel("Satellite NO$_2$")
    ax.set_ylabel("Model NO$_2$")
    ax.set_title("L2 Swath: Model vs Satellite (Density)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")

    save_figure(fig, output_dir / "scatter_density.png")
    plt.close(fig)


def plot_histograms(paired_swath: xr.Dataset, paired_grid: xr.Dataset, output_dir: Path) -> None:
    """Create histogram comparisons for both L2 and L3 data."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette()

    # L2 Swath
    ax1 = axes[0]
    obs_l2 = paired_swath["NO2"].values.flatten()
    model_l2 = paired_swath["model_NO2"].values.flatten()
    mask_l2 = ~(np.isnan(obs_l2) | np.isnan(model_l2))

    if mask_l2.sum() > 0:
        bins = np.linspace(
            min(obs_l2[mask_l2].min(), model_l2[mask_l2].min()),
            max(obs_l2[mask_l2].max(), model_l2[mask_l2].max()),
            40
        )
        ax1.hist(obs_l2[mask_l2], bins=bins, alpha=0.6, label="Satellite L2",
                 color=colors[0], edgecolor="white")
        ax1.hist(model_l2[mask_l2], bins=bins, alpha=0.6, label="Model",
                 color=colors[3], edgecolor="white")
    ax1.set_xlabel("NO$_2$")
    ax1.set_ylabel("Count")
    ax1.set_title("L2 Swath Distribution")
    ax1.legend()

    # L3 Gridded
    ax2 = axes[1]
    obs_l3 = paired_grid["NO2"].values.flatten()
    model_l3 = paired_grid["model_NO2"].values.flatten()
    mask_l3 = ~(np.isnan(obs_l3) | np.isnan(model_l3))

    if mask_l3.sum() > 0:
        bins = np.linspace(
            min(obs_l3[mask_l3].min(), model_l3[mask_l3].min()),
            max(obs_l3[mask_l3].max(), model_l3[mask_l3].max()),
            40
        )
        ax2.hist(obs_l3[mask_l3], bins=bins, alpha=0.6, label="Satellite L3",
                 color=colors[0], edgecolor="white")
        ax2.hist(model_l3[mask_l3], bins=bins, alpha=0.6, label="Model",
                 color=colors[3], edgecolor="white")
    else:
        # If no valid L3 pairs, use obs only
        obs_l3_valid = obs_l3[~np.isnan(obs_l3)]
        if len(obs_l3_valid) > 0:
            ax2.hist(obs_l3_valid, bins=40, alpha=0.6, label="Satellite L3",
                     color=colors[0], edgecolor="white")
            ax2.text(0.5, 0.5, "No model pairing\n(grid mismatch)",
                    transform=ax2.transAxes, ha="center", va="center",
                    fontsize=12, color="red")
    ax2.set_xlabel("NO$_2$")
    ax2.set_ylabel("Count")
    ax2.set_title("L3 Gridded Distribution")
    ax2.legend()

    fig.suptitle("NO$_2$ Value Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "histograms.png")
    plt.close(fig)


def plot_qa_filtering(swath: xr.Dataset, output_dir: Path) -> None:
    """Demonstrate QA flag filtering for satellite data."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    lons = swath["longitude"].values
    lats = swath["latitude"].values
    no2 = swath["NO2"].values
    qa = swath["qa_flag"].values

    # Create filtered version
    no2_filtered = np.where(qa == 0, no2, np.nan)

    if has_cartopy:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                  subplot_kw={"projection": ccrs.PlateCarree()})

        for ax in axes:
            ax.set_extent([lons.min() - 3, lons.max() + 3,
                          lats.min() - 2, lats.max() + 2],
                         crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

        vmin, vmax = np.nanpercentile(no2, [2, 98])

        # Unfiltered
        pc1 = axes[0].pcolormesh(lons, lats, no2, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree(), shading="auto")
        axes[0].set_title("All Pixels (Unfiltered)")
        plt.colorbar(pc1, ax=axes[0], orientation="horizontal", pad=0.05, shrink=0.8)

        # QA Filtered
        pc2 = axes[1].pcolormesh(lons, lats, no2_filtered, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax,
                                  transform=ccrs.PlateCarree(), shading="auto")
        axes[1].set_title("QA Filtered (qa_flag=0 only)")
        plt.colorbar(pc2, ax=axes[1], orientation="horizontal", pad=0.05, shrink=0.8)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        vmin, vmax = np.nanpercentile(no2, [2, 98])

        pc1 = axes[0].pcolormesh(lons, lats, no2, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, shading="auto")
        axes[0].set_title("All Pixels")
        plt.colorbar(pc1, ax=axes[0])

        pc2 = axes[1].pcolormesh(lons, lats, no2_filtered, cmap="YlOrRd",
                                  vmin=vmin, vmax=vmax, shading="auto")
        axes[1].set_title("QA Filtered")
        plt.colorbar(pc2, ax=axes[1])

    # Add stats
    n_total = np.sum(~np.isnan(no2))
    n_good = np.sum(qa == 0)
    fig.suptitle(f"QA Flag Filtering Demo ({n_good:,} of {n_total:,} pixels pass QA)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "qa_filtering.png")
    plt.close(fig)


def main():
    """Run satellite evaluation example."""
    print("DAVINCI-MONET: Satellite Evaluation Example")
    print("=" * 50)

    output_dir = Path("examples/output/satellite")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("\n1. Creating synthetic data...")
    model, swath, gridded = create_satellite_data()
    print(f"   Model dims: {dict(model.sizes)}")
    print(f"   L2 Swath dims: {dict(swath.sizes)}")
    print(f"   L3 Gridded dims: {dict(gridded.sizes)}")
    print(f"   Swath pixels: {swath.sizes['scanline'] * swath.sizes['pixel']:,}")

    # Pair data
    print("\n2. Pairing satellite with model...")

    print("   - Pairing L2 swath...")
    paired_swath = pair_swath_with_model(model, swath)
    print(f"     Paired variables: {list(paired_swath.data_vars)}")

    print("   - Pairing L3 gridded...")
    paired_grid = pair_gridded_with_model(model, gridded)
    print(f"     Paired variables: {list(paired_grid.data_vars)}")

    # Create plots
    print("\n3. Generating plots...")

    print("   - Swath footprint")
    plot_swath_footprint(swath, output_dir)

    print("   - Swath bias map")
    plot_swath_bias(paired_swath, swath, output_dir)

    print("   - Gridded comparison")
    plot_gridded_comparison(paired_grid, output_dir)

    print("   - Scatter density plot")
    plot_scatter_density(paired_swath, output_dir)

    print("   - Histograms")
    plot_histograms(paired_swath, paired_grid, output_dir)

    print("   - QA filtering demo")
    plot_qa_filtering(swath, output_dir)

    print("\n" + "=" * 50)
    print("Satellite evaluation complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
