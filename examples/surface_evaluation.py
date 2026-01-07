#!/usr/bin/env python
"""Surface observation evaluation example using DAVINCI-MONET.

This script demonstrates evaluating model output against surface observations
using the DAVINCI-MONET statistics and plotting modules.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

# Set seaborn style for beautiful plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def create_paired_surface_data() -> xr.Dataset:
    """Create synthetic paired surface data for demonstration.

    In a real workflow, this would come from:
    1. Loading model data with davinci_monet.models.open_model()
    2. Loading obs with davinci_monet.observations.open_observation()
    3. Pairing with davinci_monet.pairing.PairingEngine
    """
    np.random.seed(42)

    # Create realistic surface station network
    n_sites = 50
    times = np.arange("2024-07-01", "2024-07-15", dtype="datetime64[h]")

    # Station coordinates (US domain)
    site_lats = np.random.uniform(28, 48, n_sites)
    site_lons = np.random.uniform(-120, -70, n_sites)
    site_ids = [f"AQS{i:05d}" for i in range(n_sites)]

    # Generate correlated obs and model data
    time_hours = np.arange(len(times))
    hour_of_day = time_hours % 24

    # O3: Diurnal cycle with model bias
    diurnal_o3 = 25 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    base_o3 = 45 + 8 * (site_lats - 38) / 10  # Latitude gradient

    obs_o3 = base_o3[np.newaxis, :] + diurnal_o3[:, np.newaxis]
    obs_o3 += np.random.randn(*obs_o3.shape) * 8

    model_o3 = obs_o3 + 5 + np.random.randn(*obs_o3.shape) * 6  # Model has positive bias
    obs_o3 = np.clip(obs_o3, 0, 150)
    model_o3 = np.clip(model_o3, 0, 150)

    # PM2.5: Less diurnal variation, model underestimates
    base_pm25 = 12 + 5 * np.random.randn(n_sites)
    obs_pm25 = base_pm25[np.newaxis, :] + np.random.randn(len(times), n_sites) * 4
    model_pm25 = obs_pm25 * 0.85 + np.random.randn(*obs_pm25.shape) * 3  # Model underestimates
    obs_pm25 = np.clip(obs_pm25, 0, 100)
    model_pm25 = np.clip(model_pm25, 0, 100)

    # NO2: Urban enhancement, model captures well
    urban_factor = np.random.uniform(0.5, 2.0, n_sites)
    base_no2 = 15 * urban_factor
    obs_no2 = base_no2[np.newaxis, :] + np.random.randn(len(times), n_sites) * 5
    model_no2 = obs_no2 + np.random.randn(*obs_no2.shape) * 4  # Model tracks well
    obs_no2 = np.clip(obs_no2, 0, 80)
    model_no2 = np.clip(model_no2, 0, 80)

    ds = xr.Dataset(
        {
            "obs_o3": (["time", "site"], obs_o3.astype(np.float32)),
            "model_o3": (["time", "site"], model_o3.astype(np.float32)),
            "obs_pm25": (["time", "site"], obs_pm25.astype(np.float32)),
            "model_pm25": (["time", "site"], model_pm25.astype(np.float32)),
            "obs_no2": (["time", "site"], obs_no2.astype(np.float32)),
            "model_no2": (["time", "site"], model_no2.astype(np.float32)),
        },
        coords={
            "time": times,
            "site": site_ids,
            "latitude": ("site", site_lats),
            "longitude": ("site", site_lons),
        },
        attrs={
            "title": "Paired Surface Observations",
            "source": "DAVINCI-MONET Example",
        },
    )

    return ds


def compute_all_statistics(paired_ds: xr.Dataset) -> dict:
    """Compute statistics for all variables using DAVINCI-MONET stats module."""
    from davinci_monet.stats import calculate_statistics

    variables = ["o3", "pm25", "no2"]
    results = {}

    for var in variables:
        obs_var = f"obs_{var}"
        model_var = f"model_{var}"

        stats_df = calculate_statistics(
            paired_ds,
            obs_var,
            model_var,
            metrics=["N", "MO", "MP", "MB", "RMSE", "R", "NMB", "NME", "IOA"],
        )
        results[var] = stats_df.iloc[0].to_dict()

    return results


def print_statistics_table(stats_dict: dict) -> None:
    """Print formatted statistics table."""
    print("\n" + "=" * 80)
    print("SURFACE EVALUATION STATISTICS")
    print("=" * 80)

    header = f"{'Variable':>10} {'N':>8} {'Mean Obs':>10} {'Mean Mod':>10} {'MB':>8} {'RMSE':>8} {'R':>6} {'NMB%':>8} {'IOA':>6}"
    print(header)
    print("-" * 80)

    for var, stats in stats_dict.items():
        row = (
            f"{var.upper():>10} "
            f"{stats['N']:>8.0f} "
            f"{stats['MO']:>10.2f} "
            f"{stats['MP']:>10.2f} "
            f"{stats['MB']:>8.2f} "
            f"{stats['RMSE']:>8.2f} "
            f"{stats['R']:>6.3f} "
            f"{stats['NMB']:>8.1f} "
            f"{stats['IOA']:>6.3f}"
        )
        print(row)
    print("=" * 80)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure as both PNG (300 DPI) and PDF."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def create_multi_panel_plot(paired_ds: xr.Dataset, output_path: Path) -> None:
    """Create multi-panel scatter plot for all variables."""
    variables = [
        ("o3", "O$_3$ (ppbv)"),
        ("pm25", "PM$_{2.5}$ ($\\mu$g/m$^3$)"),
        ("no2", "NO$_2$ (ppbv)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = sns.color_palette()

    for ax, (var, label) in zip(axes, variables):
        obs = paired_ds[f"obs_{var}"].values.flatten()
        model = paired_ds[f"model_{var}"].values.flatten()

        # Remove NaN
        mask = ~(np.isnan(obs) | np.isnan(model))
        obs, model = obs[mask], model[mask]

        # Scatter
        ax.scatter(obs, model, alpha=0.3, s=5, color=colors[0], edgecolor="none")

        # 1:1 line
        lims = [0, max(obs.max(), model.max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="1:1")

        # Regression
        coeffs = np.polyfit(obs, model, 1)
        r = np.corrcoef(obs, model)[0, 1]
        mb = np.mean(model - obs)

        ax.plot(lims, np.polyval(coeffs, lims), color=colors[3], linewidth=2)

        # Stats annotation
        text = f"N = {len(obs)}\nR = {r:.3f}\nMB = {mb:.2f}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Modeled {label}")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def create_diurnal_plot(paired_ds: xr.Dataset, output_path: Path) -> None:
    """Create diurnal cycle plot for O3."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette()

    # Group by hour
    obs = paired_ds["obs_o3"]
    model = paired_ds["model_o3"]

    obs_hourly = obs.groupby("time.hour").mean(dim=["time", "site"])
    model_hourly = model.groupby("time.hour").mean(dim=["time", "site"])

    hours = obs_hourly.hour.values

    ax.plot(hours, obs_hourly, "-o", label="Observations", linewidth=2, markersize=7,
            color=colors[0], markeredgecolor="white", markeredgewidth=1)
    ax.plot(hours, model_hourly, "-s", label="Model", linewidth=2, markersize=7,
            color=colors[3], markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("O$_3$ (ppbv)")
    ax.set_title("Diurnal Cycle of Ozone")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def create_spatial_bias_plot(paired_ds: xr.Dataset, output_path: Path) -> None:
    """Create spatial bias map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping spatial plot")
        return

    # Compute mean bias per site
    obs_mean = paired_ds["obs_o3"].mean(dim="time")
    model_mean = paired_ds["model_o3"].mean(dim="time")
    bias = model_mean - obs_mean

    lats = paired_ds["latitude"].values
    lons = paired_ds["longitude"].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")
    ax.add_feature(cfeature.LAKES, facecolor="#e6f3ff", edgecolor="gray", linewidth=0.3)

    # Scatter plot with bias as color
    vmax = max(abs(bias.min()), abs(bias.max()))
    sc = ax.scatter(
        lons, lats,
        c=bias.values,
        cmap="RdBu_r",
        vmin=-vmax, vmax=vmax,
        s=100,
        edgecolor="k",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("O$_3$ Bias (ppbv)")

    ax.set_title("Mean O$_3$ Model Bias by Site", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def main():
    """Run surface evaluation example."""
    print("DAVINCI-MONET Surface Evaluation Example")
    print("=" * 45)

    # Create output directory
    output_dir = Path("examples/output/surface")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic paired data
    print("\n1. Creating paired surface data...")
    paired_ds = create_paired_surface_data()
    print(f"   Shape: {paired_ds.dims}")
    print(f"   Variables: {list(paired_ds.data_vars)}")

    # Compute statistics
    print("\n2. Computing statistics...")
    stats = compute_all_statistics(paired_ds)
    print_statistics_table(stats)

    # Create plots
    print("\n3. Creating plots...")
    create_multi_panel_plot(paired_ds, output_dir / "scatter_multi.png")
    create_diurnal_plot(paired_ds, output_dir / "diurnal_o3.png")
    create_spatial_bias_plot(paired_ds, output_dir / "spatial_bias.png")

    # Save statistics to CSV
    print("\n4. Saving statistics...")
    from davinci_monet.stats import write_statistics_csv
    import pandas as pd

    stats_df = pd.DataFrame(stats).T
    stats_df.index.name = "variable"
    stats_df.to_csv(output_dir / "statistics.csv")
    print(f"   Saved: {output_dir / 'statistics.csv'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
