#!/usr/bin/env python
"""Demonstration of all DAVINCI-MONET plot types.

This script creates examples of all 10 plot types available in DAVINCI-MONET.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

# Set seaborn style for beautiful plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure as both PNG (300 DPI) and PDF."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def create_synthetic_data() -> xr.Dataset:
    """Create synthetic paired data for all plot demonstrations."""
    np.random.seed(42)

    # Time and spatial dimensions
    n_sites = 30
    times = np.arange("2024-07-01", "2024-07-15", dtype="datetime64[h]")

    # Station coordinates (US domain)
    site_lats = np.random.uniform(28, 48, n_sites)
    site_lons = np.random.uniform(-120, -70, n_sites)
    site_ids = [f"SITE_{i:03d}" for i in range(n_sites)]

    # Vertical levels for curtain plots
    levels = np.array([1000, 925, 850, 700, 500, 300, 200, 100])  # hPa

    # Generate O3 data with diurnal cycle
    time_hours = np.arange(len(times))
    hour_of_day = time_hours % 24
    diurnal = 25 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    base_o3 = 45 + 8 * (site_lats - 38) / 10

    obs_o3 = base_o3[np.newaxis, :] + diurnal[:, np.newaxis]
    obs_o3 += np.random.randn(*obs_o3.shape) * 8
    model_o3 = obs_o3 + 5 + np.random.randn(*obs_o3.shape) * 6
    obs_o3 = np.clip(obs_o3, 0, 150)
    model_o3 = np.clip(model_o3, 0, 150)

    # Generate PM2.5 data
    base_pm25 = 12 + 5 * np.random.randn(n_sites)
    obs_pm25 = base_pm25[np.newaxis, :] + np.random.randn(len(times), n_sites) * 4
    model_pm25 = obs_pm25 * 0.85 + np.random.randn(*obs_pm25.shape) * 3
    obs_pm25 = np.clip(obs_pm25, 0, 100)
    model_pm25 = np.clip(model_pm25, 0, 100)

    # Generate NO2 data
    urban_factor = np.random.uniform(0.5, 2.0, n_sites)
    base_no2 = 15 * urban_factor
    obs_no2 = base_no2[np.newaxis, :] + np.random.randn(len(times), n_sites) * 5
    model_no2 = obs_no2 + np.random.randn(*obs_no2.shape) * 4
    obs_no2 = np.clip(obs_no2, 0, 80)
    model_no2 = np.clip(model_no2, 0, 80)

    # Generate vertical profile data (for curtain plots)
    # O3 increases with altitude, model has slight bias
    base_profile = 40 + 30 * (1000 - levels) / 900  # O3 increases with altitude
    obs_profile = base_profile[np.newaxis, :] + np.random.randn(len(times), len(levels)) * 5
    model_profile = obs_profile * 1.1 + np.random.randn(*obs_profile.shape) * 4

    ds = xr.Dataset(
        {
            # Surface data
            "obs_o3": (["time", "site"], obs_o3.astype(np.float32)),
            "model_o3": (["time", "site"], model_o3.astype(np.float32)),
            "obs_pm25": (["time", "site"], obs_pm25.astype(np.float32)),
            "model_pm25": (["time", "site"], model_pm25.astype(np.float32)),
            "obs_no2": (["time", "site"], obs_no2.astype(np.float32)),
            "model_no2": (["time", "site"], model_no2.astype(np.float32)),
            # Profile data
            "obs_o3_profile": (["time", "level"], obs_profile.astype(np.float32)),
            "model_o3_profile": (["time", "level"], model_profile.astype(np.float32)),
        },
        coords={
            "time": times,
            "site": site_ids,
            "latitude": ("site", site_lats),
            "longitude": ("site", site_lons),
            "level": levels,
        },
        attrs={
            "title": "Synthetic Data for Plot Demonstrations",
            "source": "DAVINCI-MONET Example",
        },
    )

    return ds


# =============================================================================
# 1. TIME SERIES PLOT
# =============================================================================
def plot_timeseries(ds: xr.Dataset, output_dir: Path) -> None:
    """Create time series plot."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = sns.color_palette()

    obs_mean = ds["obs_o3"].mean(dim="site")
    model_mean = ds["model_o3"].mean(dim="site")

    ax.plot(ds["time"], obs_mean, label="Observations", linewidth=1.5, color=colors[0])
    ax.plot(ds["time"], model_mean, label="Model", linewidth=1.5, color=colors[3])

    ax.fill_between(ds["time"].values,
                    ds["obs_o3"].quantile(0.25, dim="site"),
                    ds["obs_o3"].quantile(0.75, dim="site"),
                    alpha=0.2, color=colors[0], label="Obs IQR")

    ax.set_xlabel("Time")
    ax.set_ylabel("O$_3$ (ppbv)")
    ax.set_title("1. Time Series Plot")
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    save_figure(fig, output_dir / "01_timeseries.png")
    plt.close(fig)


# =============================================================================
# 2. DIURNAL CYCLE PLOT
# =============================================================================
def plot_diurnal(ds: xr.Dataset, output_dir: Path) -> None:
    """Create diurnal cycle plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette()

    obs_hourly = ds["obs_o3"].groupby("time.hour").mean(dim=["time", "site"])
    model_hourly = ds["model_o3"].groupby("time.hour").mean(dim=["time", "site"])
    obs_std = ds["obs_o3"].groupby("time.hour").std(dim=["time", "site"])

    hours = obs_hourly.hour.values

    ax.fill_between(hours, obs_hourly - obs_std, obs_hourly + obs_std,
                    alpha=0.3, color=colors[0])
    ax.plot(hours, obs_hourly, "-o", label="Observations", linewidth=2, markersize=7,
            color=colors[0], markeredgecolor="white", markeredgewidth=1)
    ax.plot(hours, model_hourly, "-s", label="Model", linewidth=2, markersize=7,
            color=colors[3], markeredgecolor="white", markeredgewidth=1)

    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("O$_3$ (ppbv)")
    ax.set_title("2. Diurnal Cycle Plot")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    save_figure(fig, output_dir / "02_diurnal.png")
    plt.close(fig)


# =============================================================================
# 3. SCATTER PLOT
# =============================================================================
def plot_scatter(ds: xr.Dataset, output_dir: Path) -> None:
    """Create scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = sns.color_palette()

    obs = ds["obs_o3"].values.flatten()
    model = ds["model_o3"].values.flatten()
    mask = ~(np.isnan(obs) | np.isnan(model))
    obs, model = obs[mask], model[mask]

    ax.scatter(obs, model, alpha=0.3, s=8, color=colors[0], edgecolor="none")

    lims = [0, max(obs.max(), model.max()) * 1.1]
    ax.plot(lims, lims, "k--", label="1:1", linewidth=1.5)

    coeffs = np.polyfit(obs, model, 1)
    r = np.corrcoef(obs, model)[0, 1]
    ax.plot(lims, np.polyval(coeffs, lims), color=colors[3],
            linewidth=2, label=f"Fit (R={r:.2f})")

    ax.set_xlabel("Observed O$_3$ (ppbv)")
    ax.set_ylabel("Modeled O$_3$ (ppbv)")
    ax.set_title("3. Scatter Plot")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(frameon=True, fancybox=True)
    ax.set_aspect("equal")

    plt.tight_layout()
    save_figure(fig, output_dir / "03_scatter.png")
    plt.close(fig)


# =============================================================================
# 4. TAYLOR DIAGRAM
# =============================================================================
def plot_taylor(ds: xr.Dataset, output_dir: Path) -> None:
    """Create Taylor diagram."""
    fig = plt.figure(figsize=(8, 8))
    colors = sns.color_palette()

    # Compute statistics for multiple variables
    variables = [
        ("o3", "O$_3$", colors[0]),
        ("pm25", "PM$_{2.5}$", colors[1]),
        ("no2", "NO$_2$", colors[2]),
    ]

    # Create polar axes for Taylor diagram
    ax = fig.add_subplot(111, polar=True)

    # Limit to first quadrant
    ax.set_thetamax(90)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    stats_list = []
    for var, label, color in variables:
        obs = ds[f"obs_{var}"].values.flatten()
        mod = ds[f"model_{var}"].values.flatten()
        mask = ~(np.isnan(obs) | np.isnan(mod))
        obs, mod = obs[mask], mod[mask]

        # Normalize by obs std
        obs_std = np.std(obs)
        mod_std = np.std(mod) / obs_std
        corr = np.corrcoef(obs, mod)[0, 1]

        stats_list.append((mod_std, corr, label, color))

    # Plot reference point (observations)
    ax.plot(0, 1, "ko", markersize=12, label="Reference")

    # Plot model points
    for std, corr, label, color in stats_list:
        theta = np.arccos(corr)
        ax.plot(theta, std, "o", markersize=10, color=color, label=label,
                markeredgecolor="white", markeredgewidth=1.5)

    # Add correlation lines
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        theta = np.arccos(r)
        ax.plot([theta, theta], [0, 2], "gray", alpha=0.3, linewidth=0.5)
        if r in [0.5, 0.9, 0.99]:
            ax.text(theta, 2.05, f"{r}", ha="center", fontsize=8, color="gray")

    # Add std circles
    for s in [0.5, 1.0, 1.5, 2.0]:
        circle = plt.Circle((0, 0), s, transform=ax.transData._b, fill=False,
                           color="gray", alpha=0.3, linewidth=0.5)
        ax.add_patch(circle)

    ax.set_ylim(0, 2)
    ax.set_xlabel("Standard Deviation (normalized)")
    ax.set_title("4. Taylor Diagram", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), frameon=True)

    plt.tight_layout()
    save_figure(fig, output_dir / "04_taylor.png")
    plt.close(fig)


# =============================================================================
# 5. BOX PLOT
# =============================================================================
def plot_boxplot(ds: xr.Dataset, output_dir: Path) -> None:
    """Create box plot comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette()

    # Prepare data for boxplot
    variables = ["O$_3$", "PM$_{2.5}$", "NO$_2$"]
    obs_data = [
        ds["obs_o3"].values.flatten(),
        ds["obs_pm25"].values.flatten(),
        ds["obs_no2"].values.flatten(),
    ]
    model_data = [
        ds["model_o3"].values.flatten(),
        ds["model_pm25"].values.flatten(),
        ds["model_no2"].values.flatten(),
    ]

    positions_obs = np.array([1, 4, 7]) - 0.4
    positions_mod = np.array([1, 4, 7]) + 0.4

    bp_obs = ax.boxplot(obs_data, positions=positions_obs, widths=0.6,
                        patch_artist=True, showfliers=False)
    bp_mod = ax.boxplot(model_data, positions=positions_mod, widths=0.6,
                        patch_artist=True, showfliers=False)

    # Style boxplots
    for box in bp_obs["boxes"]:
        box.set_facecolor(colors[0])
        box.set_alpha(0.7)
    for box in bp_mod["boxes"]:
        box.set_facecolor(colors[3])
        box.set_alpha(0.7)

    ax.set_xticks([1, 4, 7])
    ax.set_xticklabels(variables)
    ax.set_ylabel("Concentration")
    ax.set_title("5. Box Plot Comparison")

    # Legend
    ax.plot([], [], "s", color=colors[0], markersize=10, label="Observations")
    ax.plot([], [], "s", color=colors[3], markersize=10, label="Model")
    ax.legend(frameon=True, fancybox=True)

    plt.tight_layout()
    save_figure(fig, output_dir / "05_boxplot.png")
    plt.close(fig)


# =============================================================================
# 6. SPATIAL BIAS MAP
# =============================================================================
def plot_spatial_bias(ds: xr.Dataset, output_dir: Path) -> None:
    """Create spatial bias map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping spatial bias plot")
        return

    bias = ds["model_o3"].mean(dim="time") - ds["obs_o3"].mean(dim="time")
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

    vmax = max(abs(bias.min()), abs(bias.max()))
    sc = ax.scatter(lons, lats, c=bias.values, cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, s=120, edgecolor="k",
                    linewidth=0.8, transform=ccrs.PlateCarree(), zorder=5)

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("O$_3$ Bias (ppbv)")
    ax.set_title("6. Spatial Bias Map", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_dir / "06_spatial_bias.png")
    plt.close(fig)


# =============================================================================
# 7. SPATIAL OVERLAY MAP
# =============================================================================
def plot_spatial_overlay(ds: xr.Dataset, output_dir: Path) -> None:
    """Create spatial overlay map (model field + obs points)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping spatial overlay plot")
        return

    # Create a synthetic model grid field
    lats_grid = np.linspace(25, 50, 50)
    lons_grid = np.linspace(-125, -65, 100)
    lon_mesh, lat_mesh = np.meshgrid(lons_grid, lats_grid)

    # Synthetic model field with gradient
    model_field = 40 + 15 * (lat_mesh - 35) / 15 + 5 * np.sin(lon_mesh / 10)
    model_field += np.random.randn(*model_field.shape) * 3

    # Obs data
    obs_mean = ds["obs_o3"].mean(dim="time")
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

    # Plot model field as contour
    cf = ax.contourf(lon_mesh, lat_mesh, model_field, levels=20,
                     cmap="YlOrRd", transform=ccrs.PlateCarree(), alpha=0.8)

    # Overlay obs points
    sc = ax.scatter(lons, lats, c=obs_mean.values, cmap="YlOrRd",
                    vmin=model_field.min(), vmax=model_field.max(),
                    s=100, edgecolor="k", linewidth=1.5,
                    transform=ccrs.PlateCarree(), zorder=5)

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

    cbar = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("O$_3$ (ppbv)")
    ax.set_title("7. Spatial Overlay (Model Field + Obs)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_dir / "07_spatial_overlay.png")
    plt.close(fig)


# =============================================================================
# 8. SPATIAL DISTRIBUTION MAP
# =============================================================================
def plot_spatial_distribution(ds: xr.Dataset, output_dir: Path) -> None:
    """Create spatial distribution map showing value ranges."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, skipping spatial distribution plot")
        return

    obs_mean = ds["obs_o3"].mean(dim="time")
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

    sc = ax.scatter(lons, lats, c=obs_mean.values, cmap="viridis",
                    s=120, edgecolor="k", linewidth=0.8,
                    transform=ccrs.PlateCarree(), zorder=5)

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("Mean O$_3$ (ppbv)")
    ax.set_title("8. Spatial Distribution Map", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_dir / "08_spatial_distribution.png")
    plt.close(fig)


# =============================================================================
# 9. CURTAIN PLOT (Vertical Cross-Section)
# =============================================================================
def plot_curtain(ds: xr.Dataset, output_dir: Path) -> None:
    """Create curtain plot (vertical cross-section over time)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    times = ds["time"].values
    levels = ds["level"].values

    # Observations
    obs = ds["obs_o3_profile"].values.T  # (level, time)
    im1 = axes[0].pcolormesh(times, levels, obs, cmap="YlOrRd", shading="auto")
    axes[0].set_ylabel("Pressure (hPa)")
    axes[0].set_title("Observations")
    axes[0].invert_yaxis()
    plt.colorbar(im1, ax=axes[0], label="O$_3$ (ppbv)")

    # Model
    model = ds["model_o3_profile"].values.T
    im2 = axes[1].pcolormesh(times, levels, model, cmap="YlOrRd", shading="auto",
                              vmin=obs.min(), vmax=obs.max())
    axes[1].set_ylabel("Pressure (hPa)")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Model")
    axes[1].invert_yaxis()
    plt.colorbar(im2, ax=axes[1], label="O$_3$ (ppbv)")

    fig.suptitle("9. Curtain Plot (Vertical Cross-Section)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    save_figure(fig, output_dir / "09_curtain.png")
    plt.close(fig)


# =============================================================================
# 10. SCORECARD
# =============================================================================
def plot_scorecard(ds: xr.Dataset, output_dir: Path) -> None:
    """Create scorecard showing multiple metrics for multiple variables."""
    fig, ax = plt.subplots(figsize=(10, 6))

    variables = ["O$_3$", "PM$_{2.5}$", "NO$_2$"]
    metrics = ["R", "MB", "RMSE", "NMB (%)", "IOA"]

    # Compute statistics
    data = []
    for var_name in ["o3", "pm25", "no2"]:
        obs = ds[f"obs_{var_name}"].values.flatten()
        mod = ds[f"model_{var_name}"].values.flatten()
        mask = ~(np.isnan(obs) | np.isnan(mod))
        obs, mod = obs[mask], mod[mask]

        r = np.corrcoef(obs, mod)[0, 1]
        mb = np.mean(mod - obs)
        rmse = np.sqrt(np.mean((mod - obs) ** 2))
        nmb = 100 * np.sum(mod - obs) / np.sum(obs)
        ioa = 1 - np.sum((mod - obs) ** 2) / np.sum((np.abs(mod - obs.mean()) + np.abs(obs - obs.mean())) ** 2)

        data.append([r, mb, rmse, nmb, ioa])

    data = np.array(data)

    # Normalize for color mapping (different scales for different metrics)
    # R: higher is better (0-1), MB: closer to 0 is better, etc.
    colors_data = np.zeros_like(data)
    colors_data[:, 0] = data[:, 0]  # R: direct (higher=better)
    colors_data[:, 1] = 1 - np.abs(data[:, 1]) / np.abs(data[:, 1]).max()  # MB: inverse abs
    colors_data[:, 2] = 1 - data[:, 2] / data[:, 2].max()  # RMSE: inverse
    colors_data[:, 3] = 1 - np.abs(data[:, 3]) / np.abs(data[:, 3]).max()  # NMB: inverse abs
    colors_data[:, 4] = data[:, 4]  # IOA: direct

    im = ax.imshow(colors_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Add text annotations
    for i in range(len(variables)):
        for j in range(len(metrics)):
            val = data[i, j]
            fmt = ".3f" if j in [0, 4] else ".2f"
            text = ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                          color="black", fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(variables)))
    ax.set_yticklabels(variables)
    ax.set_title("10. Scorecard", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Performance (green=better)")

    plt.tight_layout()
    save_figure(fig, output_dir / "10_scorecard.png")
    plt.close(fig)


def main():
    """Generate all plot type examples."""
    print("DAVINCI-MONET: All Plot Types Demo")
    print("=" * 50)

    output_dir = Path("output/all_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nCreating synthetic data...")
    ds = create_synthetic_data()
    print(f"   Surface dims: {dict(ds.dims)}")

    print("\nGenerating plots...")

    print("\n1. Time Series Plot")
    plot_timeseries(ds, output_dir)

    print("\n2. Diurnal Cycle Plot")
    plot_diurnal(ds, output_dir)

    print("\n3. Scatter Plot")
    plot_scatter(ds, output_dir)

    print("\n4. Taylor Diagram")
    plot_taylor(ds, output_dir)

    print("\n5. Box Plot")
    plot_boxplot(ds, output_dir)

    print("\n6. Spatial Bias Map")
    plot_spatial_bias(ds, output_dir)

    print("\n7. Spatial Overlay Map")
    plot_spatial_overlay(ds, output_dir)

    print("\n8. Spatial Distribution Map")
    plot_spatial_distribution(ds, output_dir)

    print("\n9. Curtain Plot")
    plot_curtain(ds, output_dir)

    print("\n10. Scorecard")
    plot_scorecard(ds, output_dir)

    print("\n" + "=" * 50)
    print("All 10 plot types generated!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
