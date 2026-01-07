#!/usr/bin/env python
"""Sonde profile evaluation example using synthetic data.

This script demonstrates how to:
1. Generate synthetic ozonesonde profile observations
2. Generate synthetic 3D model output
3. Pair sonde profiles with model columns
4. Create various plots for profile evaluation:
   - Sonde launch locations map
   - Individual profile comparisons
   - Multi-profile overlay
   - Bias profile
   - Statistics by altitude layer
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

# DAVINCI-MONET imports
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_3d_model
from davinci_monet.tests.synthetic.observations import create_profile_observations

# Set seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure as both PNG (300 DPI) and PDF."""
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_path}")


def create_sonde_data() -> tuple[xr.Dataset, xr.Dataset]:
    """Create synthetic sonde and model data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (model_data, sonde_data)
    """
    # Define domain (US region)
    domain = Domain(
        lon_min=-110, lon_max=-70,
        lat_min=25, lat_max=50,
        n_lon=80, n_lat=50,
    )

    # Time configuration - week of soundings
    time_config = TimeConfig(
        start="2024-07-01T00:00:00",
        end="2024-07-07T23:59:59",
        freq="6h",
    )

    # Create 3D model with O3 and temperature
    model = create_3d_model(
        variables=["O3", "temperature"],
        domain=domain,
        time_config=time_config,
        n_levels=30,
        seed=42,
    )

    # Rename level to z for pairing strategy
    altitudes = np.linspace(0, 30000, 30)  # 0 to 30 km
    model = model.rename({"level": "z"})
    model = model.assign_coords(z=altitudes)

    # Create sonde profiles
    sonde = create_profile_observations(
        n_profiles=12,  # 12 soundings over the week
        n_levels=50,    # 50 vertical levels per sonde
        variables=["O3", "temperature"],
        domain=domain,
        time_config=time_config,
        seed=123,
    )

    # Convert sonde levels to altitude (km) for clearer plots
    sonde_altitudes = np.linspace(0, 30, 50)  # 0 to 30 km
    sonde = sonde.assign_coords(level=sonde_altitudes)
    sonde["level"].attrs["units"] = "km"
    sonde["level"].attrs["long_name"] = "Altitude"

    return model, sonde


def pair_sonde_with_model(
    model: xr.Dataset,
    sonde: xr.Dataset,
    profile_idx: int,
) -> xr.Dataset:
    """Pair a single sonde profile with model column.

    Parameters
    ----------
    model
        3D model dataset
    sonde
        Sonde dataset with multiple profiles
    profile_idx
        Index of the profile to pair

    Returns
    -------
    xr.Dataset
        Paired dataset with model and obs profiles
    """
    # Extract single profile
    single_profile = sonde.isel(time=profile_idx)
    profile_time = sonde.time.values[profile_idx]
    profile_lat = float(sonde.latitude.values[profile_idx])
    profile_lon = float(sonde.longitude.values[profile_idx])

    # Find nearest model column
    lat_idx = int(np.abs(model.lat.values - profile_lat).argmin())
    lon_idx = int(np.abs(model.lon.values - profile_lon).argmin())

    # Find nearest model time
    time_idx = int(np.abs(model.time.values - profile_time).argmin())

    # Extract model column
    model_column = model.isel(time=time_idx, lat=lat_idx, lon=lon_idx)

    # Interpolate model to sonde levels
    sonde_levels = sonde.level.values * 1000  # km to m
    model_interp = model_column.interp(z=sonde_levels, method="linear")

    # Build paired dataset
    paired = xr.Dataset(
        {
            "O3": single_profile["O3"],
            "temperature": single_profile["temperature"],
            "model_O3": (["level"], model_interp["O3"].values),
            "model_temperature": (["level"], model_interp["temperature"].values),
        },
        coords={
            "level": sonde.level,
            "latitude": profile_lat,
            "longitude": profile_lon,
            "time": profile_time,
        },
    )

    return paired


def pair_all_sondes(model: xr.Dataset, sonde: xr.Dataset) -> list[xr.Dataset]:
    """Pair all sonde profiles with model."""
    paired_list = []
    for i in range(sonde.sizes["time"]):
        paired = pair_sonde_with_model(model, sonde, i)
        paired_list.append(paired)
    return paired_list


def plot_launch_locations(sonde: xr.Dataset, output_dir: Path) -> None:
    """Plot sonde launch locations on a map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    lats = sonde.latitude.values
    lons = sonde.longitude.values
    times = sonde.time.values

    if has_cartopy:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        ax.set_extent([-115, -65, 22, 52], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#e6f3ff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.3, linestyle=":")

        # Color by launch order
        colors = plt.cm.viridis(np.linspace(0, 1, len(lats)))

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            ax.plot(lon, lat, "o", markersize=12, color=colors[i],
                   transform=ccrs.PlateCarree(), markeredgecolor="k",
                   markeredgewidth=1, zorder=5)
            ax.text(lon + 0.5, lat + 0.5, f"{i+1}",
                   transform=ccrs.PlateCarree(), fontsize=9, fontweight="bold")

        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                    norm=plt.Normalize(0, len(lats)-1))
        cbar = plt.colorbar(sm, ax=ax, orientation="horizontal",
                           pad=0.05, shrink=0.6)
        cbar.set_label("Sounding Number")

        ax.set_title(f"Ozonesonde Launch Locations (n={len(lats)})",
                    fontsize=14, fontweight="bold")
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(lons, lats, c=range(len(lats)), cmap="viridis",
                       s=100, edgecolor="k")
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            ax.text(lon + 0.5, lat + 0.5, f"{i+1}", fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Sonde Launch Locations")
        plt.colorbar(sc, label="Sounding Number")

    save_figure(fig, output_dir / "launch_locations.png")
    plt.close(fig)


def plot_single_profile(paired: xr.Dataset, idx: int, output_dir: Path) -> None:
    """Plot a single profile comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 8), sharey=True)
    colors = sns.color_palette()

    levels = paired.level.values

    # O3 profile
    ax1 = axes[0]
    ax1.plot(paired["O3"], levels, "-o", label="Sonde", linewidth=2,
            markersize=4, color=colors[0])
    ax1.plot(paired["model_O3"], levels, "-s", label="Model", linewidth=2,
            markersize=4, color=colors[3])
    ax1.set_xlabel("O$_3$ (ppbv)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Ozone")
    ax1.legend()
    ax1.set_ylim(0, 30)

    # Temperature profile
    ax2 = axes[1]
    ax2.plot(paired["temperature"], levels, "-o", label="Sonde", linewidth=2,
            markersize=4, color=colors[0])
    ax2.plot(paired["model_temperature"], levels, "-s", label="Model", linewidth=2,
            markersize=4, color=colors[3])
    ax2.set_xlabel("Temperature (K)")
    ax2.set_title("Temperature")
    ax2.legend()

    lat = float(paired.latitude)
    lon = float(paired.longitude)
    time_str = str(paired.time.values)[:16]

    fig.suptitle(f"Sonde #{idx+1}: {time_str}\n({lat:.1f}°N, {lon:.1f}°W)",
                fontsize=12, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / f"profile_{idx+1:02d}.png")
    plt.close(fig)


def plot_multi_profile_overlay(paired_list: list[xr.Dataset], output_dir: Path) -> None:
    """Plot all profiles overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    colors = sns.color_palette()

    n_profiles = len(paired_list)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_profiles))

    for i, paired in enumerate(paired_list):
        levels = paired.level.values

        # O3
        axes[0].plot(paired["O3"], levels, "-", alpha=0.7, color=cmap[i],
                    linewidth=1.5)
        axes[0].plot(paired["model_O3"], levels, "--", alpha=0.7, color=cmap[i],
                    linewidth=1.5)

        # Temperature
        axes[1].plot(paired["temperature"], levels, "-", alpha=0.7, color=cmap[i],
                    linewidth=1.5)
        axes[1].plot(paired["model_temperature"], levels, "--", alpha=0.7, color=cmap[i],
                    linewidth=1.5)

    # Add legend entries
    axes[0].plot([], [], "k-", linewidth=2, label="Sonde")
    axes[0].plot([], [], "k--", linewidth=2, label="Model")
    axes[0].set_xlabel("O$_3$ (ppbv)")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Ozone Profiles")
    axes[0].legend()
    axes[0].set_ylim(0, 30)

    axes[1].plot([], [], "k-", linewidth=2, label="Sonde")
    axes[1].plot([], [], "k--", linewidth=2, label="Model")
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_title("Temperature Profiles")
    axes[1].legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, n_profiles))
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", shrink=0.6, pad=0.02)
    cbar.set_label("Sounding Number")

    fig.suptitle(f"All Ozonesonde Profiles (n={n_profiles})",
                fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "profiles_overlay.png")
    plt.close(fig)


def plot_mean_bias_profile(paired_list: list[xr.Dataset], output_dir: Path) -> None:
    """Plot mean and bias profiles."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    colors = sns.color_palette()

    # Collect all profiles
    levels = paired_list[0].level.values
    n_profiles = len(paired_list)

    o3_obs = np.array([p["O3"].values for p in paired_list])
    o3_mod = np.array([p["model_O3"].values for p in paired_list])
    temp_obs = np.array([p["temperature"].values for p in paired_list])
    temp_mod = np.array([p["model_temperature"].values for p in paired_list])

    # Mean profiles
    ax1 = axes[0]
    ax1.fill_betweenx(levels, np.nanpercentile(o3_obs, 25, axis=0),
                      np.nanpercentile(o3_obs, 75, axis=0),
                      alpha=0.3, color=colors[0])
    ax1.plot(np.nanmean(o3_obs, axis=0), levels, "-", label="Sonde Mean",
            linewidth=2.5, color=colors[0])
    ax1.fill_betweenx(levels, np.nanpercentile(o3_mod, 25, axis=0),
                      np.nanpercentile(o3_mod, 75, axis=0),
                      alpha=0.3, color=colors[3])
    ax1.plot(np.nanmean(o3_mod, axis=0), levels, "-", label="Model Mean",
            linewidth=2.5, color=colors[3])
    ax1.set_xlabel("O$_3$ (ppbv)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Mean O$_3$ Profiles")
    ax1.legend()
    ax1.set_ylim(0, 30)

    # O3 Bias profile
    ax2 = axes[1]
    o3_bias = o3_mod - o3_obs
    o3_bias_mean = np.nanmean(o3_bias, axis=0)
    o3_bias_std = np.nanstd(o3_bias, axis=0)

    ax2.axvline(0, color="k", linestyle="--", linewidth=1)
    ax2.fill_betweenx(levels, o3_bias_mean - o3_bias_std,
                      o3_bias_mean + o3_bias_std,
                      alpha=0.3, color=colors[2])
    ax2.plot(o3_bias_mean, levels, "-", linewidth=2.5, color=colors[2])
    ax2.set_xlabel("O$_3$ Bias (ppbv)")
    ax2.set_title("O$_3$ Bias (Model - Sonde)")

    # Temperature bias profile
    ax3 = axes[2]
    temp_bias = temp_mod - temp_obs
    temp_bias_mean = np.nanmean(temp_bias, axis=0)
    temp_bias_std = np.nanstd(temp_bias, axis=0)

    ax3.axvline(0, color="k", linestyle="--", linewidth=1)
    ax3.fill_betweenx(levels, temp_bias_mean - temp_bias_std,
                      temp_bias_mean + temp_bias_std,
                      alpha=0.3, color=colors[1])
    ax3.plot(temp_bias_mean, levels, "-", linewidth=2.5, color=colors[1])
    ax3.set_xlabel("Temperature Bias (K)")
    ax3.set_title("Temperature Bias (Model - Sonde)")

    fig.suptitle(f"Mean and Bias Profiles (n={n_profiles})",
                fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "mean_bias_profiles.png")
    plt.close(fig)


def plot_layer_statistics(paired_list: list[xr.Dataset], output_dir: Path) -> None:
    """Plot statistics by altitude layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors = sns.color_palette()

    # Define altitude layers
    layer_bounds = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    layer_labels = ["0-2 km", "2-5 km", "5-10 km", "10-15 km", "15-20 km", "20-30 km"]

    levels = paired_list[0].level.values
    n_profiles = len(paired_list)

    o3_obs = np.array([p["O3"].values for p in paired_list])
    o3_mod = np.array([p["model_O3"].values for p in paired_list])

    # Compute layer statistics
    layer_mb = []
    layer_rmse = []
    layer_r = []

    for low, high in layer_bounds:
        mask = (levels >= low) & (levels < high)
        obs_layer = o3_obs[:, mask].flatten()
        mod_layer = o3_mod[:, mask].flatten()

        valid = ~(np.isnan(obs_layer) | np.isnan(mod_layer))
        obs_v = obs_layer[valid]
        mod_v = mod_layer[valid]

        if len(obs_v) > 0:
            layer_mb.append(np.mean(mod_v - obs_v))
            layer_rmse.append(np.sqrt(np.mean((mod_v - obs_v) ** 2)))
            layer_r.append(np.corrcoef(obs_v, mod_v)[0, 1] if len(obs_v) > 1 else np.nan)
        else:
            layer_mb.append(np.nan)
            layer_rmse.append(np.nan)
            layer_r.append(np.nan)

    # Plot MB and RMSE
    ax1 = axes[0]
    x = np.arange(len(layer_labels))
    width = 0.35

    bars1 = ax1.bar(x - width/2, layer_mb, width, label="Mean Bias", color=colors[0])
    bars2 = ax1.bar(x + width/2, layer_rmse, width, label="RMSE", color=colors[3])

    ax1.axhline(0, color="k", linestyle="--", linewidth=1)
    ax1.set_xlabel("Altitude Layer")
    ax1.set_ylabel("O$_3$ (ppbv)")
    ax1.set_title("O$_3$ Error Statistics by Layer")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax1.legend()

    # Plot correlation
    ax2 = axes[1]
    bars3 = ax2.bar(x, layer_r, width=0.6, color=colors[2])
    ax2.axhline(0.9, color="r", linestyle="--", linewidth=1, label="R=0.9")
    ax2.set_xlabel("Altitude Layer")
    ax2.set_ylabel("Correlation (R)")
    ax2.set_title("O$_3$ Correlation by Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax2.set_ylim(0, 1.1)
    ax2.legend()

    fig.suptitle(f"Layer Statistics (n={n_profiles} profiles)",
                fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "layer_statistics.png")
    plt.close(fig)


def plot_scatter_by_layer(paired_list: list[xr.Dataset], output_dir: Path) -> None:
    """Plot scatter plots for different altitude layers."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    colors = sns.color_palette()

    layer_bounds = [(0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30)]
    layer_labels = ["0-2 km\n(Boundary Layer)", "2-5 km\n(Lower Free Trop.)",
                   "5-10 km\n(Mid Troposphere)", "10-15 km\n(Upper Trop./Trop.)",
                   "15-20 km\n(Lower Strat.)", "20-30 km\n(Stratosphere)"]

    levels = paired_list[0].level.values
    o3_obs = np.array([p["O3"].values for p in paired_list])
    o3_mod = np.array([p["model_O3"].values for p in paired_list])

    for i, ((low, high), label) in enumerate(zip(layer_bounds, layer_labels)):
        ax = axes[i]
        mask = (levels >= low) & (levels < high)

        obs_layer = o3_obs[:, mask].flatten()
        mod_layer = o3_mod[:, mask].flatten()

        valid = ~(np.isnan(obs_layer) | np.isnan(mod_layer))
        obs_v = obs_layer[valid]
        mod_v = mod_layer[valid]

        if len(obs_v) > 0:
            ax.scatter(obs_v, mod_v, alpha=0.5, s=20, color=colors[i % len(colors)],
                      edgecolor="none")

            lims = [min(obs_v.min(), mod_v.min()) * 0.9,
                   max(obs_v.max(), mod_v.max()) * 1.1]
            ax.plot(lims, lims, "k--", linewidth=1.5)

            r = np.corrcoef(obs_v, mod_v)[0, 1]
            mb = np.mean(mod_v - obs_v)
            ax.text(0.05, 0.95, f"R={r:.2f}\nMB={mb:.1f}",
                   transform=ax.transAxes, va="top", fontsize=10,
                   bbox=dict(facecolor="white", alpha=0.8))

            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_xlabel("Sonde O$_3$")
        ax.set_ylabel("Model O$_3$")
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle("O$_3$ Scatter by Altitude Layer", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_figure(fig, output_dir / "scatter_by_layer.png")
    plt.close(fig)


def main():
    """Run sonde evaluation example."""
    print("DAVINCI-MONET: Sonde Profile Evaluation Example")
    print("=" * 50)

    output_dir = Path("examples/output/sonde")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("\n1. Creating synthetic data...")
    model, sonde = create_sonde_data()
    print(f"   Model dims: {dict(model.sizes)}")
    print(f"   Sonde dims: {dict(sonde.sizes)}")
    print(f"   Number of profiles: {sonde.sizes['time']}")
    print(f"   Vertical levels per profile: {sonde.sizes['level']}")

    # Pair all sondes
    print("\n2. Pairing sondes with model...")
    paired_list = pair_all_sondes(model, sonde)
    print(f"   Paired {len(paired_list)} profiles")

    # Create plots
    print("\n3. Generating plots...")

    print("   - Launch locations map")
    plot_launch_locations(sonde, output_dir)

    print("   - Individual profile plots (first 3)")
    for i in range(min(3, len(paired_list))):
        plot_single_profile(paired_list[i], i, output_dir)

    print("   - Multi-profile overlay")
    plot_multi_profile_overlay(paired_list, output_dir)

    print("   - Mean and bias profiles")
    plot_mean_bias_profile(paired_list, output_dir)

    print("   - Layer statistics")
    plot_layer_statistics(paired_list, output_dir)

    print("   - Scatter by layer")
    plot_scatter_by_layer(paired_list, output_dir)

    print("\n" + "=" * 50)
    print("Sonde evaluation complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
