#!/usr/bin/env python
"""Aircraft evaluation example using synthetic data.

This script demonstrates how to:
1. Generate synthetic aircraft track observations
2. Generate synthetic 3D model output
3. Pair aircraft data with model data along the flight track
4. Create various plots for aircraft evaluation:
   - Flight path map
   - Curtain plot (altitude vs time)
   - Scatter plot (model vs obs)
   - Time series along track
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

# DAVINCI-MONET imports
from davinci_monet.tests.synthetic.generators import Domain, TimeConfig
from davinci_monet.tests.synthetic.models import create_3d_model
from davinci_monet.tests.synthetic.observations import create_track_observations
from davinci_monet.pairing.strategies.track import TrackStrategy

# Set seaborn style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def create_aircraft_data() -> tuple[xr.Dataset, xr.Dataset]:
    """Create synthetic aircraft track and model data.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (model_data, aircraft_data)
    """
    # Define domain (US region)
    domain = Domain(
        lon_min=-105, lon_max=-85,
        lat_min=30, lat_max=45,
        n_lon=40, n_lat=30,
    )

    # Time configuration - 6 hour flight
    time_config = TimeConfig(
        start="2024-07-15T12:00:00",
        end="2024-07-15T18:00:00",
        freq="10min",
    )

    # Create 3D model with O3 and CO
    model = create_3d_model(
        variables=["O3", "CO"],
        domain=domain,
        time_config=time_config,
        n_levels=20,
        seed=42,
    )

    # Create aircraft track
    aircraft = create_track_observations(
        n_points=200,
        variables=["O3", "CO"],
        domain=domain,
        time_config=time_config,
        altitude_range=(1000.0, 10000.0),  # meters
        seed=123,
    )

    return model, aircraft


def pair_aircraft_with_model(
    model: xr.Dataset,
    aircraft: xr.Dataset
) -> xr.Dataset:
    """Pair aircraft observations with model output.

    Parameters
    ----------
    model
        3D model dataset
    aircraft
        Aircraft track dataset

    Returns
    -------
    xr.Dataset
        Paired dataset with model values along flight track
    """
    # Rename model vertical dimension to 'z' for pairing strategy
    if "level" in model.dims:
        # Create altitude coordinate from level index (simplified)
        n_levels = model.sizes["level"]
        altitudes = np.linspace(0, 15000, n_levels)  # 0 to 15km
        model = model.rename({"level": "z"})
        model = model.assign_coords(z=altitudes)

    strategy = TrackStrategy()
    paired = strategy.pair(
        model,
        aircraft,
        vertical_method="linear",
        horizontal_method="nearest",
    )

    return paired


def plot_flight_path(aircraft: xr.Dataset, output_dir: Path) -> None:
    """Plot the aircraft flight path on a map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not available, creating simple flight path plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(
            aircraft["longitude"],
            aircraft["latitude"],
            c=aircraft["altitude"] / 1000,
            cmap="viridis",
            s=20,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Aircraft Flight Path")
        plt.colorbar(sc, label="Altitude (km)")
        fig.savefig(output_dir / "flight_path.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = aircraft["longitude"].values
    lats = aircraft["latitude"].values
    alts = aircraft["altitude"].values / 1000  # Convert to km

    # Set extent with some padding
    lon_pad = (lons.max() - lons.min()) * 0.2
    lat_pad = (lats.max() - lats.min()) * 0.2
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

    # Plot flight track colored by altitude
    sc = ax.scatter(
        lons, lats,
        c=alts,
        cmap="viridis",
        s=30,
        edgecolor="k",
        linewidth=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    # Mark start and end points
    ax.plot(lons[0], lats[0], "go", markersize=12, transform=ccrs.PlateCarree(),
            label="Start", zorder=6)
    ax.plot(lons[-1], lats[-1], "rs", markersize=12, transform=ccrs.PlateCarree(),
            label="End", zorder=6)

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    cbar.set_label("Altitude (km)")

    ax.set_title("Aircraft Flight Path", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right")

    fig.savefig(output_dir / "flight_path.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'flight_path.png'}")
    plt.close(fig)


def plot_curtain(paired: xr.Dataset, aircraft: xr.Dataset, output_dir: Path) -> None:
    """Create curtain plot showing O3 along the flight track."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = sns.color_palette()

    times = paired["time"].values
    alts = aircraft["altitude"].values / 1000  # km

    obs_o3 = paired["O3"].values
    model_o3 = paired["model_O3"].values
    bias = model_o3 - obs_o3

    # Panel 1: Observations
    sc1 = axes[0].scatter(times, alts, c=obs_o3, cmap="YlOrRd", s=30, edgecolor="none")
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Observed O$_3$")
    plt.colorbar(sc1, ax=axes[0], label="O$_3$ (ppbv)")

    # Panel 2: Model
    sc2 = axes[1].scatter(times, alts, c=model_o3, cmap="YlOrRd", s=30, edgecolor="none",
                          vmin=np.nanmin(obs_o3), vmax=np.nanmax(obs_o3))
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_title("Model O$_3$")
    plt.colorbar(sc2, ax=axes[1], label="O$_3$ (ppbv)")

    # Panel 3: Bias
    vmax = np.nanmax(np.abs(bias))
    sc3 = axes[2].scatter(times, alts, c=bias, cmap="RdBu_r", s=30, edgecolor="none",
                          vmin=-vmax, vmax=vmax)
    axes[2].set_ylabel("Altitude (km)")
    axes[2].set_xlabel("Time (UTC)")
    axes[2].set_title("Bias (Model - Obs)")
    plt.colorbar(sc3, ax=axes[2], label="Bias (ppbv)")

    # Rotate x-axis labels
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("O$_3$ Curtain Plot Along Flight Track", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_dir / "curtain_o3.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'curtain_o3.png'}")
    plt.close(fig)


def plot_scatter(paired: xr.Dataset, output_dir: Path) -> None:
    """Create scatter plot of model vs observations."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = sns.color_palette()

    for i, var in enumerate(["O3", "CO"]):
        ax = axes[i]
        obs = paired[var].values
        model = paired[f"model_{var}"].values

        # Remove NaNs
        mask = ~(np.isnan(obs) | np.isnan(model))
        obs_clean = obs[mask]
        model_clean = model[mask]

        ax.scatter(obs_clean, model_clean, alpha=0.5, s=20, color=colors[i], edgecolor="none")

        # 1:1 line
        lims = [
            min(obs_clean.min(), model_clean.min()),
            max(obs_clean.max(), model_clean.max())
        ]
        ax.plot(lims, lims, "k--", label="1:1", linewidth=1.5)

        # Linear fit
        coeffs = np.polyfit(obs_clean, model_clean, 1)
        r = np.corrcoef(obs_clean, model_clean)[0, 1]
        rmse = np.sqrt(np.mean((model_clean - obs_clean) ** 2))
        mb = np.mean(model_clean - obs_clean)

        ax.plot(lims, np.polyval(coeffs, lims), color=colors[3],
                linewidth=2, label=f"Fit (R={r:.3f})")

        # Stats text
        stats_text = f"N = {len(obs_clean)}\nR = {r:.3f}\nMB = {mb:.2f}\nRMSE = {rmse:.2f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        units = "ppbv" if var == "O3" else "ppbv"
        ax.set_xlabel(f"Observed {var} ({units})")
        ax.set_ylabel(f"Modeled {var} ({units})")
        ax.set_title(f"{var} Comparison")
        ax.legend(loc="lower right")
        ax.set_aspect("equal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    fig.suptitle("Model vs Observation Scatter Plots", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_dir / "scatter.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'scatter.png'}")
    plt.close(fig)


def plot_timeseries(paired: xr.Dataset, aircraft: xr.Dataset, output_dir: Path) -> None:
    """Create time series plot along the flight track."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = sns.color_palette()

    times = paired["time"].values
    alts = aircraft["altitude"].values / 1000

    # Panel 1: O3
    ax1 = axes[0]
    ax1.plot(times, paired["O3"], label="Observed", linewidth=1.5, color=colors[0])
    ax1.plot(times, paired["model_O3"], label="Model", linewidth=1.5, color=colors[3])
    ax1.set_ylabel("O$_3$ (ppbv)")
    ax1.legend(loc="upper right")
    ax1.set_title("O$_3$ Along Flight Track")

    # Panel 2: CO
    ax2 = axes[1]
    ax2.plot(times, paired["CO"], label="Observed", linewidth=1.5, color=colors[0])
    ax2.plot(times, paired["model_CO"], label="Model", linewidth=1.5, color=colors[3])
    ax2.set_ylabel("CO (ppbv)")
    ax2.legend(loc="upper right")
    ax2.set_title("CO Along Flight Track")

    # Panel 3: Altitude profile
    ax3 = axes[2]
    ax3.fill_between(times, 0, alts, alpha=0.3, color=colors[2])
    ax3.plot(times, alts, linewidth=1.5, color=colors[2])
    ax3.set_ylabel("Altitude (km)")
    ax3.set_xlabel("Time (UTC)")
    ax3.set_title("Flight Altitude")
    ax3.set_ylim(0, None)

    # Rotate x-axis labels
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Time Series Along Flight Track", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_dir / "timeseries.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'timeseries.png'}")
    plt.close(fig)


def plot_vertical_profile(paired: xr.Dataset, aircraft: xr.Dataset, output_dir: Path) -> None:
    """Create vertical profile plot (binned by altitude)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    colors = sns.color_palette()

    alts = aircraft["altitude"].values / 1000  # km

    # Altitude bins
    alt_bins = np.arange(0, 12, 1)  # 1 km bins
    alt_centers = (alt_bins[:-1] + alt_bins[1:]) / 2

    for i, var in enumerate(["O3", "CO"]):
        ax = axes[i]
        obs = paired[var].values
        model = paired[f"model_{var}"].values

        obs_means = []
        obs_stds = []
        model_means = []
        model_stds = []

        for j in range(len(alt_bins) - 1):
            mask = (alts >= alt_bins[j]) & (alts < alt_bins[j+1])
            if mask.sum() > 0:
                obs_means.append(np.nanmean(obs[mask]))
                obs_stds.append(np.nanstd(obs[mask]))
                model_means.append(np.nanmean(model[mask]))
                model_stds.append(np.nanstd(model[mask]))
            else:
                obs_means.append(np.nan)
                obs_stds.append(np.nan)
                model_means.append(np.nan)
                model_stds.append(np.nan)

        obs_means = np.array(obs_means)
        obs_stds = np.array(obs_stds)
        model_means = np.array(model_means)
        model_stds = np.array(model_stds)

        # Plot with error bars
        ax.fill_betweenx(alt_centers, obs_means - obs_stds, obs_means + obs_stds,
                        alpha=0.3, color=colors[0])
        ax.plot(obs_means, alt_centers, "-o", label="Observed",
                linewidth=2, markersize=6, color=colors[0])

        ax.fill_betweenx(alt_centers, model_means - model_stds, model_means + model_stds,
                        alpha=0.3, color=colors[3])
        ax.plot(model_means, alt_centers, "-s", label="Model",
                linewidth=2, markersize=6, color=colors[3])

        units = "ppbv"
        ax.set_xlabel(f"{var} ({units})")
        ax.set_ylabel("Altitude (km)")
        ax.set_title(f"{var} Vertical Profile")
        ax.legend(loc="upper right")
        ax.set_ylim(0, 10)

    fig.suptitle("Vertical Profiles (Altitude-Binned)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    fig.savefig(output_dir / "vertical_profile.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {output_dir / 'vertical_profile.png'}")
    plt.close(fig)


def main():
    """Run aircraft evaluation example."""
    print("DAVINCI-MONET: Aircraft Evaluation Example")
    print("=" * 50)

    output_dir = Path("examples/output/aircraft")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("\n1. Creating synthetic data...")
    model, aircraft = create_aircraft_data()
    print(f"   Model dims: {dict(model.dims)}")
    print(f"   Aircraft dims: {dict(aircraft.dims)}")
    print(f"   Aircraft points: {aircraft.sizes['time']}")
    print(f"   Altitude range: {aircraft['altitude'].min().values:.0f} - "
          f"{aircraft['altitude'].max().values:.0f} m")

    # Pair data
    print("\n2. Pairing aircraft with model...")
    paired = pair_aircraft_with_model(model, aircraft)
    print(f"   Paired variables: {list(paired.data_vars)}")

    # Create plots
    print("\n3. Generating plots...")

    print("   - Flight path map")
    plot_flight_path(aircraft, output_dir)

    print("   - Curtain plot")
    plot_curtain(paired, aircraft, output_dir)

    print("   - Scatter plot")
    plot_scatter(paired, output_dir)

    print("   - Time series")
    plot_timeseries(paired, aircraft, output_dir)

    print("   - Vertical profile")
    plot_vertical_profile(paired, aircraft, output_dir)

    print("\n" + "=" * 50)
    print("Aircraft evaluation complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
