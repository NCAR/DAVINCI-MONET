#!/usr/bin/env python
"""
Run CESM model evaluation against AirNow and AERONET observations.

This script performs:
1. Load CESM model data
2. Load AirNow and AERONET observations
3. Pair model with observations (spatial/temporal matching)
4. Generate comparison plots
5. Calculate statistics

Usage:
    python run_evaluation.py
"""

from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# DAVINCI-MONET imports
from davinci_monet.models.cesm import open_cesm
from davinci_monet.pairing.engine import PairingEngine
from davinci_monet.stats.calculator import StatisticsCalculator

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "evaluation"
MODEL_DIR = Path.home() / "Data" / "ASIA-AQ"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis period
START_DATE = "2024-02-01"
END_DATE = "2024-02-03"

# ASIA-AQ domain
DOMAIN = {"lat_min": 0, "lat_max": 45, "lon_min": 90, "lon_max": 140}


def load_model_data() -> xr.Dataset:
    """Load CESM model data."""
    print("Loading CESM model data...")

    files = sorted(MODEL_DIR.glob("f.e3b06m.FCnudged.t6s.01x01.01.cam.h2i.*.nc"))
    print(f"  Found {len(files)} model files")

    model_data = open_cesm(
        files=files,
        variables=["O3", "NO2", "CO", "PM25", "AODVISdn", "PS"],
        label="cesm_asiaq",
    )

    ds = model_data.data
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Variables: {list(ds.data_vars)}")

    return ds


def load_airnow_data() -> xr.Dataset:
    """Load AirNow observation data."""
    print("\nLoading AirNow data...")

    filepath = DATA_DIR / "airnow_asiaq_2024-02-01_2024-02-03.nc"
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        print("  Run download_airnow.py first")
        return None

    ds = xr.open_dataset(filepath)
    print(f"  Sites: {len(ds.site)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    print(f"  Variables: {list(ds.data_vars)}")

    return ds


def load_aeronet_data() -> xr.Dataset:
    """Load AERONET observation data."""
    print("\nLoading AERONET data...")

    filepath = DATA_DIR / "AERONET_L15_20240201_20240203.nc"
    if not filepath.exists():
        print(f"  ERROR: File not found: {filepath}")
        print("  Run: davinci-monet get aeronet -s 2024-02-01 -e 2024-02-03 -d analyses/asia-aq/data")
        return None

    ds = xr.open_dataset(filepath)

    # Filter to ASIA-AQ domain
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    asia_mask = (
        (lat >= DOMAIN["lat_min"]) & (lat <= DOMAIN["lat_max"]) &
        (lon >= DOMAIN["lon_min"]) & (lon <= DOMAIN["lon_max"])
    )
    asia_sites = ds["site"].values[asia_mask]

    ds_asia = ds.sel(site=asia_sites)
    print(f"  Sites in ASIA-AQ domain: {len(ds_asia.site)}")
    print(f"  Time range: {ds_asia.time.values[0]} to {ds_asia.time.values[-1]}")

    return ds_asia


def extract_model_at_obs_locations(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    model_var: str,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract model values at observation locations and times.

    Returns arrays of matched (model, obs, lat, lon) values.
    """
    # Get observation coordinates
    obs_lat = obs_ds["latitude"].values
    obs_lon = obs_ds["longitude"].values
    obs_times = pd.to_datetime(obs_ds["time"].values)

    # Get model coordinates
    model_lat_coord = "latitude" if "latitude" in model_ds.coords else "lat"
    model_lon_coord = "longitude" if "longitude" in model_ds.coords else "lon"
    model_lat = model_ds[model_lat_coord].values
    model_lon = model_ds[model_lon_coord].values
    model_times = pd.to_datetime(model_ds["time"].values)

    # Handle 2D lat/lon
    if model_lat.ndim == 2:
        # Take center column/row for 1D approximation
        model_lat_1d = model_lat[:, model_lat.shape[1] // 2]
        model_lon_1d = model_lon[model_lon.shape[0] // 2, :]
    else:
        model_lat_1d = model_lat
        model_lon_1d = model_lon

    # Get model variable (surface level)
    # Note: After open_cesm processing, z=0 is surface (reader inverts vertical coord)
    model_data = model_ds[model_var]
    if "z" in model_data.dims:
        model_data = model_data.isel(z=0)  # Surface level
    elif "lev" in model_data.dims:
        model_data = model_data.isel(lev=-1)  # Raw CESM: surface is last index

    # Lists to collect matched values
    model_vals = []
    obs_vals = []
    lats = []
    lons = []

    # Loop over observation sites
    for site_idx in range(len(obs_lat)):
        site_lat = obs_lat[site_idx]
        site_lon = obs_lon[site_idx]

        # Skip if outside domain
        if not (DOMAIN["lat_min"] <= site_lat <= DOMAIN["lat_max"] and
                DOMAIN["lon_min"] <= site_lon <= DOMAIN["lon_max"]):
            continue

        # Find nearest model grid point
        lat_idx = int(np.abs(model_lat_1d - site_lat).argmin())
        lon_idx = int(np.abs(model_lon_1d - site_lon).argmin())

        # Loop over observation times
        for time_idx, obs_time in enumerate(obs_times):
            # Find nearest model time
            time_diff = np.abs(model_times - obs_time)
            model_time_idx = time_diff.argmin()

            # Skip if time difference > 1 hour
            if time_diff[model_time_idx] > pd.Timedelta(hours=1):
                continue

            # Get observation value
            obs_var_names = list(obs_ds.data_vars)
            if len(obs_var_names) == 0:
                continue

            # Try to get the first available variable
            obs_val = None
            for var_name in obs_var_names:
                try:
                    val = obs_ds[var_name].isel(time=time_idx, site=site_idx).values
                    if np.isfinite(val):
                        obs_val = val
                        break
                except Exception:
                    continue

            if obs_val is None or not np.isfinite(obs_val):
                continue

            # Get model value
            try:
                if model_lat.ndim == 2:
                    model_val = model_data.isel(time=model_time_idx, y=lat_idx, x=lon_idx).values
                else:
                    model_val = model_data.isel(time=model_time_idx, lat=lat_idx, lon=lon_idx).values

                model_val = float(model_val) * scale

                if np.isfinite(model_val):
                    model_vals.append(model_val)
                    obs_vals.append(obs_val)
                    lats.append(site_lat)
                    lons.append(site_lon)
            except Exception:
                continue

    return np.array(model_vals), np.array(obs_vals), np.array(lats), np.array(lons)


def pair_model_obs_simple(
    model_ds: xr.Dataset,
    obs_ds: xr.Dataset,
    model_var: str,
    obs_var: str,
    model_scale: float = 1.0,
) -> dict:
    """
    Simple pairing of model and observations.

    Returns dictionary with paired values and metadata.
    """
    print(f"\n  Pairing {model_var} (model) with {obs_var} (obs)...")

    # Get observation data
    if obs_var not in obs_ds:
        print(f"    WARNING: {obs_var} not in observations")
        return None

    obs_data = obs_ds[obs_var]
    obs_lat = obs_ds["latitude"].values
    obs_lon = obs_ds["longitude"].values
    obs_times = pd.to_datetime(obs_ds["time"].values)

    # Get model coordinates
    model_lat_coord = "latitude" if "latitude" in model_ds.coords else "lat"
    model_lon_coord = "longitude" if "longitude" in model_ds.coords else "lon"
    model_lat = model_ds[model_lat_coord].values
    model_lon = model_ds[model_lon_coord].values
    model_times = pd.to_datetime(model_ds["time"].values)

    # Handle 2D lat/lon
    if model_lat.ndim == 2:
        model_lat_1d = model_lat[:, model_lat.shape[1] // 2]
        model_lon_1d = model_lon[model_lon.shape[0] // 2, :]
    else:
        model_lat_1d = model_lat
        model_lon_1d = model_lon

    # Get model variable (surface level)
    # Note: After open_cesm processing, z=0 is surface (reader inverts vertical coord)
    model_data = model_ds[model_var]
    if "z" in model_data.dims:
        model_data = model_data.isel(z=0)  # Surface level
    elif "lev" in model_data.dims:
        model_data = model_data.isel(lev=-1)  # Raw CESM: surface is last index

    # Collect paired values
    pairs = []

    for site_idx in range(len(obs_lat)):
        site_lat = obs_lat[site_idx]
        site_lon = obs_lon[site_idx]

        # Skip if outside domain
        if not (DOMAIN["lat_min"] <= site_lat <= DOMAIN["lat_max"] and
                DOMAIN["lon_min"] <= site_lon <= DOMAIN["lon_max"]):
            continue

        # Find nearest model grid point
        lat_idx = int(np.abs(model_lat_1d - site_lat).argmin())
        lon_idx = int(np.abs(model_lon_1d - site_lon).argmin())

        for time_idx, obs_time in enumerate(obs_times):
            # Find nearest model time
            time_diff = np.abs(model_times - obs_time)
            model_time_idx = int(time_diff.argmin())

            if time_diff[model_time_idx] > pd.Timedelta(hours=1):
                continue

            # Get values
            try:
                obs_val = float(obs_data.isel(time=time_idx, site=site_idx).values)
                if model_lat.ndim == 2:
                    model_val = float(model_data.isel(time=model_time_idx, y=lat_idx, x=lon_idx).values)
                else:
                    model_val = float(model_data.isel(time=model_time_idx).values[lat_idx, lon_idx])

                model_val *= model_scale

                if np.isfinite(obs_val) and np.isfinite(model_val):
                    pairs.append({
                        "time": obs_time,
                        "lat": site_lat,
                        "lon": site_lon,
                        "obs": obs_val,
                        "model": model_val,
                    })
            except Exception:
                continue

    if len(pairs) == 0:
        print(f"    WARNING: No valid pairs found")
        return None

    df = pd.DataFrame(pairs)
    print(f"    Found {len(df)} valid pairs")

    return {
        "model_var": model_var,
        "obs_var": obs_var,
        "data": df,
        "model_values": df["model"].values,
        "obs_values": df["obs"].values,
        "latitudes": df["lat"].values,
        "longitudes": df["lon"].values,
        "times": df["time"].values,
    }


def calculate_stats(paired: dict) -> dict:
    """Calculate evaluation statistics."""
    if paired is None:
        return None

    obs = paired["obs_values"]
    mod = paired["model_values"]

    n = len(obs)
    if n == 0:
        return None

    # Basic statistics
    mean_obs = np.mean(obs)
    mean_mod = np.mean(mod)
    bias = mod - obs
    mb = np.mean(bias)
    rmse = np.sqrt(np.mean(bias**2))

    # Correlation
    if np.std(obs) > 0 and np.std(mod) > 0:
        r = np.corrcoef(obs, mod)[0, 1]
    else:
        r = np.nan

    # Normalized metrics
    if mean_obs > 0:
        nmb = 100 * mb / mean_obs
        nme = 100 * np.mean(np.abs(bias)) / mean_obs
    else:
        nmb = np.nan
        nme = np.nan

    # Index of agreement
    diff_obs = obs - mean_obs
    diff_mod = mod - mean_mod
    denom = np.sum((np.abs(diff_mod) + np.abs(diff_obs))**2)
    if denom > 0:
        ioa = 1 - np.sum(bias**2) / denom
    else:
        ioa = np.nan

    return {
        "N": n,
        "Mean_Obs": mean_obs,
        "Mean_Model": mean_mod,
        "MB": mb,
        "RMSE": rmse,
        "R": r,
        "NMB_%": nmb,
        "NME_%": nme,
        "IOA": ioa,
    }


def plot_scatter(paired: dict, title: str, xlabel: str, ylabel: str, outfile: Path):
    """Create scatter plot of model vs observations."""
    if paired is None:
        return

    obs = paired["obs_values"]
    mod = paired["model_values"]
    stats = calculate_stats(paired)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(obs, mod, alpha=0.6, s=30, edgecolors="none")

    # 1:1 line
    vmin = min(obs.min(), mod.min())
    vmax = max(obs.max(), mod.max())
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="1:1")

    # Linear regression
    if len(obs) > 2:
        m, b = np.polyfit(obs, mod, 1)
        x_fit = np.array([vmin, vmax])
        ax.plot(x_fit, m * x_fit + b, "r-", linewidth=1, label=f"Fit: y={m:.2f}x+{b:.2f}")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
    ax.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
    ax.legend(loc="upper left")

    # Add stats text
    if stats:
        stats_text = (
            f"N = {stats['N']}\n"
            f"R = {stats['R']:.3f}\n"
            f"MB = {stats['MB']:.2f}\n"
            f"RMSE = {stats['RMSE']:.2f}\n"
            f"NMB = {stats['NMB_%']:.1f}%"
        )
        ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment="bottom", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_aspect("equal")
    plt.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    print(f"    Saved: {outfile}")
    plt.close(fig)


def plot_spatial_bias(paired: dict, title: str, outfile: Path, vmin: float = None, vmax: float = None):
    """Create spatial map of model bias."""
    if paired is None:
        return

    lats = paired["latitudes"]
    lons = paired["longitudes"]
    bias = paired["model_values"] - paired["obs_values"]

    # Average bias by location
    df = pd.DataFrame({"lat": lats, "lon": lons, "bias": bias})
    site_bias = df.groupby(["lat", "lon"]).mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine color limits
    if vmin is None:
        vmax_abs = max(abs(site_bias["bias"].min()), abs(site_bias["bias"].max()))
        vmin = -vmax_abs
        vmax = vmax_abs

    scatter = ax.scatter(
        site_bias["lon"], site_bias["lat"],
        c=site_bias["bias"], cmap="RdBu_r",
        s=100, vmin=vmin, vmax=vmax,
        edgecolors="black", linewidths=0.5
    )

    plt.colorbar(scatter, ax=ax, label="Bias (Model - Obs)", shrink=0.8)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(DOMAIN["lon_min"], DOMAIN["lon_max"])
    ax.set_ylim(DOMAIN["lat_min"], DOMAIN["lat_max"])

    # Add simple coastline approximation
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    print(f"    Saved: {outfile}")
    plt.close(fig)


def plot_timeseries(paired: dict, title: str, ylabel: str, outfile: Path):
    """Create time series comparison plot."""
    if paired is None:
        return

    df = paired["data"]

    # Average by time
    time_avg = df.groupby("time")[["obs", "model"]].mean()

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(time_avg.index, time_avg["obs"], "b-", linewidth=1.5, label="Observations", marker="o", markersize=4)
    ax.plot(time_avg.index, time_avg["model"], "r-", linewidth=1.5, label="Model", marker="s", markersize=4)

    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    fig.savefig(outfile.with_suffix(".pdf"), bbox_inches="tight")
    print(f"    Saved: {outfile}")
    plt.close(fig)


def main():
    """Main evaluation routine."""
    print("=" * 70)
    print("CESM/CAM-chem ASIA-AQ Model Evaluation")
    print("=" * 70)

    # Load data
    model_ds = load_model_data()
    airnow_ds = load_airnow_data()
    aeronet_ds = load_aeronet_data()

    if model_ds is None:
        print("ERROR: Could not load model data")
        return

    # Store all statistics
    all_stats = {}

    # =========================================================================
    # AirNow PM2.5 evaluation
    # =========================================================================
    if airnow_ds is not None and "pm25" in airnow_ds:
        print("\n" + "=" * 70)
        print("PM2.5 Evaluation (AirNow)")
        print("=" * 70)

        # PM25 conversion: kg/kg to μg/m³ = ρ_air × 1e9 ≈ 1.2e9
        paired_pm25 = pair_model_obs_simple(
            model_ds, airnow_ds,
            model_var="PM25", obs_var="pm25",
            model_scale=1.2e9,  # kg/kg to μg/m³
        )

        if paired_pm25 is not None:
            stats = calculate_stats(paired_pm25)
            all_stats["PM2.5"] = stats
            print(f"\n  Statistics:")
            for k, v in stats.items():
                print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

            # Plots
            print("\n  Creating plots...")
            plot_scatter(
                paired_pm25,
                title="PM2.5: CESM vs AirNow",
                xlabel="AirNow PM2.5 (μg/m³)",
                ylabel="CESM PM2.5 (μg/m³)",
                outfile=OUTPUT_DIR / "pm25_scatter.png"
            )
            plot_spatial_bias(
                paired_pm25,
                title="PM2.5 Bias (CESM - AirNow)",
                outfile=OUTPUT_DIR / "pm25_spatial_bias.png",
                vmin=-50, vmax=50
            )
            plot_timeseries(
                paired_pm25,
                title="PM2.5 Time Series: CESM vs AirNow",
                ylabel="PM2.5 (μg/m³)",
                outfile=OUTPUT_DIR / "pm25_timeseries.png"
            )

    # =========================================================================
    # AirNow Ozone evaluation
    # =========================================================================
    if airnow_ds is not None and "ozone" in airnow_ds:
        print("\n" + "=" * 70)
        print("Ozone Evaluation (AirNow)")
        print("=" * 70)

        paired_o3 = pair_model_obs_simple(
            model_ds, airnow_ds,
            model_var="O3", obs_var="ozone",
            model_scale=1e9,  # mol/mol to ppb
        )

        if paired_o3 is not None:
            stats = calculate_stats(paired_o3)
            all_stats["O3"] = stats
            print(f"\n  Statistics:")
            for k, v in stats.items():
                print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

            print("\n  Creating plots...")
            plot_scatter(
                paired_o3,
                title="O3: CESM vs AirNow",
                xlabel="AirNow O3 (ppb)",
                ylabel="CESM O3 (ppb)",
                outfile=OUTPUT_DIR / "o3_scatter.png"
            )
            plot_timeseries(
                paired_o3,
                title="O3 Time Series: CESM vs AirNow",
                ylabel="O3 (ppb)",
                outfile=OUTPUT_DIR / "o3_timeseries.png"
            )

    # =========================================================================
    # AERONET AOD evaluation
    # =========================================================================
    if aeronet_ds is not None:
        print("\n" + "=" * 70)
        print("AOD Evaluation (AERONET)")
        print("=" * 70)

        # AERONET has aod_500nm, model has AODVISdn (550nm)
        # Use 500nm as approximation
        aod_var = None
        for var in ["aod_550nm", "aod_500nm", "aod_440nm"]:
            if var in aeronet_ds:
                aod_var = var
                break

        if aod_var and "AODVISdn" in model_ds:
            paired_aod = pair_model_obs_simple(
                model_ds, aeronet_ds,
                model_var="AODVISdn", obs_var=aod_var,
                model_scale=1.0,
            )

            if paired_aod is not None:
                stats = calculate_stats(paired_aod)
                all_stats["AOD"] = stats
                print(f"\n  Statistics:")
                for k, v in stats.items():
                    print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")

                print("\n  Creating plots...")
                plot_scatter(
                    paired_aod,
                    title="AOD: CESM vs AERONET",
                    xlabel=f"AERONET {aod_var}",
                    ylabel="CESM AODVISdn",
                    outfile=OUTPUT_DIR / "aod_scatter.png"
                )
                plot_spatial_bias(
                    paired_aod,
                    title="AOD Bias (CESM - AERONET)",
                    outfile=OUTPUT_DIR / "aod_spatial_bias.png",
                    vmin=-0.5, vmax=0.5
                )

    # =========================================================================
    # Summary statistics table
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)

    if all_stats:
        stats_df = pd.DataFrame(all_stats).T
        print(stats_df.to_string())

        # Save to CSV
        stats_file = OUTPUT_DIR / "statistics_summary.csv"
        stats_df.to_csv(stats_file)
        print(f"\nSaved: {stats_file}")

    print("\n" + "=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
