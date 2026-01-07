"""Track-to-grid pairing strategy.

This module implements pairing for track observations (aircraft, mobile
platforms) with gridded model output, including 3D interpolation.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class TrackStrategy(BasePairingStrategy):
    """Pairing strategy for track observations.

    Matches moving observations (aircraft, mobile platforms) to model
    grid cells along their trajectory, with 3D interpolation.

    The strategy:
    1. For each track point, finds nearest model grid cell
    2. Interpolates model temporally to track time
    3. Interpolates model vertically to track altitude
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = TrackStrategy()
    >>> paired = strategy.pair(model_data, aircraft_data,
    ...                        vertical_method='linear')
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return TRACK geometry."""
        return DataGeometry.TRACK

    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "linear",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair track observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, z, lat, lon).
        obs
            Observation Dataset with dims (time,) and lat/lon/alt coords.
        radius_of_influence
            Maximum horizontal distance in meters.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Vertical interpolation method ('nearest', 'linear', 'log').
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        **kwargs
            Additional options:
            - pressure_var: str, name of pressure variable for vertical interp
            - altitude_var: str, name of altitude variable in obs

        Returns
        -------
        xr.Dataset
            Paired dataset with model values along track.
        """
        pressure_var = kwargs.get("pressure_var", "pressure")
        altitude_var = kwargs.get("altitude_var", "altitude")

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Get observation times
        if "time" not in obs.dims:
            raise PairingError("Track observations must have 'time' dimension")
        obs_times = obs["time"]
        n_points = len(obs_times)

        # Find nearest model grid cell for each track point
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat, model_lon, obs_lat, obs_lon,
            radius_of_influence=radius_of_influence,
        )

        # Determine vertical coordinate
        obs_altitude = self._get_altitude(obs, altitude_var)
        model_has_vertical = "z" in model.dims

        # Extract and interpolate model values along track
        model_along_track = self._extract_along_track(
            model, model_lat, model_lon,
            lat_idx.values, lon_idx.values,
            obs_times, obs_altitude,
            vertical_method=vertical_method,
            model_has_vertical=model_has_vertical,
        )

        # Create paired output
        paired = self._create_paired_output(obs, model_along_track)

        return paired

    def _get_altitude(
        self, obs: xr.Dataset, altitude_var: str
    ) -> xr.DataArray | None:
        """Get altitude/pressure coordinate from observations.

        Parameters
        ----------
        obs
            Observation dataset.
        altitude_var
            Name of altitude/pressure variable.

        Returns
        -------
        xr.DataArray | None
            Altitude values or None if not present.
        """
        # Try various altitude/pressure variable names
        for name in [altitude_var, "altitude", "alt", "z", "pressure", "pressure_obs"]:
            if name in obs.coords:
                return obs.coords[name]
            if name in obs.data_vars:
                return obs[name]
        return None

    def _extract_along_track(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        obs_times: xr.DataArray,
        obs_altitude: xr.DataArray | None,
        vertical_method: str,
        model_has_vertical: bool,
    ) -> xr.Dataset:
        """Extract model values along the track.

        Parameters
        ----------
        model
            Model dataset.
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Nearest grid indices for each track point.
        obs_times
            Track observation times.
        obs_altitude
            Track altitudes (optional).
        vertical_method
            Vertical interpolation method.
        model_has_vertical
            Whether model has vertical dimension.

        Returns
        -------
        xr.Dataset
            Model values along track with time dimension.
        """
        n_points = len(obs_times)

        # Determine lat/lon dimension names
        if model_lat.ndim == 1:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lon.dims[0]
        else:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lat.dims[1]

        # First, interpolate model to track times
        model_time_interp = self._interpolate_time(model, obs_times, method="linear")

        # Build output for each variable
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in model_time_interp.data_vars:
            var_data = model_time_interp[var]

            # Initialize output array
            out_data = np.full(n_points, np.nan)

            for i in range(n_points):
                if lat_idx[i] < 0 or lon_idx[i] < 0:
                    continue

                # Extract at horizontal location
                if model_lat.ndim == 1:
                    point_data = var_data.isel(
                        {lat_dim: lat_idx[i], lon_dim: lon_idx[i], "time": i}
                    )
                else:
                    point_data = var_data.isel(
                        {lat_dim: lat_idx[i], lon_dim: lon_idx[i], "time": i}
                    )

                # Handle vertical interpolation
                if model_has_vertical and "z" in point_data.dims:
                    if obs_altitude is not None:
                        # Interpolate to observation altitude
                        target_alt = float(obs_altitude.values[i])
                        z_vals = point_data["z"].values
                        if vertical_method == "nearest":
                            z_idx = np.argmin(np.abs(z_vals - target_alt))
                            out_data[i] = float(point_data.isel(z=z_idx).values)
                        else:
                            # Linear interpolation
                            out_data[i] = float(
                                np.interp(target_alt, z_vals, point_data.values)
                            )
                    else:
                        # Take surface level
                        out_data[i] = float(point_data.isel(z=0).values)
                else:
                    out_data[i] = float(point_data.values)

            data_vars[str(var)] = (("time",), out_data)

        # Build output dataset
        coords = {"time": obs_times.values}

        return xr.Dataset(data_vars, coords=coords)

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_along_track: xr.Dataset,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_along_track
            Model values along track.

        Returns
        -------
        xr.Dataset
            Combined dataset.
        """
        # Combine coordinates
        coords = dict(obs.coords)

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables with prefix
        for var in model_along_track.data_vars:
            model_var_name = f"model_{var}"
            data_vars[model_var_name] = model_along_track[var]

        return xr.Dataset(data_vars, coords=coords)
