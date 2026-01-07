"""Point-to-grid pairing strategy.

This module implements pairing for point observations (surface stations,
ground sites) with gridded model output.
"""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np
import xarray as xr

from davinci_monet.core.exceptions import PairingError
from davinci_monet.core.protocols import DataGeometry
from davinci_monet.core.types import TimeDelta
from davinci_monet.pairing.strategies.base import BasePairingStrategy


class PointStrategy(BasePairingStrategy):
    """Pairing strategy for point observations.

    Matches fixed-location observations (surface stations, ground sites)
    to the nearest model grid cell within the radius of influence.

    The strategy:
    1. Finds nearest model grid cell for each observation site
    2. Extracts surface level from model (if 3D)
    3. Interpolates model to observation times
    4. Creates paired dataset with aligned values

    Examples
    --------
    >>> strategy = PointStrategy()
    >>> paired = strategy.pair(model_data, obs_data,
    ...                        radius_of_influence=12000)
    """

    @property
    def geometry(self) -> DataGeometry:
        """Return POINT geometry."""
        return DataGeometry.POINT

    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        radius_of_influence: float | None = None,
        time_tolerance: TimeDelta | None = None,
        vertical_method: str = "nearest",
        horizontal_method: str = "nearest",
        **kwargs: Any,
    ) -> xr.Dataset:
        """Pair point observations with model grid.

        Parameters
        ----------
        model
            Model Dataset with dims (time, [z], lat, lon).
        obs
            Observation Dataset with dims (time, site).
        radius_of_influence
            Maximum distance in meters for matching. Default 12000m.
        time_tolerance
            Maximum time difference for matching.
        vertical_method
            Not used for surface observations.
        horizontal_method
            Horizontal matching method ('nearest' only currently).
        **kwargs
            Additional options:
            - extract_surface: bool, whether to extract surface level (default True)

        Returns
        -------
        xr.Dataset
            Paired dataset with model values at observation locations.
        """
        if radius_of_influence is None:
            radius_of_influence = 12000.0

        extract_surface = kwargs.get("extract_surface", True)

        # Get coordinates
        model_lat, model_lon = self._get_model_coords(model)
        obs_lat, obs_lon = self._get_obs_coords(obs)

        # Extract surface level if model is 3D
        if extract_surface:
            model_surface = self._extract_surface(model)
        else:
            model_surface = model

        # Find site dimension in obs
        site_dim = self._get_site_dim(obs)

        # Get unique site locations
        if obs_lat.dims == (site_dim,):
            site_lats = obs_lat.values
            site_lons = obs_lon.values
        else:
            # Lat/lon may be (time, site) - take first time
            site_lats = obs_lat.isel(time=0).values if "time" in obs_lat.dims else obs_lat.values
            site_lons = obs_lon.isel(time=0).values if "time" in obs_lon.dims else obs_lon.values

        # Find nearest model grid indices for each site
        lat_idx, lon_idx = self._find_nearest_indices(
            model_lat, model_lon,
            xr.DataArray(site_lats), xr.DataArray(site_lons),
            radius_of_influence=radius_of_influence,
        )

        # Extract model values at observation sites
        model_at_sites = self._extract_at_sites(
            model_surface, model_lat, model_lon, lat_idx.values, lon_idx.values, site_dim
        )

        # Interpolate model to observation times
        if "time" in model_at_sites.dims and "time" in obs.dims:
            obs_times = obs["time"]
            model_at_sites = self._interpolate_time(
                model_at_sites, obs_times, method="nearest"
            )

        # Combine into paired dataset
        paired = self._create_paired_output(obs, model_at_sites, site_dim)

        return paired

    def _get_site_dim(self, obs: xr.Dataset) -> str:
        """Find the site dimension name in observations.

        Parameters
        ----------
        obs
            Observation dataset.

        Returns
        -------
        str
            Site dimension name.
        """
        for dim in ["site", "station", "x", "location"]:
            if dim in obs.dims:
                return dim

        raise PairingError(
            f"Cannot find site dimension in observations. "
            f"Available dims: {list(obs.dims)}"
        )

    def _extract_at_sites(
        self,
        model: xr.Dataset,
        model_lat: xr.DataArray,
        model_lon: xr.DataArray,
        lat_idx: np.ndarray[Any, np.dtype[Any]],
        lon_idx: np.ndarray[Any, np.dtype[Any]],
        site_dim: str,
    ) -> xr.Dataset:
        """Extract model values at observation site locations.

        Parameters
        ----------
        model
            Model dataset (surface level).
        model_lat, model_lon
            Model coordinate arrays.
        lat_idx, lon_idx
            Indices of nearest model grid cells.
        site_dim
            Name of site dimension.

        Returns
        -------
        xr.Dataset
            Model values at site locations with site dimension.
        """
        n_sites = len(lat_idx)

        # Determine lat/lon dimension names
        if model_lat.ndim == 1:
            lat_dim = model_lat.dims[0]
            lon_dim = model_lon.dims[0]
        else:
            # Curvilinear grid - assume (y, x) or similar
            lat_dim = model_lat.dims[0]
            lon_dim = model_lat.dims[1]

        # Create site coordinate
        site_coord = np.arange(n_sites)

        # Extract values for each variable
        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, np.dtype[Any]]]] = {}

        for var in model.data_vars:
            var_data = model[var]

            # Determine output dimensions
            out_dims: list[str] = []
            for dim in var_data.dims:
                if dim not in (lat_dim, lon_dim):
                    out_dims.append(str(dim))
            out_dims.append(site_dim)

            # Build output shape
            out_shape: list[int] = []
            for dim in out_dims[:-1]:
                out_shape.append(int(var_data.sizes[dim]))
            out_shape.append(n_sites)

            # Extract values
            out_data = np.full(out_shape, np.nan)

            for i in range(n_sites):
                if lat_idx[i] < 0 or lon_idx[i] < 0:
                    # Outside radius of influence
                    continue

                if model_lat.ndim == 1:
                    # Regular grid
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}
                else:
                    # Curvilinear grid
                    selection = {lat_dim: lat_idx[i], lon_dim: lon_idx[i]}

                site_vals = var_data.isel(selection).values
                out_data[..., i] = site_vals

            data_vars[str(var)] = (tuple(out_dims), out_data)

        # Build output dataset
        coords = {site_dim: site_coord}

        # Add time coordinate if present
        if "time" in model.coords:
            coords["time"] = model.coords["time"].values

        return xr.Dataset(data_vars, coords=coords)

    def _create_paired_output(
        self,
        obs: xr.Dataset,
        model_at_sites: xr.Dataset,
        site_dim: str,
    ) -> xr.Dataset:
        """Create the final paired output dataset.

        Parameters
        ----------
        obs
            Observation dataset.
        model_at_sites
            Model values at site locations.
        site_dim
            Site dimension name.

        Returns
        -------
        xr.Dataset
            Combined dataset with both obs and model values.
        """
        # Align dimensions
        common_dims = set(obs.dims) & set(model_at_sites.dims)

        # Combine coordinates
        coords = dict(obs.coords)
        for coord in model_at_sites.coords:
            if coord not in coords:
                coords[coord] = model_at_sites.coords[coord]

        # Combine data variables
        data_vars: dict[str, Any] = {}

        # Add observation variables
        for var in obs.data_vars:
            data_vars[str(var)] = obs[var]

        # Add model variables (with model_ prefix to avoid conflicts)
        for var in model_at_sites.data_vars:
            model_var_name = f"model_{var}"
            data_vars[model_var_name] = model_at_sites[var]

        return xr.Dataset(data_vars, coords=coords)
