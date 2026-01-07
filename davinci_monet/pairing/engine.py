"""Unified pairing engine for model-observation matching.

This module provides the main pairing orchestrator that dispatches to
geometry-specific strategies based on observation data type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.base import PairedData, create_paired_dataset
from davinci_monet.core.exceptions import (
    GeometryMismatchError,
    NoOverlapError,
    PairingError,
)
from davinci_monet.core.protocols import DataGeometry, PairingStrategy
from davinci_monet.core.types import TimeDelta


@dataclass
class PairingConfig:
    """Configuration for pairing operations.

    Attributes
    ----------
    radius_of_influence : float
        Spatial search radius in meters.
    time_tolerance : TimeDelta | None
        Maximum time difference for matching.
    vertical_method : str
        Vertical interpolation method ('nearest', 'linear', 'log').
    horizontal_method : str
        Horizontal interpolation method ('nearest', 'bilinear').
    apply_averaging_kernel : bool
        Whether to apply satellite averaging kernels.
    require_overlap : bool
        Whether to require temporal overlap.
    """

    radius_of_influence: float = 12000.0
    time_tolerance: TimeDelta | None = None
    vertical_method: str = "nearest"
    horizontal_method: str = "nearest"
    apply_averaging_kernel: bool = False
    require_overlap: bool = True


class PairingEngine:
    """Unified pairing engine that dispatches to geometry-specific strategies.

    The engine automatically selects the appropriate pairing strategy based
    on the observation data's geometry attribute.

    Examples
    --------
    >>> engine = PairingEngine()
    >>> paired = engine.pair(model_data, obs_data,
    ...                      obs_vars=['O3'], model_vars=['OZONE'])
    """

    def __init__(self, register_defaults: bool = True) -> None:
        """Initialize pairing engine.

        Parameters
        ----------
        register_defaults
            If True, register default strategies for all geometries.
        """
        self._strategies: dict[DataGeometry, PairingStrategy] = {}

        if register_defaults:
            self._register_default_strategies()

    def _register_default_strategies(self) -> None:
        """Register the default strategies for each geometry type."""
        # Import here to avoid circular imports
        from davinci_monet.pairing.strategies import (
            GridStrategy,
            PointStrategy,
            ProfileStrategy,
            SwathStrategy,
            TrackStrategy,
        )

        self.register_strategy(PointStrategy())
        self.register_strategy(TrackStrategy())
        self.register_strategy(ProfileStrategy())
        self.register_strategy(SwathStrategy())
        self.register_strategy(GridStrategy())

    def register_strategy(self, strategy: PairingStrategy) -> None:
        """Register a pairing strategy for a geometry type.

        Parameters
        ----------
        strategy
            Strategy instance implementing PairingStrategy protocol.
        """
        self._strategies[strategy.geometry] = strategy

    def get_strategy(self, geometry: DataGeometry) -> PairingStrategy:
        """Get the strategy for a given geometry.

        Parameters
        ----------
        geometry
            The data geometry type.

        Returns
        -------
        PairingStrategy
            The registered strategy.

        Raises
        ------
        PairingError
            If no strategy registered for geometry.
        """
        if geometry not in self._strategies:
            raise PairingError(
                f"No pairing strategy registered for geometry {geometry.name}. "
                f"Available: {[g.name for g in self._strategies.keys()]}"
            )
        return self._strategies[geometry]

    def pair(
        self,
        model: xr.Dataset,
        obs: xr.Dataset,
        obs_vars: Sequence[str],
        model_vars: Sequence[str],
        config: PairingConfig | None = None,
        **kwargs: Any,
    ) -> PairedData:
        """Pair model output with observations.

        Automatically detects observation geometry and dispatches to
        the appropriate strategy.

        Parameters
        ----------
        model
            Model Dataset with dims (time, level, lat, lon) or similar.
        obs
            Observation Dataset with geometry-specific dimensions.
        obs_vars
            List of observation variable names to pair.
        model_vars
            List of model variable names (same order as obs_vars).
        config
            Pairing configuration. If None, uses defaults.
        **kwargs
            Additional strategy-specific options.

        Returns
        -------
        PairedData
            Paired data container with aligned model and observation variables.

        Raises
        ------
        PairingError
            If pairing fails.
        GeometryMismatchError
            If observation geometry not recognized.
        NoOverlapError
            If no temporal/spatial overlap exists.
        """
        if config is None:
            config = PairingConfig()

        # Detect observation geometry
        geometry = self._detect_geometry(obs)

        # Check temporal overlap if required
        if config.require_overlap:
            self._check_temporal_overlap(model, obs)

        # Get appropriate strategy
        strategy = self.get_strategy(geometry)

        # Perform pairing
        paired_ds = strategy.pair(
            model=model,
            obs=obs,
            radius_of_influence=config.radius_of_influence,
            time_tolerance=config.time_tolerance,
            vertical_method=config.vertical_method,
            horizontal_method=config.horizontal_method,
            **kwargs,
        )

        # Create paired dataset with proper prefixes
        result_ds = create_paired_dataset(
            obs_data=paired_ds,
            model_data=paired_ds,
            obs_vars=obs_vars,
            model_vars=model_vars,
        )

        # Get labels from attributes if available
        model_label = model.attrs.get("label", "model")
        obs_label = obs.attrs.get("label", "obs")

        return PairedData(
            data=result_ds,
            model_label=model_label,
            obs_label=obs_label,
            geometry=geometry,
            pairing_info={
                "radius_of_influence": config.radius_of_influence,
                "time_tolerance": config.time_tolerance,
                "vertical_method": config.vertical_method,
                "horizontal_method": config.horizontal_method,
                "strategy": strategy.__class__.__name__,
            },
        )

    def _detect_geometry(self, obs: xr.Dataset) -> DataGeometry:
        """Detect observation geometry from dataset attributes or structure.

        Parameters
        ----------
        obs
            Observation dataset.

        Returns
        -------
        DataGeometry
            Detected geometry type.

        Raises
        ------
        GeometryMismatchError
            If geometry cannot be determined.
        """
        # Check explicit geometry attribute
        if "geometry" in obs.attrs:
            geom = obs.attrs["geometry"]
            if isinstance(geom, DataGeometry):
                return geom
            if isinstance(geom, str):
                try:
                    return DataGeometry[geom.upper()]
                except KeyError:
                    pass

        # Infer from dimensions
        dims = set(obs.dims)

        # Grid: has lat/lon as dimensions
        if ("lat" in dims or "latitude" in dims) and (
            "lon" in dims or "longitude" in dims
        ):
            return DataGeometry.GRID

        # Swath: has scanline/pixel or similar
        if "scanline" in dims or "pixel" in dims or "cross_track" in dims:
            return DataGeometry.SWATH

        # Profile: has level/z dimension with time
        if "time" in dims and ("level" in dims or "z" in dims or "altitude" in dims):
            return DataGeometry.PROFILE

        # Point: has site/station dimension
        if "site" in dims or "station" in dims or "x" in dims:
            return DataGeometry.POINT

        # Track: has time dimension with lat/lon as coordinates
        if "time" in dims:
            coords = set(obs.coords)
            if ("lat" in coords or "latitude" in coords) and (
                "lon" in coords or "longitude" in coords
            ):
                return DataGeometry.TRACK

        raise GeometryMismatchError(
            f"Cannot determine observation geometry from dims {dims}. "
            "Please set the 'geometry' attribute on the dataset."
        )

    def _check_temporal_overlap(
        self, model: xr.Dataset, obs: xr.Dataset
    ) -> None:
        """Check if model and observation have temporal overlap.

        Parameters
        ----------
        model
            Model dataset.
        obs
            Observation dataset.

        Raises
        ------
        NoOverlapError
            If no temporal overlap exists.
        """
        if "time" not in model.dims or "time" not in obs.dims:
            return

        model_times = model["time"].values
        obs_times = obs["time"].values

        if len(model_times) == 0 or len(obs_times) == 0:
            return

        model_start = model_times.min()
        model_end = model_times.max()
        obs_start = obs_times.min()
        obs_end = obs_times.max()

        if model_end < obs_start or obs_end < model_start:
            raise NoOverlapError(
                f"No temporal overlap between model ({model_start} to {model_end}) "
                f"and observations ({obs_start} to {obs_end})"
            )


def create_default_engine() -> PairingEngine:
    """Create a pairing engine with all default strategies registered.

    Returns
    -------
    PairingEngine
        Engine with point, track, profile, swath, and grid strategies.
    """
    from davinci_monet.pairing.strategies.grid import GridStrategy
    from davinci_monet.pairing.strategies.point import PointStrategy
    from davinci_monet.pairing.strategies.profile import ProfileStrategy
    from davinci_monet.pairing.strategies.swath import SwathStrategy
    from davinci_monet.pairing.strategies.track import TrackStrategy

    engine = PairingEngine()
    engine.register_strategy(PointStrategy())
    engine.register_strategy(TrackStrategy())
    engine.register_strategy(ProfileStrategy())
    engine.register_strategy(SwathStrategy())
    engine.register_strategy(GridStrategy())

    return engine
