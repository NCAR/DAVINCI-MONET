"""CESM (Community Earth System Model) readers.

This module provides readers for CESM output, including:
- CESM-FV: Finite volume dynamical core (regular lat-lon grid)
- CESM-SE: Spectral element dynamical core (unstructured grid)
- CAM-chem: Chemistry component
- MUSICA: Multi-Scale Infrastructure for Chemistry and Aerosols
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from davinci_monet.core.exceptions import DataFormatError, DataNotFoundError
from davinci_monet.core.registry import model_registry
from davinci_monet.models.base import ModelData, create_model_data


# Standard variable name mappings for CESM/CAM-chem
CESM_VARIABLE_MAPPING: dict[str, str] = {
    "ozone": "O3",
    "pm25": "PM25",
    "no2": "NO2",
    "no": "NO",
    "co": "CO",
    "so2": "SO2",
    # Aerosols
    "bc": "bc_a4",
    "oc": "pom_a4",
    "so4": "so4_a1",
    "dust": "dst_a1",
    "sea_salt": "ncl_a1",
    # Meteorology
    "temperature": "T",
    "temperature_k": "T",
    "pressure": "PS",
    "pres_pa_mid": "P",
    "relative_humidity": "RELHUM",
    "specific_humidity": "Q",
    "wind_speed_u": "U",
    "wind_speed_v": "V",
    "geopotential_height": "Z3",
}

# Reverse mapping
CESM_STANDARD_NAMES: dict[str, str] = {v: k for k, v in CESM_VARIABLE_MAPPING.items()}


@model_registry.register("cesm_fv")
class CESMFVReader:
    """Reader for CESM Finite Volume (FV) output.

    Reads CESM/CAM-chem output on the regular latitude-longitude grid
    used by the finite volume dynamical core.

    Examples
    --------
    >>> reader = CESMFVReader()
    >>> ds = reader.open(["cam.h0.2024-01.nc"])
    >>> print(ds.dims)
    Frozen({'time': 31, 'lev': 32, 'lat': 192, 'lon': 288})
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cesm_fv"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV output files.

        Parameters
        ----------
        file_paths
            Paths to CESM output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options passed to monetio or xarray.

        Returns
        -------
        xr.Dataset
            CESM data with standardized dimensions (time, z, lat, lon).
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No CESM files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"CESM files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV files using monetio."""
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cesm_fv_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-FV files using xarray."""
        if len(file_paths) > 1:
            ds = xr.open_mfdataset(
                [str(f) for f in file_paths],
                combine="by_coords",
                **kwargs,
            )
        else:
            ds = xr.open_dataset(str(file_paths[0]), **kwargs)

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-FV dataset dimensions."""
        dim_renames: dict[str, str] = {}

        # CESM standard dimension names
        if "lev" in ds.dims:
            dim_renames["lev"] = "z"
        if "ilev" in ds.dims:
            dim_renames["ilev"] = "z_interface"

        if dim_renames:
            ds = ds.rename(dim_renames)

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return CESM variable name mapping."""
        return CESM_VARIABLE_MAPPING


@model_registry.register("cesm_se")
class CESMSEReader:
    """Reader for CESM Spectral Element (SE) output.

    Reads CESM/CAM-chem output on the unstructured grid used by the
    spectral element dynamical core. Requires a SCRIP file for
    coordinate mapping.

    Examples
    --------
    >>> reader = CESMSEReader()
    >>> ds = reader.open(
    ...     ["cam.h0.2024-01.nc"],
    ...     scrip_file="ne30pg2_scrip.nc"
    ... )
    """

    @property
    def name(self) -> str:
        """Return reader name."""
        return "cesm_se"

    def open(
        self,
        file_paths: Sequence[str | Path],
        variables: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE output files.

        Parameters
        ----------
        file_paths
            Paths to CESM output files.
        variables
            Variables to load. If None, loads all variables.
        **kwargs
            Additional options:
            - scrip_file: Path to SCRIP grid file (required for coordinate mapping)

        Returns
        -------
        xr.Dataset
            CESM data with unstructured grid dimensions.
        """
        file_list = [Path(f) for f in file_paths]

        if not file_list:
            raise DataNotFoundError("No CESM files provided")

        missing = [f for f in file_list if not f.exists()]
        if missing:
            raise DataNotFoundError(f"CESM files not found: {missing}")

        # Try monetio first
        try:
            ds = self._open_with_monetio(file_list, variables, **kwargs)
        except ImportError:
            warnings.warn(
                "monetio not available, using basic xarray reader. "
                "SE grid coordinate mapping may not work correctly.",
                UserWarning,
            )
            ds = self._open_with_xarray(file_list, variables, **kwargs)

        ds = self._standardize_dataset(ds)

        return ds

    def _open_with_monetio(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE files using monetio."""
        import monetio as mio

        mio_kwargs: dict[str, Any] = {}

        if variables is not None:
            mio_kwargs["var_list"] = list(variables)

        # SCRIP file is required for SE grid
        if "scrip_file" in kwargs:
            mio_kwargs["scrip_file"] = kwargs.pop("scrip_file")

        mio_kwargs.update(kwargs)

        files = [str(f) for f in file_paths]
        ds: xr.Dataset = mio.models._cesm_se_mm.open_mfdataset(files, **mio_kwargs)

        return ds

    def _open_with_xarray(
        self,
        file_paths: list[Path],
        variables: Sequence[str] | None,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Open CESM-SE files using xarray."""
        # Remove our custom kwargs
        xr_kwargs = {k: v for k, v in kwargs.items() if k != "scrip_file"}

        if len(file_paths) > 1:
            ds = xr.open_mfdataset(
                [str(f) for f in file_paths],
                combine="by_coords",
                **xr_kwargs,
            )
        else:
            ds = xr.open_dataset(str(file_paths[0]), **xr_kwargs)

        # If SCRIP file provided, add coordinates
        if "scrip_file" in kwargs:
            scrip_path = kwargs["scrip_file"]
            if Path(scrip_path).exists():
                scrip = xr.open_dataset(scrip_path)
                if "grid_center_lat" in scrip:
                    ds = ds.assign_coords(lat=("ncol", scrip["grid_center_lat"].values))
                if "grid_center_lon" in scrip:
                    ds = ds.assign_coords(lon=("ncol", scrip["grid_center_lon"].values))

        if variables is not None:
            available = [v for v in variables if v in ds.data_vars]
            if available:
                ds = ds[available]

        return ds

    def _standardize_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize CESM-SE dataset dimensions."""
        dim_renames: dict[str, str] = {}

        if "lev" in ds.dims:
            dim_renames["lev"] = "z"
        if "ilev" in ds.dims:
            dim_renames["ilev"] = "z_interface"

        if dim_renames:
            ds = ds.rename(dim_renames)

        return ds

    def get_variable_mapping(self) -> Mapping[str, str]:
        """Return CESM variable name mapping."""
        return CESM_VARIABLE_MAPPING


def open_cesm(
    files: str | Path | Sequence[str | Path],
    variables: Sequence[str] | None = None,
    label: str = "cesm",
    grid_type: str = "fv",
    **kwargs: Any,
) -> ModelData:
    """Convenience function to open CESM model data.

    Parameters
    ----------
    files
        File path(s) or glob pattern.
    variables
        Variables to load.
    label
        Model label.
    grid_type
        Grid type: 'fv' for finite volume, 'se' for spectral element.
    **kwargs
        Additional reader options.

    Returns
    -------
    ModelData
        CESM model data container.
    """
    if grid_type.lower() == "se":
        reader: CESMFVReader | CESMSEReader = CESMSEReader()
        mod_type = "cesm_se"
    else:
        reader = CESMFVReader()
        mod_type = "cesm_fv"

    # Handle glob pattern
    if isinstance(files, (str, Path)):
        file_str = str(files)
        if "*" in file_str or "?" in file_str:
            from glob import glob
            file_list = sorted(glob(file_str))
            if not file_list:
                raise DataNotFoundError(f"No files match pattern: {files}")
            file_paths: Sequence[str | Path] = file_list
        else:
            file_paths = [files]
    else:
        file_paths = list(files)

    ds = reader.open(file_paths, variables, **kwargs)

    return create_model_data(
        label=label,
        mod_type=mod_type,
        data=ds,
        files=file_paths,
    )
