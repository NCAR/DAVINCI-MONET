"""I/O module for reading and writing data files.

This module provides functions for reading and writing various
data formats including NetCDF, pickle, CSV, and ICARTT.
"""

from davinci_monet.io.readers import (
    read_csv,
    read_csv_to_xarray,
    read_dataset,
    read_icartt,
    read_mfdataset,
    read_pickle,
    read_saved_analysis,
)
from davinci_monet.io.writers import (
    write_analysis_results,
    write_csv,
    write_dataset,
    write_paired_data,
    write_pickle,
    write_statistics,
)

__all__ = [
    # Readers
    "read_dataset",
    "read_mfdataset",
    "read_pickle",
    "read_csv",
    "read_csv_to_xarray",
    "read_icartt",
    "read_saved_analysis",
    # Writers
    "write_dataset",
    "write_pickle",
    "write_csv",
    "write_paired_data",
    "write_statistics",
    "write_analysis_results",
]
