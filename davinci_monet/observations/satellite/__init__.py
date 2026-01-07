"""Satellite observation readers.

This subpackage provides readers for satellite-based observations including
TROPOMI, GOES, MODIS, and other satellite products.
"""

from davinci_monet.observations.satellite.goes import GOESReader, open_goes
from davinci_monet.observations.satellite.tropomi import TROPOMIReader, open_tropomi

__all__ = [
    "TROPOMIReader",
    "GOESReader",
    "open_tropomi",
    "open_goes",
]
