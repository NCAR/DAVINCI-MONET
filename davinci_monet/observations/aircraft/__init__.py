"""Aircraft observation readers.

This subpackage provides readers for aircraft-based observations including
ICARTT format files from field campaigns.
"""

from davinci_monet.observations.aircraft.icartt import ICARTTReader, open_icartt

__all__ = [
    "ICARTTReader",
    "open_icartt",
]
