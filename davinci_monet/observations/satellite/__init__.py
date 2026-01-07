"""Satellite observation readers.

This subpackage provides readers for satellite-based observations.

Satellite-Specific Readers
--------------------------
These readers are optimized for specific satellite products and rely on
monetio for full functionality:

- **TROPOMIReader** / **open_tropomi**: TROPOMI L2 products (NO2, O3, CO, HCHO, SO2)
  Requires: monetio.sat._tropomi_l2_no2_mm

- **GOESL3AODReader** / **open_goes_l3_aod**: GOES-ABI L3 AOD products
  Requires: monetio.sat.goes

Generic Readers
---------------
These readers work with any satellite product but lack satellite-specific
features like projection handling and specialized QA filtering:

- **GenericL2Reader** / **open_satellite_l2**: Generic L2 swath products
- **GenericL3Reader** / **open_satellite_l3**: Generic L3 gridded products

Note
----
The satellite-specific readers fall back to basic xarray reading if monetio
is not available, but some features (projections, swath geometry) may not
work correctly without monetio.
"""

# Satellite-specific readers (require monetio for full functionality)
from davinci_monet.observations.satellite.tropomi import (
    TROPOMIReader,
    open_tropomi,
    TROPOMI_VARIABLE_MAPPING,
)
from davinci_monet.observations.satellite.goes_l3_aod import (
    GOESL3AODReader,
    GOESReader,  # Backward compatibility alias
    open_goes_l3_aod,
    open_goes,  # Backward compatibility alias (deprecated)
    GOES_AOD_VARIABLE_MAPPING,
    GOES_VARIABLE_MAPPING,  # Backward compatibility alias
)

# Generic readers (pure xarray, no monetio dependency)
from davinci_monet.observations.satellite.generic_l2 import (
    GenericL2Reader,
    open_satellite_l2,
)
from davinci_monet.observations.satellite.generic_l3 import (
    GenericL3Reader,
    open_satellite_l3,
)

__all__ = [
    # TROPOMI
    "TROPOMIReader",
    "open_tropomi",
    "TROPOMI_VARIABLE_MAPPING",
    # GOES L3 AOD
    "GOESL3AODReader",
    "GOESReader",  # Backward compatibility
    "open_goes_l3_aod",
    "open_goes",  # Backward compatibility (deprecated)
    "GOES_AOD_VARIABLE_MAPPING",
    "GOES_VARIABLE_MAPPING",  # Backward compatibility
    # Generic L2
    "GenericL2Reader",
    "open_satellite_l2",
    # Generic L3
    "GenericL3Reader",
    "open_satellite_l3",
]
