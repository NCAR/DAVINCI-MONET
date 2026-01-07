"""Observation data handling module.

This module provides classes for loading and processing observational
data from various sources (surface stations, aircraft, satellites, etc.).
"""

from davinci_monet.observations.base import (
    GriddedObservation,
    ObservationData,
    PointObservation,
    ProfileObservation,
    SwathObservation,
    TrackObservation,
    create_observation_data,
)

# Surface observation readers
from davinci_monet.observations.surface.aeronet import AERONETReader, open_aeronet
from davinci_monet.observations.surface.airnow import AirNowReader, open_airnow
from davinci_monet.observations.surface.aqs import AQSReader, open_aqs
from davinci_monet.observations.surface.openaq import OpenAQReader, open_openaq

# Aircraft observation readers
from davinci_monet.observations.aircraft.icartt import ICARTTReader, open_icartt

# Satellite observation readers
from davinci_monet.observations.satellite.goes import GOESReader, open_goes
from davinci_monet.observations.satellite.tropomi import TROPOMIReader, open_tropomi

# Sonde observation readers
from davinci_monet.observations.sonde.ozonesonde import OzonesondeReader, open_ozonesonde

__all__ = [
    # Base classes
    "ObservationData",
    "PointObservation",
    "TrackObservation",
    "ProfileObservation",
    "SwathObservation",
    "GriddedObservation",
    "create_observation_data",
    # Surface readers
    "AQSReader",
    "AirNowReader",
    "AERONETReader",
    "OpenAQReader",
    "open_aqs",
    "open_airnow",
    "open_aeronet",
    "open_openaq",
    # Aircraft readers
    "ICARTTReader",
    "open_icartt",
    # Satellite readers
    "TROPOMIReader",
    "GOESReader",
    "open_tropomi",
    "open_goes",
    # Sonde readers
    "OzonesondeReader",
    "open_ozonesonde",
]
