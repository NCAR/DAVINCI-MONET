"""Core module containing protocols, registry, and base classes.

This module provides the foundational components for DAVINCI-MONET:
- Protocol definitions for all pluggable components
- Plugin registry system
- Custom exceptions
- Type aliases
"""

from davinci_monet.core.protocols import (
    Configurable,
    DataGeometry,
    DataReader,
    DataWriter,
    ModelProcessor,
    ModelReader,
    ObservationProcessor,
    ObservationReader,
    PairingEngine,
    PairingStrategy,
    Pipeline,
    PipelineStage,
    Plotter,
    SpatialPlotter,
    StatisticMetric,
    StatisticsCalculator,
)

__all__ = [
    # Data geometry enum
    "DataGeometry",
    # Model protocols
    "ModelReader",
    "ModelProcessor",
    # Observation protocols
    "ObservationReader",
    "ObservationProcessor",
    # Pairing protocols
    "PairingStrategy",
    "PairingEngine",
    # Plotting protocols
    "Plotter",
    "SpatialPlotter",
    # Statistics protocols
    "StatisticMetric",
    "StatisticsCalculator",
    # Pipeline protocols
    "PipelineStage",
    "Pipeline",
    # I/O protocols
    "DataReader",
    "DataWriter",
    # Configuration protocol
    "Configurable",
]
