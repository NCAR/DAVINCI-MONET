"""Synthetic data generators for testing.

This module provides utilities to generate synthetic model output and
observation data for testing the pairing, plotting, and statistics
components without requiring external datasets.

Usage:
    from davinci_monet.tests.synthetic import (
        create_model_dataset,
        create_point_observations,
        create_track_observations,
        PerfectMatchScenario,
    )

    # Generate synthetic model data
    model = create_model_dataset(
        variables=["O3", "PM25"],
        time_range=("2024-01-01", "2024-01-03"),
    )

    # Generate matching observations
    obs = create_point_observations(
        n_sites=10,
        variables=["O3", "PM25"],
        time_range=("2024-01-01", "2024-01-03"),
    )
"""

from davinci_monet.tests.synthetic.generators import (
    Domain,
    TimeConfig,
    create_coordinate_grid,
    create_time_axis,
    random_locations_in_domain,
)
from davinci_monet.tests.synthetic.models import (
    ModelConfig,
    create_model_dataset,
    create_variable_field,
)
from davinci_monet.tests.synthetic.observations import (
    create_point_observations,
    create_profile_observations,
    create_swath_observations,
    create_track_observations,
)
from davinci_monet.tests.synthetic.scenarios import (
    BiasScenario,
    MismatchScenario,
    PerfectMatchScenario,
    Scenario,
)

__all__ = [
    # Generators
    "Domain",
    "TimeConfig",
    "create_coordinate_grid",
    "create_time_axis",
    "random_locations_in_domain",
    # Models
    "ModelConfig",
    "create_model_dataset",
    "create_variable_field",
    # Observations
    "create_point_observations",
    "create_track_observations",
    "create_profile_observations",
    "create_swath_observations",
    # Scenarios
    "Scenario",
    "PerfectMatchScenario",
    "BiasScenario",
    "MismatchScenario",
]
