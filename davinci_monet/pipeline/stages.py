"""Pipeline stage definitions.

This module provides the Stage protocol and concrete stage implementations
for the analysis pipeline. Each stage is a composable unit of work that
transforms data through the analysis workflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence, TypeVar, runtime_checkable

import xarray as xr

from davinci_monet.core.exceptions import PipelineError
from davinci_monet.core.protocols import DataGeometry


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class StageResult:
    """Result of a pipeline stage execution.

    Attributes
    ----------
    stage_name
        Name of the stage that produced this result.
    status
        Execution status.
    data
        Output data from the stage.
    metadata
        Additional metadata about the execution.
    error
        Error message if the stage failed.
    duration_seconds
        Execution time in seconds.
    """

    stage_name: str
    status: StageStatus
    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0


@runtime_checkable
class Stage(Protocol):
    """Protocol for pipeline stages.

    A stage is a single unit of work in the analysis pipeline.
    Stages can be composed and chained together.
    """

    @property
    def name(self) -> str:
        """Stage name."""
        ...

    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage.

        Parameters
        ----------
        context
            Pipeline context containing configuration and data.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        ...

    def validate(self, context: PipelineContext) -> bool:
        """Validate that the stage can run with the given context.

        Parameters
        ----------
        context
            Pipeline context to validate.

        Returns
        -------
        bool
            True if validation passes.
        """
        ...


@dataclass
class PipelineContext:
    """Context passed between pipeline stages.

    Contains configuration, data, and state that flows through the pipeline.

    Attributes
    ----------
    config
        Configuration dictionary from YAML or programmatic setup.
    models
        Dictionary of loaded model data.
    observations
        Dictionary of loaded observation data.
    paired
        Dictionary of paired model-observation data.
    results
        Results from completed stages.
    metadata
        Pipeline metadata (start time, etc.).
    """

    config: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    paired: dict[str, Any] = field(default_factory=dict)
    results: dict[str, StageResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_model(self, label: str) -> Any:
        """Get a model by label."""
        if label not in self.models:
            raise KeyError(f"Model '{label}' not found in context")
        return self.models[label]

    def get_observation(self, label: str) -> Any:
        """Get an observation by label."""
        if label not in self.observations:
            raise KeyError(f"Observation '{label}' not found in context")
        return self.observations[label]

    def get_paired(self, key: str) -> Any:
        """Get paired data by key."""
        if key not in self.paired:
            raise KeyError(f"Paired data '{key}' not found in context")
        return self.paired[key]


class BaseStage(ABC):
    """Abstract base class for pipeline stages.

    Provides common functionality for stage implementations.
    """

    def __init__(self, name: str | None = None) -> None:
        """Initialize stage.

        Parameters
        ----------
        name
            Optional custom name. If None, uses class name.
        """
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Stage name."""
        return self._name

    def validate(self, context: PipelineContext) -> bool:
        """Default validation - always passes.

        Override in subclasses for specific validation.
        """
        return True

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute the stage."""
        ...

    def _create_result(
        self,
        status: StageStatus,
        data: Any = None,
        error: str | None = None,
        duration: float = 0.0,
        **metadata: Any,
    ) -> StageResult:
        """Create a stage result."""
        return StageResult(
            stage_name=self.name,
            status=status,
            data=data,
            error=error,
            duration_seconds=duration,
            metadata=metadata,
        )


class LoadModelsStage(BaseStage):
    """Stage for loading model data.

    Reads model configuration and loads model files into the context.
    """

    def __init__(self) -> None:
        super().__init__("load_models")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that model config exists."""
        return "model" in context.config or "models" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        """Load model data from configuration."""
        import time

        from davinci_monet.models import open_model

        start = time.time()
        model_config = context.config.get("model") or context.config.get("models", {})

        loaded_count = 0
        for label, config in model_config.items():
            try:
                files = config.get("files", config.get("filename"))
                mod_type = config.get("mod_type", "generic")
                variables = config.get("variables")

                if isinstance(variables, dict):
                    var_list = list(variables.keys())
                else:
                    var_list = variables

                model_data = open_model(
                    files=files,
                    mod_type=mod_type,
                    variables=var_list,
                    label=label,
                )
                context.models[label] = model_data
                loaded_count += 1
            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load model '{label}': {e}",
                    duration=time.time() - start,
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_models": list(context.models.keys())},
            duration=time.time() - start,
            count=loaded_count,
        )


class LoadObservationsStage(BaseStage):
    """Stage for loading observation data.

    Reads observation configuration and loads observation files into the context.
    """

    def __init__(self) -> None:
        super().__init__("load_observations")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that observation config exists."""
        return "obs" in context.config or "observations" in context.config

    def execute(self, context: PipelineContext) -> StageResult:
        """Load observation data from configuration."""
        import time

        from davinci_monet.observations import create_observation_data

        start = time.time()
        obs_config = context.config.get("obs") or context.config.get("observations", {})

        loaded_count = 0
        for label, config in obs_config.items():
            try:
                obs_type = config.get("obs_type", "pt_sfc")
                filename = config.get("filename")
                variables = config.get("variables", {})

                obs_data = create_observation_data(
                    label=label,
                    obs_type=obs_type,
                    filename=filename,
                    variables=variables,
                )
                context.observations[label] = obs_data
                loaded_count += 1
            except Exception as e:
                return self._create_result(
                    StageStatus.FAILED,
                    error=f"Failed to load observation '{label}': {e}",
                    duration=time.time() - start,
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"loaded_observations": list(context.observations.keys())},
            duration=time.time() - start,
            count=loaded_count,
        )


class PairingStage(BaseStage):
    """Stage for pairing model and observation data.

    Uses the pairing engine to match model output with observations.
    """

    def __init__(self) -> None:
        super().__init__("pairing")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that models and observations are loaded."""
        return bool(context.models) and bool(context.observations)

    def execute(self, context: PipelineContext) -> StageResult:
        """Pair model and observation data."""
        import time

        from davinci_monet.pairing import PairingEngine

        start = time.time()
        paired_count = 0

        # Get pairing configuration
        pairing_config = context.config.get("pairing", {})

        engine = PairingEngine()

        for model_label, model_data in context.models.items():
            for obs_label, obs_data in context.observations.items():
                try:
                    pair_key = f"{model_label}_{obs_label}"

                    # Skip if not in model-obs mapping
                    model_config = context.config.get("model", {}).get(model_label, {})
                    mapping = model_config.get("mapping", {})
                    if mapping and obs_label not in mapping:
                        continue

                    # Get model and obs datasets
                    model_ds = model_data.data if hasattr(model_data, "data") else model_data
                    obs_ds = obs_data.data if hasattr(obs_data, "data") else obs_data

                    if model_ds is None or obs_ds is None:
                        continue

                    # Pair data
                    paired_ds = engine.pair(
                        model_ds,
                        obs_ds,
                        radius=pairing_config.get("radius_of_influence", 1e6),
                        time_tolerance=pairing_config.get("time_tolerance", "1h"),
                    )

                    context.paired[pair_key] = paired_ds
                    paired_count += 1

                except Exception as e:
                    # Log but continue with other pairs
                    context.metadata.setdefault("pairing_errors", []).append(
                        f"{pair_key}: {e}"
                    )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"paired_keys": list(context.paired.keys())},
            duration=time.time() - start,
            count=paired_count,
        )


class StatisticsStage(BaseStage):
    """Stage for calculating statistics on paired data."""

    def __init__(self) -> None:
        super().__init__("statistics")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that paired data exists."""
        return bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Calculate statistics on paired data."""
        import time

        start = time.time()
        stats_results: dict[str, Any] = {}

        stats_config = context.config.get("stats", {})

        for pair_key, paired_data in context.paired.items():
            try:
                # Calculate basic statistics
                pair_stats = self._calculate_stats(paired_data, stats_config)
                stats_results[pair_key] = pair_stats
            except Exception as e:
                context.metadata.setdefault("stats_errors", []).append(
                    f"{pair_key}: {e}"
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data=stats_results,
            duration=time.time() - start,
        )

    def _calculate_stats(
        self, paired_data: xr.Dataset, config: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate statistics for a paired dataset."""
        import numpy as np

        stats: dict[str, Any] = {}

        # Find model and obs variable pairs
        model_vars = [v for v in paired_data.data_vars if v.endswith("_model")]
        obs_vars = [v for v in paired_data.data_vars if v.endswith("_obs")]

        for model_var in model_vars:
            base_name = model_var.replace("_model", "")
            obs_var = f"{base_name}_obs"

            if obs_var not in paired_data:
                continue

            model_vals = paired_data[model_var].values.flatten()
            obs_vals = paired_data[obs_var].values.flatten()

            # Remove NaNs
            mask = ~(np.isnan(model_vals) | np.isnan(obs_vals))
            model_vals = model_vals[mask]
            obs_vals = obs_vals[mask]

            if len(model_vals) == 0:
                continue

            # Calculate metrics
            diff = model_vals - obs_vals
            stats[base_name] = {
                "n": len(model_vals),
                "mean_bias": float(np.mean(diff)),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "correlation": float(np.corrcoef(model_vals, obs_vals)[0, 1])
                if len(model_vals) > 1
                else np.nan,
                "model_mean": float(np.mean(model_vals)),
                "obs_mean": float(np.mean(obs_vals)),
            }

        return stats


class PlottingStage(BaseStage):
    """Stage for generating plots from paired data."""

    def __init__(self) -> None:
        super().__init__("plotting")

    def validate(self, context: PipelineContext) -> bool:
        """Validate that paired data exists."""
        return bool(context.paired)

    def execute(self, context: PipelineContext) -> StageResult:
        """Generate plots from paired data."""
        import time

        start = time.time()
        plots_generated: list[str] = []

        plot_config = context.config.get("plots", {})

        # Plotting is optional - if no config, skip
        if not plot_config:
            return self._create_result(
                StageStatus.SKIPPED,
                data={"message": "No plot configuration found"},
                duration=time.time() - start,
            )

        # TODO: Implement plotting when Phase 9 is complete
        # For now, return completed with placeholder
        return self._create_result(
            StageStatus.COMPLETED,
            data={"plots_generated": plots_generated},
            duration=time.time() - start,
        )


class SaveResultsStage(BaseStage):
    """Stage for saving analysis results."""

    def __init__(self) -> None:
        super().__init__("save_results")

    def execute(self, context: PipelineContext) -> StageResult:
        """Save analysis results to files."""
        import time

        from davinci_monet.io import write_dataset

        start = time.time()
        saved_files: list[str] = []

        output_config = context.config.get("output", {})
        output_dir = output_config.get("dir", ".")

        # Save paired data
        for pair_key, paired_data in context.paired.items():
            try:
                if isinstance(paired_data, xr.Dataset):
                    filename = f"{output_dir}/{pair_key}_paired.nc"
                    write_dataset(paired_data, filename)
                    saved_files.append(filename)
            except Exception as e:
                context.metadata.setdefault("save_errors", []).append(
                    f"{pair_key}: {e}"
                )

        return self._create_result(
            StageStatus.COMPLETED,
            data={"saved_files": saved_files},
            duration=time.time() - start,
        )


# Convenience function to create a standard analysis pipeline
def create_standard_pipeline() -> list[BaseStage]:
    """Create a standard analysis pipeline with all stages.

    Returns
    -------
    list[BaseStage]
        List of stages for a complete analysis.
    """
    return [
        LoadModelsStage(),
        LoadObservationsStage(),
        PairingStage(),
        StatisticsStage(),
        PlottingStage(),
        SaveResultsStage(),
    ]
