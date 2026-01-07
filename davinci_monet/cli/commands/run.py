"""Run command for DAVINCI-MONET CLI.

This module implements the main analysis execution command.
"""

from __future__ import annotations

from pathlib import Path

import typer

from davinci_monet.cli.app import (
    DEBUG,
    ERROR_COLOR,
    INFO_COLOR,
    SUCCESS_COLOR,
    timer,
)


def run_analysis(control_path: str, debug: bool = False) -> None:
    """Execute DAVINCI-MONET analysis from a control file.

    Parameters
    ----------
    control_path
        Path to the YAML control file.
    debug
        If True, show full tracebacks on error.
    """
    # Update global debug flag
    import davinci_monet.cli.app as app_module

    app_module.DEBUG = debug

    p = Path(control_path)
    if not p.is_file():
        typer.secho(f"Error: control file {control_path!r} does not exist", fg=ERROR_COLOR)
        raise typer.Exit(2)

    typer.secho(f"Using control file: {control_path!r}", fg=INFO_COLOR)
    typer.secho(f"Full path: {p.absolute().as_posix()}", fg=INFO_COLOR)

    with timer("Loading configuration"):
        from davinci_monet.config import load_config

        config = load_config(p)

    with timer("Initializing pipeline"):
        from davinci_monet.pipeline.runner import PipelineRunner
        from davinci_monet.pipeline.stages import PipelineContext

        context = PipelineContext(config=config)
        runner = PipelineRunner(fail_fast=True)

    with timer("Opening model(s)"):
        from davinci_monet.pipeline.stages import LoadModelsStage

        models_stage = LoadModelsStage()
        models_result = models_stage.execute(context)
        if not models_result.success:
            raise RuntimeError(f"Failed to load models: {models_result.error}")

    with timer("Opening observation(s)"):
        from davinci_monet.pipeline.stages import LoadObservationsStage

        obs_stage = LoadObservationsStage()
        obs_result = obs_stage.execute(context)
        if not obs_result.success:
            raise RuntimeError(f"Failed to load observations: {obs_result.error}")

    with timer("Pairing data"):
        from davinci_monet.pipeline.stages import PairingStage

        pairing_stage = PairingStage()
        pairing_result = pairing_stage.execute(context)
        if not pairing_result.success:
            raise RuntimeError(f"Failed to pair data: {pairing_result.error}")

    # Check if plotting is configured
    if config.plots:
        with timer("Generating plots"):
            from davinci_monet.pipeline.stages import PlottingStage

            plotting_stage = PlottingStage()
            plotting_result = plotting_stage.execute(context)
            if not plotting_result.success:
                typer.secho(
                    f"Warning: Some plots failed: {plotting_result.error}",
                    fg=typer.colors.YELLOW,
                )

    # Check if statistics are configured
    if config.stats:
        with timer("Computing statistics"):
            from davinci_monet.pipeline.stages import StatisticsStage

            stats_stage = StatisticsStage()
            stats_result = stats_stage.execute(context)
            if not stats_result.success:
                typer.secho(
                    f"Warning: Statistics computation failed: {stats_result.error}",
                    fg=typer.colors.YELLOW,
                )

    # Save results if output directory configured
    if config.analysis and config.analysis.output_dir:
        with timer("Saving results"):
            from davinci_monet.pipeline.stages import SaveResultsStage

            save_stage = SaveResultsStage()
            save_result = save_stage.execute(context)
            if not save_result.success:
                typer.secho(
                    f"Warning: Failed to save some results: {save_result.error}",
                    fg=typer.colors.YELLOW,
                )

    typer.secho("\nAnalysis complete!", fg=SUCCESS_COLOR)
