#!/usr/bin/env python
"""
Run CESM model evaluation against AirNow and AERONET observations.

This script uses the DAVINCI-MONET pipeline to:
1. Load CESM model data
2. Load AirNow and AERONET observations
3. Pair model with observations
4. Generate comparison plots
5. Calculate statistics

Usage:
    python run_evaluation.py

Or use the CLI directly:
    davinci-monet run ../configs/cesm_airnow_aeronet.yaml
"""

from pathlib import Path

from davinci_monet.pipeline.runner import run_analysis


def main():
    """Run the ASIA-AQ model evaluation pipeline."""
    # Path to config file
    config_path = Path(__file__).parent.parent / "configs" / "cesm_airnow_aeronet.yaml"

    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    print("=" * 70)
    print("CESM/CAM-chem ASIA-AQ Model Evaluation")
    print("=" * 70)
    print(f"\nUsing config: {config_path}")
    print()

    # Run the pipeline
    result = run_analysis(str(config_path))

    # Report results
    print()
    print("=" * 70)
    if result.success:
        print("Pipeline completed successfully!")
        print(f"Total time: {result.total_duration_seconds:.1f} seconds")
        print(f"Stages completed: {', '.join(result.completed_stages)}")
    else:
        print("Pipeline failed!")
        for failed in result.failed_stages:
            print(f"  {failed.stage_name}: {failed.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
