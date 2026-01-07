# DAVINCI-MONET Wiki

**Diagnostic Analysis and Validation Infrastructure for Numerical Chemistry Investigation - Model and ObservatioN Evaluation Toolkit**

Welcome to the DAVINCI-MONET documentation wiki.

## Getting Started

- [[Installation]] - Setup and dependencies
- [[Quick Start]] - Your first analysis in 5 minutes
- [[Examples]] - Complete working examples

## User Guide

- [[Configuration]] - YAML configuration guide
- [[CLI Reference]] - Command-line interface
- [[Pairing Strategies]] - Understanding data geometry pairing

## Reference

- [[API Reference]] - Python API documentation
- [[Statistics Metrics]] - All 27 supported metrics
- [[Plot Types]] - All 10 plot types

## Migration

- [[Migration Guide]] - Migrating from MELODIES-MONET

---

## Overview

DAVINCI-MONET is a modern rewrite of MELODIES-MONET with:

- **Unified Pairing Engine** - Single system based on data geometry (point, track, profile, swath, grid)
- **Type Safety** - Full type hints with mypy strict mode
- **Modular Architecture** - Small, focused modules (<500 lines each)
- **Comprehensive Testing** - 732+ tests with synthetic data generation

## Supported Data Types

### Models
- CMAQ
- WRF-Chem
- UFS/RRFS
- CESM (FV and SE grids)
- Generic NetCDF

### Observations
- **Surface**: AirNow, AQS, AERONET, OpenAQ
- **Aircraft**: ICARTT format
- **Satellite**: TROPOMI, GOES-ABI
- **Sondes**: Ozonesonde (WOUDC, SHADOZ)

## Quick Links

- [GitHub Repository](https://github.com/NCAR/DAVINCI-MONET)
- [MELODIES-MONET](https://github.com/NOAA-CSL/MELODIES-MONET) (original project)
