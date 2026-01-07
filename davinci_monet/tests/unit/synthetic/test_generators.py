"""Tests for base synthetic data generators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from davinci_monet.tests.synthetic.generators import (
    VARIABLE_SPECS,
    Domain,
    TimeConfig,
    VariableSpec,
    add_diurnal_cycle,
    add_random_noise,
    create_coordinate_grid,
    create_level_axis,
    create_time_axis,
    generate_random_field,
    get_variable_spec,
    random_locations_in_domain,
)


class TestDomain:
    """Tests for Domain dataclass."""

    def test_default_domain(self) -> None:
        """Test default domain values."""
        domain = Domain()
        assert domain.lon_min == -130.0
        assert domain.lon_max == -60.0
        assert domain.lat_min == 20.0
        assert domain.lat_max == 55.0

    def test_lon_range(self) -> None:
        """Test longitude range calculation."""
        domain = Domain(lon_min=-100, lon_max=-90)
        assert domain.lon_range == 10.0

    def test_lat_range(self) -> None:
        """Test latitude range calculation."""
        domain = Domain(lat_min=30, lat_max=45)
        assert domain.lat_range == 15.0

    def test_resolution(self) -> None:
        """Test grid resolution calculation."""
        domain = Domain(lon_min=0, lon_max=10, lat_min=0, lat_max=5, n_lon=11, n_lat=6)
        assert domain.lon_resolution == 1.0
        assert domain.lat_resolution == 1.0

    def test_center(self) -> None:
        """Test center calculation."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40)
        lon_center, lat_center = domain.center
        assert lon_center == -95.0
        assert lat_center == 35.0


class TestTimeConfig:
    """Tests for TimeConfig dataclass."""

    def test_default_time_config(self) -> None:
        """Test default time configuration."""
        config = TimeConfig()
        assert len(config.time_range) > 0

    def test_time_range_length(self) -> None:
        """Test time range generates correct number of steps."""
        config = TimeConfig(start="2024-01-01", end="2024-01-02", freq="1h")
        assert config.n_times == 25  # 24 hours + 1 endpoint

    def test_time_range_hourly(self) -> None:
        """Test hourly time range."""
        config = TimeConfig(start="2024-01-01 00:00", end="2024-01-01 03:00", freq="1h")
        assert len(config.time_range) == 4

    def test_time_range_daily(self) -> None:
        """Test daily time range."""
        config = TimeConfig(start="2024-01-01", end="2024-01-05", freq="1D")
        assert len(config.time_range) == 5


class TestVariableSpec:
    """Tests for VariableSpec dataclass."""

    def test_default_values(self) -> None:
        """Test default variable specification."""
        spec = VariableSpec(name="test")
        assert spec.name == "test"
        assert spec.units == "units"
        assert spec.mean == 50.0
        assert spec.std == 10.0

    def test_long_name_auto_fill(self) -> None:
        """Test long_name defaults to name."""
        spec = VariableSpec(name="O3")
        assert spec.long_name == "O3"

    def test_predefined_specs(self) -> None:
        """Test predefined variable specifications exist."""
        assert "O3" in VARIABLE_SPECS
        assert "PM25" in VARIABLE_SPECS
        assert "NO2" in VARIABLE_SPECS

    def test_get_variable_spec_known(self) -> None:
        """Test getting known variable spec."""
        spec = get_variable_spec("O3")
        assert spec.name == "O3"
        assert spec.units == "ppbv"

    def test_get_variable_spec_unknown(self) -> None:
        """Test getting unknown variable returns default."""
        spec = get_variable_spec("unknown_var")
        assert spec.name == "unknown_var"


class TestCoordinateGrid:
    """Tests for coordinate grid creation."""

    def test_create_coordinate_grid(self) -> None:
        """Test basic coordinate grid creation."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40, n_lon=11, n_lat=11)
        lon, lat = create_coordinate_grid(domain)

        assert len(lon) == 11
        assert len(lat) == 11
        assert float(lon[0]) == -100.0
        assert float(lon[-1]) == -90.0
        assert float(lat[0]) == 30.0
        assert float(lat[-1]) == 40.0

    def test_coordinate_attributes(self) -> None:
        """Test coordinate arrays have proper attributes."""
        domain = Domain(n_lon=5, n_lat=5)
        lon, lat = create_coordinate_grid(domain)

        assert lon.attrs["units"] == "degrees_east"
        assert lat.attrs["units"] == "degrees_north"


class TestTimeAxis:
    """Tests for time axis creation."""

    def test_create_time_axis(self) -> None:
        """Test time axis creation."""
        config = TimeConfig(start="2024-01-01", end="2024-01-02", freq="6h")
        time = create_time_axis(config)

        assert len(time) == 5
        assert "time" in time.dims


class TestLevelAxis:
    """Tests for vertical level axis creation."""

    def test_create_level_axis(self) -> None:
        """Test level axis creation."""
        level = create_level_axis(n_levels=10)
        assert len(level) == 10
        assert "level" in level.dims

    def test_level_axis_range(self) -> None:
        """Test level axis spans surface to top."""
        level = create_level_axis(n_levels=20, surface_pressure=1013, top_pressure=10)
        assert float(level[0]) == pytest.approx(1013, rel=0.01)
        assert float(level[-1]) == pytest.approx(10, rel=0.01)

    def test_level_axis_log_spacing(self) -> None:
        """Test levels are logarithmically spaced."""
        level = create_level_axis(n_levels=10)
        # Log-spaced means ratio between consecutive levels is roughly constant
        ratios = level.values[:-1] / level.values[1:]
        assert np.std(ratios) < 0.1  # Ratios should be similar


class TestRandomLocations:
    """Tests for random location generation."""

    def test_random_locations_count(self) -> None:
        """Test correct number of locations generated."""
        domain = Domain()
        lons, lats = random_locations_in_domain(domain, 10)
        assert len(lons) == 10
        assert len(lats) == 10

    def test_random_locations_in_bounds(self) -> None:
        """Test locations are within domain."""
        domain = Domain(lon_min=-100, lon_max=-90, lat_min=30, lat_max=40)
        lons, lats = random_locations_in_domain(domain, 50)

        assert np.all(lons >= domain.lon_min)
        assert np.all(lons <= domain.lon_max)
        assert np.all(lats >= domain.lat_min)
        assert np.all(lats <= domain.lat_max)

    def test_random_locations_reproducible(self) -> None:
        """Test random locations are reproducible with seed."""
        domain = Domain()
        lons1, lats1 = random_locations_in_domain(domain, 10, seed=42)
        lons2, lats2 = random_locations_in_domain(domain, 10, seed=42)

        np.testing.assert_array_equal(lons1, lons2)
        np.testing.assert_array_equal(lats1, lats2)


class TestRandomField:
    """Tests for random field generation."""

    def test_generate_random_field_shape(self) -> None:
        """Test field has correct shape."""
        spec = get_variable_spec("O3")
        field = generate_random_field((10, 20), spec, seed=42)
        assert field.shape == (10, 20)

    def test_generate_random_field_bounds(self) -> None:
        """Test field values are within bounds."""
        spec = VariableSpec(name="test", min_val=0, max_val=100)
        field = generate_random_field((100, 100), spec, seed=42)

        assert np.all(field >= spec.min_val)
        assert np.all(field <= spec.max_val)

    def test_generate_random_field_reproducible(self) -> None:
        """Test field generation is reproducible."""
        spec = get_variable_spec("O3")
        field1 = generate_random_field((10, 10), spec, seed=42)
        field2 = generate_random_field((10, 10), spec, seed=42)

        np.testing.assert_array_equal(field1, field2)


class TestDiurnalCycle:
    """Tests for diurnal cycle addition."""

    def test_add_diurnal_cycle(self) -> None:
        """Test diurnal cycle is added."""
        # Create 24-hour data
        time = xr.DataArray(
            np.arange(24),
            dims=["time"],
            attrs={"units": "hours since 2024-01-01"},
        )
        time = xr.DataArray(
            np.datetime64("2024-01-01") + np.arange(24) * np.timedelta64(1, "h"),
            dims=["time"],
        )
        data = xr.DataArray(
            np.ones(24) * 100,
            dims=["time"],
            coords={"time": time},
        )

        result = add_diurnal_cycle(data, amplitude=0.2, peak_hour=14)

        # Values should vary
        assert result.max() > result.min()
        # Mean should be close to original
        assert float(result.mean()) == pytest.approx(100, rel=0.05)


class TestRandomNoise:
    """Tests for random noise addition."""

    def test_add_random_noise(self) -> None:
        """Test noise is added to data."""
        data = xr.DataArray(np.ones((10, 10)) * 100)
        result = add_random_noise(data, noise_fraction=0.1, seed=42)

        # Values should vary
        assert float(result.std()) > 0
        # Mean should be close to original
        assert float(result.mean()) == pytest.approx(100, rel=0.05)

    def test_noise_reproducible(self) -> None:
        """Test noise is reproducible with seed."""
        data = xr.DataArray(np.ones((10, 10)) * 100)
        result1 = add_random_noise(data, noise_fraction=0.1, seed=42)
        result2 = add_random_noise(data, noise_fraction=0.1, seed=42)

        xr.testing.assert_equal(result1, result2)
