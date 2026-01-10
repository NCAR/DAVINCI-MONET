"""Tests for davinci_monet.plots.base utilities."""

import pytest
import xarray as xr
import numpy as np

from davinci_monet.plots.base import (
    VARIABLE_DISPLAY_NAMES,
    format_variable_display_name,
    format_label_with_units,
    get_variable_label,
    get_variable_units,
)


class TestVariableDisplayNames:
    """Tests for the VARIABLE_DISPLAY_NAMES lookup table."""

    def test_common_pollutants_have_subscripts(self):
        """Chemical formulas should use Unicode subscripts."""
        assert VARIABLE_DISPLAY_NAMES["pm25"] == "PM₂.₅"
        assert VARIABLE_DISPLAY_NAMES["o3"] == "O₃"
        assert VARIABLE_DISPLAY_NAMES["no2"] == "NO₂"
        assert VARIABLE_DISPLAY_NAMES["so2"] == "SO₂"
        assert VARIABLE_DISPLAY_NAMES["co2"] == "CO₂"

    def test_aod_variables(self):
        """AOD variables should have wavelength in name."""
        assert VARIABLE_DISPLAY_NAMES["aod_500nm"] == "AOD (500 nm)"
        assert VARIABLE_DISPLAY_NAMES["aod_550nm"] == "AOD (550 nm)"

    def test_column_variables(self):
        """Column variables should have descriptive names."""
        assert VARIABLE_DISPLAY_NAMES["no2_trop_column"] == "Tropospheric NO₂ Column"
        assert VARIABLE_DISPLAY_NAMES["no2_column"] == "NO₂ Column"

    def test_case_variants(self):
        """Both lowercase and uppercase variants should be present."""
        assert "pm25" in VARIABLE_DISPLAY_NAMES
        assert "PM25" in VARIABLE_DISPLAY_NAMES
        assert VARIABLE_DISPLAY_NAMES["pm25"] == VARIABLE_DISPLAY_NAMES["PM25"]


class TestFormatVariableDisplayName:
    """Tests for format_variable_display_name function."""

    def test_lookup_table_match(self):
        """Variables in lookup table should return formatted name."""
        assert format_variable_display_name("pm25") == "PM₂.₅"
        assert format_variable_display_name("o3") == "O₃"
        assert format_variable_display_name("aod_500nm") == "AOD (500 nm)"

    def test_case_insensitive_lookup(self):
        """Lookup should be case-insensitive."""
        assert format_variable_display_name("PM25") == "PM₂.₅"
        assert format_variable_display_name("O3") == "O₃"
        assert format_variable_display_name("NO2") == "NO₂"

    def test_obs_prefix_with_include_prefix_true(self):
        """obs_ prefix should add 'Observed ' when include_prefix=True."""
        assert format_variable_display_name("obs_pm25", include_prefix=True) == "Observed PM₂.₅"
        assert format_variable_display_name("obs_o3", include_prefix=True) == "Observed O₃"

    def test_model_prefix_with_include_prefix_true(self):
        """model_ prefix should add 'Modeled ' when include_prefix=True."""
        assert format_variable_display_name("model_pm25", include_prefix=True) == "Modeled PM₂.₅"
        assert format_variable_display_name("model_o3", include_prefix=True) == "Modeled O₃"

    def test_obs_prefix_with_include_prefix_false(self):
        """obs_ prefix should not add 'Observed ' when include_prefix=False."""
        assert format_variable_display_name("obs_pm25", include_prefix=False) == "PM₂.₅"
        assert format_variable_display_name("obs_o3", include_prefix=False) == "O₃"

    def test_model_prefix_with_include_prefix_false(self):
        """model_ prefix should not add 'Modeled ' when include_prefix=False."""
        assert format_variable_display_name("model_pm25", include_prefix=False) == "PM₂.₅"
        assert format_variable_display_name("model_o3", include_prefix=False) == "O₃"

    def test_unknown_variable_formatting(self):
        """Unknown variables should get basic formatting."""
        # Underscores replaced, title case
        assert format_variable_display_name("some_variable") == "Some Variable"
        assert format_variable_display_name("my_custom_var") == "My Custom Var"

    def test_unknown_variable_with_prefix(self):
        """Unknown variables with prefix should still format correctly."""
        assert format_variable_display_name("obs_some_var", include_prefix=True) == "Observed Some Var"
        assert format_variable_display_name("model_some_var", include_prefix=True) == "Modeled Some Var"


class TestFormatLabelWithUnits:
    """Tests for format_label_with_units function."""

    def test_label_with_units(self):
        """Label with units should include parentheses."""
        assert format_label_with_units("PM₂.₅", "μg/m³") == "PM₂.₅ (μg/m³)"
        assert format_label_with_units("O₃", "ppb") == "O₃ (ppb)"

    def test_label_without_units(self):
        """Label without units should not have parentheses."""
        assert format_label_with_units("PM₂.₅", None) == "PM₂.₅"

    def test_dimensionless_units_omitted(self):
        """Dimensionless units ('1') should be omitted."""
        assert format_label_with_units("AOD", "1") == "AOD"

    def test_empty_string_units(self):
        """Empty string units should be treated as no units."""
        assert format_label_with_units("PM₂.₅", "") == "PM₂.₅"


class TestGetVariableLabel:
    """Tests for get_variable_label function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        return xr.Dataset({
            "obs_pm25": xr.DataArray(
                [1, 2, 3],
                attrs={"long_name": "PM2.5 Concentration", "units": "μg/m³"}
            ),
            "model_pm25": xr.DataArray(
                [1.1, 2.1, 3.1],
                attrs={"display_name": "Custom PM₂.₅", "units": "μg/m³"}
            ),
            "obs_temp": xr.DataArray(
                [20, 21, 22],
                attrs={"standard_name": "air_temperature"}
            ),
            "no_attrs_var": xr.DataArray([1, 2, 3]),
        })

    def test_custom_label_takes_precedence(self, sample_dataset):
        """Custom label should override everything."""
        result = get_variable_label(sample_dataset, "obs_pm25", custom_label="My Label")
        assert result == "My Label"

    def test_display_name_attr_used(self, sample_dataset):
        """display_name attribute should be used if present."""
        result = get_variable_label(sample_dataset, "model_pm25")
        assert result == "Custom PM₂.₅"

    def test_long_name_attr_used(self, sample_dataset):
        """long_name attribute should be used if display_name not present."""
        result = get_variable_label(sample_dataset, "obs_pm25")
        assert result == "PM2.5 Concentration"

    def test_standard_name_attr_used(self, sample_dataset):
        """standard_name should be used if long_name not present."""
        result = get_variable_label(sample_dataset, "obs_temp")
        assert result == "air_temperature"

    def test_fallback_to_lookup_table(self, sample_dataset):
        """Should fall back to lookup table if no attrs."""
        result = get_variable_label(sample_dataset, "no_attrs_var")
        # "no_attrs_var" not in lookup, gets basic formatting
        assert result == "No Attrs Var"

    def test_variable_not_in_dataset(self):
        """Variables not in dataset should use lookup table."""
        empty_ds = xr.Dataset()
        result = get_variable_label(empty_ds, "obs_pm25")
        assert result == "Observed PM₂.₅"

    def test_include_prefix_parameter(self):
        """include_prefix should control prefix for lookup table fallback."""
        empty_ds = xr.Dataset()
        with_prefix = get_variable_label(empty_ds, "obs_pm25", include_prefix=True)
        without_prefix = get_variable_label(empty_ds, "obs_pm25", include_prefix=False)
        assert with_prefix == "Observed PM₂.₅"
        assert without_prefix == "PM₂.₅"

    def test_none_display_name_attr_ignored(self):
        """display_name attr set to None should be ignored."""
        ds = xr.Dataset({
            "obs_pm25": xr.DataArray(
                [1, 2, 3],
                attrs={"display_name": None, "units": "μg/m³"}
            )
        })
        result = get_variable_label(ds, "obs_pm25")
        # Should fall back to lookup table, not return "None"
        assert result == "Observed PM₂.₅"
        assert result != "None"


class TestGetVariableUnits:
    """Tests for get_variable_units function."""

    def test_units_from_attrs(self):
        """Should return units from variable attrs."""
        ds = xr.Dataset({
            "pm25": xr.DataArray([1, 2, 3], attrs={"units": "μg/m³"})
        })
        assert get_variable_units(ds, "pm25") == "μg/m³"

    def test_no_units_attr(self):
        """Should return None if no units attr."""
        ds = xr.Dataset({
            "pm25": xr.DataArray([1, 2, 3])
        })
        assert get_variable_units(ds, "pm25") is None

    def test_variable_not_in_dataset(self):
        """Should return None if variable not in dataset."""
        ds = xr.Dataset()
        assert get_variable_units(ds, "pm25") is None
