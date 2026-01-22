"""
Tests for the statistics standard schema validation.
"""

import pytest

from pmd_beamphysics.standards.statistics import (
    YAML_PATH,
    load_standard,
    validate_standard,
    get_statistic,
    get_category,
)


class TestStatisticsStandard:
    """Tests for the statistics standard YAML schema."""

    @pytest.fixture
    def standard(self):
        """Load the statistics standard."""
        return load_standard()

    def test_yaml_file_exists(self):
        """Test that the YAML file exists."""
        assert YAML_PATH.exists(), f"Statistics YAML not found at {YAML_PATH}"

    def test_yaml_loads_successfully(self, standard):
        """Test that the YAML file loads without errors."""
        assert standard is not None
        assert isinstance(standard, dict)

    def test_schema_version_exists(self, standard):
        """Test that schema version is defined."""
        assert "schema_version" in standard
        assert isinstance(standard["schema_version"], str)

    def test_categories_exist(self, standard):
        """Test that categories are defined."""
        assert "categories" in standard
        assert isinstance(standard["categories"], list)
        assert len(standard["categories"]) > 0

    def test_statistics_exist(self, standard):
        """Test that statistics are defined."""
        assert "statistics" in standard
        assert isinstance(standard["statistics"], list)
        assert len(standard["statistics"]) > 0

    def test_schema_validation_passes(self, standard):
        """Test that the schema validation passes with no errors."""
        errors = validate_standard(standard)
        assert errors == [], f"Schema validation errors: {errors}"

    def test_all_categories_have_required_fields(self, standard):
        """Test that all categories have required fields."""
        required_fields = ["id", "name", "description"]
        for cat in standard["categories"]:
            for field in required_fields:
                assert field in cat, f"Category missing field '{field}': {cat}"

    def test_all_statistics_have_required_fields(self, standard):
        """Test that all statistics have required fields."""
        required_fields = [
            "label",
            "mathlabel",
            "units",
            "description",
            "reference",
            "category",
        ]
        for stat in standard["statistics"]:
            for field in required_fields:
                assert (
                    field in stat
                ), f"Statistic '{stat.get('label', '?')}' missing field '{field}'"

    def test_all_statistics_reference_valid_categories(self, standard):
        """Test that all statistics reference existing categories."""
        category_ids = {cat["id"] for cat in standard["categories"]}
        for stat in standard["statistics"]:
            cat_id = stat.get("category")
            assert (
                cat_id in category_ids
            ), f"Statistic '{stat.get('label')}' references unknown category '{cat_id}'"

    def test_no_duplicate_labels(self, standard):
        """Test that there are no duplicate statistic labels."""
        labels = [stat["label"] for stat in standard["statistics"]]
        duplicates = [label for label in labels if labels.count(label) > 1]
        assert len(duplicates) == 0, f"Duplicate labels found: {set(duplicates)}"

    def test_no_duplicate_category_ids(self, standard):
        """Test that there are no duplicate category IDs."""
        ids = [cat["id"] for cat in standard["categories"]]
        duplicates = [id_ for id_ in ids if ids.count(id_) > 1]
        assert len(duplicates) == 0, f"Duplicate category IDs found: {set(duplicates)}"

    def test_get_statistic_found(self, standard):
        """Test that get_statistic returns a statistic when found."""
        # Get any label from the standard
        label = standard["statistics"][0]["label"]
        stat = get_statistic(label, standard)
        assert stat is not None
        assert stat["label"] == label

    def test_get_statistic_not_found(self, standard):
        """Test that get_statistic returns None when not found."""
        stat = get_statistic("nonexistent_label_xyz", standard)
        assert stat is None

    def test_get_category_found(self, standard):
        """Test that get_category returns a category when found."""
        cat_id = standard["categories"][0]["id"]
        cat = get_category(cat_id, standard)
        assert cat is not None
        assert cat["id"] == cat_id

    def test_get_category_not_found(self, standard):
        """Test that get_category returns None when not found."""
        cat = get_category("nonexistent_category_xyz", standard)
        assert cat is None

    def test_known_statistics_exist(self, standard):
        """Test that key statistics are defined."""
        expected_labels = [
            "x",
            "y",
            "z",
            "px",
            "py",
            "pz",
            "t",
            "energy",
            "kinetic_energy",
            "gamma",
            "beta",
            "norm_emit_x",
            "norm_emit_y",
            "charge",
        ]
        defined_labels = {stat["label"] for stat in standard["statistics"]}
        for label in expected_labels:
            assert label in defined_labels, f"Expected statistic '{label}' not defined"


class TestComputedStatistics:
    """Tests for computed statistics generation."""

    @pytest.fixture
    def computed(self):
        """Load computed statistics."""
        from pmd_beamphysics.standards.statistics import load_computed_statistics

        return load_computed_statistics()

    def test_computed_statistics_generated(self, computed):
        """Test that computed statistics are generated."""
        assert "statistics" in computed
        assert len(computed["statistics"]) > 0

    def test_computed_statistics_count(self, computed):
        """Test that the expected number of statistics are generated."""
        # 6 operators * 31 keys = 186 operator stats
        # 31 * 31 = 961 covariance stats
        # Total = 1147
        assert len(computed["statistics"]) == 1147

    def test_computed_categories_exist(self, computed):
        """Test that computed statistics have categories."""
        assert "categories" in computed
        cat_ids = {cat["id"] for cat in computed["categories"]}
        assert "computed_operators" in cat_ids
        assert "computed_covariance" in cat_ids

    def test_operator_statistics_format(self, computed):
        """Test that operator statistics have correct format."""
        from pmd_beamphysics.standards.statistics import get_computed_statistic

        stat = get_computed_statistic("sigma_x")
        assert stat is not None
        assert stat["label"] == "sigma_x"
        assert stat["units"] == "m"
        assert stat["category"] == "computed_operators"
        assert stat["base_statistic"] == "x"
        assert stat["operator"] == "sigma"
        assert "mathlabel" in stat
        assert "description" in stat

    def test_covariance_statistics_format(self, computed):
        """Test that covariance statistics have correct format."""
        from pmd_beamphysics.standards.statistics import get_computed_statistic

        stat = get_computed_statistic("cov_x__px")
        assert stat is not None
        assert stat["label"] == "cov_x__px"
        assert stat["units"] == "m*eV/c"
        assert stat["category"] == "computed_covariance"
        assert stat["base_statistics"] == ["x", "px"]
        assert "mathlabel" in stat
        assert "description" in stat

    def test_get_computed_statistic_not_found(self):
        """Test that get_computed_statistic returns None for unknown labels."""
        from pmd_beamphysics.standards.statistics import get_computed_statistic

        stat = get_computed_statistic("nonexistent_computed_stat")
        assert stat is None

    def test_computed_is_cached(self):
        """Test that computed statistics are cached (same object returned)."""
        from pmd_beamphysics.standards.statistics import load_computed_statistics

        computed1 = load_computed_statistics()
        computed2 = load_computed_statistics()
        assert computed1 is computed2  # Same cached object


class TestUnitsParsingWithPmdUnit:
    """Tests that all units strings can be parsed by pmd_unit."""

    def test_base_statistics_units_parseable(self):
        """Test that all base statistics units can be parsed by pmd_unit."""
        from pmd_beamphysics.standards.statistics import load_standard
        from pmd_beamphysics.units import pmd_unit

        standard = load_standard()
        failed = []

        for stat in standard["statistics"]:
            units_str = stat.get("units", "")
            if not units_str:
                continue
            try:
                pmd_unit(units_str)
            except Exception as e:
                failed.append((stat["label"], units_str, str(e)))

        assert failed == [], f"Failed to parse units: {failed}"

    def test_computed_statistics_units_parseable(self):
        """Test that all computed statistics units can be parsed by pmd_unit."""
        from pmd_beamphysics.standards.statistics import load_computed_statistics
        from pmd_beamphysics.units import pmd_unit

        computed = load_computed_statistics()
        failed = []

        for stat in computed["statistics"]:
            units_str = stat.get("units", "")
            if not units_str:
                continue
            try:
                pmd_unit(units_str)
            except Exception as e:
                failed.append((stat["label"], units_str, str(e)))

        assert failed == [], f"Failed to parse units: {failed}"
