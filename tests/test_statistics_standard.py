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
