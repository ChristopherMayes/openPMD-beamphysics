import pytest

import beamphysics

LAZY_ATTRIBUTES: list[str] = list(beamphysics._LAZY_IMPORTS.keys())


@pytest.mark.parametrize("attribute_name", LAZY_ATTRIBUTES, ids=LAZY_ATTRIBUTES)
def test_lazy_imports_resolve_correctly(attribute_name: str) -> None:
    resolved_object = getattr(beamphysics, attribute_name)
    assert resolved_object is not None
    assert attribute_name in beamphysics.__dir__()


def test_lazy_import_invalid_attribute_raises_error() -> None:
    with pytest.raises(AttributeError) as exc_info:
        getattr(beamphysics, "bad_attr")

    assert "bad_attr" in str(exc_info.value)
