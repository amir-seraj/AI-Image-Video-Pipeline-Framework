"""Tests for build_material_grid in workflows/shared/image_utils.py."""
import sys
from pathlib import Path
from PIL import Image as PILImage

# Add shared utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows" / "shared"))
from image_utils import build_material_grid


def _solid(color, size=(200, 300)):
    """Create a solid color test image."""
    return PILImage.new("RGB", size, color)


def test_single_material_grid():
    materials = [{"name": "Leather", "image": _solid((139, 69, 19)), "is_color": False}]
    grid, names = build_material_grid(materials)
    assert names == ["Leather"]
    assert grid.width == grid.height  # square output
    assert grid.width >= 500  # at least one tile wide


def test_auto_naming_material_and_color():
    materials = [
        {"name": None, "image": _solid((200, 0, 0)), "is_color": False},
        {"name": None, "image": _solid((0, 0, 255)), "is_color": True},
        {"name": None, "image": _solid((0, 128, 0)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    assert names == ["Material 1", "Color 1", "Material 2"]
    assert grid.width == grid.height  # square


def test_custom_name_preserved():
    materials = [
        {"name": "Suede A", "image": _solid((100, 100, 100)), "is_color": False},
        {"name": None, "image": _solid((50, 50, 50)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    assert names == ["Suede A", "Material 1"]


def test_two_materials_grid_is_2x1():
    materials = [
        {"name": "M1", "image": _solid((255, 0, 0)), "is_color": False},
        {"name": "M2", "image": _solid((0, 255, 0)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    # 2 cols x 1 row = 1000 x 530, padded to square = 1000 x 1000
    assert grid.width == grid.height


def test_resolution_cap():
    # 5x5 grid = 25 materials, 2500x2650 raw → should be downscaled
    materials = [
        {"name": f"M{i}", "image": _solid((i * 10, i * 10, i * 10)), "is_color": False}
        for i in range(25)
    ]
    grid, names = build_material_grid(materials)
    assert grid.width <= 2048
    assert grid.height <= 2048
    assert len(names) == 25


def test_empty_raises():
    try:
        build_material_grid([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
