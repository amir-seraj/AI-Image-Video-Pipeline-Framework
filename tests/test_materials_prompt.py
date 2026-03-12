"""Tests for build_materials_prompt in sketch_to_shoe_gemini pipeline."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "workflows" / "shared"))
sys.path.insert(0, str(_project / "workflows" / "sketch_to_shoe" / "scripts"))

from workflows.sketch_to_shoe_gemini.pipeline import build_materials_prompt


def test_single_material_no_placement():
    materials = [{"name": None, "placement": None, "note": None, "is_color": False}]
    names = ["Material 1"]
    result = build_materials_prompt(materials, names)
    assert "material" in result.lower()
    assert "reference image" in result.lower()
    assert "Material 1" not in result  # single mode doesn't use name


def test_single_color_no_placement():
    materials = [{"name": None, "placement": None, "note": None, "is_color": True}]
    names = ["Color 1"]
    result = build_materials_prompt(materials, names)
    assert "color" in result.lower()


def test_single_material_with_placement():
    materials = [{"name": "Suede", "placement": "toe", "note": None, "is_color": False}]
    names = ["Suede"]
    result = build_materials_prompt(materials, names)
    assert "toe" in result


def test_single_material_with_note():
    materials = [{"name": None, "placement": None, "note": "matte finish in daylight", "is_color": False}]
    names = ["Material 1"]
    result = build_materials_prompt(materials, names)
    assert "matte finish in daylight" in result


def test_multiple_materials():
    materials = [
        {"name": "Suede A", "placement": "toe", "note": "soft nubuck", "is_color": False},
        {"name": None, "placement": "heel", "note": None, "is_color": True},
    ]
    names = ["Suede A", "Color 1"]
    result = build_materials_prompt(materials, names)
    assert '"Suede A"' in result
    assert '"Color 1"' in result
    assert "toe" in result
    assert "heel" in result
    assert "soft nubuck" in result
    assert "material" in result.lower()
    assert "color" in result.lower()


def test_multiple_all_colors():
    materials = [
        {"name": None, "placement": "upper", "note": None, "is_color": True},
        {"name": None, "placement": "sole", "note": None, "is_color": True},
    ]
    names = ["Color 1", "Color 2"]
    result = build_materials_prompt(materials, names)
    assert '"Color 1"' in result
    assert '"Color 2"' in result


from PIL import Image as PILImage
from workflows.sketch_to_shoe_gemini.pipeline import build_pipeline


def _solid(color=(128, 128, 128), size=(200, 200)):
    return PILImage.new("RGB", size, color)


def test_build_pipeline_materials_mode_returns_pipeline():
    """build_pipeline with spec['materials'] should use materials prompt template."""
    from casadei import ImageMedia
    spec = {
        "material": "ignored",
        "camera_angle": "3/4",
        "extra": {},
        "materials": [
            {"name": "Test Mat", "image": _solid(), "placement": "toe", "note": None, "is_color": False},
        ],
    }
    import unittest.mock as mock
    vlm = mock.MagicMock()
    pipeline, agent, sessions, grid_img = build_pipeline(spec, vlm, foot="pair", temperature=0.8)
    assert pipeline.name == "sketch_to_shoe_gemini"
    assert grid_img is not None
    # The agent's prompt template should contain materials_instructions
    assert "materials" in agent.config.prompt_template.lower() or "reference image" in agent.config.prompt_template.lower()


def test_build_pipeline_text_mode_unchanged():
    """build_pipeline without spec['materials'] should use text prompt template."""
    import unittest.mock as mock
    spec = {
        "material": "red leather",
        "camera_angle": "3/4",
        "extra": {},
    }
    vlm = mock.MagicMock()
    pipeline, agent, sessions, grid_img = build_pipeline(spec, vlm, foot="pair")
    assert grid_img is None
    assert "$material" in agent.config.prompt_template
