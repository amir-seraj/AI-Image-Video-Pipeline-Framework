"""Tests for the material judge."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "workflows" / "shared"))
sys.path.insert(0, str(_project / "workflows" / "sketch_to_shoe" / "scripts"))

import unittest.mock as mock
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from casadei.media import ImageMedia, TextMedia

from judge import _MaterialJudgeResult, _SpecJudgeResult, make_material_judge


def test_material_judge_result_valid():
    result = _MaterialJudgeResult(
        observation="Black patent leather with glossy finish",
        score=4,
        repair="none",
    )
    assert result.score == 4
    assert result.repair == "none"


def test_material_judge_result_score_bounds():
    import pytest
    with pytest.raises(Exception):
        _MaterialJudgeResult(observation="test", score=0, repair="none")
    with pytest.raises(Exception):
        _MaterialJudgeResult(observation="test", score=6, repair="none")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate():
    """Create a dummy candidate ImageMedia."""
    return ImageMedia(image=PILImage.new("RGB", (100, 100), (0, 0, 0)))


def _mock_vlm_text_mode(raw_json):
    """Return a mock VLM session that returns the given JSON string."""
    session = MagicMock()
    model = MagicMock()
    session.acquire.return_value = model
    model.run.return_value = MagicMock(
        items={"text": TextMedia(text=raw_json)}
    )
    return session, model


# ---------------------------------------------------------------------------
# Text mode tests
# ---------------------------------------------------------------------------

def test_text_mode_accept():
    """Text mode: score >= threshold -> accept."""
    result_json = _MaterialJudgeResult(
        observation="Red leather upper with matte finish",
        score=4,
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    assert feedback == "none"
    assert "_judge_metadata_material" in ctx


def test_text_mode_reject():
    """Text mode: score below threshold -> reject with repair."""
    result_json = _MaterialJudgeResult(
        observation="Blue suede instead of red leather",
        score=2,
        repair="Change material from blue suede to red leather",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="moderate",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "red leather" in feedback


def test_text_mode_no_grid_image():
    """Text mode is selected when grid_image is None."""
    result_json = _MaterialJudgeResult(
        observation="Black patent leather", score=5, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="black patent leather",
        grid_image=None,
    )
    ctx = {"image": _make_candidate()}
    judge(ctx)
    # Bundle should NOT contain material_ref key
    call_args = model.run.call_args
    bundle = call_args[0][0]
    assert "material_ref" not in bundle.items


# ---------------------------------------------------------------------------
# Image mode tests
# ---------------------------------------------------------------------------

def test_single_image_mode_accept():
    """Image mode with single material: grid_image provided, no material_names."""
    result_json = _MaterialJudgeResult(
        observation="Material matches the swatch — brown leather",
        score=5,
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (500, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply the material shown in the reference image to the shoe.",
        grid_image=grid,
        material_names=None,
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    # Bundle should contain material_ref
    call_args = model.run.call_args
    bundle = call_args[0][0]
    assert "material_ref" in bundle.items


def test_single_image_mode_with_one_name():
    """Image mode with material_names=["Suede A"] -> still single mode (len <= 1)."""
    result_json = _MaterialJudgeResult(
        observation="Suede matches", score=4, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (500, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply the material shown in the reference image to the shoe.",
        grid_image=grid,
        material_names=["Suede A"],
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, _ = judge(ctx)
    assert accepted is True
    # Metadata should have single "material" key, not per-name keys
    meta = ctx["_judge_metadata_material"]
    assert "material" in meta["scores"]


def test_multi_image_mode_accept():
    """Multi-material image mode: grid_image + len(material_names) > 1."""
    result_json = _SpecJudgeResult(
        observations={
            "Suede_A_match": "Brown suede matches swatch",
            "Suede_A_placement": "Applied to upper correctly",
            "Color_1_match": "Red matches swatch",
            "Color_1_placement": "Applied to heel correctly",
        },
        scores={
            "Suede_A_match": 5,
            "Suede_A_placement": 4,
            "Color_1_match": 5,
            "Color_1_placement": 5,
        },
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (1000, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply Suede A to upper, Color 1 to heel.",
        grid_image=grid,
        material_names=["Suede A", "Color 1"],
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    meta = ctx["_judge_metadata_material"]
    assert "Suede_A_match" in meta["scores"]
    assert "Color_1_placement" in meta["scores"]


def test_multi_image_mode_reject():
    """Multi-material: one score below min_floor -> reject."""
    result_json = _SpecJudgeResult(
        observations={
            "Suede_A_match": "Wrong material", "Suede_A_placement": "Wrong spot",
            "Color_1_match": "Good", "Color_1_placement": "Good",
        },
        scores={
            "Suede_A_match": 1, "Suede_A_placement": 2,
            "Color_1_match": 5, "Color_1_placement": 5,
        },
        repair="Suede A is completely wrong — use brown suede on the upper.",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (1000, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply Suede A to upper, Color 1 to heel.",
        grid_image=grid,
        material_names=["Suede A", "Color 1"],
        tolerance="moderate",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "Suede" in feedback


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_hash_dedup_auto_accepts():
    """Same image called twice -> second call auto-accepts without VLM call."""
    result_json = _MaterialJudgeResult(
        observation="ok", score=4, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",
    )
    candidate = _make_candidate()
    ctx1 = {"image": candidate}
    judge(ctx1)
    assert model.run.call_count == 1

    # Same image again
    ctx2 = {"image": candidate}
    accepted, feedback = judge(ctx2)
    assert accepted is True
    assert feedback == "none"
    assert model.run.call_count == 1  # No second VLM call


def test_parse_failure_returns_rejection():
    """VLM returns garbage -> graceful rejection, not crash."""
    session, model = _mock_vlm_text_mode("not valid json at all")
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "parse error" in feedback.lower()


def test_empty_material_spec_raises():
    """material_spec="" should raise ValueError."""
    import pytest
    session = MagicMock()
    with pytest.raises(ValueError, match="non-empty"):
        make_material_judge(session=session, material_spec="")


# ---------------------------------------------------------------------------
# best_fn scoring tests
# ---------------------------------------------------------------------------

from judge import make_best_fn
from casadei.loop import LoopIteration


def test_best_fn_includes_material_avg():
    """best_fn should factor material_avg into candidate ranking."""
    session = MagicMock()
    best_fn = make_best_fn(session=session, output_key="image")

    img_good_camera = _make_candidate()
    img_good_material = _make_candidate()
    # Modify the second image so it has a different identity
    img_good_material.image.putpixel((0, 0), (255, 0, 0))

    history = [
        LoopIteration(
            index=0,
            accepted=False,
            feedback="camera bad",
            duration_ms=100.0,
            outputs={"image": img_good_camera},
            metadata={"sketch_avg": None, "spec_avg": 5.0, "material_avg": 1.0},
        ),
        LoopIteration(
            index=1,
            accepted=False,
            feedback="camera ok-ish",
            duration_ms=100.0,
            outputs={"image": img_good_material},
            metadata={"sketch_avg": None, "spec_avg": 3.0, "material_avg": 5.0},
        ),
    ]
    result = best_fn(history, {})
    # iter 0: non-zero components = [5.0, 1.0] -> avg = 3.0
    # iter 1: non-zero components = [3.0, 5.0] -> avg = 4.0
    # iter 1 should win
    assert result.get("best_selection_index") == 2  # 1-indexed


def test_best_fn_backward_compatible_no_material():
    """best_fn still works when material_avg is absent (older pipelines)."""
    session = MagicMock()
    best_fn = make_best_fn(session=session, output_key="image")

    img = _make_candidate()
    history = [
        LoopIteration(
            index=0,
            accepted=False,
            feedback="rejected",
            duration_ms=100.0,
            outputs={"image": img},
            metadata={"sketch_avg": None, "spec_avg": 4.0},
        ),
    ]
    result = best_fn(history, {})
    assert result.get("best_selection_index") == 1
