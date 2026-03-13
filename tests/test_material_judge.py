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
