"""Tests for the material judge."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "workflows" / "shared"))
sys.path.insert(0, str(_project / "workflows" / "sketch_to_shoe" / "scripts"))

from judge import _MaterialJudgeResult


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
