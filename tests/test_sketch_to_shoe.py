"""Unit tests for sketch-to-shoe — pure functions, no GPU required."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from PIL import Image as PILImage
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows" / "sketch_to_shoe" / "scripts"))


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_parses_valid_scores(self):
        from judge import _parse_scores
        result = _parse_scores("SCORES: shape=4, proportions=3, toe_shape=5")
        assert result == {"shape": 4.0, "proportions": 3.0, "toe_shape": 5.0}

    def test_raises_on_missing_scores_line(self):
        from judge import _parse_scores
        with pytest.raises(ValueError, match="No 'SCORES:'"):
            _parse_scores("REPAIR: fix the toe")

    def test_raises_on_empty_scores(self):
        from judge import _parse_scores
        with pytest.raises(ValueError, match="Could not parse any scores"):
            _parse_scores("SCORES: ")

    def test_float_scores(self):
        from judge import _parse_scores
        result = _parse_scores("SCORES: color=4.5, material=3.0")
        assert result["color"] == 4.5


class TestParseRepair:
    def test_extracts_repair_text(self):
        from judge import _parse_repair
        result = _parse_repair("SCORES: shape=3\nREPAIR: The toe is too round.")
        assert result == "The toe is too round."

    def test_returns_full_text_when_no_repair_tag(self):
        from judge import _parse_repair
        result = _parse_repair("something without repair tag")
        assert result == "something without repair tag"


# ---------------------------------------------------------------------------
# Feature extraction fallback
# ---------------------------------------------------------------------------

class TestExtractSketchFeaturesFallback:
    def _make_session(self, response_text: str):
        session = MagicMock()
        model = MagicMock()
        model.run_streaming.side_effect = lambda bundle: iter([response_text])
        session.acquire.return_value = model
        return session

    def test_returns_fallback_on_invalid_json(self):
        from judge import extract_sketch_features, _FALLBACK_SKETCH_FEATURES
        from casadei.media import ImageMedia
        session = self._make_session("not json at all")
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == list(_FALLBACK_SKETCH_FEATURES)

    def test_returns_extracted_features_on_valid_json(self):
        from judge import extract_sketch_features
        from casadei.media import ImageMedia
        session = self._make_session('["toe shape", "heel height", "sole thickness"]')
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == ["toe shape", "heel height", "sole thickness"]

    def test_returns_fallback_on_too_few_features(self):
        from judge import extract_sketch_features, _FALLBACK_SKETCH_FEATURES
        from casadei.media import ImageMedia
        session = self._make_session('["only one"]')
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == list(_FALLBACK_SKETCH_FEATURES)


# ---------------------------------------------------------------------------
# Dual judge combiner
# ---------------------------------------------------------------------------

class TestMakeDualJudge:
    def _img(self):
        from casadei.media import ImageMedia
        return ImageMedia(image=PILImage.new("RGB", (64, 64)))

    def test_both_accept_returns_accepted(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (True, "Sketch OK."), lambda ctx: (True, "Spec OK."))
        accepted, feedback = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is True
        assert "[Sketch feedback]" in feedback
        assert "[Spec feedback]" in feedback

    def test_sketch_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (False, "Wrong toe."), lambda ctx: (True, "Spec OK."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_spec_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (True, "Sketch OK."), lambda ctx: (False, "Wrong material."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_both_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (False, "Bad sketch."), lambda ctx: (False, "Bad spec."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_both_judges_always_run(self):
        """Spec judge must run even when sketch judge rejects."""
        from judge import make_dual_judge
        spec_ran = [False]

        def spec_j(ctx):
            spec_ran[0] = True
            return True, "OK"

        dual = make_dual_judge(lambda ctx: (False, "Sketch bad."), spec_j)
        dual({"sketch": self._img(), "image": self._img()})
        assert spec_ran[0] is True

    def test_metadata_merged_into_context(self):
        from judge import make_dual_judge
        from casadei.media import ImageMedia

        def sketch_j(ctx):
            ctx["_judge_metadata_sketch"] = {"scores": {"shape": 4.0}, "avg_score": 4.0, "lowest_attr": "shape"}
            return True, "OK"

        def spec_j(ctx):
            ctx["_judge_metadata_spec"] = {"scores": {"material": 5.0}, "avg_score": 5.0, "lowest_attr": "material"}
            return True, "OK"

        dual = make_dual_judge(sketch_j, spec_j)
        ctx = {"sketch": self._img(), "image": self._img()}
        dual(ctx)
        assert "_judge_metadata" in ctx
        assert "sketch_scores" in ctx["_judge_metadata"]
        assert "spec_scores" in ctx["_judge_metadata"]
        # Temp keys must be cleaned up
        assert "_judge_metadata_sketch" not in ctx
        assert "_judge_metadata_spec" not in ctx

    def test_feedback_format(self):
        from judge import make_dual_judge
        dual = make_dual_judge(
            lambda ctx: (True, "heel is off"),
            lambda ctx: (True, "color is wrong"),
        )
        _, feedback = dual({"sketch": self._img(), "image": self._img()})
        assert feedback == "[Sketch feedback]: heel is off\n[Spec feedback]: color is wrong"


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

class TestBuildSketchGrid:
    def _sketch(self, w, h, color=(200, 200, 200)):
        return PILImage.new("RGB", (w, h), color)

    def test_single_square_sketch_unchanged(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 100)], spacing=0)
        assert result.size == (100, 100)

    def test_single_non_square_padded_to_square(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 60)], spacing=0)
        w, h = result.size
        assert w == h, f"Expected square, got {w}x{h}"

    def test_two_sketches_output_is_square(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 100), self._sketch(100, 100)], spacing=0)
        w, h = result.size
        assert w == h

    def test_four_sketches_form_2x2_grid(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(50, 50) for _ in range(4)], spacing=0)
        assert result.size == (100, 100)

    def test_padding_area_is_white(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        # non-square black image → padded to square with white border
        result = _build_sketch_grid([self._sketch(50, 80, (0, 0, 0))], spacing=0)
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_raises_on_empty_list(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        with pytest.raises(ValueError, match="No sketch images"):
            _build_sketch_grid([], spacing=0)


class TestParseSpecArgs:
    def test_parses_key_value_pairs(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["style=elegant", "note=chunky sole"])
        assert result == {"style": "elegant", "note": "chunky sole"}

    def test_empty_list_returns_empty_dict(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        assert _parse_spec_args([]) == {}

    def test_ignores_entries_without_equals(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["valid=yes", "no_equals", "also=good"])
        assert "valid" in result and "also" in result
        assert "no_equals" not in result

    def test_value_with_embedded_equals_preserved(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["desc=a=b"])
        assert result["desc"] == "a=b"


class TestBuildExtraSpecsText:
    def test_formats_as_bullet_lines(self):
        from run_sketch_to_shoe_loop import _build_extra_specs_text
        result = _build_extra_specs_text({"style": "elegant", "note": "chunky"})
        assert "- Style: elegant" in result
        assert "- Note: chunky" in result

    def test_empty_dict_returns_empty_string(self):
        from run_sketch_to_shoe_loop import _build_extra_specs_text
        assert _build_extra_specs_text({}) == ""
