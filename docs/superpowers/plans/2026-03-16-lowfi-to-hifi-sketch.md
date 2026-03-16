# Low-Fi to High-Fi Sketch Workflow Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a single-pass Gemini workflow that transforms rough hand-drawn shoe sketches into high-fidelity pencil sketches, optionally using a 3D volume image for shape context.

**Architecture:** A `workflows/lowfi_to_hifi/` directory with two prompt YAML files and a `pipeline.py` that dynamically selects the prompt based on whether a volume image is provided. Single `AgentStep` wrapping Gemini Flash image edit — no loops or judges.

**Tech Stack:** Python 3.12+, Gemini Flash Image Edit API, Casadei pipeline framework (Agent, AgentStep, Pipeline), PyYAML, PIL.

**Spec:** `docs/superpowers/specs/2026-03-16-lowfi-to-hifi-sketch-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `workflows/lowfi_to_hifi/prompt_sketch_only.yaml` | Prompt for sketch-only mode |
| Create | `workflows/lowfi_to_hifi/prompt_with_volume.yaml` | Prompt for volume+sketch mode |
| Create | `workflows/lowfi_to_hifi/pipeline.py` | `build_pipeline()`, `save_results()`, prompt loading |
| Create | `tests/test_lowfi_to_hifi_pipeline.py` | Unit tests for pipeline construction and save_results |

---

## Chunk 1: Prompt YAML Files and Pipeline Module

### Task 1: Create prompt YAML files

**Files:**
- Create: `workflows/lowfi_to_hifi/prompt_sketch_only.yaml`
- Create: `workflows/lowfi_to_hifi/prompt_with_volume.yaml`

- [ ] **Step 1: Create `prompt_sketch_only.yaml`**

```yaml
prompt_template: |
  You are given a rough hand-drawn sketch of a shoe design. Study every line,
  strap, opening, and structural element carefully. Generate a high-fidelity
  professional pencil sketch of a single shoe that faithfully renders the design.

  Output style: clean black-and-white pencil rendering on off-white paper,
  professional fashion sketch quality, single shoe, fine line work with shading
  for depth.
  $extra_specs

default_params:
  temperature: 0.8

negative_prompt: ""
```

- [ ] **Step 2: Create `prompt_with_volume.yaml`**

```yaml
prompt_template: |
  You are given two images: a 3D volume showing the shoe's shape and proportions,
  and a rough hand-drawn sketch showing the design. Study the volume to understand
  the depth, curves, and structure. Study the sketch to understand every design
  detail — lines, straps, openings, and structural elements. Generate a
  high-fidelity professional pencil sketch of a single shoe that combines the 3D
  understanding from the volume with the design from the sketch.

  Output style: clean black-and-white pencil rendering on off-white paper,
  professional fashion sketch quality, single shoe, fine line work with shading
  for depth.
  $extra_specs

default_params:
  temperature: 0.8

negative_prompt: ""
```

- [ ] **Step 3: Commit prompt files**

```bash
git add workflows/lowfi_to_hifi/prompt_sketch_only.yaml workflows/lowfi_to_hifi/prompt_with_volume.yaml
git commit -m "feat: add prompt YAML files for lowfi-to-hifi sketch workflow"
```

---

### Task 2: Write failing tests for `build_pipeline`

**Files:**
- Create: `tests/test_lowfi_to_hifi_pipeline.py`

- [ ] **Step 1: Write tests for `build_pipeline`**

Tests cover: sketch-only mode, volume+sketch mode, extra_specs passthrough, custom temperature.

```python
"""Tests for workflows/lowfi_to_hifi/pipeline.py."""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia
from casadei.pipeline import AgentStep, Pipeline


# Import path for the module under test
import sys
from pathlib import Path

_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows" / "lowfi_to_hifi"
if str(_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW_DIR))


class TestBuildPipeline:
    def test_sketch_only_mode(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "lowfi_to_hifi"
        assert len(pipeline.steps) == 1

        step = pipeline.steps[0]
        assert isinstance(step, AgentStep)
        assert step.input_map == {"image": "sketch"}
        assert step.output_map == {"image": "image"}

    def test_volume_mode(self):
        from pipeline import build_pipeline

        spec = {"volume": True}
        pipeline, agent = build_pipeline(spec)

        step = pipeline.steps[0]
        assert step.input_map == {"image": "sketch", "volume": "volume"}

    def test_extra_specs_passed(self):
        from pipeline import build_pipeline

        spec = {"extra": {"style": "minimal", "brand": "Casadei"}}
        pipeline, agent = build_pipeline(spec)

        step = pipeline.steps[0]
        assert "extra_specs" in step.template_kwargs
        assert "Style: minimal" in step.template_kwargs["extra_specs"]
        assert "Brand: Casadei" in step.template_kwargs["extra_specs"]

    def test_custom_temperature(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec, temperature=1.0)

        assert agent.config.params["temperature"] == 1.0

    def test_default_temperature(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec)

        assert agent.config.params["temperature"] == 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lowfi_to_hifi_pipeline.py -v`
Expected: FAIL — `pipeline` module not found / `build_pipeline` not importable.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_lowfi_to_hifi_pipeline.py
git commit -m "test: add failing tests for lowfi-to-hifi build_pipeline"
```

---

### Task 3: Implement `pipeline.py` — `build_pipeline` and prompt loading

**Files:**
- Create: `workflows/lowfi_to_hifi/pipeline.py`

- [ ] **Step 1: Implement `pipeline.py`**

```python
"""Pipeline module for lowfi_to_hifi workflow.

Provides build_pipeline and save_results for transforming rough hand-drawn
sketches into high-fidelity pencil sketches, optionally with a 3D volume image.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, Pipeline
from casadei.providers.gemini_pricing import format_usage_summary

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WORKFLOW_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Prompt config loading
# ---------------------------------------------------------------------------

_PROMPT_SKETCH_ONLY_PATH = _WORKFLOW_DIR / "prompt_sketch_only.yaml"
_PROMPT_WITH_VOLUME_PATH = _WORKFLOW_DIR / "prompt_with_volume.yaml"

_cached_sketch_only: dict | None = None
_cached_with_volume: dict | None = None


def load_prompt_config(with_volume: bool) -> dict:
    """Load and cache the appropriate prompt config."""
    global _cached_sketch_only, _cached_with_volume
    if with_volume:
        if _cached_with_volume is None:
            with open(_PROMPT_WITH_VOLUME_PATH) as f:
                _cached_with_volume = yaml.safe_load(f)
        return _cached_with_volume
    else:
        if _cached_sketch_only is None:
            with open(_PROMPT_SKETCH_ONLY_PATH) as f:
                _cached_sketch_only = yaml.safe_load(f)
        return _cached_sketch_only


# ---------------------------------------------------------------------------
# Spec utilities
# ---------------------------------------------------------------------------

def build_extra_specs_text(extra: dict) -> str:
    """Build a formatted string of extra spec key-value pairs."""
    if not extra:
        return ""
    return "\n".join(f"- {k.capitalize()}: {v}" for k, v in extra.items())


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(
    spec: dict,
    temperature: float = 0.8,
) -> tuple[Pipeline, Agent]:
    """Build the lowfi-to-hifi sketch generation pipeline.

    Args:
        spec: Dict optionally containing 'volume' (truthy = volume image provided)
              and 'extra' (dict of additional spec key-value pairs).
        temperature: Generation temperature for Gemini.

    Returns:
        Tuple of (Pipeline, Agent).
    """
    has_volume = bool(spec.get("volume"))
    prompt_config = load_prompt_config(with_volume=has_volume)

    agent = Agent(AgentConfig(
        name="gemini_lowfi_to_hifi",
        model="gemini_flash_image_edit",
        description="Gemini Flash image edit for lowfi-to-hifi sketch generation",
        prompt_template=prompt_config["prompt_template"],
        negative_prompt=prompt_config.get("negative_prompt", ""),
        params={"temperature": temperature},
    ))

    input_map: dict[str, str] = {"image": "sketch"}
    if has_volume:
        input_map["volume"] = "volume"

    extra_specs_text = build_extra_specs_text(spec.get("extra", {}))

    step = AgentStep(
        name="generate_hifi_sketch",
        agent=agent,
        input_map=input_map,
        output_map={"image": "image"},
        template_kwargs={
            "extra_specs": extra_specs_text,
        },
    )

    return Pipeline(name="lowfi_to_hifi", steps=[step]), agent


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    run_dir: Path,
    result_image: ImageMedia | None,
    spec: dict,
    total_elapsed: float,
    token_records: list[dict] | None = None,
) -> None:
    """Save generation results to disk.

    Args:
        run_dir: Directory to write results into (created if needed).
        result_image: The generated high-fidelity sketch image.
        spec: The spec dict used for generation.
        total_elapsed: Total wall-clock time in seconds.
        token_records: Optional list of token usage dicts for cost tracking.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    has_volume = bool(spec.get("volume"))
    mode = "with_volume" if has_volume else "sketch_only"

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "model": "gemini_flash_image_edit",
        "mode": mode,
        "spec": {k: v for k, v in spec.items() if k != "volume"},
    }

    if result_image is not None and isinstance(result_image, ImageMedia):
        result_image.image.save(run_dir / "result.png")
        results_data["result"] = "result.png"

    if token_records:
        usage_summary = format_usage_summary(token_records)
        results_data["token_usage"] = {
            "records": token_records,
            "summary": usage_summary,
        }

    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    lines = [
        "Low-Fi to High-Fi Sketch — Gemini",
        "=" * 50,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Model: gemini_flash_image_edit",
        f"Mode: {mode}",
    ]
    if spec.get("extra"):
        for k, v in spec["extra"].items():
            lines.append(f"  {k.capitalize()}: {v}")

    if token_records:
        usage_summary = format_usage_summary(token_records)
        lines.append("")
        lines.append("Token Usage & Pricing")
        lines.append("-" * 40)
        gt = usage_summary["grand_total"]
        lines.append(
            f"  Total: {gt['total_tokens']:,} tokens  |  "
            f"${gt['cost_usd']:.6f}  |  {gt['calls']} API calls"
        )

    lines.append(f"Output: {run_dir}")
    summary = "\n".join(lines)
    print(f"\n{summary}")
    (run_dir / "summary.txt").write_text(summary)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_lowfi_to_hifi_pipeline.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add workflows/lowfi_to_hifi/pipeline.py
git commit -m "feat: implement lowfi-to-hifi pipeline module"
```

---

### Task 4: Write and run tests for `save_results`

**Files:**
- Modify: `tests/test_lowfi_to_hifi_pipeline.py`

- [ ] **Step 1: Add `save_results` tests**

Append to `tests/test_lowfi_to_hifi_pipeline.py`:

```python
class TestSaveResults:
    def test_saves_result_image(self, tmp_path):
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run1",
            result_image=img,
            spec={},
            total_elapsed=2.5,
        )
        assert (tmp_path / "run1" / "result.png").exists()
        assert (tmp_path / "run1" / "results.json").exists()
        assert (tmp_path / "run1" / "summary.txt").exists()

    def test_metadata_contains_mode_sketch_only(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run2",
            result_image=img,
            spec={},
            total_elapsed=1.0,
        )
        data = json.loads((tmp_path / "run2" / "results.json").read_text())
        assert data["mode"] == "sketch_only"

    def test_metadata_contains_mode_with_volume(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run3",
            result_image=img,
            spec={"volume": True},
            total_elapsed=1.0,
        )
        data = json.loads((tmp_path / "run3" / "results.json").read_text())
        assert data["mode"] == "with_volume"

    def test_token_records_saved(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        records = [{"model": "gemini-3.1-flash-image-preview", "input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "cached_tokens": 0, "total_tokens": 150}]
        save_results(
            run_dir=tmp_path / "run4",
            result_image=img,
            spec={},
            total_elapsed=1.0,
            token_records=records,
        )
        data = json.loads((tmp_path / "run4" / "results.json").read_text())
        assert "token_usage" in data
        assert data["token_usage"]["records"] == records

    def test_none_image_no_crash(self, tmp_path):
        from pipeline import save_results

        save_results(
            run_dir=tmp_path / "run5",
            result_image=None,
            spec={},
            total_elapsed=0.5,
        )
        assert (tmp_path / "run5" / "results.json").exists()
        assert not (tmp_path / "run5" / "result.png").exists()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_lowfi_to_hifi_pipeline.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_lowfi_to_hifi_pipeline.py
git commit -m "test: add save_results tests for lowfi-to-hifi workflow"
```
