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
