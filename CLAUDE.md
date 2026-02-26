# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
pip install -e .              # Install in editable mode
pip install -e ".[dev]"       # Install with dev dependencies (pytest, pytest-cov)

pytest                        # Run all tests
pytest tests/test_api_app.py -v -s   # Run a single test file
pytest -k "test_name" -v -s          # Run a specific test by name
pytest --cov=src/casadei              # Run with coverage
```

The API server runs via uvicorn:
```bash
uvicorn casadei.api.app:app --reload
```

## Architecture

Casadei is an AI pipeline framework for image/video generation and editing, targeting NVIDIA Jetson (CUDA 13.0). It has three layers: a core library, a pipeline engine, and a REST API.

### Core Abstractions

- **Media types** (`media.py`) — `ImageMedia`, `TextMedia`, `VideoMedia`, `MediaBundle` (named dict of media items). These flow through pipelines as the universal data format.
- **Model system** (`models/`) — Abstract base classes (`ImageEditModel`, `ImageToVideoModel`, `VideoEditModel`, `ReferenceInpaintModel`) with a capability declaration system (`ModelCapability`, `MediaConstraint`). Models register themselves in `ModelRegistry` (`default_registry` singleton).
- **Providers** (`providers/`) — Concrete model implementations. Each model type has multiple quantization variants (full, FP8, INT8, NF4, GGUF). Provider `_base.py` has shared utilities for safetensors verification and step clamping.
- **Agent** (`agent.py`) — Wraps a model with YAML config containing prompt templates (using `$variable` syntax via `string.Template`), default params, and negative prompts. Handles model load/unload and CUDA memory cleanup.
- **Pipeline** (`pipeline.py`) — Sequential composition of `AgentStep`, `CodeStep`, and nested `PipelineStep`. Steps communicate through a shared context dict. `input_map`/`output_map` dicts rename media keys between pipeline context and step inputs/outputs.

### REST API (`api/`)

FastAPI application with:
- **Products/Sketches/Generations** — CRUD for the product catalog
- **Agents** — Global agents stored as YAML in `agents/` directory
- **Pipelines** — Workflows with local agents, stored in `workflows/{id}/`
- **Workbench** — Run agents/pipelines on uploaded images, returns job ID
- **Jobs** — Background execution in daemon threads via `JobManager`, with SSE streaming for progress
- **Storage** — `JsonStore` persists to `data/store.json`; files go to `data/uploads/` and `data/results/`

### Code Step Convention

Pipeline code steps reference a Python script + function. The function signature is:
```python
def process(context: dict[str, Media]) -> dict[str, Media]:
```

### Agent YAML Format

```yaml
name: agent_name
model: registered_model_name
description: What the agent does
prompt_template: "Do $action to the image"
negative_prompt: "low quality, blurry"
params:
  num_inference_steps: 40
```

## Key Patterns

- Models declare capabilities upfront (accepted/produced media types with constraints) for validation before execution
- Pipelines use context-based data flow — steps read from and write to a shared dict, with key remapping via input/output maps
- Global agents live in `agents/*.yaml`; pipeline-local agents live in `workflows/{pipeline_id}/agents/*.yaml`
- `LoggedPipeline` wraps `Pipeline` to capture per-step timing and I/O metadata
- Python 3.12+ required; uses Pydantic v2 for all API models
