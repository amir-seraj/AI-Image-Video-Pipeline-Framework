from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
import yaml
from pathlib import Path

import fastapi
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .jobs import JobManager
from .models import (
    AddProductToCollectionRequest,
    AgentConfigRequest,
    AgentConfigResponse,
    AgentInfo,
    Collection,
    CreateCollectionRequest,
    CreateProductRequest,
    CreateVariationRequest,
    DuplicateAgentRequest,
    GenerateRequest,
    Generation,
    JobStatus,
    PipelineCreateRequest,
    PipelineDetailResponse,
    PipelineInputDeclaration,
    PipelinePreset,
    PipelineStepRequest,
    PipelineStepResponse,
    PipelineUpdateRequest,
    Product,
    RefineRequest,
    RegenerateVariationRequest,
    ResultFile,
    RunResponse,
    Sketch,
    UpdateCollectionRequest,
    UpdateVariationMetaRequest,
    Variation,
)
from .store import JsonStore

_DEFAULT_DATA_DIR = Path("data")


def create_app(
    data_dir: Path = _DEFAULT_DATA_DIR,
    *,
    workflows_dir: Path | None = None,
    agents_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="Casadei API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = JsonStore(data_dir / "store.json")
    uploads_dir = data_dir / "uploads"
    results_dir = data_dir / "results"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store results_dir on app state so tests can inspect output files
    app.state.results_dir = results_dir

    job_manager = JobManager()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if agents_dir is None:
        agents_dir = project_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    if workflows_dir is None:
        workflows_dir = project_root / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    # --- Product CRUD ---

    @app.post("/api/products", status_code=201)
    def create_product(req: CreateProductRequest) -> Product:
        product = Product(name=req.name)
        store.save_product(product)
        return product

    @app.get("/api/products")
    def list_products_endpoint() -> list[Product]:
        return store.list_products()

    @app.get("/api/products/{product_id}")
    def get_product(product_id: str) -> Product:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product

    @app.delete("/api/products/{product_id}", status_code=204)
    def delete_product(product_id: str) -> None:
        if not store.delete_product(product_id):
            raise HTTPException(status_code=404, detail="Product not found")

    # --- Sketch upload ---

    @app.post("/api/products/{product_id}/sketches", status_code=201)
    async def upload_sketch(product_id: str, file: UploadFile) -> Sketch:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        sketch = Sketch(filename=file.filename or "sketch.png")
        sketch_dir = uploads_dir / product_id
        sketch_dir.mkdir(parents=True, exist_ok=True)
        dest = sketch_dir / f"{sketch.id}_{sketch.filename}"
        dest.write_bytes(await file.read())

        product.sketches.append(sketch)
        store.save_product(product)
        return sketch

    @app.delete(
        "/api/products/{product_id}/sketches/{sketch_id}", status_code=204
    )
    def delete_sketch(product_id: str, sketch_id: str) -> None:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        original_len = len(product.sketches)
        product.sketches = [s for s in product.sketches if s.id != sketch_id]
        if len(product.sketches) == original_len:
            raise HTTPException(status_code=404, detail="Sketch not found")

        sketch_dir = uploads_dir / product_id
        for f in sketch_dir.glob(f"{sketch_id}_*"):
            f.unlink()

        store.save_product(product)

    @app.get("/api/uploads/{product_id}/{filename}")
    def serve_upload(product_id: str, filename: str) -> FileResponse:
        path = uploads_dir / product_id / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path)

    @app.get("/api/results/{filename:path}")
    def serve_result(filename: str) -> FileResponse:
        path = results_dir / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path)

    # --- Analyze (VLM) ---

    _VLM_MODEL = "qwen3_vl_8b"

    _ANALYZE_PROMPT = (
        "Analyze this shoe design sketch. Provide:\n"
        "1. A comma-separated list of visual feature tags (e.g. pointed toe, stiletto heel, ankle strap)\n"
        "2. A brief description of the design.\n\n"
        "Format your response exactly as:\n"
        "LABELS: tag1, tag2, tag3\n"
        "DESCRIPTION: Your description here"
    )

    def _run_analysis(product: Product, job_id: str) -> None:
        """Background thread: runs VLM analysis on product sketches."""
        try:
            from casadei import ImageMedia, MediaBundle, TextMedia
            from casadei.models.registry import default_registry

            job_manager.update_progress(job_id, 0.1, "Loading VLM...")

            model_cls = default_registry.get(_VLM_MODEL)
            model = model_cls()
            model.load_model()

            job_manager.update_progress(job_id, 0.3, "Analyzing sketch...")

            sketch_dir = uploads_dir / product.id
            image = None
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    image = ImageMedia.load(f)
                    break
                if image:
                    break

            if not image:
                model.unload_model()
                job_manager.fail(job_id, "No sketch images found")
                return

            try:
                bundle = MediaBundle(items={
                    "image": image,
                    "prompt": TextMedia(text=_ANALYZE_PROMPT),
                })
                output = model.run(bundle)
            finally:
                model.unload_model()

            job_manager.update_progress(job_id, 0.9, "Parsing results...")

            # Extract text from output
            response_text = ""
            for _key, media in output.items.items():
                if isinstance(media, TextMedia):
                    response_text = media.text
                    break

            # Parse LABELS and DESCRIPTION
            label = ""
            description = ""
            for line in response_text.split("\n"):
                line = line.strip()
                if line.upper().startswith("LABELS:"):
                    label = line[len("LABELS:"):].strip()
                elif line.upper().startswith("DESCRIPTION:"):
                    description = line[len("DESCRIPTION:"):].strip()

            # Fallback: if parsing failed, use full response as description
            if not label and not description and response_text:
                description = response_text.strip()

            product.label = label
            product.description = description
            store.save_product(product)

            job_manager.complete(job_id)

        except Exception as e:
            job_manager.fail(job_id, str(e))

    @app.post("/api/products/{product_id}/analyze", status_code=202)
    def analyze_product(product_id: str) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        if not product.sketches:
            raise HTTPException(
                status_code=400, detail="No sketches uploaded"
            )

        job_id = job_manager.create(product_id, "analyze")

        thread = threading.Thread(
            target=_run_analysis,
            args=(product, job_id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    # --- Agent & Pipeline info ---

    @app.get("/api/agents")
    def list_agents() -> list[AgentInfo]:
        from casadei import load_agent

        result = []
        if agents_dir.exists():
            for yaml_file in sorted(agents_dir.glob("*.yaml")):
                config = load_agent(yaml_file)
                variables = re.findall(
                    r"\$\{?(\w+)\}?", config.prompt_template
                )
                result.append(
                    AgentInfo(
                        name=config.name,
                        model=config.model,
                        description=config.description,
                        template_variables=variables,
                    )
                )
        return result

    @app.get("/api/agents/{agent_name}")
    def get_agent(agent_name: str) -> AgentConfigResponse:
        from casadei import load_agent

        for yaml_file in agents_dir.glob("*.yaml"):
            config = load_agent(yaml_file)
            if config.name == agent_name:
                variables = re.findall(r"\$\{?(\w+)\}?", config.prompt_template)
                return AgentConfigResponse(
                    name=config.name,
                    model=config.model,
                    description=config.description,
                    prompt_template=config.prompt_template,
                    negative_prompt=config.negative_prompt,
                    params=config.params,
                    template_variables=variables,
                )
        raise HTTPException(status_code=404, detail="Agent not found")

    @app.post("/api/agents", status_code=201)
    def create_agent(req: AgentConfigRequest) -> AgentConfigResponse:
        from casadei.agent import AgentConfig, save_agent

        dest = agents_dir / f"{req.name}.yaml"
        if dest.exists():
            raise HTTPException(status_code=409, detail="Agent already exists")

        config = AgentConfig(
            name=req.name,
            model=req.model,
            description=req.description,
            prompt_template=req.prompt_template,
            negative_prompt=req.negative_prompt,
            params=req.params,
        )
        save_agent(config, dest)

        variables = re.findall(r"\$\{?(\w+)\}?", config.prompt_template)
        return AgentConfigResponse(
            name=config.name,
            model=config.model,
            description=config.description,
            prompt_template=config.prompt_template,
            negative_prompt=config.negative_prompt,
            params=config.params,
            template_variables=variables,
        )

    @app.put("/api/agents/{agent_name}")
    def update_agent(agent_name: str, req: AgentConfigRequest) -> AgentConfigResponse:
        from casadei.agent import AgentConfig, save_agent, load_agent

        found_path = None
        for yaml_file in agents_dir.glob("*.yaml"):
            config = load_agent(yaml_file)
            if config.name == agent_name:
                found_path = yaml_file
                break

        if not found_path:
            raise HTTPException(status_code=404, detail="Agent not found")

        new_config = AgentConfig(
            name=req.name,
            model=req.model,
            description=req.description,
            prompt_template=req.prompt_template,
            negative_prompt=req.negative_prompt,
            params=req.params,
        )

        if req.name != agent_name:
            found_path.unlink()
            found_path = agents_dir / f"{req.name}.yaml"

        save_agent(new_config, found_path)

        variables = re.findall(r"\$\{?(\w+)\}?", new_config.prompt_template)
        return AgentConfigResponse(
            name=new_config.name,
            model=new_config.model,
            description=new_config.description,
            prompt_template=new_config.prompt_template,
            negative_prompt=new_config.negative_prompt,
            params=new_config.params,
            template_variables=variables,
        )

    @app.delete("/api/agents/{agent_name}", status_code=204)
    def delete_agent(agent_name: str) -> None:
        from casadei import load_agent

        for yaml_file in agents_dir.glob("*.yaml"):
            config = load_agent(yaml_file)
            if config.name == agent_name:
                yaml_file.unlink()
                return
        raise HTTPException(status_code=404, detail="Agent not found")

    @app.post("/api/agents/{agent_name}/duplicate", status_code=201)
    def duplicate_agent(agent_name: str, req: DuplicateAgentRequest) -> AgentConfigResponse:
        from casadei.agent import AgentConfig, save_agent, load_agent

        dest = agents_dir / f"{req.new_name}.yaml"
        if dest.exists():
            raise HTTPException(status_code=409, detail="Agent with that name already exists")

        source_config = None
        for yaml_file in agents_dir.glob("*.yaml"):
            config = load_agent(yaml_file)
            if config.name == agent_name:
                source_config = config
                break

        if not source_config:
            raise HTTPException(status_code=404, detail="Agent not found")

        new_config = AgentConfig(
            name=req.new_name,
            model=source_config.model,
            description=source_config.description,
            prompt_template=source_config.prompt_template,
            negative_prompt=source_config.negative_prompt,
            params=source_config.params,
        )
        save_agent(new_config, dest)

        variables = re.findall(r"\$\{?(\w+)\}?", new_config.prompt_template)
        return AgentConfigResponse(
            name=new_config.name,
            model=new_config.model,
            description=new_config.description,
            prompt_template=new_config.prompt_template,
            negative_prompt=new_config.negative_prompt,
            params=new_config.params,
            template_variables=variables,
        )

    @app.get("/api/models")
    def list_models() -> list[dict]:
        from casadei.models.registry import default_registry
        result = []
        for name in default_registry.list_models():
            cls = default_registry.get(name)
            all_params = cls.get_all_params()
            result.append({"name": name, "default_params": all_params})
        return result

    @app.get("/api/pipelines")
    def list_pipelines() -> list[PipelinePreset]:
        from casadei import load_agent as _load_agent

        results = []
        for pipeline_yaml in sorted(workflows_dir.glob("*/pipeline.yaml")):
            with open(pipeline_yaml) as f:
                data = yaml.safe_load(f)

            template_vars: list[str] = []
            pipeline_dir = pipeline_yaml.parent
            for step in data.get("steps", []):
                if step.get("type") == "agent":
                    agent_name = step["agent"]
                    local_yaml = pipeline_dir / "agents" / f"{agent_name}.yaml"
                    config = None
                    if local_yaml.exists():
                        config = _load_agent(local_yaml)
                    else:
                        for gf in agents_dir.glob("*.yaml"):
                            gc = _load_agent(gf)
                            if gc.name == agent_name:
                                config = gc
                                break
                    if config:
                        step_vars = re.findall(r"\$\{?(\w+)\}?", config.prompt_template)
                        for v in step_vars:
                            if v not in template_vars:
                                template_vars.append(v)

            agent_names = [s["agent"] for s in data.get("steps", []) if s.get("type") == "agent"]

            raw_inputs = data.get("inputs", {})
            pipeline_inputs = {
                k: PipelineInputDeclaration(type=v.get("type", "image"), label=v.get("label", k))
                for k, v in raw_inputs.items()
            }

            results.append(PipelinePreset(
                id=data["id"],
                name=data["name"],
                description=data.get("description", ""),
                agents=agent_names,
                template_variables=template_vars,
                inputs=pipeline_inputs,
            ))
        return results

    @app.get("/api/pipelines/{pipeline_id}")
    def get_pipeline(pipeline_id: str) -> PipelineDetailResponse:
        from casadei import load_agent

        pipeline_dir = workflows_dir / pipeline_id
        pipeline_yaml = pipeline_dir / "pipeline.yaml"
        if not pipeline_yaml.exists():
            raise HTTPException(status_code=404, detail="Pipeline not found")

        with open(pipeline_yaml) as f:
            data = yaml.safe_load(f)

        steps = []
        template_vars: list[str] = []
        for step in data.get("steps", []):
            if step["type"] == "agent":
                agent_name = step["agent"]
                local_yaml = pipeline_dir / "agents" / f"{agent_name}.yaml"
                if local_yaml.exists():
                    config = load_agent(local_yaml)
                else:
                    config = None
                    for gf in agents_dir.glob("*.yaml"):
                        gc = load_agent(gf)
                        if gc.name == agent_name:
                            config = gc
                            break
                if config:
                    step_vars = re.findall(r"\$\{?(\w+)\}?", config.prompt_template)
                    for v in step_vars:
                        if v not in template_vars:
                            template_vars.append(v)
                steps.append(PipelineStepResponse(type="agent", agent=agent_name))
            elif step["type"] == "code":
                script_path = pipeline_dir / "scripts" / step["script"]
                steps.append(PipelineStepResponse(
                    type="code",
                    script=step["script"],
                    function=step.get("function"),
                    exists=script_path.exists(),
                ))

        local_agents_dir = pipeline_dir / "agents"
        local_agents = []
        if local_agents_dir.exists():
            for la in local_agents_dir.glob("*.yaml"):
                config = load_agent(la)
                local_agents.append(config.name)

        raw_inputs = data.get("inputs", {})
        pipeline_inputs = {
            k: PipelineInputDeclaration(type=v.get("type", "image"), label=v.get("label", k))
            for k, v in raw_inputs.items()
        }

        return PipelineDetailResponse(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            steps=steps,
            local_agents=local_agents,
            template_variables=template_vars,
            inputs=pipeline_inputs,
        )

    @app.post("/api/pipelines", status_code=201)
    def create_pipeline(req: PipelineCreateRequest) -> PipelineDetailResponse:
        pipeline_dir = workflows_dir / req.id
        if pipeline_dir.exists():
            raise HTTPException(status_code=409, detail="Pipeline already exists")

        pipeline_dir.mkdir(parents=True)

        data = {
            "id": req.id,
            "name": req.name,
            "description": req.description,
            "steps": [s.model_dump(exclude_none=True) for s in req.steps],
        }
        with open(pipeline_dir / "pipeline.yaml", "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return get_pipeline(req.id)

    @app.put("/api/pipelines/{pipeline_id}")
    def update_pipeline(pipeline_id: str, req: PipelineUpdateRequest) -> PipelineDetailResponse:
        pipeline_dir = workflows_dir / pipeline_id
        pipeline_yaml = pipeline_dir / "pipeline.yaml"
        if not pipeline_yaml.exists():
            raise HTTPException(status_code=404, detail="Pipeline not found")

        with open(pipeline_yaml) as f:
            data = yaml.safe_load(f)

        if req.name is not None:
            data["name"] = req.name
        if req.description is not None:
            data["description"] = req.description
        if req.steps is not None:
            data["steps"] = [s.model_dump(exclude_none=True) for s in req.steps]

        with open(pipeline_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        return get_pipeline(pipeline_id)

    @app.delete("/api/pipelines/{pipeline_id}", status_code=204)
    def delete_pipeline(pipeline_id: str) -> None:
        import shutil
        pipeline_dir = workflows_dir / pipeline_id
        if not pipeline_dir.exists():
            raise HTTPException(status_code=404, detail="Pipeline not found")
        shutil.rmtree(pipeline_dir)

    @app.post("/api/pipelines/{pipeline_id}/agents", status_code=201)
    def create_pipeline_local_agent(
        pipeline_id: str, req: AgentConfigRequest
    ) -> AgentConfigResponse:
        from casadei.agent import AgentConfig, save_agent

        pipeline_dir = workflows_dir / pipeline_id
        if not pipeline_dir.exists():
            raise HTTPException(status_code=404, detail="Pipeline not found")

        local_agents_dir = pipeline_dir / "agents"
        local_agents_dir.mkdir(exist_ok=True)

        config = AgentConfig(
            name=req.name,
            model=req.model,
            description=req.description,
            prompt_template=req.prompt_template,
            negative_prompt=req.negative_prompt,
            params=req.params,
        )
        save_agent(config, local_agents_dir / f"{req.name}.yaml")

        variables = re.findall(r"\$\{?(\w+)\}?", config.prompt_template)
        return AgentConfigResponse(
            name=config.name,
            model=config.model,
            description=config.description,
            prompt_template=config.prompt_template,
            negative_prompt=config.negative_prompt,
            params=config.params,
            template_variables=variables,
        )

    @app.delete("/api/pipelines/{pipeline_id}/agents/{agent_name}", status_code=204)
    def delete_pipeline_local_agent(pipeline_id: str, agent_name: str) -> None:
        pipeline_dir = workflows_dir / pipeline_id
        local_yaml = pipeline_dir / "agents" / f"{agent_name}.yaml"
        if not local_yaml.exists():
            raise HTTPException(status_code=404, detail="Local agent not found")
        local_yaml.unlink()

    @app.post("/api/pipelines/{pipeline_id}/scripts/{filename}/open", status_code=200)
    def open_pipeline_script(pipeline_id: str, filename: str) -> dict:
        import subprocess
        pipeline_dir = workflows_dir / pipeline_id
        scripts_dir = pipeline_dir / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / filename

        if not script_path.exists():
            script_path.write_text(
                f'"""Pipeline code step: {filename}"""\n\n'
                f"from casadei.media import Media\n\n\n"
                f"def process(context: dict[str, Media]) -> dict[str, Media]:\n"
                f"    # TODO: implement\n"
                f"    return context\n"
            )

        try:
            subprocess.Popen(["xdg-open", str(script_path)])
        except FileNotFoundError:
            subprocess.Popen(["open", str(script_path)])

        return {"opened": str(script_path)}

    # --- Scratch run (workbench) ---

    def _resolve_agent_yaml(agent_name: str, pipeline_dir: Path | None = None) -> Path | None:
        """Find agent YAML: pipeline-local first, then global."""
        from casadei import load_agent
        if pipeline_dir:
            local_yaml = pipeline_dir / "agents" / f"{agent_name}.yaml"
            if local_yaml.exists():
                return local_yaml
        for gf in agents_dir.glob("*.yaml"):
            gc = load_agent(gf)
            if gc.name == agent_name:
                return gf
        return None

    def _run_scratch(
        run_id: str,
        run_type: str,
        name: str,
        template_variables: dict[str, str],
        named_images: dict[str, Path],
        job_id: str,
    ) -> None:
        """Background thread: runs a single agent or pipeline on scratch images.

        Uses a named context dict so pipelines can work with multiple images.
        Backward-compatible: single-image pipelines use the "image" key.
        """
        try:
            import importlib.util
            from casadei import Agent, ImageMedia, MediaBundle, TextMedia, load_agent

            job_manager.update_progress(job_id, 0.05, "Loading images...")

            # Build named context from uploaded images
            context: dict = {
                key: ImageMedia.load(path) for key, path in named_images.items()
            }

            # Backward compat: if single image without "image" key, alias it
            if len(context) == 1 and "image" not in context:
                only_key = next(iter(context))
                context["image"] = context[only_key]

            if run_type == "agent":
                steps = [{"type": "agent", "agent": name}]
                pipeline_dir = None
            else:
                pipeline_dir = workflows_dir / name
                pipeline_yaml = pipeline_dir / "pipeline.yaml"
                if not pipeline_yaml.exists():
                    job_manager.fail(job_id, f"Unknown pipeline: {name}")
                    return
                with open(pipeline_yaml) as f:
                    data = yaml.safe_load(f)
                steps = data.get("steps", [])

            total = len(steps)

            for i, step in enumerate(steps):
                progress = 0.1 + (0.8 * i / max(total, 1))

                if step["type"] == "agent":
                    agent_name = step["agent"]
                    job_manager.update_progress(job_id, progress, f"Running {agent_name}...")

                    yaml_path = _resolve_agent_yaml(agent_name, pipeline_dir)
                    if not yaml_path:
                        job_manager.fail(job_id, f"Agent not found: {agent_name}")
                        return

                    agent_config = load_agent(yaml_path)
                    agent = Agent(agent_config)
                    agent.load()

                    try:
                        # If step has input mapping, build bundle from mapped context keys
                        step_inputs = step.get("inputs")
                        if step_inputs:
                            bundle_items: dict = {}
                            for bundle_key, context_key in step_inputs.items():
                                if context_key not in context:
                                    job_manager.fail(
                                        job_id,
                                        f"Step '{agent_name}': context key '{context_key}' "
                                        f"not found. Available: {list(context.keys())}",
                                    )
                                    return
                                bundle_items[bundle_key] = context[context_key]
                        else:
                            # Legacy passthrough: use "image" key
                            bundle_items = {}
                            if "image" in context:
                                bundle_items["image"] = context["image"]

                        template_kwargs = dict(template_variables)

                        output = agent.execute(
                            MediaBundle(items=bundle_items),
                            **template_kwargs,
                        )

                        # Merge agent outputs back into context
                        for out_key, media in output.items.items():
                            context[out_key] = media
                    finally:
                        agent.unload()

                elif step["type"] == "code":
                    script_name = step["script"]
                    function_name = step.get("function", "process")
                    job_manager.update_progress(job_id, progress, f"Running {script_name}:{function_name}...")

                    if not pipeline_dir:
                        job_manager.fail(job_id, "Code steps only supported in pipelines")
                        return

                    script_path = pipeline_dir / "scripts" / script_name
                    if not script_path.exists():
                        job_manager.fail(job_id, f"Script not found: {script_name}")
                        return

                    spec = importlib.util.spec_from_file_location(
                        f"pipeline_script_{name}_{script_name}", script_path
                    )
                    if not spec or not spec.loader:
                        job_manager.fail(job_id, f"Cannot load script: {script_name}")
                        return

                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    func = getattr(module, function_name, None)
                    if not func:
                        job_manager.fail(job_id, f"Function {function_name} not found in {script_name}")
                        return

                    # Code steps always receive the full context
                    result_context = func(context)
                    if isinstance(result_context, dict):
                        context.update(result_context)

            job_manager.update_progress(job_id, 0.95, "Saving results...")
            scratch_results_dir = results_dir / "scratch" / run_id
            scratch_results_dir.mkdir(parents=True, exist_ok=True)

            # Collect all ImageMedia from context for output
            from casadei import ImageMedia as _IM
            output_images = [v for v in context.values() if isinstance(v, _IM)]
            # Prefer the "image" key as primary output
            if "image" in context and isinstance(context["image"], _IM):
                output_images = [context["image"]]

            for idx, img in enumerate(output_images):
                fname = f"output_{idx}.png"
                img.save(scratch_results_dir / fname)

            job_manager.complete(job_id)

        except Exception as e:
            job_manager.fail(job_id, str(e))

    @app.post("/api/run", status_code=202)
    async def run_workbench(
        type: str = fastapi.Form(...),
        name: str = fastapi.Form(...),
        template_variables: str = fastapi.Form("{}"),
        image: UploadFile | None = fastapi.File(None),
        images: list[UploadFile] = fastapi.File([]),
        image_keys: str = fastapi.Form("[]"),
    ) -> RunResponse:
        if type not in ("agent", "pipeline"):
            raise HTTPException(status_code=400, detail="type must be 'agent' or 'pipeline'")

        try:
            vars_dict = json.loads(template_variables)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="template_variables must be valid JSON")

        try:
            keys_list = json.loads(image_keys)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="image_keys must be valid JSON")

        run_id = uuid.uuid4().hex[:12]
        scratch_dir = uploads_dir / "scratch" / run_id
        scratch_dir.mkdir(parents=True, exist_ok=True)

        named_images: dict[str, Path] = {}

        if images and keys_list:
            # Multi-image mode: pair each uploaded image with its key
            if len(images) != len(keys_list):
                raise HTTPException(
                    status_code=400,
                    detail=f"images count ({len(images)}) must match image_keys count ({len(keys_list)})",
                )
            for key, upload in zip(keys_list, images):
                path = scratch_dir / f"{key}_{upload.filename or 'input.png'}"
                path.write_bytes(await upload.read())
                named_images[key] = path
        elif image:
            # Legacy single-image mode
            image_path = scratch_dir / (image.filename or "input.png")
            image_path.write_bytes(await image.read())
            named_images["image"] = image_path
        else:
            raise HTTPException(status_code=400, detail="No image(s) provided")

        job_id = job_manager.create("scratch", run_id)

        thread = threading.Thread(
            target=_run_scratch,
            args=(run_id, type, name, vars_dict, named_images, job_id),
            daemon=True,
        )
        thread.start()

        return RunResponse(job_id=job_id, run_id=run_id)

    # --- Generation ---

    def _run_generation(
        product: Product,
        generation: Generation,
        job_id: str,
    ) -> None:
        """Background thread: runs the Casadei pipeline."""
        try:
            from casadei import (
                Agent,
                ImageMedia,
                MediaBundle,
                TextMedia,
                load_agent,
            )

            job_manager.update_progress(job_id, 0.1, "Loading models...")

            agent_name_to_yaml: dict[str, Path] = {}
            if agents_dir.exists():
                for yaml_file in agents_dir.glob("*.yaml"):
                    config = load_agent(yaml_file)
                    agent_name_to_yaml[config.name] = yaml_file

            preset = None
            for p in list_pipelines():
                if p.id == generation.pipeline:
                    preset = p
                    break

            if not preset:
                job_manager.fail(
                    job_id, f"Unknown pipeline: {generation.pipeline}"
                )
                return

            # Pipelines with named inputs require the Workbench (multi-image upload)
            if preset.inputs:
                input_labels = ", ".join(
                    v.label for v in preset.inputs.values()
                )
                job_manager.fail(
                    job_id,
                    f"'{preset.name}' requires multiple image inputs ({input_labels}). "
                    f"Use the Workbench to run this pipeline.",
                )
                return

            sketch_dir = uploads_dir / product.id
            images = []
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    images.append(ImageMedia.load(f))

            if not images:
                job_manager.fail(job_id, "No sketch images found")
                return

            total_agents = len(preset.agents)
            result_images = images

            for i, agent_name in enumerate(preset.agents):
                progress = 0.1 + (0.8 * i / total_agents)
                job_manager.update_progress(
                    job_id, progress, f"Running {agent_name}..."
                )

                yaml_path = agent_name_to_yaml.get(agent_name)
                if not yaml_path:
                    job_manager.fail(
                        job_id, f"Agent not found: {agent_name}"
                    )
                    return

                agent_config = load_agent(yaml_path)
                agent = Agent(agent_config)
                agent.load()

                try:
                    bundle_items: dict = {"image": result_images[0]}
                    if len(result_images) > 1:
                        bundle_items["image_2"] = result_images[1]
                    bundle_items["prompt"] = TextMedia(
                        text=generation.prompt
                    )
                    if generation.negative_prompt:
                        bundle_items["negative_prompt"] = TextMedia(
                            text=generation.negative_prompt
                        )

                    template_kwargs = {
                        "prompt": generation.prompt,
                        "style": generation.prompt,
                    }

                    output = agent.execute(
                        MediaBundle(items=bundle_items),
                        **template_kwargs,
                    )

                    for _key, media in output.items.items():
                        if isinstance(media, ImageMedia):
                            result_images = [media]
                finally:
                    agent.unload()

            job_manager.update_progress(job_id, 0.95, "Saving results...")
            gen_results_dir = results_dir / product.id / generation.id
            gen_results_dir.mkdir(parents=True, exist_ok=True)

            result_files = []
            for idx, img in enumerate(
                result_images[: generation.num_outputs]
            ):
                fname = f"output_{idx}.png"
                img.save(gen_results_dir / fname)
                result_files.append(ResultFile(filename=fname))

            generation.results = result_files
            generation.status = JobStatus.completed
            store.save_product(product)

            job_manager.complete(job_id)

        except Exception as e:
            generation.status = JobStatus.failed
            store.save_product(product)
            job_manager.fail(job_id, str(e))

    @app.post("/api/products/{product_id}/generate", status_code=202)
    def start_generation(product_id: str, req: GenerateRequest) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        if not product.sketches:
            raise HTTPException(
                status_code=400, detail="No sketches uploaded"
            )

        generation = Generation(
            pipeline=req.pipeline,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_outputs=req.num_outputs,
            status=JobStatus.pending,
        )
        product.generations.append(generation)
        store.save_product(product)

        job_id = job_manager.create(product_id, generation.id)

        thread = threading.Thread(
            target=_run_generation,
            args=(product, generation, job_id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "generation_id": generation.id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        state = job_manager.get(job_id)
        if not state:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": state.job_id,
            "status": state.status,
            "progress": state.progress,
            "message": state.message,
            "error": state.error,
            "product_id": state.product_id,
            "generation_id": state.generation_id,
        }

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job(job_id: str):
        state = job_manager.get(job_id)
        if not state:
            raise HTTPException(status_code=404, detail="Job not found")

        async def event_generator():
            event = job_manager.subscribe(job_id)
            try:
                while True:
                    current = job_manager.get(job_id)
                    if not current:
                        break
                    yield {
                        "event": "progress",
                        "data": json.dumps(
                            {
                                "status": current.status,
                                "progress": current.progress,
                                "message": current.message,
                                "error": current.error,
                            }
                        ),
                    }
                    if current.status in ("completed", "failed"):
                        break
                    event.clear()
                    # threading.Event.wait with timeout — runs in threadpool
                    await asyncio.get_event_loop().run_in_executor(
                        None, lambda: event.wait(timeout=30.0)
                    )
            finally:
                job_manager.unsubscribe(job_id, event)

        return EventSourceResponse(event_generator())

    # --- Refine ---

    @app.post("/api/generations/{generation_id}/refine", status_code=202)
    def refine_generation(generation_id: str, req: RefineRequest) -> dict:
        for product in store.list_products():
            for gen in product.generations:
                if gen.id == generation_id:
                    new_gen = Generation(
                        pipeline=gen.pipeline,
                        prompt=req.prompt,
                        negative_prompt=req.negative_prompt
                        or gen.negative_prompt,
                        num_outputs=gen.num_outputs,
                        parent_generation_id=generation_id,
                        status=JobStatus.pending,
                    )
                    product.generations.append(new_gen)
                    store.save_product(product)
                    job_id = job_manager.create(product.id, new_gen.id)
                    thread = threading.Thread(
                        target=_run_generation,
                        args=(product, new_gen, job_id),
                        daemon=True,
                    )
                    thread.start()
                    return {"job_id": job_id, "generation_id": new_gen.id}
        raise HTTPException(status_code=404, detail="Generation not found")

    # --- Variations ---

    def _run_variation(
        product: Product,
        variation: Variation,
        job_id: str,
        change_request: str = "",
    ) -> None:
        """Background thread: generates images for a variation."""
        try:
            from casadei import (
                Agent,
                ImageMedia,
                MediaBundle,
                TextMedia,
                load_agent,
            )

            job_manager.update_progress(job_id, 0.1, "Loading models...")

            agent_name_to_yaml: dict[str, Path] = {}
            if agents_dir.exists():
                for yaml_file in agents_dir.glob("*.yaml"):
                    config = load_agent(yaml_file)
                    agent_name_to_yaml[config.name] = yaml_file

            preset = None
            for p in list_pipelines():
                if p.id == variation.pipeline:
                    preset = p
                    break

            if not preset:
                job_manager.fail(
                    job_id, f"Unknown pipeline: {variation.pipeline}"
                )
                return

            if preset.inputs:
                input_labels = ", ".join(
                    v.label for v in preset.inputs.values()
                )
                job_manager.fail(
                    job_id,
                    f"'{preset.name}' requires multiple image inputs ({input_labels}). "
                    f"Use the Workbench to run this pipeline.",
                )
                return

            sketch_dir = uploads_dir / product.id
            images = []
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    images.append(ImageMedia.load(f))

            if not images:
                job_manager.fail(job_id, "No sketch images found")
                return

            parts = []
            if variation.material or variation.color:
                parts.append(
                    f"A photorealistic {variation.material} shoe in {variation.color}."
                )
            if variation.note:
                parts.append(variation.note + ".")
            if change_request:
                parts.append(change_request + ".")
            prompt = " ".join(parts) if parts else "Generate a shoe design."

            total_agents = len(preset.agents)
            result_images = images

            for i, agent_name in enumerate(preset.agents):
                progress = 0.1 + (0.8 * i / total_agents)
                job_manager.update_progress(
                    job_id, progress, f"Running {agent_name}..."
                )

                yaml_path = agent_name_to_yaml.get(agent_name)
                if not yaml_path:
                    job_manager.fail(
                        job_id, f"Agent not found: {agent_name}"
                    )
                    return

                agent_config = load_agent(yaml_path)
                agent = Agent(agent_config)
                agent.load()

                try:
                    bundle_items: dict = {"image": result_images[0]}
                    if len(result_images) > 1:
                        bundle_items["image_2"] = result_images[1]
                    bundle_items["prompt"] = TextMedia(text=prompt)

                    template_kwargs = {
                        "prompt": prompt,
                        "style": prompt,
                    }

                    output = agent.execute(
                        MediaBundle(items=bundle_items),
                        **template_kwargs,
                    )

                    for _key, media in output.items.items():
                        if isinstance(media, ImageMedia):
                            result_images = [media]
                finally:
                    agent.unload()

            job_manager.update_progress(job_id, 0.95, "Saving results...")
            var_results_dir = results_dir / product.id / variation.id
            var_results_dir.mkdir(parents=True, exist_ok=True)

            result_files = []
            for idx, img in enumerate(
                result_images[: variation.num_outputs]
            ):
                fname = f"output_{idx}.png"
                img.save(var_results_dir / fname)
                result_files.append(ResultFile(filename=fname))

            variation.results = result_files
            variation.status = JobStatus.completed
            store.save_product(product)

            job_manager.complete(job_id)

        except Exception as e:
            variation.status = JobStatus.failed
            store.save_product(product)
            job_manager.fail(job_id, str(e))

    @app.post("/api/products/{product_id}/variations", status_code=202)
    def create_variation(
        product_id: str, req: CreateVariationRequest,
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        if not product.sketches:
            raise HTTPException(
                status_code=400, detail="No sketches uploaded"
            )

        variation = Variation(
            material=req.material,
            color=req.color,
            note=req.note,
            pipeline=req.pipeline,
            num_outputs=req.num_outputs,
            status=JobStatus.pending,
        )
        product.variations.append(variation)
        store.save_product(product)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_variation,
            args=(product, variation, job_id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "variation_id": variation.id}

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/regenerate",
        status_code=202,
    )
    def regenerate_variation(
        product_id: str,
        variation_id: str,
        req: RegenerateVariationRequest,
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(
                status_code=404, detail="Variation not found"
            )

        variation.status = JobStatus.pending
        variation.results = []
        store.save_product(product)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_variation,
            args=(product, variation, job_id, req.change_request),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_add_image(
        product: Product,
        variation: Variation,
        job_id: str,
        change_request: str,
    ) -> None:
        """Background thread: generates one more image and appends to variation results."""
        try:
            from casadei import (
                Agent,
                ImageMedia,
                MediaBundle,
                TextMedia,
                load_agent,
            )

            job_manager.update_progress(job_id, 0.1, "Loading models...")

            agent_name_to_yaml: dict[str, Path] = {}
            if agents_dir.exists():
                for yaml_file in agents_dir.glob("*.yaml"):
                    config = load_agent(yaml_file)
                    agent_name_to_yaml[config.name] = yaml_file

            preset = None
            for p in list_pipelines():
                if p.id == variation.pipeline:
                    preset = p
                    break

            if not preset:
                job_manager.fail(
                    job_id, f"Unknown pipeline: {variation.pipeline}"
                )
                variation.status = JobStatus.completed
                store.save_product(product)
                return

            sketch_dir = uploads_dir / product.id
            images = []
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    images.append(ImageMedia.load(f))

            if not images:
                job_manager.fail(job_id, "No sketch images found")
                variation.status = JobStatus.completed
                store.save_product(product)
                return

            parts = []
            if variation.material or variation.color:
                parts.append(
                    f"A photorealistic {variation.material} shoe in {variation.color}."
                )
            if variation.note:
                parts.append(variation.note + ".")
            if change_request:
                parts.append(change_request + ".")
            prompt = " ".join(parts) if parts else "Generate a shoe design."

            total_agents = len(preset.agents)
            result_images = images

            for i, agent_name in enumerate(preset.agents):
                progress = 0.1 + (0.8 * i / total_agents)
                job_manager.update_progress(
                    job_id, progress, f"Running {agent_name}..."
                )

                yaml_path = agent_name_to_yaml.get(agent_name)
                if not yaml_path:
                    job_manager.fail(
                        job_id, f"Agent not found: {agent_name}"
                    )
                    variation.status = JobStatus.completed
                    store.save_product(product)
                    return

                agent_config = load_agent(yaml_path)
                agent = Agent(agent_config)
                agent.load()

                try:
                    bundle_items: dict = {"image": result_images[0]}
                    if len(result_images) > 1:
                        bundle_items["image_2"] = result_images[1]
                    bundle_items["prompt"] = TextMedia(text=prompt)

                    template_kwargs = {
                        "prompt": prompt,
                        "style": prompt,
                    }

                    output = agent.execute(
                        MediaBundle(items=bundle_items),
                        **template_kwargs,
                    )

                    for _key, media in output.items.items():
                        if isinstance(media, ImageMedia):
                            result_images = [media]
                finally:
                    agent.unload()

            job_manager.update_progress(job_id, 0.95, "Saving results...")
            var_results_dir = results_dir / product.id / variation.id
            var_results_dir.mkdir(parents=True, exist_ok=True)

            # Find next index by checking existing results
            existing_count = len(variation.results)
            next_idx = existing_count
            fname = f"output_{next_idx}.png"
            result_images[0].save(var_results_dir / fname)
            variation.results.append(ResultFile(filename=fname))

            variation.status = JobStatus.completed
            store.save_product(product)

            job_manager.complete(job_id)

        except Exception as e:
            variation.status = JobStatus.completed
            store.save_product(product)
            job_manager.fail(job_id, str(e))

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/add-image",
        status_code=202,
    )
    def add_image_to_variation(
        product_id: str,
        variation_id: str,
        req: RegenerateVariationRequest,
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(
                status_code=404, detail="Variation not found"
            )

        variation.status = JobStatus.running
        store.save_product(product)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_add_image,
            args=(product, variation, job_id, req.change_request),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/generate-360",
        status_code=202,
    )
    def generate_360(
        product_id: str,
        variation_id: str,
        provider: str = "zero123pp",
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(
                status_code=404, detail="Variation not found"
            )

        if not variation.results:
            raise HTTPException(
                status_code=400,
                detail="Variation has no rendered images to generate 360 from",
            )

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_generate_360,
            args=(product, variation, job_id, provider),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_generate_360(
        prod: Product,
        var: Variation,
        jid: str,
        provider_name: str,
    ) -> None:
        """Background thread: generates 360° spin frames for a variation."""
        try:
            var_results_dir = results_dir / prod.id / var.id

            if provider_name == "existing":
                _run_generate_360_from_existing(prod, var, jid, var_results_dir)
            else:
                _run_generate_360_ai(prod, var, jid, provider_name, var_results_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            var.status = JobStatus.failed
            store.save_product(prod)
            job_manager.fail(jid, str(e))

    def _run_generate_360_from_existing(
        prod: Product,
        var: Variation,
        jid: str,
        var_results_dir: Path,
    ) -> None:
        """Use the variation's existing result images as spin frames."""
        import shutil

        job_manager.update_progress(jid, 0.2, "Copying existing images as spin frames...")

        spin_files = []
        for i, rf in enumerate(var.results):
            src = var_results_dir / rf.filename
            if not src.exists():
                continue
            fname = f"spin_frame_{i}.png"
            dst = var_results_dir / fname
            shutil.copy2(src, dst)
            spin_files.append(ResultFile(filename=fname))

        var.spin_frames = spin_files
        store.save_product(prod)

        job_manager.update_progress(jid, 1.0, f"Done — {len(spin_files)} frames")
        job_manager.complete(jid)

    def _run_generate_360_ai(
        prod: Product,
        var: Variation,
        jid: str,
        provider_name: str,
        var_results_dir: Path,
    ) -> None:
        """Generate spin frames using an AI multi-view model."""
        from casadei import ImageMedia
        from casadei.media import MediaBundle
        from casadei.models.registry import default_registry

        job_manager.update_progress(jid, 0.05, "Loading source image...")

        source_path = var_results_dir / var.results[0].filename
        source = ImageMedia.load(source_path)

        job_manager.update_progress(jid, 0.1, f"Loading {provider_name} model...")

        model_cls = default_registry.get(provider_name)
        model = model_cls()
        model.load_model()

        try:
            job_manager.update_progress(jid, 0.2, "Generating views...")
            result = model.run(
                MediaBundle(items={"image": source}),
                num_views=6,
            )

            job_manager.update_progress(jid, 0.8, "Saving frames...")

            var_results_dir.mkdir(parents=True, exist_ok=True)

            spin_files = []
            for i, (key, media) in enumerate(sorted(result.items.items())):
                if isinstance(media, ImageMedia):
                    fname = f"spin_frame_{i}.png"
                    media.save(var_results_dir / fname)
                    spin_files.append(ResultFile(filename=fname))

            var.spin_frames = spin_files
            store.save_product(prod)

            job_manager.update_progress(jid, 1.0, "Done!")
            job_manager.complete(jid)

        finally:
            model.unload_model()

    @app.patch(
        "/api/products/{product_id}/variations/{variation_id}/meta",
        status_code=200,
    )
    def update_variation_meta(
        product_id: str,
        variation_id: str,
        req: UpdateVariationMetaRequest,
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        for v in product.variations:
            if v.id == variation_id:
                if req.tags is not None:
                    v.tags = req.tags
                if req.price is not None:
                    v.price = req.price
                elif req.model_fields_set and "price" in req.model_fields_set:
                    v.price = None
                store.save_product(product)
                return {"tags": v.tags, "price": v.price}

        raise HTTPException(status_code=404, detail="Variation not found")

    @app.delete(
        "/api/products/{product_id}/variations/{variation_id}",
        status_code=204,
    )
    def delete_variation(
        product_id: str, variation_id: str,
    ):
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        original_len = len(product.variations)
        product.variations = [
            v for v in product.variations if v.id != variation_id
        ]
        if len(product.variations) == original_len:
            raise HTTPException(
                status_code=404, detail="Variation not found"
            )
        store.save_product(product)

    @app.put(
        "/api/products/{product_id}/variations/{variation_id}/set-cover",
        status_code=200,
    )
    def set_variation_cover(
        product_id: str,
        variation_id: str,
        req: dict,
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(status_code=404, detail="Variation not found")

        filename = req.get("filename")
        if not filename:
            raise HTTPException(status_code=400, detail="filename required")

        idx = next(
            (i for i, r in enumerate(variation.results) if r.filename == filename),
            None,
        )
        if idx is None:
            raise HTTPException(status_code=404, detail="Result not found")

        # Move the chosen result to the front
        chosen = variation.results.pop(idx)
        variation.results.insert(0, chosen)
        store.save_product(product)
        return {"ok": True}

    # ── Collections ──────────────────────────────────────────────

    @app.get("/api/collections")
    def list_collections() -> list[Collection]:
        return store.list_collections()

    @app.post("/api/collections", status_code=201)
    def create_collection(req: CreateCollectionRequest) -> Collection:
        collection = Collection(name=req.name)
        store.save_collection(collection)
        return collection

    @app.get("/api/collections/{collection_id}")
    def get_collection(collection_id: str) -> Collection:
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        return collection

    @app.put("/api/collections/{collection_id}")
    def update_collection(
        collection_id: str, req: UpdateCollectionRequest
    ) -> Collection:
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        for field, value in req.model_dump(exclude_none=True).items():
            setattr(collection, field, value)
        store.save_collection(collection)
        return collection

    @app.delete("/api/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: str):
        if not store.delete_collection(collection_id):
            raise HTTPException(status_code=404, detail="Collection not found")

    @app.post("/api/collections/{collection_id}/products", status_code=200)
    def add_product_to_collection(
        collection_id: str, req: AddProductToCollectionRequest
    ) -> Collection:
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        product = store.get_product(req.product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        if req.product_id not in collection.product_ids:
            collection.product_ids.append(req.product_id)
            store.save_collection(collection)
        return collection

    @app.delete(
        "/api/collections/{collection_id}/products/{product_id}",
        status_code=200,
    )
    def remove_product_from_collection(
        collection_id: str, product_id: str
    ) -> Collection:
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        if product_id not in collection.product_ids:
            raise HTTPException(
                status_code=404, detail="Product not in collection"
            )
        collection.product_ids.remove(product_id)
        store.save_collection(collection)
        return collection

    @app.put("/api/collections/{collection_id}/reorder", status_code=200)
    def reorder_collection_products(
        collection_id: str, body: dict
    ) -> Collection:
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        new_order = body.get("product_ids", [])
        if set(new_order) != set(collection.product_ids):
            raise HTTPException(
                status_code=400,
                detail="product_ids must contain the same products",
            )
        collection.product_ids = new_order
        store.save_collection(collection)
        return collection

    return app
