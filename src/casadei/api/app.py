from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
import yaml
from pathlib import Path

import os
import logging

import fastapi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
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
    IndexStats,
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
    PromoteToVariationRequest,
    RegenerateVariationRequest,
    GeneratedResult,
    HeroCandidate,
    ResultFile,
    RunResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    Sketch,
    UpdateCollectionRequest,
    SelectHeroCandidateRequest,
    UpdateVariationMetaRequest,
    User,
    UserRole,
    LoginRequest,
    CreateUserRequest,
    ChangePasswordRequest,
    CostRecord,
    ResetPasswordRequest,
    Variation,
)
from .auth import hash_password, verify_password, create_token, decode_token
from casadei.providers.gemini_pricing import calculate_cost, extract_token_usage, calculate_veo_cost, calculate_voyage_cost
from .store import JsonStore
from .vectordb import VariantVectorDB, build_variant_text, normalize_text

_DEFAULT_DATA_DIR = Path("data")


def create_app(
    data_dir: Path = _DEFAULT_DATA_DIR,
    *,
    workflows_dir: Path | None = None,
    agents_dir: Path | None = None,
    embedding_provider: object | None = None,
) -> FastAPI:
    load_dotenv()
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

    # Store refs on app state so tests can inspect
    app.state.results_dir = results_dir
    app.state.store = store

    job_manager = JobManager()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if agents_dir is None:
        agents_dir = project_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    if workflows_dir is None:
        workflows_dir = project_root / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    # --- Bootstrap admin user from env vars ---
    logger = logging.getLogger(__name__)
    _admin_email = os.environ.get("ADMIN_EMAIL")
    _admin_password = os.environ.get("ADMIN_PASSWORD")
    if _admin_email and _admin_password:
        existing_admin = store.get_user_by_email(_admin_email)
        if not existing_admin:
            admin_user = User(
                email=_admin_email,
                name="Admin",
                password_hash=hash_password(_admin_password),
                role=UserRole.admin,
            )
            store.save_user(admin_user)
            logger.info("Admin user created: %s", _admin_email)
        else:
            logger.info("Admin user already exists: %s", _admin_email)

    # --- Auth helpers ---

    def _get_current_user(request: Request) -> User:
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing token")
        token = auth_header[7:]
        payload = decode_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user = store.get_user(payload["sub"])
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user

    def _require_admin(request: Request) -> User:
        user = _get_current_user(request)
        if user.role != UserRole.admin:
            raise HTTPException(status_code=403, detail="Admin access required")
        return user

    # --- Auth endpoints ---

    @app.post("/api/auth/login")
    def auth_login(req: LoginRequest):
        user = store.get_user_by_email(req.email)
        if not user or not verify_password(req.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_token(user.id, user.role.value)
        return {
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "role": user.role.value,
                "created_at": user.created_at,
            },
        }

    @app.get("/api/auth/me")
    def auth_me(request: Request):
        user = _get_current_user(request)
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "created_at": user.created_at,
        }

    @app.put("/api/auth/change-password")
    def auth_change_password(req: ChangePasswordRequest, request: Request):
        user = _get_current_user(request)
        if not verify_password(req.current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        user.password_hash = hash_password(req.new_password)
        store.save_user(user)
        return {"ok": True}

    # --- Admin user management ---

    @app.get("/api/admin/users")
    def admin_list_users(request: Request):
        _require_admin(request)
        users = store.list_users()
        return [
            {
                "id": u.id,
                "email": u.email,
                "name": u.name,
                "role": u.role.value,
                "created_at": u.created_at,
            }
            for u in users
        ]

    @app.post("/api/admin/users", status_code=201)
    def admin_create_user(req: CreateUserRequest, request: Request):
        _require_admin(request)
        if store.get_user_by_email(req.email):
            raise HTTPException(status_code=409, detail="Email already exists")
        user = User(
            email=req.email,
            name=req.name,
            password_hash=hash_password(req.password),
            role=UserRole.designer,
        )
        store.save_user(user)
        return {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value,
            "created_at": user.created_at,
        }

    @app.delete("/api/admin/users/{user_id}", status_code=204)
    def admin_delete_user(user_id: str, request: Request):
        admin = _require_admin(request)
        if user_id == admin.id:
            raise HTTPException(status_code=400, detail="Cannot delete yourself")
        if not store.delete_user(user_id):
            raise HTTPException(status_code=404, detail="User not found")

    @app.put("/api/admin/users/{user_id}/reset-password")
    def admin_reset_password(user_id: str, req: ResetPasswordRequest, request: Request):
        _require_admin(request)
        user = store.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        user.password_hash = hash_password(req.new_password)
        store.save_user(user)
        return {"ok": True}

    # --- Cost logging ---

    def _log_cost(
        user_id: str,
        operation: str,
        model: str,
        product_id: str = "",
        variation_id: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        video_seconds: float = 0,
        cost_usd: float = 0.0,
    ) -> None:
        record = CostRecord(
            user_id=user_id,
            operation=operation,
            model=model,
            product_id=product_id,
            variation_id=variation_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            video_seconds=video_seconds,
            cost_usd=cost_usd,
        )
        store.append_cost(record)

    # --- Product CRUD ---

    @app.post("/api/products", status_code=201)
    def create_product(req: CreateProductRequest, request: Request) -> Product:
        _get_current_user(request)
        product = Product(name=req.name)
        store.save_product(product)
        return product

    @app.get("/api/products")
    def list_products_endpoint(request: Request) -> list[Product]:
        _get_current_user(request)
        return store.list_products()

    @app.get("/api/products/{product_id}")
    def get_product(product_id: str, request: Request) -> Product:
        _get_current_user(request)
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product

    @app.delete("/api/products/{product_id}", status_code=204)
    def delete_product(product_id: str, request: Request) -> None:
        _get_current_user(request)
        if not store.delete_product(product_id):
            raise HTTPException(status_code=404, detail="Product not found")

    # --- Sketch upload ---

    @app.post("/api/products/{product_id}/sketches", status_code=201)
    async def upload_sketch(product_id: str, file: UploadFile, request: Request) -> Sketch:
        _get_current_user(request)
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
    def delete_sketch(product_id: str, sketch_id: str, request: Request) -> None:
        _get_current_user(request)
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

    _VLM_MODEL = "gemini_flash_lite"

    _ANALYZE_PROMPT = (
        "Analyze this shoe design sketch. Provide:\n"
        "1. A comma-separated list of visual feature tags (e.g. pointed toe, stiletto heel, ankle strap)\n"
        "2. A brief description of the design.\n\n"
        "Format your response exactly as:\n"
        "LABELS: tag1, tag2, tag3\n"
        "DESCRIPTION: Your description here"
    )

    def _run_analysis(product: Product, job_id: str, user_id: str = "") -> None:
        """Background thread: runs VLM analysis on product sketches."""
        try:
            import torch
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
            # Reset CUDA state on failure to prevent corrupted context
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            job_manager.fail(job_id, str(e))

    @app.post("/api/products/{product_id}/analyze", status_code=202)
    def analyze_product(product_id: str, request: Request) -> dict:
        user = _get_current_user(request)
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
            args=(product, job_id, user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    # --- Agent & Pipeline info ---

    @app.get("/api/agents")
    def list_agents(request: Request) -> list[AgentInfo]:
        _get_current_user(request)
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
    def get_agent(agent_name: str, request: Request) -> AgentConfigResponse:
        _get_current_user(request)
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
    def create_agent(req: AgentConfigRequest, request: Request) -> AgentConfigResponse:
        _get_current_user(request)
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
    def update_agent(agent_name: str, req: AgentConfigRequest, request: Request) -> AgentConfigResponse:
        _get_current_user(request)
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
    def delete_agent(agent_name: str, request: Request) -> None:
        _get_current_user(request)
        from casadei import load_agent

        for yaml_file in agents_dir.glob("*.yaml"):
            config = load_agent(yaml_file)
            if config.name == agent_name:
                yaml_file.unlink()
                return
        raise HTTPException(status_code=404, detail="Agent not found")

    @app.post("/api/agents/{agent_name}/duplicate", status_code=201)
    def duplicate_agent(agent_name: str, req: DuplicateAgentRequest, request: Request) -> AgentConfigResponse:
        _get_current_user(request)
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
    def list_models(request: Request) -> list[dict]:
        _get_current_user(request)
        from casadei.models.registry import default_registry
        result = []
        for name in default_registry.list_models():
            cls = default_registry.get(name)
            all_params = cls.get_all_params()
            result.append({"name": name, "default_params": all_params})
        return result

    def _list_pipelines() -> list[PipelinePreset]:
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

    @app.get("/api/pipelines")
    def list_pipelines(request: Request) -> list[PipelinePreset]:
        _get_current_user(request)
        return _list_pipelines()

    def _get_pipeline(pipeline_id: str) -> PipelineDetailResponse:
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

    @app.get("/api/pipelines/{pipeline_id}")
    def get_pipeline(pipeline_id: str, request: Request) -> PipelineDetailResponse:
        _get_current_user(request)
        return _get_pipeline(pipeline_id)

    @app.post("/api/pipelines", status_code=201)
    def create_pipeline(req: PipelineCreateRequest, request: Request) -> PipelineDetailResponse:
        _get_current_user(request)
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

        return _get_pipeline(req.id)

    @app.put("/api/pipelines/{pipeline_id}")
    def update_pipeline(pipeline_id: str, req: PipelineUpdateRequest, request: Request) -> PipelineDetailResponse:
        _get_current_user(request)
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

        return _get_pipeline(pipeline_id)

    @app.delete("/api/pipelines/{pipeline_id}", status_code=204)
    def delete_pipeline(pipeline_id: str, request: Request) -> None:
        _get_current_user(request)
        import shutil
        pipeline_dir = workflows_dir / pipeline_id
        if not pipeline_dir.exists():
            raise HTTPException(status_code=404, detail="Pipeline not found")
        shutil.rmtree(pipeline_dir)

    @app.post("/api/pipelines/{pipeline_id}/agents", status_code=201)
    def create_pipeline_local_agent(
        pipeline_id: str, req: AgentConfigRequest, request: Request,
    ) -> AgentConfigResponse:
        _get_current_user(request)
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
    def delete_pipeline_local_agent(pipeline_id: str, agent_name: str, request: Request) -> None:
        _get_current_user(request)
        pipeline_dir = workflows_dir / pipeline_id
        local_yaml = pipeline_dir / "agents" / f"{agent_name}.yaml"
        if not local_yaml.exists():
            raise HTTPException(status_code=404, detail="Local agent not found")
        local_yaml.unlink()

    @app.post("/api/pipelines/{pipeline_id}/scripts/{filename}/open", status_code=200)
    def open_pipeline_script(pipeline_id: str, filename: str, request: Request) -> dict:
        _get_current_user(request)
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
        user_id: str = "",
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
        request: Request = None,
    ) -> RunResponse:
        user = _get_current_user(request)
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
            args=(run_id, type, name, vars_dict, named_images, job_id, user.id),
            daemon=True,
        )
        thread.start()

        return RunResponse(job_id=job_id, run_id=run_id)

    # --- Generation ---

    def _run_generation(
        product: Product,
        generation: Generation,
        job_id: str,
        user_id: str = "",
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
            for p in _list_pipelines():
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
    def start_generation(product_id: str, req: GenerateRequest, request: Request) -> dict:
        user = _get_current_user(request)
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
            args=(product, generation, job_id, user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "generation_id": generation.id}

    @app.get("/api/jobs/active")
    def list_active_jobs(request: Request) -> list[dict]:
        _get_current_user(request)
        return [
            {
                "job_id": j.job_id,
                "status": j.status,
                "progress": j.progress,
                "message": j.message,
                "product_id": j.product_id,
                "generation_id": j.generation_id,
            }
            for j in job_manager.list_active()
        ]

    @app.get("/api/jobs/all")
    def list_all_jobs(request: Request) -> list[dict]:
        _get_current_user(request)
        jobs = job_manager.list_all()
        return [
            {
                "job_id": j.job_id,
                "product_id": j.product_id,
                "generation_id": j.generation_id,
                "status": j.status,
                "progress": j.progress,
                "message": j.message,
                "error": j.error,
                "detail": j.detail,
            }
            for j in jobs
        ]

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, request: Request) -> dict:
        _get_current_user(request)
        success = job_manager.cancel(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not found or already finished")
        return {"status": "cancelled"}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, request: Request) -> dict:
        _get_current_user(request)
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
                                "detail": current.detail,
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
    def refine_generation(generation_id: str, req: RefineRequest, request: Request) -> dict:
        user = _get_current_user(request)
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
                        args=(product, new_gen, job_id, user.id),
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
        user_id: str = "",
    ) -> None:
        """Background thread: generates images for a variation."""
        if variation.pipeline == "sketch_to_shoe_gemini":
            return _run_variation_gemini(
                product, variation, job_id, change_request, user_id
            )
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
            for p in _list_pipelines():
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

    def _run_variation_gemini(
        product: Product,
        variation: Variation,
        job_id: str,
        change_request: str = "",
        user_id: str = "",
    ) -> None:
        """Background thread: Gemini sketch-to-shoe direct generation with camera angle judge."""
        try:
            import sys as _sys
            from PIL import Image as PILImage
            from casadei import ImageMedia, LoggedPipeline

            # Add workflow paths for imports
            _project_root = Path(__file__).resolve().parent.parent.parent.parent
            _workflows_dir = _project_root / "workflows"
            for p in [str(_workflows_dir / "shared"), str(_workflows_dir / "sketch_to_shoe" / "scripts")]:
                if p not in _sys.path:
                    _sys.path.insert(0, p)

            from workflows.sketch_to_shoe_gemini.pipeline import build_pipeline
            from image_utils import build_sketch_grid
            from judge import VLMSession

            job_manager.update_progress(job_id, 0.05, "Loading sketches...")

            # Load sketch images
            sketch_dir = uploads_dir / product.id
            raw_sketches = []
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    raw_sketches.append(PILImage.open(f).convert("RGB"))

            if not raw_sketches:
                job_manager.fail(job_id, "No sketch images found")
                return

            sketch_grid = build_sketch_grid(raw_sketches)
            sketch_media = ImageMedia(image=sketch_grid)

            # Build material spec from variation fields
            # Frontend may send a single prompt in note, or separate material/color fields
            material_parts = []
            if variation.color:
                material_parts.append(variation.color)
            if variation.material:
                material_parts.append(variation.material)
            if variation.note and not material_parts:
                # Single-prompt mode: note IS the full description
                material_str = variation.note
            else:
                material_str = " ".join(material_parts) if material_parts else "black patent leather"

            extra_spec: dict[str, str] = {}
            if variation.note and material_parts:
                # Only pass note as extra if material/color are also present
                extra_spec["note"] = variation.note
            if change_request:
                extra_spec["change_request"] = change_request

            # Load reference images if available (may be uploaded concurrently)
            import time as _time
            ref_images: dict[str, "ImageMedia"] = {}
            var_results_dir = results_dir / product.id / variation.id
            var_results_dir.mkdir(parents=True, exist_ok=True)
            # Brief wait for concurrent ref image uploads to land
            _time.sleep(1.5)
            # Re-read variation in case ref images were uploaded after creation
            _fresh = store.get_product(product.id)
            if _fresh:
                for _v in _fresh.variations:
                    if _v.id == variation.id:
                        variation.material_image = _v.material_image
                        variation.color_image = _v.color_image
                        break
            if variation.material_image:
                ref_path = var_results_dir / variation.material_image
                if ref_path.exists():
                    ref_images["material_ref"] = ImageMedia(
                        image=PILImage.open(ref_path).convert("RGB")
                    )
            if variation.color_image:
                ref_path = var_results_dir / variation.color_image
                if ref_path.exists():
                    ref_images["color_ref"] = ImageMedia(
                        image=PILImage.open(ref_path).convert("RGB")
                    )

            # Load material images if available (material-image mode)
            materials_list = None
            materials_meta_path = var_results_dir / "materials_meta.json"
            if materials_meta_path.exists():
                import json as _json
                meta_list = _json.loads(materials_meta_path.read_text())
                if meta_list:
                    materials_list = []
                    for entry in sorted(meta_list, key=lambda e: e["index"]):
                        img_path = var_results_dir / f"ref_mat_{entry['index']}.png"
                        if img_path.exists():
                            materials_list.append({
                                "name": entry.get("name"),
                                "image": PILImage.open(img_path).convert("RGB"),
                                "placement": entry.get("placement"),
                                "note": entry.get("note"),
                                "is_color": entry.get("is_color", False),
                            })

            spec = {
                "material": material_str,
                "camera_angle": "3/4",
                "extra": extra_spec,
                "ref_images": ref_images,
            }
            if materials_list:
                spec["materials"] = materials_list

            NUM_HERO_CANDIDATES = 2

            job_manager.update_progress(job_id, 0.05, "Generating hero candidates concurrently...")
            var_results_dir.mkdir(parents=True, exist_ok=True)

            from casadei.loop import LoopResult as _LoopResult
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Track progress across concurrent runs
            _run_progress: dict[int, float] = {}
            _progress_lock = threading.Lock()

            def _update_combined_progress():
                with _progress_lock:
                    if _run_progress:
                        avg = sum(_run_progress.values()) / NUM_HERO_CANDIDATES
                        pct = 0.05 + avg * 0.85
                    else:
                        pct = 0.05
                job_manager.update_progress(job_id, min(pct, 0.89), "Generating candidates...")

            def _run_single_candidate(run_idx: int) -> list[HeroCandidate]:
                """Run one candidate pipeline and return its HeroCandidate list."""
                run_candidates: list[HeroCandidate] = []

                vlm_session = VLMSession("gemini_flash_lite")
                try:
                    pipeline, edit_agent, _vlm_sessions, grid_image = build_pipeline(
                        spec=spec,
                        vlm_session=vlm_session,
                        foot="pair",
                        temperature=0.8,
                    )
                    logged = LoggedPipeline(pipeline)
                    context: dict = {
                        "sketch": sketch_media,
                        "image": sketch_media,
                    }
                    if grid_image is not None:
                        context["materials_grid"] = ImageMedia(image=grid_image)
                    else:
                        for ref_key, ref_media in ref_images.items():
                            context[ref_key] = ref_media

                    # Progress monitor for this run
                    _progress_done = threading.Event()
                    def _make_monitor(idx=run_idx, done=_progress_done):
                        def _monitor():
                            step = 0
                            while not done.is_set():
                                step += 1
                                with _progress_lock:
                                    _run_progress[idx] = min(step * 0.25, 0.95)
                                _update_combined_progress()
                                done.wait(timeout=20)
                        return _monitor

                    monitor = threading.Thread(target=_make_monitor(), daemon=True)
                    monitor.start()

                    try:
                        result, exec_log = logged.run(context)
                    finally:
                        _progress_done.set()
                        monitor.join(timeout=5)
                        with _progress_lock:
                            _run_progress[run_idx] = 1.0
                        _update_combined_progress()

                finally:
                    vlm_session.unload()

                # Log costs
                if hasattr(edit_agent, "token_usage_log"):
                    for usage_entry in edit_agent.token_usage_log:
                        cost = calculate_cost(usage_entry.get("model", ""), usage_entry)
                        _log_cost(user_id=user_id, operation="variation", model=usage_entry.get("model", ""),
                                  product_id=product.id, variation_id=variation.id,
                                  input_tokens=usage_entry.get("input_tokens", 0),
                                  output_tokens=usage_entry.get("output_tokens", 0),
                                  thinking_tokens=usage_entry.get("thinking_tokens", 0), cost_usd=cost)

                # Save ALL iteration images as candidates
                loop_history = result.get("angle_correction_loop_history")
                if isinstance(loop_history, _LoopResult) and loop_history.iterations:
                    for it in loop_history.iterations:
                        it_img = it.outputs.get("image")
                        if it_img is None or not isinstance(it_img, ImageMedia):
                            continue
                        spec_avg = it.metadata.get("spec_avg") or 0.0
                        sketch_avg = it.metadata.get("sketch_avg") or 0.0
                        it_score = (spec_avg + sketch_avg) / 2.0 if (spec_avg or sketch_avg) else 0.0

                        c_fname = f"hero_candidate_{run_idx}_{it.index}.png"
                        it_img.image.save(var_results_dir / c_fname)
                        run_candidates.append(HeroCandidate(
                            filename=c_fname,
                            iteration=run_idx,
                            accepted=it.accepted,
                            score=round(it_score, 3),
                            feedback=it.feedback[:200],
                            selected=False,
                        ))
                else:
                    # Fallback: save just the final image if no loop history
                    run_img = result.get("image")
                    if run_img is not None and isinstance(run_img, ImageMedia):
                        c_fname = f"hero_candidate_{run_idx}.png"
                        run_img.image.save(var_results_dir / c_fname)
                        run_candidates.append(HeroCandidate(
                            filename=c_fname,
                            iteration=run_idx,
                            accepted=False,
                            score=0.0,
                            feedback="",
                            selected=False,
                        ))

                return run_candidates

            # Run all candidates concurrently
            candidates: list[HeroCandidate] = []
            with ThreadPoolExecutor(max_workers=NUM_HERO_CANDIDATES) as executor:
                futures = {
                    executor.submit(_run_single_candidate, idx): idx
                    for idx in range(NUM_HERO_CANDIDATES)
                }
                for future in as_completed(futures):
                    try:
                        run_candidates = future.result()
                        candidates.extend(run_candidates)
                    except Exception as e:
                        logger.warning(f"Candidate {futures[future]} failed: {e}")

            # Find the best candidate
            best_score = -1.0
            best_candidate_idx = 0
            for i, c in enumerate(candidates):
                if c.score > best_score:
                    best_score = c.score
                    best_candidate_idx = i

            if not candidates:
                job_manager.fail(job_id, "All generation runs produced no output")
                return

            # Mark the best candidate
            candidates[best_candidate_idx].selected = True

            job_manager.update_progress(job_id, 0.92, "Saving hero image...")

            # Copy best candidate as hero.png
            import shutil
            best_fname = candidates[best_candidate_idx].filename
            shutil.copy2(var_results_dir / best_fname, var_results_dir / "hero.png")

            variation.results = [ResultFile(filename="hero.png")]
            variation.hero_candidates = candidates
            variation.status = JobStatus.completed
            store.save_product(product)

            job_manager.complete(job_id)

        except Exception as e:
            variation.status = JobStatus.failed
            store.save_product(product)
            job_manager.fail(job_id, str(e))

    @app.post("/api/products/{product_id}/variations", status_code=202)
    def create_variation(
        product_id: str, req: CreateVariationRequest, request: Request,
    ) -> dict:
        user = _get_current_user(request)
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
            material_image=req.material_image,
            color_image=req.color_image,
            pipeline=req.pipeline,
            num_outputs=req.num_outputs,
            status=JobStatus.pending,
        )
        product.variations.append(variation)
        store.save_product(product)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_variation,
            args=(product, variation, job_id, "", user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id, "variation_id": variation.id}

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/ref-image",
        status_code=200,
    )
    async def upload_variation_ref_image(
        product_id: str,
        variation_id: str,
        file: UploadFile,
        kind: str = fastapi.Query(..., regex="^(material|color)$"),
        request: Request = None,
    ):
        """Upload a reference image (material or color) for a variation."""
        _get_current_user(request)
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

        var_dir = results_dir / product_id / variation_id
        var_dir.mkdir(parents=True, exist_ok=True)
        filename = f"ref_{kind}.png"
        dest = var_dir / filename
        content = await file.read()
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(content)).convert("RGB")
        img.save(str(dest), "PNG")

        if kind == "material":
            variation.material_image = filename
        else:
            variation.color_image = filename
        store.save_product(product)

        return {"filename": filename}

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/ref-material",
        status_code=200,
    )
    async def upload_variation_ref_material(
        product_id: str,
        variation_id: str,
        file: UploadFile,
        index: int = fastapi.Query(...),
        name: str = fastapi.Query(""),
        placement: str = fastapi.Query(""),
        note: str = fastapi.Query(""),
        is_color: bool = fastapi.Query(False),
        request: Request = None,
    ):
        """Upload a material/color reference image for a variation."""
        _get_current_user(request)
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

        var_dir = results_dir / product_id / variation_id
        var_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        filename = f"ref_mat_{index}.png"
        content = await file.read()
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(content)).convert("RGB")
        img.save(str(var_dir / filename), "PNG")

        # Update materials_meta.json
        import json as _json
        meta_path = var_dir / "materials_meta.json"
        meta_list = []
        if meta_path.exists():
            meta_list = _json.loads(meta_path.read_text())

        # Remove existing entry with same index (overwrite)
        meta_list = [e for e in meta_list if e["index"] != index]
        meta_list.append({
            "index": index,
            "name": name or None,
            "placement": placement or None,
            "note": note or None,
            "is_color": is_color,
        })
        meta_list.sort(key=lambda e: e["index"])
        meta_path.write_text(_json.dumps(meta_list, indent=2))

        return {"filename": filename, "index": index}

    @app.delete(
        "/api/products/{product_id}/variations/{variation_id}/ref-material/{index}",
        status_code=200,
    )
    def delete_variation_ref_material(
        product_id: str,
        variation_id: str,
        index: int,
        request: Request = None,
    ):
        """Delete a material/color reference image from a variation."""
        _get_current_user(request)
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

        var_dir = results_dir / product_id / variation_id

        # Remove image file
        img_path = var_dir / f"ref_mat_{index}.png"
        if img_path.exists():
            img_path.unlink()

        # Update materials_meta.json
        import json as _json
        meta_path = var_dir / "materials_meta.json"
        if meta_path.exists():
            meta_list = _json.loads(meta_path.read_text())
            meta_list = [e for e in meta_list if e["index"] != index]
            if meta_list:
                meta_path.write_text(_json.dumps(meta_list, indent=2))
            else:
                meta_path.unlink()

        return {"deleted": index}

    @app.post(
        "/api/products/{product_id}/variations/from-result",
        status_code=201,
    )
    def promote_to_variation(
        product_id: str, req: PromoteToVariationRequest, request: Request,
    ) -> dict:
        """Create a new variation by copying an existing generated result as its hero."""
        _get_current_user(request)
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Find source variation
        source_var = None
        for v in product.variations:
            if v.id == req.source_variation_id:
                source_var = v
                break
        if not source_var:
            raise HTTPException(status_code=404, detail="Source variation not found")

        # Find the source file
        source_path = results_dir / product_id / source_var.id / req.filename
        if not source_path.exists():
            raise HTTPException(status_code=404, detail="Source image not found")

        # Create new variation
        new_variation = Variation(
            note=req.note or source_var.note,
            pipeline=source_var.pipeline,
            status=JobStatus.completed,
        )

        # Copy the image as hero.png
        new_results_dir = results_dir / product_id / new_variation.id
        new_results_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dest = new_results_dir / "hero.png"
        shutil.copy2(source_path, dest)

        new_variation.results = [ResultFile(filename="hero.png")]
        product.variations.append(new_variation)
        store.save_product(product)

        return {"variation_id": new_variation.id}

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/regenerate",
        status_code=202,
    )
    def regenerate_variation(
        product_id: str,
        variation_id: str,
        req: RegenerateVariationRequest,
        request: Request,
    ) -> dict:
        user = _get_current_user(request)
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
        if req.note is not None:
            variation.note = req.note
            variation.material = ""
            variation.color = ""
        store.save_product(product)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_variation,
            args=(product, variation, job_id, req.change_request, user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    # --- Generate all angles from hero image ---

    def _run_generate_angles(
        product: Product,
        variation: Variation,
        job_id: str,
        foot: str = "right",
        single: bool = False,
        selected_angles: list[str] | None = None,
        judged: bool = True,
        user_id: str = "",
    ) -> None:
        """Background thread: generates all camera angles from the hero image."""
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from PIL import Image as PILImage
            from google import genai

            from workflows.shoe_angles.pipeline import (
                generate_angle,
                generate_angle_with_judge,
                get_canonical_angles,
            )

            job_manager.update_progress(job_id, 0.05, "Loading images...")

            # Load sketch
            sketch_dir = uploads_dir / product.id
            sketch_imgs = []
            for sketch in product.sketches:
                for f in sketch_dir.glob(f"{sketch.id}_*"):
                    sketch_imgs.append(PILImage.open(f).convert("RGB"))
            if not sketch_imgs:
                job_manager.fail(job_id, "No sketch images found")
                return

            sketch = sketch_imgs[0]

            # Load hero (first result)
            var_results_dir = results_dir / product.id / variation.id
            if not variation.results:
                job_manager.fail(job_id, "No hero image to use as reference")
                return

            hero_path = var_results_dir / variation.results[0].filename
            if not hero_path.exists():
                job_manager.fail(job_id, f"Hero image not found: {hero_path}")
                return

            reference = PILImage.open(hero_path).convert("RGB")

            job_manager.update_progress(job_id, 0.1, "Generating angles...")

            client = genai.Client()
            generated_images: dict[str, PILImage.Image] = {}

            canonical = get_canonical_angles()
            angles_to_generate = selected_angles if selected_angles is not None else canonical

            total = len(angles_to_generate)
            done = 0

            if judged:
                # Judged mode: sequential with 3 parallel judges per angle
                output_dir = var_results_dir / "_angle_judge_logs"
                output_dir.mkdir(parents=True, exist_ok=True)

                for angle in angles_to_generate:
                    if job_manager.is_cancelled(job_id):
                        break
                    try:
                        _, img, cost_usd = generate_angle_with_judge(
                            client, sketch, reference, angle, foot, single=single,
                            output_dir=output_dir,
                        )
                        generated_images[angle] = img
                        prefix = "single_" if single else ""
                        fname = f"{prefix}{angle.replace('/', '_')}.png"
                        img.save(var_results_dir / fname)
                        _log_cost(user_id=user_id, operation="angles", model="gemini-3.1-flash-image-preview",
                                  product_id=product.id, variation_id=variation.id, cost_usd=cost_usd)
                    except Exception:
                        pass
                    done += 1
                    job_manager.update_progress(
                        job_id,
                        0.1 + 0.7 * (done / total),
                        f"Generated {done}/{total} angles (judged)...",
                    )
            else:
                # Fast mode: parallel generation without judges
                def _gen_angle(angle: str) -> tuple[str, PILImage.Image | None]:
                    try:
                        if job_manager.is_cancelled(job_id):
                            return angle, None
                        _, img, _usage = generate_angle(client, sketch, reference, angle, foot, single=single)
                        return angle, img
                    except Exception:
                        return angle, None

                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = {pool.submit(_gen_angle, a): a for a in angles_to_generate}
                    for fut in as_completed(futures):
                        angle, img = fut.result()
                        done += 1
                        if img is not None:
                            generated_images[angle] = img
                            prefix = "single_" if single else ""
                            fname = f"{prefix}{angle.replace('/', '_')}.png"
                            img.save(var_results_dir / fname)
                            _log_cost(user_id=user_id, operation="angles", model="gemini-3.1-flash-image-preview",
                                      product_id=product.id, variation_id=variation.id, cost_usd=0.0)
                        job_manager.update_progress(
                            job_id,
                            0.1 + 0.7 * (done / total),
                            f"Generated {done}/{total} angles...",
                        )

            if job_manager.is_cancelled(job_id):
                return

            # Update variation results
            job_manager.update_progress(job_id, 0.9, "Saving results...")
            result_files = []
            # Keep original hero first
            for rf in variation.results:
                result_files.append(rf)
            # Add all new angle images
            for angle in angles_to_generate:
                prefix = "single_" if single else ""
                fname = f"{prefix}{angle.replace('/', '_')}.png"
                if (var_results_dir / fname).exists():
                    result_files.append(ResultFile(filename=fname))

            variation.results = result_files
            variation.status = JobStatus.completed
            store.save_product(product)
            job_manager.complete(job_id)

        except Exception as e:
            variation.status = JobStatus.failed
            store.save_product(product)
            job_manager.fail(job_id, str(e))

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/generate-angles",
        status_code=202,
    )
    def generate_angles(
        product_id: str,
        variation_id: str,
        foot: str = Query("right", pattern="^(left|right)$"),
        single: bool = Query(False),
        angles: list[str] = Query(None, alias="angle"),
        judged: bool = Query(True),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
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

        if not variation.results:
            raise HTTPException(
                status_code=400, detail="No hero image — generate one first"
            )

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_generate_angles,
            args=(product, variation, job_id),
            kwargs={"foot": foot, "single": single, "selected_angles": angles, "judged": judged, "user_id": user.id},
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_add_image(
        product: Product,
        variation: Variation,
        job_id: str,
        change_request: str,
        user_id: str = "",
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
            for p in _list_pipelines():
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
        request: Request,
    ) -> dict:
        user = _get_current_user(request)
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
            args=(product, variation, job_id, req.change_request, user.id),
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
        source: list[str] = fastapi.Query(default=[]),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
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
            args=(product, variation, job_id, source or None, user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_generate_360(
        prod: Product,
        var: Variation,
        jid: str,
        source_filenames: list[str] | None = None,
        user_id: str = "",
    ) -> None:
        """Background thread: generates a 360° spin video via Veo."""
        try:
            var_results_dir = results_dir / prod.id / var.id
            _run_generate_360_veo(prod, var, jid, var_results_dir, source_filenames, user_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            var.status = JobStatus.failed
            store.save_product(prod)
            job_manager.fail(jid, str(e))

    def _run_generate_360_veo(
        prod: Product,
        var: Variation,
        jid: str,
        var_results_dir: Path,
        source_filenames: list[str] | None = None,
        user_id: str = "",
    ) -> None:
        """Generate a 360° spin video using Google Veo."""
        import io
        import time as _time

        from PIL import Image as PILImage

        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError(
                "google-genai is required for 360 video. "
                "Install: pip install google-genai"
            )

        job_manager.update_progress(jid, 0.05, "Loading source images...")

        # Collect source images
        source_paths: list[Path] = []
        if source_filenames:
            for fn in source_filenames:
                p = var_results_dir / fn
                if p.exists():
                    source_paths.append(p)
        if not source_paths:
            # Default: prefer front-facing angles
            for candidate in ["front.png", "hero-front-left.png", "hero-front-right.png"]:
                p = var_results_dir / candidate
                if p.exists():
                    source_paths.append(p)
                    break
            if not source_paths:
                source_paths.append(var_results_dir / var.results[0].filename)

        source_imgs = [PILImage.open(p).convert("RGB") for p in source_paths]
        source_img = source_imgs[0]  # primary frame

        # Pad to 16:9
        job_manager.update_progress(jid, 0.1, "Preparing image for Veo...")
        import numpy as np

        w, h = source_img.size
        target_ratio = 16 / 9
        current_ratio = w / h
        if current_ratio < target_ratio:
            new_w = round(h * target_ratio)
            # Sample background from corner regions
            arr = np.array(source_img)
            corners = np.concatenate([
                arr[:20, :20].reshape(-1, 3),
                arr[:20, -20:].reshape(-1, 3),
                arr[-20:, :20].reshape(-1, 3),
                arr[-20:, -20:].reshape(-1, 3),
            ])
            bg = tuple(int(v) for v in corners.mean(axis=0))
            canvas = PILImage.new("RGB", (new_w, h), bg)
            canvas.paste(source_img, ((new_w - w) // 2, 0))
        else:
            canvas = source_img
        source_img = canvas

        # Convert to bytes
        buf = io.BytesIO()
        source_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        job_manager.update_progress(jid, 0.15, "Starting Veo video generation...")

        client = genai.Client()
        prompt = (
            "Fast-spinning product turntable video. "
            "The shoe shown in the image spins rapidly on a motorized turntable. "
            "The turntable completes exactly one full 360-degree revolution at constant speed. "
            "The rotation is quick, steady, and never stops or reverses. "
            "Fixed camera, clean gray studio background, soft product lighting. "
            "Professional e-commerce turntable spin, photorealistic."
        )

        operation = client.models.generate_videos(
            model="veo-3.1-fast-generate-preview",
            prompt=prompt,
            image=types.Image(
                image_bytes=img_bytes,
                mime_type="image/png",
            ),
            config=types.GenerateVideosConfig(
                aspect_ratio="16:9",
                resolution="4k",
                last_frame=types.Image(
                    image_bytes=img_bytes,
                    mime_type="image/png",
                ),
            ),
        )

        # Poll until done
        poll_count = 0
        retries = 0
        while not operation.done:
            poll_count += 1
            job_manager.update_progress(
                jid,
                min(0.15 + poll_count * 0.05, 0.85),
                "Generating video... (this may take a minute)",
            )
            _time.sleep(10)
            try:
                operation = client.operations.get(operation)
                retries = 0
            except Exception:
                retries += 1
                if retries >= 5:
                    raise
                _time.sleep(5)

        response = operation.response
        if not response or not response.generated_videos:
            raise RuntimeError(
                f"Veo video generation failed or was filtered. Response: {response}"
            )

        job_manager.update_progress(jid, 0.9, "Downloading and saving video...")

        generated_video = response.generated_videos[0]
        client.files.download(file=generated_video.video)
        video_bytes = generated_video.video.video_bytes

        var_results_dir.mkdir(parents=True, exist_ok=True)
        video_filename = "spin_360.mp4"
        (var_results_dir / video_filename).write_bytes(video_bytes)

        video_duration = 8.0
        cost = calculate_veo_cost("veo-3.1-fast-generate-preview", "4k", video_duration)
        _log_cost(user_id=user_id, operation="360_spin", model="veo-3.1-fast-generate-preview",
                  product_id=prod.id, variation_id=var.id,
                  video_seconds=video_duration, cost_usd=cost)

        var.spin_video = video_filename
        store.save_product(prod)

        job_manager.update_progress(jid, 1.0, "Done!")
        job_manager.complete(jid)

    @app.get(
        "/api/products/{product_id}/variations/{variation_id}/spin-video",
    )
    def serve_spin_video(product_id: str, variation_id: str, request: Request):
        _get_current_user(request)
        video_path = results_dir / product_id / variation_id / "spin_360.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Spin video not found")
        return FileResponse(video_path, media_type="video/mp4")

    # ----------------------------------------------------------------
    # Hunyuan3D 2.1 — 3D model generation
    # ----------------------------------------------------------------

    _hunyuan3d_provider = None

    def _get_hunyuan3d():
        """Lazy-load the Hunyuan3D provider."""
        nonlocal _hunyuan3d_provider
        if _hunyuan3d_provider is None:
            from casadei.providers.hunyuan3d import Hunyuan3DProvider
            repo_path = os.environ.get("HUNYUAN3D_REPO_PATH", "")
            _hunyuan3d_provider = Hunyuan3DProvider(
                hunyuan3d_repo=repo_path if repo_path else None
            )
            _hunyuan3d_provider.load_model()
        return _hunyuan3d_provider

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/generate-3d",
        status_code=202,
    )
    def generate_3d(
        product_id: str,
        variation_id: str,
        source: str = fastapi.Query(default=""),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
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

        if not variation.results:
            raise HTTPException(
                status_code=400,
                detail="Variation has no rendered images to generate 3D from",
            )

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_generate_3d,
            args=(product, variation, job_id, source or None, user.id),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_generate_3d(
        prod: Product,
        var: Variation,
        jid: str,
        source_filename: str | None = None,
        user_id: str = "",
    ) -> None:
        """Background thread: generates a 3D mesh via Hunyuan3D 2.1."""
        try:
            var_results_dir = results_dir / prod.id / var.id
            var_results_dir.mkdir(parents=True, exist_ok=True)

            job_manager.update_progress(jid, 0.05, "Loading source image...")

            # Pick source image
            source_path: Path | None = None
            if source_filename:
                p = var_results_dir / source_filename
                if p.exists():
                    source_path = p
            if not source_path:
                # Prefer front-facing angles
                for candidate in ["front.png", "3_4.png", "hero-front-left.png"]:
                    p = var_results_dir / candidate
                    if p.exists():
                        source_path = p
                        break
                if not source_path:
                    source_path = var_results_dir / var.results[0].filename

            if not source_path.exists():
                raise FileNotFoundError(f"Source image not found: {source_path}")

            job_manager.update_progress(jid, 0.1, "Loading Hunyuan3D model...")

            provider = _get_hunyuan3d()

            job_manager.update_progress(jid, 0.2, "Generating 3D shape...")

            output_dir = var_results_dir / "_3d_temp"
            output_dir.mkdir(parents=True, exist_ok=True)

            glb_path = provider.generate(
                image_path=source_path,
                output_dir=output_dir,
                num_inference_steps=30,
                guidance_scale=7.5,
                octree_resolution=256,
                texture=True,
            )

            job_manager.update_progress(jid, 0.9, "Saving 3D model...")

            # Move final GLB to variation directory
            final_glb = var_results_dir / "model_3d.glb"
            if glb_path != final_glb:
                import shutil
                shutil.move(str(glb_path), str(final_glb))

            # Clean up temp dir
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)

            # Update variation
            var.model_3d = "model_3d.glb"
            store.save_product(prod)

            # Log cost (local GPU — no API cost, but track for usage stats)
            _log_cost(
                user_id=user_id,
                operation="3d_model",
                model="hunyuan3d-2.1",
                product_id=prod.id,
                variation_id=var.id,
                cost_usd=0.0,
            )

            job_manager.update_progress(jid, 1.0, "3D model ready!")
            job_manager.complete(jid)

        except Exception as e:
            import traceback
            traceback.print_exc()
            var.status = JobStatus.failed
            store.save_product(prod)
            job_manager.fail(jid, str(e))

    @app.get(
        "/api/products/{product_id}/variations/{variation_id}/model-3d",
    )
    def serve_model_3d(product_id: str, variation_id: str):
        var_dir = results_dir / product_id / variation_id
        # Check multiple possible locations
        for candidate in [
            var_dir / "model_3d.glb",
            var_dir / "_3d_hq" / "model_3d.glb",
        ]:
            if candidate.exists():
                return FileResponse(candidate, media_type="model/gltf-binary")
        raise HTTPException(status_code=404, detail="3D model not found")

    # ----------------------------------------------------------------
    # Gemini-powered variation tools: try-on, context, accessories
    # ----------------------------------------------------------------

    def _gemini_edit(images: list, prompt: str) -> tuple[bytes, dict]:
        """Call Gemini Flash Image Edit and return the result as PNG bytes."""
        import io as _io

        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise RuntimeError("google-genai required. pip install google-genai")

        client = genai.Client()

        # Convert PIL images to padded versions at 1K
        _RATIOS = [
            (1, 1), (1, 4), (1, 8), (2, 3), (3, 2), (3, 4), (4, 3),
            (4, 5), (5, 4), (8, 1), (9, 16), (16, 9), (21, 9),
        ]

        def find_ratio(w, h):
            target = w / h
            return min(_RATIOS, key=lambda r: abs(r[0] / r[1] - target))

        def pad(img, ratio):
            from PIL import Image as PILImage
            wr, hr = ratio
            ow, oh = img.size
            if ow / oh <= wr / hr:
                cw, ch = round(oh * wr / hr), oh
            else:
                cw, ch = ow, round(ow * hr / wr)
            scale = 1024 / max(cw, ch)
            fw, fh = round(cw * scale), round(ch * scale)
            iw, ih = round(ow * scale), round(oh * scale)
            scaled = img.resize((iw, ih), PILImage.LANCZOS)
            canvas = PILImage.new("RGB", (fw, fh), (255, 255, 255))
            canvas.paste(scaled, ((fw - iw) // 2, (fh - ih) // 2))
            return canvas

        from PIL import Image as PILImage
        ratio = find_ratio(*images[0].size)
        padded = [pad(img, ratio) for img in images]
        aspect_str = f"{ratio[0]}:{ratio[1]}"

        response = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[prompt] + padded,
            config=genai_types.GenerateContentConfig(
                temperature=1.0,
                image_config=genai_types.ImageConfig(
                    image_size="1K",
                    aspect_ratio=aspect_str,
                ),
            ),
        )

        usage = extract_token_usage(getattr(response, "usage_metadata", None))

        for part in response.parts:
            if part.inline_data is not None:
                return part.inline_data.data, usage

        raise RuntimeError("Gemini returned no image — request may have been refused.")

    def _find_variation(product_id: str, variation_id: str):
        """Lookup product + variation or raise 404."""
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        for v in product.variations:
            if v.id == variation_id:
                return product, v
        raise HTTPException(status_code=404, detail="Variation not found")

    def _save_gemini_result(
        prod: Product,
        var: Variation,
        img_bytes: bytes,
        label: str,
        pipeline: str,
    ) -> str:
        """Save image bytes as a new generated result on the variation."""
        var_dir = results_dir / prod.id / var.id
        var_dir.mkdir(parents=True, exist_ok=True)

        idx = len(var.generated_results)
        fname = f"{label}_{idx}.png"
        (var_dir / fname).write_bytes(img_bytes)

        var.generated_results.append(GeneratedResult(
            filename=fname,
            pipeline=pipeline,
            label=label,
        ))
        store.save_product(prod)
        return fname

    # --- Try on Model ---

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/try-on",
        status_code=202,
    )
    async def try_on_model(
        product_id: str,
        variation_id: str,
        model_photo: UploadFile = fastapi.File(...),
        mode: str = Query("fast", pattern="^(fast|quality)$"),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
        product, variation = _find_variation(product_id, variation_id)
        if not variation.results:
            raise HTTPException(status_code=400, detail="Variation has no images")

        # Save uploaded model photo
        scratch = uploads_dir / "scratch" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        photo_path = scratch / (model_photo.filename or "model.png")
        photo_path.write_bytes(await model_photo.read())

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_try_on,
            args=(product, variation, job_id, photo_path, user.id, mode),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    def _run_try_on(
        prod: Product,
        var: Variation,
        jid: str,
        photo_path: Path,
        user_id: str = "",
        mode: str = "fast",
    ) -> None:
        try:
            from PIL import Image as PILImage

            job_manager.update_progress(jid, 0.1, "Loading images...")

            model_img = PILImage.open(photo_path).convert("RGB")
            shoe_path = results_dir / prod.id / var.id / var.results[0].filename
            shoe_img = PILImage.open(shoe_path).convert("RGB")

            if mode == "quality":
                # Quality mode: agentic loop with judge (3 iterations max)
                from casadei import ImageMedia, LoggedPipeline
                from workflows.shoe_tryon_gemini.pipeline import (
                    build_pipeline as build_tryon_pipeline,
                    VLMSession,
                    extract_features,
                )

                job_manager.update_progress(jid, 0.15, "Extracting shoe features...")

                vlm_session = VLMSession("gemini_flash_lite")
                try:
                    shoe_media = ImageMedia(image=shoe_img)
                    features = extract_features(vlm_session, shoe_media)

                    job_manager.update_progress(jid, 0.2, "Running quality try-on loop...")

                    pipeline, edit_agent = build_tryon_pipeline(
                        max_iterations=3,
                        vlm_session=vlm_session,
                        features=features,
                        tolerance="moderate",
                    )
                    logged = LoggedPipeline(pipeline)
                    person_media = ImageMedia(image=model_img)
                    context = {
                        "person": person_media,
                        "shoe": shoe_media,
                        "image": person_media,
                    }
                    result, exec_log = logged.run(context)
                finally:
                    vlm_session.unload()

                final_img = result.get("image")
                if final_img is not None and isinstance(final_img, ImageMedia):
                    import io as _io
                    buf = _io.BytesIO()
                    final_img.image.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                else:
                    job_manager.fail(jid, "Quality try-on produced no image")
                    return

                # Log costs
                token_records = vlm_session.token_usage_log + edit_agent.token_usage_log
                for usage_entry in token_records:
                    cost = calculate_cost(usage_entry.get("model", ""), usage_entry)
                    _log_cost(user_id=user_id, operation="try_on", model=usage_entry.get("model", ""),
                              product_id=prod.id, variation_id=var.id,
                              input_tokens=usage_entry.get("input_tokens", 0),
                              output_tokens=usage_entry.get("output_tokens", 0),
                              thinking_tokens=usage_entry.get("thinking_tokens", 0), cost_usd=cost)
            else:
                # Fast mode: single-call Gemini edit (no loop, no judge)
                job_manager.update_progress(jid, 0.2, "Generating try-on with Gemini...")

                desc = f"{var.material} {var.color}".strip() or "the shoe"
                prompt = (
                    f"Replace the shoes the model is wearing with {desc} shown in the "
                    f"second image. Keep the model's pose, outfit, and background exactly "
                    f"the same. Only change the shoes. The result should look like a natural "
                    f"fashion photograph. Photorealistic, high quality."
                )

                img_bytes, usage = _gemini_edit([model_img, shoe_img], prompt)

                cost = calculate_cost("gemini-3.1-flash-image-preview", usage)
                _log_cost(user_id=user_id, operation="try_on", model="gemini-3.1-flash-image-preview",
                          product_id=prod.id, variation_id=var.id,
                          input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                          thinking_tokens=usage.get("thinking_tokens", 0), cost_usd=cost)

            job_manager.update_progress(jid, 0.9, "Saving result...")
            _save_gemini_result(prod, var, img_bytes, "try-on", "gemini_tryon")

            job_manager.update_progress(jid, 1.0, "Done!")
            job_manager.complete(jid)

        except Exception as e:
            import traceback
            traceback.print_exc()
            job_manager.fail(jid, str(e))

    # --- Create Context ---

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/create-context",
        status_code=202,
    )
    def create_context(
        product_id: str,
        variation_id: str,
        scene: str = fastapi.Query(...),
        prompt: str = fastapi.Query(""),
        source: str = fastapi.Query(""),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
        product, variation = _find_variation(product_id, variation_id)
        if not variation.results:
            raise HTTPException(status_code=400, detail="Variation has no images")

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_create_context,
            args=(product, variation, job_id, scene, prompt, source, user.id),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    def _run_create_context(
        prod: Product,
        var: Variation,
        jid: str,
        scene: str,
        extra_prompt: str,
        source_filename: str = "",
        user_id: str = "",
    ) -> None:
        try:
            from PIL import Image as PILImage

            job_manager.update_progress(jid, 0.1, "Loading shoe image...")

            # Use selected source or fall back to first result
            src = source_filename or var.results[0].filename
            shoe_path = results_dir / prod.id / var.id / src
            if not shoe_path.exists():
                shoe_path = results_dir / prod.id / var.id / var.results[0].filename
            shoe_img = PILImage.open(shoe_path).convert("RGB")

            # Detect if source is a single shoe or a pair
            is_single = src.startswith("single_")

            job_manager.update_progress(jid, 0.2, f"Creating {scene} scene with Gemini...")

            SCENE_PROMPTS = {
                "store-shelf": "on a luxury boutique store shelf with elegant wood and glass display",
                "garden": "in a beautiful garden setting with lush greenery and soft natural light",
                "studio": "in a professional fashion photography studio with dramatic lighting",
                "street": "on a stylish urban street with architectural background",
                "runway": "on a fashion runway with dramatic stage lighting",
                "custom": "",
            }
            scene_desc = SCENE_PROMPTS.get(scene, scene)

            desc = f"{var.material} {var.color}".strip() or "this luxury shoe"
            if is_single:
                prompt = (
                    f"Place this single {desc} shoe {scene_desc}. "
                    f"Create a professional lifestyle product photograph. "
                    f"The shoe must be the hero of the image, clearly visible and in focus. "
                    f"Show only one shoe, elegantly positioned. "
                    f"Photorealistic, editorial quality, beautiful composition."
                )
            else:
                prompt = (
                    f"Place this pair of {desc} shoes {scene_desc}. "
                    f"Create a professional lifestyle product photograph. "
                    f"The pair of shoes must be the hero of the image, clearly visible and in focus. "
                    f"Keep both shoes together as a pair. "
                    f"Photorealistic, editorial quality, beautiful composition."
                )
            if extra_prompt:
                prompt += f" {extra_prompt}"

            img_bytes, usage = _gemini_edit([shoe_img], prompt)

            cost = calculate_cost("gemini-3.1-flash-image-preview", usage)
            _log_cost(user_id=user_id, operation="context", model="gemini-3.1-flash-image-preview",
                      product_id=prod.id, variation_id=var.id,
                      input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                      thinking_tokens=usage.get("thinking_tokens", 0), cost_usd=cost)

            job_manager.update_progress(jid, 0.9, "Saving result...")
            _save_gemini_result(prod, var, img_bytes, "context", "gemini_context")

            job_manager.update_progress(jid, 1.0, "Done!")
            job_manager.complete(jid)

        except Exception as e:
            import traceback
            traceback.print_exc()
            job_manager.fail(jid, str(e))

    # --- Apply Accessories ---

    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/apply-accessories",
        status_code=202,
    )
    async def apply_accessories(
        product_id: str,
        variation_id: str,
        accessory_images: list[UploadFile] = fastapi.File(...),
        instruction: str = fastapi.Form(""),
        request: Request = None,
    ) -> dict:
        user = _get_current_user(request)
        product, variation = _find_variation(product_id, variation_id)
        if not variation.results:
            raise HTTPException(status_code=400, detail="Variation has no images")

        # Save uploaded accessory photos
        scratch = uploads_dir / "scratch" / uuid.uuid4().hex[:12]
        scratch.mkdir(parents=True, exist_ok=True)
        acc_paths = []
        for i, upload in enumerate(accessory_images):
            path = scratch / f"acc_{i}_{upload.filename or 'input.png'}"
            path.write_bytes(await upload.read())
            acc_paths.append(path)

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_apply_accessories,
            args=(product, variation, job_id, acc_paths, instruction.strip(), user.id),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    def _run_apply_accessories(
        prod: Product,
        var: Variation,
        jid: str,
        accessory_paths: list[Path],
        instruction: str = "",
        user_id: str = "",
    ) -> None:
        try:
            from PIL import Image as PILImage

            job_manager.update_progress(jid, 0.1, "Loading images...")

            shoe_path = results_dir / prod.id / var.id / var.results[0].filename
            shoe_img = PILImage.open(shoe_path).convert("RGB")

            acc_imgs = [PILImage.open(p).convert("RGB") for p in accessory_paths]

            job_manager.update_progress(jid, 0.2, "Composing with accessories via Gemini...")

            placement = instruction if instruction else "on or next to the shoe"

            prompt = (
                f"Keep the shoe in the first image exactly as it is — same angle, same background, same lighting. "
                f"Take the item(s) from the other image(s) and place them {placement}. "
                f"Do not change anything else about the photo."
            )

            img_bytes, usage = _gemini_edit([shoe_img] + acc_imgs, prompt)

            cost = calculate_cost("gemini-3.1-flash-image-preview", usage)
            _log_cost(user_id=user_id, operation="accessories", model="gemini-3.1-flash-image-preview",
                      product_id=prod.id, variation_id=var.id,
                      input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                      thinking_tokens=usage.get("thinking_tokens", 0), cost_usd=cost)

            job_manager.update_progress(jid, 0.9, "Saving result...")
            _save_gemini_result(prod, var, img_bytes, "accessories", "gemini_accessories")

            job_manager.update_progress(jid, 1.0, "Done!")
            job_manager.complete(jid)

        except Exception as e:
            import traceback
            traceback.print_exc()
            job_manager.fail(jid, str(e))

    @app.patch(
        "/api/products/{product_id}/variations/{variation_id}/meta",
        status_code=200,
    )
    def update_variation_meta(
        product_id: str,
        variation_id: str,
        req: UpdateVariationMetaRequest,
        request: Request,
    ) -> dict:
        _get_current_user(request)
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        for v in product.variations:
            if v.id == variation_id:
                if req.name is not None:
                    v.name = req.name
                if req.price_tier is not None:
                    v.price_tier = req.price_tier
                if req.theme is not None:
                    v.theme = req.theme
                if req.feature is not None:
                    v.feature = req.feature
                store.save_product(product)
                return {
                    "name": v.name,
                    "price_tier": v.price_tier,
                    "theme": v.theme,
                    "feature": v.feature,
                }

        raise HTTPException(status_code=404, detail="Variation not found")

    @app.put("/api/products/{product_id}/variations/{variation_id}/select-hero")
    def select_hero_candidate(
        product_id: str,
        variation_id: str,
        req: SelectHeroCandidateRequest,
        request: Request,
    ) -> dict:
        """Switch the active hero to a different candidate."""
        _get_current_user(request)
        fname = req.filename

        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        for v in product.variations:
            if v.id == variation_id:
                # Verify the candidate file exists
                candidate_path = results_dir / product_id / variation_id / fname
                if not candidate_path.exists():
                    raise HTTPException(status_code=404, detail="Candidate file not found")

                # Copy candidate to hero.png
                import shutil
                hero_path = results_dir / product_id / variation_id / "hero.png"
                shutil.copy2(candidate_path, hero_path)

                # Update selected flags
                for c in v.hero_candidates:
                    c.selected = (c.filename == fname)

                store.save_product(product)
                return {"ok": True, "selected": fname}

        raise HTTPException(status_code=404, detail="Variation not found")

    @app.delete(
        "/api/products/{product_id}/variations/{variation_id}",
        status_code=204,
    )
    def delete_variation(
        product_id: str, variation_id: str, request: Request,
    ):
        _get_current_user(request)
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

        # Clean up result files on disk
        import shutil
        var_dir = results_dir / product_id / variation_id
        if var_dir.exists():
            shutil.rmtree(var_dir, ignore_errors=True)

    @app.delete(
        "/api/products/{product_id}/variations/{variation_id}/results/{filename}",
        status_code=204,
    )
    def delete_variation_result(
        product_id: str, variation_id: str, filename: str, request: Request,
    ):
        _get_current_user(request)
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

        original_len = len(variation.results)
        variation.results = [r for r in variation.results if r.filename != filename]
        variation.generated_results = [
            r for r in variation.generated_results if r.filename != filename
        ]
        if len(variation.results) == original_len:
            raise HTTPException(status_code=404, detail="Result not found")

        # Delete the file from disk
        result_path = results_dir / product_id / variation_id / filename
        if result_path.exists():
            result_path.unlink()

        store.save_product(product)

    @app.put(
        "/api/products/{product_id}/variations/{variation_id}/set-cover",
        status_code=200,
    )
    def set_variation_cover(
        product_id: str,
        variation_id: str,
        req: dict,
        request: Request,
    ) -> dict:
        _get_current_user(request)
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
    def list_collections(request: Request) -> list[Collection]:
        _get_current_user(request)
        return store.list_collections()

    @app.post("/api/collections", status_code=201)
    def create_collection(req: CreateCollectionRequest, request: Request) -> Collection:
        _get_current_user(request)
        collection = Collection(name=req.name)
        store.save_collection(collection)
        return collection

    @app.get("/api/collections/{collection_id}")
    def get_collection(collection_id: str, request: Request) -> Collection:
        _get_current_user(request)
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        return collection

    @app.put("/api/collections/{collection_id}")
    def update_collection(
        collection_id: str, req: UpdateCollectionRequest, request: Request,
    ) -> Collection:
        _get_current_user(request)
        collection = store.get_collection(collection_id)
        if not collection:
            raise HTTPException(status_code=404, detail="Collection not found")
        for field, value in req.model_dump(exclude_none=True).items():
            setattr(collection, field, value)
        store.save_collection(collection)
        return collection

    @app.delete("/api/collections/{collection_id}", status_code=204)
    def delete_collection(collection_id: str, request: Request):
        _get_current_user(request)
        if not store.delete_collection(collection_id):
            raise HTTPException(status_code=404, detail="Collection not found")

    @app.post("/api/collections/{collection_id}/products", status_code=200)
    def add_product_to_collection(
        collection_id: str, req: AddProductToCollectionRequest, request: Request,
    ) -> Collection:
        _get_current_user(request)
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
        collection_id: str, product_id: str, request: Request,
    ) -> Collection:
        _get_current_user(request)
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
        collection_id: str, body: dict, request: Request,
    ) -> Collection:
        _get_current_user(request)
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

    # --- Semantic Search ---

    vector_db: VariantVectorDB | None = None
    if embedding_provider is not None:
        # Test mode: use injected provider
        vector_db = VariantVectorDB(data_dir, embedding_provider)
    else:
        load_dotenv()
        _voyage_key = os.environ.get("VOYAGE_API_KEY")
        if _voyage_key:
            vector_db = VariantVectorDB(data_dir)
        else:
            logging.getLogger(__name__).warning(
                "VOYAGE_API_KEY not set — semantic search disabled"
            )

    def _require_vector_db() -> VariantVectorDB:
        if vector_db is None:
            raise HTTPException(
                status_code=503,
                detail="Semantic search unavailable (VOYAGE_API_KEY not configured)",
            )
        return vector_db

    @app.post("/api/search/index", status_code=200)
    def index_all_variants(request: Request) -> dict:
        _get_current_user(request)
        """Index all product variations incrementally.

        Only variants whose text has changed or that are new will be
        sent to the embedding API.  Unchanged variants are skipped.
        """
        db = _require_vector_db()
        products = store.list_products()

        items: list[tuple[str, str, str]] = []
        current_keys: set[str] = set()

        for product in products:
            for variation in product.variations:
                text = build_variant_text(
                    product.name,
                    product.label,
                    product.description,
                    variation,
                )
                items.append((product.id, variation.id, text))
                current_keys.add(f"{product.id}:{variation.id}")

        # Remove variants that no longer exist in the store
        stale_keys = db.indexed_keys() - current_keys
        for key in stale_keys:
            pid, vid = key.split(":", 1)
            db.remove_variant(pid, vid)

        embedded = db.index_variants_batch(items)
        return {
            "total_variants": len(items),
            "newly_embedded": embedded,
            "removed_stale": len(stale_keys),
            "indexed_total": db.indexed_count,
        }

    @app.post("/api/search/index/{product_id}/{variation_id}", status_code=200)
    def index_single_variant(product_id: str, variation_id: str, request: Request) -> dict:
        _get_current_user(request)
        """Index (or re-index) a single variation."""
        db = _require_vector_db()
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = next(
            (v for v in product.variations if v.id == variation_id), None
        )
        if not variation:
            raise HTTPException(status_code=404, detail="Variation not found")

        text = build_variant_text(
            product.name, product.label, product.description, variation
        )
        created = db.index_variant(product_id, variation_id, text)
        return {"indexed": created, "text": text}

    @app.delete("/api/search/index/{product_id}/{variation_id}", status_code=200)
    def remove_variant_from_index(product_id: str, variation_id: str, request: Request) -> dict:
        _get_current_user(request)
        """Remove a single variation from the search index."""
        db = _require_vector_db()
        removed = db.remove_variant(product_id, variation_id)
        if not removed:
            raise HTTPException(
                status_code=404, detail="Variation not found in index"
            )
        return {"removed": True}

    @app.post("/api/search")
    def search_variants(req: SearchRequest, request: Request) -> SearchResponse:
        _get_current_user(request)
        """Semantic search over indexed product variations.

        Normalizes the query, checks cache, embeds if needed,
        then returns the top-k results above the similarity threshold.
        """
        db = _require_vector_db()
        normalized = normalize_text(req.query)
        cached = normalized in db._query_cache

        results = db.search(
            query=req.query,
            top_k=req.top_k,
            min_similarity=req.min_similarity,
        )

        return SearchResponse(
            query=req.query,
            normalized_query=normalized,
            results=[SearchResult(**r) for r in results],
            cached=cached,
        )

    @app.get("/api/search/stats")
    def search_stats(request: Request) -> IndexStats:
        _get_current_user(request)
        """Return current index statistics."""
        db = _require_vector_db()
        return IndexStats(
            indexed_variants=db.indexed_count,
            cached_queries=db.cached_queries_count,
        )

    # --- Admin cost endpoints ---

    @app.get("/api/admin/costs/summary")
    def admin_costs_summary(request: Request):
        _require_admin(request)
        from datetime import date
        today = date.today().isoformat()
        all_costs = store.list_costs()
        today_costs = [c for c in all_costs if c.timestamp.startswith(today)]

        user_totals: dict[str, float] = {}
        user_calls: dict[str, int] = {}
        for c in all_costs:
            user_totals[c.user_id] = user_totals.get(c.user_id, 0) + c.cost_usd
            user_calls[c.user_id] = user_calls.get(c.user_id, 0) + 1

        per_user = []
        for uid, total in sorted(user_totals.items(), key=lambda x: -x[1]):
            u = store.get_user(uid)
            per_user.append({
                "user_id": uid,
                "name": u.name if u else "Unknown",
                "total_cost": round(total, 4),
                "calls": user_calls.get(uid, 0),
            })

        return {
            "total_cost": round(sum(c.cost_usd for c in all_costs), 4),
            "today_cost": round(sum(c.cost_usd for c in today_costs), 4),
            "today_calls": len(today_costs),
            "total_calls": len(all_costs),
            "per_user": per_user,
        }

    @app.get("/api/admin/costs/daily")
    def admin_costs_daily(request: Request, days: int = Query(default=30)):
        _require_admin(request)
        from datetime import date, timedelta
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        costs = store.list_costs(since=cutoff)

        daily: dict[str, dict] = {}
        for c in costs:
            day = c.timestamp[:10]
            if day not in daily:
                daily[day] = {"date": day, "cost": 0.0, "calls": 0}
            daily[day]["cost"] += c.cost_usd
            daily[day]["calls"] += 1

        result = sorted(daily.values(), key=lambda x: x["date"], reverse=True)
        for r in result:
            r["cost"] = round(r["cost"], 4)
        return result

    @app.get("/api/admin/costs/details")
    def admin_costs_details(request: Request, date_str: str = Query(..., alias="date")):
        _require_admin(request)
        costs = store.list_costs(since=date_str, until=date_str + "T23:59:59")
        result = []
        for c in costs:
            u = store.get_user(c.user_id)
            product = store.get_product(c.product_id) if c.product_id else None
            variation_label = ""
            if product and c.variation_id:
                for v in product.variations:
                    if v.id == c.variation_id:
                        parts = [p for p in [v.material, v.color] if p]
                        variation_label = " / ".join(parts) if parts else f"Variation {v.id[:6]}"
                        break
            result.append({
                "id": c.id,
                "timestamp": c.timestamp,
                "user_name": u.name if u else "Unknown",
                "operation": c.operation,
                "model": c.model,
                "product_id": c.product_id,
                "variation_id": c.variation_id,
                "product_name": product.name if product else "",
                "variation_label": variation_label,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "thinking_tokens": c.thinking_tokens,
                "video_seconds": c.video_seconds,
                "cost_usd": round(c.cost_usd, 6),
            })
        return result

    return app
