from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class Sketch(BaseModel):
    id: str = Field(default_factory=_new_id)
    filename: str
    uploaded_at: datetime = Field(default_factory=_utcnow)


class ResultFile(BaseModel):
    filename: str


class Generation(BaseModel):
    id: str = Field(default_factory=_new_id)
    type: str = "image"
    pipeline: str
    prompt: str
    negative_prompt: str = ""
    num_outputs: int = 1
    results: list[ResultFile] = []
    parent_generation_id: str | None = None
    status: JobStatus = JobStatus.pending
    created_at: datetime = Field(default_factory=_utcnow)


class GeneratedResult(BaseModel):
    filename: str
    pipeline: str = ""
    label: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class Variation(BaseModel):
    id: str = Field(default_factory=_new_id)
    material: str = ""
    color: str = ""
    note: str = ""
    pipeline: str = ""
    num_outputs: int = 1
    results: list[ResultFile] = []
    generated_results: list[GeneratedResult] = []
    spin_frames: list[ResultFile] = []
    spin_video: str = ""
    price_tier: str = ""
    theme: str = ""
    feature: str = ""
    status: JobStatus = JobStatus.pending
    created_at: datetime = Field(default_factory=_utcnow)


class UpdateVariationMetaRequest(BaseModel):
    price_tier: str | None = None
    theme: str | None = None
    feature: str | None = None


class Product(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    label: str = ""
    description: str = ""
    sketches: list[Sketch] = []
    generations: list[Generation] = []
    variations: list[Variation] = []
    created_at: datetime = Field(default_factory=_utcnow)


class CreateProductRequest(BaseModel):
    name: str


class CreateVariationRequest(BaseModel):
    material: str = ""
    color: str = ""
    note: str = ""
    pipeline: str
    num_outputs: int = 1


class RegenerateVariationRequest(BaseModel):
    change_request: str = ""


class Collection(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    product_ids: list[str] = []
    target_price_tiers: list[str] = []
    target_themes: list[str] = []
    target_features: list[str] = []
    target_description: str = ""
    created_at: datetime = Field(default_factory=_utcnow)


class CreateCollectionRequest(BaseModel):
    name: str


class UpdateCollectionRequest(BaseModel):
    name: str | None = None
    target_price_tiers: list[str] | None = None
    target_themes: list[str] | None = None
    target_features: list[str] | None = None
    target_description: str | None = None


class AddProductToCollectionRequest(BaseModel):
    product_id: str


class GenerateRequest(BaseModel):
    pipeline: str
    prompt: str
    num_outputs: int = 1
    negative_prompt: str = ""


class RefineRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""


class PipelineInputDeclaration(BaseModel):
    type: str
    label: str


class PipelinePreset(BaseModel):
    id: str
    name: str
    description: str
    agents: list[str]
    template_variables: list[str]
    inputs: dict[str, PipelineInputDeclaration] = {}


class AgentInfo(BaseModel):
    name: str
    model: str
    description: str
    template_variables: list[str]


class RunResponse(BaseModel):
    job_id: str
    run_id: str


class AgentConfigResponse(BaseModel):
    """Full agent configuration for detail/edit views."""
    name: str
    model: str
    description: str
    prompt_template: str
    negative_prompt: str
    params: dict[str, Any]
    template_variables: list[str]


class AgentConfigRequest(BaseModel):
    """Request body for creating/updating an agent."""
    name: str
    model: str
    description: str = ""
    prompt_template: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = {}


class DuplicateAgentRequest(BaseModel):
    new_name: str


class PipelineStepResponse(BaseModel):
    type: str
    agent: str | None = None
    script: str | None = None
    function: str | None = None
    exists: bool | None = None


class PipelineDetailResponse(BaseModel):
    id: str
    name: str
    description: str
    steps: list[PipelineStepResponse]
    local_agents: list[str]
    template_variables: list[str]
    inputs: dict[str, PipelineInputDeclaration] = {}


class PipelineStepRequest(BaseModel):
    type: str
    agent: str | None = None
    script: str | None = None
    function: str | None = None


class PipelineCreateRequest(BaseModel):
    id: str
    name: str
    description: str = ""
    steps: list[PipelineStepRequest] = []


class PipelineUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    steps: list[PipelineStepRequest] | None = None


# --- Search ---


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    min_similarity: float = 0.5


class SearchResult(BaseModel):
    product_id: str
    variation_id: str
    text: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    results: list[SearchResult]
    cached: bool = False


class IndexStats(BaseModel):
    indexed_variants: int
    cached_queries: int
