from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

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


class Product(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    sketches: list[Sketch] = []
    generations: list[Generation] = []
    created_at: datetime = Field(default_factory=_utcnow)


class CreateProductRequest(BaseModel):
    name: str


class GenerateRequest(BaseModel):
    pipeline: str
    prompt: str
    num_outputs: int = 1
    negative_prompt: str = ""


class RefineRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""


class PipelinePreset(BaseModel):
    id: str
    name: str
    description: str
    agents: list[str]
    template_variables: list[str]


class AgentInfo(BaseModel):
    name: str
    model: str
    description: str
    template_variables: list[str]
