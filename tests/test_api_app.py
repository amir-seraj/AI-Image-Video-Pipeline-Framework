import io
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from casadei.api.app import create_app


@pytest.fixture
def client(tmp_path: Path):
    app = create_app(data_dir=tmp_path)
    return TestClient(app)


def test_create_product(client: TestClient):
    resp = client.post("/api/products", json={"name": "Summer Dress"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Summer Dress"
    assert "id" in data


def test_list_products(client: TestClient):
    client.post("/api/products", json={"name": "A"})
    client.post("/api/products", json={"name": "B"})
    resp = client.get("/api/products")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_product(client: TestClient):
    create_resp = client.post("/api/products", json={"name": "Test"})
    pid = create_resp.json()["id"]
    resp = client.get(f"/api/products/{pid}")
    assert resp.status_code == 200
    assert resp.json()["name"] == "Test"


def test_get_product_not_found(client: TestClient):
    resp = client.get("/api/products/nonexistent")
    assert resp.status_code == 404


def test_delete_product(client: TestClient):
    create_resp = client.post("/api/products", json={"name": "Delete Me"})
    pid = create_resp.json()["id"]
    resp = client.delete(f"/api/products/{pid}")
    assert resp.status_code == 204
    assert client.get(f"/api/products/{pid}").status_code == 404


def test_list_pipelines(client: TestClient):
    resp = client.get("/api/pipelines")
    assert resp.status_code == 200
    pipelines = resp.json()
    assert isinstance(pipelines, list)
    assert len(pipelines) > 0
    assert "id" in pipelines[0]
    assert "name" in pipelines[0]


def test_generate_requires_sketches(client: TestClient):
    create_resp = client.post("/api/products", json={"name": "Empty"})
    pid = create_resp.json()["id"]
    resp = client.post(
        f"/api/products/{pid}/generate",
        json={"pipeline": "style_transfer", "prompt": "watercolor"},
    )
    assert resp.status_code == 400


def test_generate_returns_job_id(client: TestClient):
    create_resp = client.post(
        "/api/products", json={"name": "With Sketch"}
    )
    pid = create_resp.json()["id"]
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(buf, format="PNG")
    buf.seek(0)
    client.post(
        f"/api/products/{pid}/sketches",
        files={"file": ("sketch.png", buf, "image/png")},
    )
    resp = client.post(
        f"/api/products/{pid}/generate",
        json={"pipeline": "style_transfer", "prompt": "watercolor"},
    )
    assert resp.status_code == 202
    assert "job_id" in resp.json()
    assert "generation_id" in resp.json()
