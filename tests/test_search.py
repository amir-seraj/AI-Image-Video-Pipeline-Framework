"""Tests for semantic search (vectordb + API endpoints).

Uses a mock voyageai.Client so no Voyage API calls are made.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from casadei.api.app import create_app
from casadei.api.models import Variation
from casadei.api.vectordb import (
    VariantVectorDB,
    build_variant_text,
    normalize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 1024


def _fake_embedding(text: str) -> list[float]:
    """Deterministic pseudo-embedding from text (seeded hash)."""
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    vec = rng.randn(DIMENSION).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


def _make_mock_voyage_client():
    """Create a mock voyageai.Client with working multimodal_embed."""
    client = MagicMock()

    def fake_multimodal_embed(inputs, model, input_type=None, **kwargs):
        # inputs is a list of lists; first element of each inner list is text
        embeddings = []
        for content_list in inputs:
            # Extract text parts only (skip PIL images)
            text_parts = [item for item in content_list if isinstance(item, str)]
            combined = " ".join(text_parts)
            embeddings.append(_fake_embedding(combined))
        result = MagicMock()
        result.embeddings = embeddings
        return result

    client.multimodal_embed = fake_multimodal_embed
    return client


@pytest.fixture(autouse=True)
def _mock_voyage_client(monkeypatch):
    """Patch voyageai.Client globally for all tests."""
    monkeypatch.setattr("voyageai.Client", lambda **kw: _make_mock_voyage_client())


# ---------------------------------------------------------------------------
# normalize_text tests
# ---------------------------------------------------------------------------


class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_whitespace(self):
        assert normalize_text("  foo   bar  ") == "foo bar"

    def test_removes_accents(self):
        assert normalize_text("café résumé") == "cafe resume"

    def test_removes_punctuation(self):
        assert normalize_text("hello, world! how?") == "hello world how"

    def test_preserves_hyphens(self):
        assert normalize_text("block-heel") == "block-heel"

    def test_empty(self):
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


# ---------------------------------------------------------------------------
# build_variant_text tests
# ---------------------------------------------------------------------------


class TestBuildVariantText:
    def test_combines_fields(self):
        class FakeVariation:
            material = "Nappa Leather"
            color = "Black"
            note = "Classic look"
            price_tier = "premium"
            theme = "evening"
            feature = "stiletto"

        result = build_variant_text("Blade", "Pump", "Iconic pump", FakeVariation())
        assert "Blade" in result
        assert "Pump" in result
        assert "Iconic pump" in result
        assert "Nappa Leather" in result
        assert "Black" in result
        assert "stiletto" in result

    def test_skips_empty_fields(self):
        class FakeVariation:
            material = ""
            color = "Red"
            note = ""
            price_tier = ""
            theme = ""
            feature = ""

        result = build_variant_text("Shoe", "", "", FakeVariation())
        assert result == "Red Shoe"


# ---------------------------------------------------------------------------
# VariantVectorDB unit tests
# ---------------------------------------------------------------------------


class TestVariantVectorDB:
    @pytest.fixture
    def db(self, tmp_path):
        return VariantVectorDB(tmp_path)

    def test_index_and_search(self, db):
        db.index_variant("p1", "v1", "red leather stiletto heel evening shoe")
        db.index_variant("p1", "v2", "blue suede casual sneaker")
        db.index_variant("p2", "v1", "black patent leather pump")

        results = db.search("red heel shoe", top_k=3, min_similarity=-1.0)
        assert len(results) > 0
        assert results[0]["product_id"] in ("p1", "p2")

    def test_incremental_skip(self, db):
        # First index
        created = db.index_variant("p1", "v1", "red leather shoe")
        assert created is True

        # Same text → should skip
        created = db.index_variant("p1", "v1", "red leather shoe")
        assert created is False

    def test_reindex_on_text_change(self, db):
        db.index_variant("p1", "v1", "red leather shoe")
        assert db.indexed_count == 1

        # Changed text → re-embed
        created = db.index_variant("p1", "v1", "blue suede boot")
        assert created is True
        assert db.indexed_count == 1  # still 1 (replaced, not added)

    def test_remove_variant(self, db):
        db.index_variant("p1", "v1", "red shoe")
        assert db.indexed_count == 1

        removed = db.remove_variant("p1", "v1")
        assert removed is True
        assert db.indexed_count == 0

        # Remove again → False
        removed = db.remove_variant("p1", "v1")
        assert removed is False

    def test_batch_index(self, db):
        items = [
            ("p1", "v1", "red leather shoe"),
            ("p1", "v2", "blue suede boot"),
            ("p2", "v1", "black patent pump"),
        ]
        count = db.index_variants_batch(items)
        assert count == 3
        assert db.indexed_count == 3

        # Re-run same batch → 0 new
        count = db.index_variants_batch(items)
        assert count == 0

    def test_similarity_threshold(self, db):
        db.index_variant("p1", "v1", "red leather shoe")

        # With impossibly high threshold → no results
        results = db.search("red shoe", top_k=3, min_similarity=0.99)
        assert len(results) == 0 or all(r["similarity"] >= 0.99 for r in results)

    def test_query_caching(self, db):
        db.index_variant("p1", "v1", "red leather shoe")

        # First search → caches
        db.search("red shoe", top_k=1, min_similarity=-1.0)
        assert db.cached_queries_count >= 1

        # Same query (normalized) → uses cache
        db.search("  Red  Shoe!  ", top_k=1, min_similarity=-1.0)
        # Should still be 1 cached entry (same normalized form)
        assert db.cached_queries_count == 1

    def test_persistence(self, tmp_path):
        db1 = VariantVectorDB(tmp_path)
        db1.index_variant("p1", "v1", "red shoe")
        assert db1.indexed_count == 1

        # Load from disk
        db2 = VariantVectorDB(tmp_path)
        assert db2.indexed_count == 1

    def test_empty_search(self, db):
        results = db.search("anything", top_k=3, min_similarity=-1.0)
        assert results == []


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


def _create_app_with_fake_voyage(tmp_path: Path) -> TestClient:
    """Create a test app with fake embedding (no Voyage API calls)."""
    # Pass a dummy provider to signal "search enabled" without needing VOYAGE_API_KEY
    app = create_app(data_dir=tmp_path, embedding_provider="fake")
    return TestClient(app)


class TestSearchAPI:
    @pytest.fixture
    def client(self, tmp_path):
        return _create_app_with_fake_voyage(tmp_path)

    def _create_product_with_variation(self, client: TestClient) -> tuple[str, str]:
        """Helper: create a product with one variation, return (product_id, variation_id)."""
        resp = client.post("/api/products", json={"name": "Blade Pump"})
        pid = resp.json()["id"]

        # Add variation directly via the store (the API endpoint requires sketches)
        store = client.app.state.store
        product = store.get_product(pid)
        variation = Variation(
            material="Nappa Leather",
            color="Black",
            pipeline="test-pipe",
        )
        product.variations.append(variation)
        store.save_product(product)

        return pid, variation.id

    def test_search_stats_empty(self, client):
        resp = client.get("/api/search/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["indexed_variants"] == 0
        assert data["cached_queries"] == 0

    def test_index_all_variants(self, client):
        pid, vid = self._create_product_with_variation(client)

        resp = client.post("/api/search/index")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_variants"] == 1
        assert data["newly_embedded"] == 1
        assert data["indexed_total"] == 1

        # Second call → 0 new (incremental)
        resp = client.post("/api/search/index")
        data = resp.json()
        assert data["newly_embedded"] == 0

    def test_index_single_variant(self, client):
        pid, vid = self._create_product_with_variation(client)

        resp = client.post(f"/api/search/index/{pid}/{vid}")
        assert resp.status_code == 200
        assert resp.json()["indexed"] is True

    def test_index_single_not_found(self, client):
        resp = client.post("/api/search/index/bad_pid/bad_vid")
        assert resp.status_code == 404

    def test_remove_from_index(self, client):
        pid, vid = self._create_product_with_variation(client)
        client.post("/api/search/index")

        resp = client.delete(f"/api/search/index/{pid}/{vid}")
        assert resp.status_code == 200

        resp = client.delete(f"/api/search/index/{pid}/{vid}")
        assert resp.status_code == 404

    def test_search(self, client):
        pid, vid = self._create_product_with_variation(client)
        client.post("/api/search/index")

        resp = client.post(
            "/api/search",
            json={"query": "black leather shoe", "top_k": 3, "min_similarity": -1.0},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "black leather shoe"
        assert data["normalized_query"] == "black leather shoe"
        assert len(data["results"]) >= 1

    def test_search_caching(self, client):
        pid, vid = self._create_product_with_variation(client)
        client.post("/api/search/index")

        # First search
        client.post(
            "/api/search",
            json={"query": "leather pump", "top_k": 1, "min_similarity": -1.0},
        )

        # Second search with same normalized text → cached
        resp = client.post(
            "/api/search",
            json={"query": "  Leather  Pump!  ", "top_k": 1, "min_similarity": -1.0},
        )
        assert resp.json()["cached"] is True

    def test_search_empty_index(self, client):
        resp = client.post(
            "/api/search",
            json={"query": "anything", "top_k": 3, "min_similarity": -1.0},
        )
        assert resp.status_code == 200
        assert resp.json()["results"] == []
