"""Vector database for semantic search over product variations.

Stores variant embeddings in a FAISS index with JSON metadata.
Uses voyage-multimodal-3 to embed text + the first result image per variant.
Supports incremental indexing, query normalization, and embedding caching.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import unicodedata
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

import voyageai

logger = logging.getLogger(__name__)

VOYAGE_DIMENSION = 1024
MULTIMODAL_MODEL = "voyage-multimodal-3"


def normalize_text(text: str) -> str:
    """Normalize search text to reduce redundant embedding calls.

    Lowercases, strips accents, collapses whitespace, removes punctuation.
    """
    text = text.strip().lower()
    # Decompose unicode and remove combining marks (accents)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Remove punctuation except hyphens (useful for "block-heel" etc.)
    text = re.sub(r"[^\w\s-]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_variant_text(
    product_name: str,
    product_label: str,
    product_description: str,
    variation: object,
) -> str:
    """Build a searchable text string from product + variation fields.

    Variant-specific fields come first (material, color, note) since they
    are the most distinguishing for search. Product description is truncated
    to avoid drowning out variant details in the embedding.
    """
    # Variant fields first — these differentiate variants of the same product
    variant_parts = []
    for field in ("material", "color", "note", "price_tier", "theme", "feature"):
        val = getattr(variation, field, "")
        if val:
            variant_parts.append(val)

    # Product context second
    product_parts = [product_name]
    if product_label:
        product_parts.append(product_label)
    if product_description:
        # Truncate long descriptions to keep variant fields prominent
        desc = product_description[:200].rstrip()
        product_parts.append(desc)

    return " ".join(variant_parts + product_parts)


def _image_hash(path: Path) -> str:
    """Quick hash of an image file for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        # Read first 64KB — enough to detect changes without reading huge files
        h.update(f.read(65536))
    return h.hexdigest()[:12]


class VariantVectorDB:
    """FAISS-backed vector database for product variations.

    Uses voyage-multimodal-3 to create embeddings from text + image.

    Persistence files:
        - {data_dir}/variants.index  — FAISS binary index
        - {data_dir}/variants_meta.json — variant mapping + query cache

    Metadata schema:
        {
            "variants": {
                "<product_id>:<variation_id>": {
                    "text": "...",
                    "image_hash": "abc123" | null,
                    "faiss_id": 0
                }
            },
            "query_cache": {
                "<normalized_query>": [0.1, 0.2, ...]
            },
            "next_faiss_id": 5
        }
    """

    def __init__(self, data_dir: Path, provider: object | None = None) -> None:
        self._data_dir = Path(data_dir)
        self._results_dir = self._data_dir / "results"
        self._index_path = self._data_dir / "variants.index"
        self._meta_path = self._data_dir / "variants_meta.json"
        self._lock = threading.Lock()

        # Initialize Voyage client (provider kept for backward compat but unused)
        self._client = voyageai.Client()

        # variant_key -> {"text": str, "image_hash": str|None, "faiss_id": int}
        self._variants: dict[str, dict] = {}
        # normalized_query -> embedding vector
        self._query_cache: dict[str, list[float]] = {}
        self._next_faiss_id: int = 0

        if self._index_path.exists() and self._meta_path.exists():
            logger.info("Loading existing variant vector index...")
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._variants = meta.get("variants", {})
            self._query_cache = meta.get("query_cache", {})
            self._next_faiss_id = meta.get("next_faiss_id", 0)
        else:
            logger.info("Creating new variant vector index...")
            self._index = faiss.IndexIDMap(faiss.IndexFlatIP(VOYAGE_DIMENSION))

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _find_variant_image(self, product_id: str, variation_id: str) -> Path | None:
        """Find the first result image for a variant."""
        variant_dir = self._results_dir / product_id / variation_id
        if not variant_dir.is_dir():
            return None
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            files = sorted(variant_dir.glob(ext))
            if files:
                return files[0]
        return None

    def _embed_multimodal(self, text: str, image_path: Path | None) -> list[float]:
        """Embed text + optional image using voyage-multimodal-3."""
        content: list = [text]
        if image_path:
            img = Image.open(image_path)
            img.load()
            content.append(img)

        result = self._client.multimodal_embed(
            inputs=[content],
            model=MULTIMODAL_MODEL,
            input_type="document",
        )
        return result.embeddings[0]

    def _embed_query_text(self, text: str) -> list[float]:
        """Embed a search query (text only) using voyage-multimodal-3."""
        result = self._client.multimodal_embed(
            inputs=[[text]],
            model=MULTIMODAL_MODEL,
            input_type="query",
        )
        return result.embeddings[0]

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _variant_changed(self, key: str, text: str, img_hash: str | None) -> bool:
        """Check if a variant needs re-indexing."""
        existing = self._variants.get(key)
        if not existing:
            return True
        if existing["text"] != text:
            return True
        if existing.get("image_hash") != img_hash:
            return True
        return False

    def index_variant(
        self,
        product_id: str,
        variation_id: str,
        text: str,
    ) -> bool:
        """Index a single variant. Returns True if a new embedding was created."""
        key = f"{product_id}:{variation_id}"
        image_path = self._find_variant_image(product_id, variation_id)
        img_hash = _image_hash(image_path) if image_path else None

        with self._lock:
            if not self._variant_changed(key, text, img_hash):
                return False

            existing = self._variants.get(key)
            if existing:
                self._remove_from_index(existing["faiss_id"])

        embedding = self._embed_multimodal(text, image_path)

        with self._lock:
            faiss_id = self._next_faiss_id
            self._next_faiss_id += 1

            emb_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(emb_np)
            self._index.add_with_ids(emb_np, np.array([faiss_id], dtype=np.int64))

            self._variants[key] = {
                "text": text,
                "image_hash": img_hash,
                "faiss_id": faiss_id,
            }
            self._save()
            return True

    def index_variants_batch(
        self,
        items: list[tuple[str, str, str]],
    ) -> int:
        """Index variants one by one, saving after each.

        Each item is (product_id, variation_id, text).
        Skips unchanged variants (same text AND same image).
        If embedding fails (e.g. rate limit), stops early.
        Progress is preserved on disk after each variant.
        """
        embedded = 0

        for product_id, variation_id, text in items:
            key = f"{product_id}:{variation_id}"
            image_path = self._find_variant_image(product_id, variation_id)
            img_hash = _image_hash(image_path) if image_path else None

            with self._lock:
                if not self._variant_changed(key, text, img_hash):
                    continue
                existing = self._variants.get(key)
                if existing:
                    self._remove_from_index(existing["faiss_id"])

            # Embed outside the lock (API call)
            try:
                embedding = self._embed_multimodal(text, image_path)
            except Exception as e:
                logger.warning(
                    "Embedding failed for %s (rate limit?): %s — stopping batch",
                    key, e,
                )
                break

            with self._lock:
                faiss_id = self._next_faiss_id
                self._next_faiss_id += 1

                emb_np = np.array([embedding], dtype=np.float32)
                faiss.normalize_L2(emb_np)
                self._index.add_with_ids(emb_np, np.array([faiss_id], dtype=np.int64))

                self._variants[key] = {
                    "text": text,
                    "image_hash": img_hash,
                    "faiss_id": faiss_id,
                }
                self._save()

            has_img = "+" if image_path else "-"
            embedded += 1
            logger.info("Indexed %s [img:%s] (%d done)", key, has_img, embedded)

        return embedded

    def remove_variant(self, product_id: str, variation_id: str) -> bool:
        """Remove a variant from the index. Returns True if it was present."""
        key = f"{product_id}:{variation_id}"
        with self._lock:
            existing = self._variants.pop(key, None)
            if not existing:
                return False
            self._remove_from_index(existing["faiss_id"])
            self._save()
            return True

    def _remove_from_index(self, faiss_id: int) -> None:
        """Remove a single vector from the FAISS index by ID."""
        self._index.remove_ids(np.array([faiss_id], dtype=np.int64))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.5,
    ) -> list[dict]:
        """Search for variants matching the query.

        Uses voyage-multimodal-3 with input_type="query" for the search text.
        Embeddings are in the same space as the indexed text+image embeddings.
        """
        normalized = normalize_text(query)
        if not normalized:
            return []

        with self._lock:
            if self._index.ntotal == 0:
                return []

            # Check query cache
            if normalized in self._query_cache:
                logger.info("Query cache hit for: %s", normalized)
                embedding = self._query_cache[normalized]
            else:
                logger.info("Embedding new query: %s", normalized)
                try:
                    embedding = self._embed_query_text(normalized)
                except Exception as e:
                    logger.warning("Embedding failed (rate limit?): %s", e)
                    return []
                self._query_cache[normalized] = embedding
                self._save()

            q_np = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(q_np)

            distances, indices = self._index.search(q_np, min(top_k, self._index.ntotal))

            # Build reverse lookup: faiss_id -> variant_key
            id_to_key = {
                v["faiss_id"]: k for k, v in self._variants.items()
            }

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue
                score = float(dist)
                if score < min_similarity:
                    continue
                key = id_to_key.get(int(idx))
                if not key:
                    continue
                product_id, variation_id = key.split(":", 1)
                results.append({
                    "product_id": product_id,
                    "variation_id": variation_id,
                    "text": self._variants[key]["text"],
                    "similarity": round(score, 4),
                })

            return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Write index and metadata to disk."""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "variants": self._variants,
                    "query_cache": self._query_cache,
                    "next_faiss_id": self._next_faiss_id,
                },
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def indexed_count(self) -> int:
        return len(self._variants)

    @property
    def cached_queries_count(self) -> int:
        return len(self._query_cache)

    def indexed_keys(self) -> set[str]:
        """Return set of 'product_id:variation_id' keys currently indexed."""
        return set(self._variants.keys())
