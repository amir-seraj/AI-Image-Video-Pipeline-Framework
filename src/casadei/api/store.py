from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import Collection, Product


class JsonStore:
    """Thread-safe JSON file store for products."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._products: dict[str, Product] = {}
        self._collections: dict[str, Collection] = {}
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self._path.read_text())
        self._products = {
            pid: Product.model_validate(data)
            for pid, data in raw.get("products", {}).items()
        }
        self._collections = {
            cid: Collection.model_validate(data)
            for cid, data in raw.get("collections", {}).items()
        }

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "products": {
                pid: p.model_dump(mode="json")
                for pid, p in self._products.items()
            },
            "collections": {
                cid: c.model_dump(mode="json")
                for cid, c in self._collections.items()
            },
        }
        self._path.write_text(json.dumps(raw, indent=2, default=str))

    def save_product(self, product: Product) -> None:
        with self._lock:
            self._products[product.id] = product
            self._flush()

    def get_product(self, product_id: str) -> Product | None:
        with self._lock:
            return self._products.get(product_id)

    def list_products(self) -> list[Product]:
        with self._lock:
            return list(self._products.values())

    def delete_product(self, product_id: str) -> bool:
        with self._lock:
            if product_id in self._products:
                del self._products[product_id]
                self._flush()
                return True
            return False

    # --- Collections ---

    def save_collection(self, collection: Collection) -> None:
        with self._lock:
            self._collections[collection.id] = collection
            self._flush()

    def get_collection(self, collection_id: str) -> Collection | None:
        with self._lock:
            return self._collections.get(collection_id)

    def list_collections(self) -> list[Collection]:
        with self._lock:
            return list(self._collections.values())

    def delete_collection(self, collection_id: str) -> bool:
        with self._lock:
            if collection_id in self._collections:
                del self._collections[collection_id]
                self._flush()
                return True
            return False
