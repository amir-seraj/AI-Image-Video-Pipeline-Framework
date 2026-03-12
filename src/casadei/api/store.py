from __future__ import annotations

import json
import threading
from pathlib import Path

from .models import Collection, CostRecord, Product, User


class JsonStore:
    """Thread-safe JSON file store for products."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._products: dict[str, Product] = {}
        self._collections: dict[str, Collection] = {}
        self._users: dict[str, User] = {}
        self._cost_log: list[CostRecord] = []
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
        self._users = {
            uid: User.model_validate(data)
            for uid, data in raw.get("users", {}).items()
        }
        self._cost_log = [
            CostRecord.model_validate(r)
            for r in raw.get("cost_log", [])
        ]

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
            "users": {
                uid: u.model_dump(mode="json")
                for uid, u in self._users.items()
            },
            "cost_log": [r.model_dump(mode="json") for r in self._cost_log],
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

    # --- Users ---

    def save_user(self, user: User) -> None:
        with self._lock:
            self._users[user.id] = user
            self._flush()

    def get_user(self, user_id: str) -> User | None:
        with self._lock:
            return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> User | None:
        with self._lock:
            for u in self._users.values():
                if u.email == email:
                    return u
            return None

    def list_users(self) -> list[User]:
        with self._lock:
            return list(self._users.values())

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            if user_id in self._users:
                del self._users[user_id]
                self._flush()
                return True
            return False

    # --- Costs ---

    def append_cost(self, record: CostRecord) -> None:
        with self._lock:
            self._cost_log.append(record)
            self._flush()

    def list_costs(self, since: str | None = None, until: str | None = None) -> list[CostRecord]:
        with self._lock:
            records = self._cost_log
            if since:
                records = [r for r in records if r.timestamp >= since]
            if until:
                records = [r for r in records if r.timestamp <= until]
            return list(records)
