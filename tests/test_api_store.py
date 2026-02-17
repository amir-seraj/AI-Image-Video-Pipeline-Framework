import json
from pathlib import Path
from casadei.api.store import JsonStore
from casadei.api.models import Product


def test_create_and_get_product(tmp_path: Path):
    store = JsonStore(tmp_path / "data.json")
    product = Product(name="Summer Dress")
    store.save_product(product)
    loaded = store.get_product(product.id)
    assert loaded is not None
    assert loaded.name == "Summer Dress"


def test_list_products(tmp_path: Path):
    store = JsonStore(tmp_path / "data.json")
    store.save_product(Product(name="A"))
    store.save_product(Product(name="B"))
    products = store.list_products()
    assert len(products) == 2


def test_delete_product(tmp_path: Path):
    store = JsonStore(tmp_path / "data.json")
    product = Product(name="To Delete")
    store.save_product(product)
    store.delete_product(product.id)
    assert store.get_product(product.id) is None


def test_update_product(tmp_path: Path):
    store = JsonStore(tmp_path / "data.json")
    product = Product(name="Original")
    store.save_product(product)
    product.name = "Updated"
    store.save_product(product)
    loaded = store.get_product(product.id)
    assert loaded.name == "Updated"


def test_persistence(tmp_path: Path):
    path = tmp_path / "data.json"
    store1 = JsonStore(path)
    store1.save_product(Product(name="Persistent"))
    store2 = JsonStore(path)
    assert len(store2.list_products()) == 1
