"""Tests for multi-view generation components."""

from pathlib import Path

import pytest
from PIL import Image

from casadei.media import ImageMedia, MediaBundle
from casadei.models.base import ModelCapability, ImageConstraint
from casadei.models.image_to_multiview import ImageToMultiViewModel
from casadei.providers.zero123pp import _split_grid, GRID_COLS, GRID_ROWS, TILE_SIZE


class ConcreteMultiView(ImageToMultiViewModel):
    """Test-only concrete implementation."""

    capability = ModelCapability(
        inputs=[ImageConstraint(required=True, max_count=1)],
        outputs=[ImageConstraint(required=True, max_count=6)],
    )

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def _generate_views(self, image, num_views=6, **kwargs):
        return [Image.new("RGB", (320, 320), "red") for _ in range(num_views)]


class TestImageToMultiViewModel:
    def test_run_returns_6_views(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={"image": ImageMedia(image=Image.new("RGB", (320, 320)))})
        result = model.run(bundle, num_views=6)
        assert len(result.items) == 6
        assert all(isinstance(v, ImageMedia) for v in result.items.values())

    def test_run_validates_inputs(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={})
        with pytest.raises(ValueError, match="Required ImageMedia input is missing"):
            model.run(bundle)

    def test_run_custom_num_views(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={"image": ImageMedia(image=Image.new("RGB", (320, 320)))})
        result = model.run(bundle, num_views=3)
        assert len(result.items) == 3


class TestSplitGrid:
    def test_split_produces_6_tiles(self):
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS))
        tiles = _split_grid(grid)
        assert len(tiles) == 6

    def test_each_tile_correct_size(self):
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS))
        tiles = _split_grid(grid)
        for tile in tiles:
            assert tile.size == (TILE_SIZE, TILE_SIZE)

    def test_split_preserves_content(self):
        """Each tile should contain the correct region of the grid."""
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS), "white")
        # Paint top-left tile red
        for x in range(TILE_SIZE):
            for y in range(TILE_SIZE):
                grid.putpixel((x, y), (255, 0, 0))
        tiles = _split_grid(grid)
        # First tile should be red
        assert tiles[0].getpixel((0, 0)) == (255, 0, 0)
        # Second tile should be white
        assert tiles[1].getpixel((0, 0)) == (255, 255, 255)


class TestGenerate360Endpoint:
    """Test the API endpoint."""

    @pytest.fixture
    def client(self, tmp_path):
        from casadei.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app(data_dir=tmp_path)
        return TestClient(app)

    def test_404_product_not_found(self, client):
        resp = client.post("/api/products/bad/variations/bad/generate-360")
        assert resp.status_code == 404

    def test_404_variation_not_found(self, client):
        create = client.post("/api/products", json={"name": "Test"})
        pid = create.json()["id"]
        resp = client.post(f"/api/products/{pid}/variations/bad/generate-360")
        assert resp.status_code == 404
