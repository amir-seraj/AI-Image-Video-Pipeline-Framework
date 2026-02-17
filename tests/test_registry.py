# tests/test_registry.py
import pytest
from casadei.models.registry import ModelRegistry
from casadei.models.base import AIModel, ModelCapability, TextConstraint
from casadei.media import MediaBundle


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry()

    def test_register_and_get(self):
        class DummyModel(AIModel):
            capability = ModelCapability(inputs=[TextConstraint()], outputs=[TextConstraint()])
            def load_model(self): pass
            def unload_model(self): pass
            def run(self, inputs: MediaBundle) -> MediaBundle: return inputs
        self.registry.register("dummy", DummyModel)
        assert self.registry.get("dummy") is DummyModel

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="dummy"):
            self.registry.get("dummy")

    def test_list_models(self):
        class DummyModel(AIModel):
            capability = ModelCapability(inputs=[TextConstraint()], outputs=[TextConstraint()])
            def load_model(self): pass
            def unload_model(self): pass
            def run(self, inputs: MediaBundle) -> MediaBundle: return inputs
        self.registry.register("dummy", DummyModel)
        assert "dummy" in self.registry.list_models()

    def test_builtin_registry_has_qwen(self):
        from casadei.models.registry import default_registry
        cls = default_registry.get("qwen_image_edit")
        from casadei.providers.qwen_image_edit import QwenImageEdit
        assert cls is QwenImageEdit
