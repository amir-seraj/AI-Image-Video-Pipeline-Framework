"""Shared test fixtures."""

import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    """Mock GPU-dependent packages before any test module is imported.

    The torchao/diffusers version mismatch in this environment causes
    diffusers to fail at import time. Mocking diffusers here allows
    unit tests that only need casadei.media, casadei.loop, etc. to
    import successfully without a working GPU stack.
    """
    if "diffusers" not in sys.modules:
        sys.modules["diffusers"] = MagicMock()


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default="qwen_image_edit",
        help="Which Qwen model variant to test (default: qwen_image_edit)",
    )
    parser.addoption(
        "--source-image",
        default="model001.jpeg",
        help="Filename in tests/Image/ for the person/model image (default: model001.jpeg)",
    )
