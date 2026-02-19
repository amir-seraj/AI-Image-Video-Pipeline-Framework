"""Shared test fixtures."""


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
