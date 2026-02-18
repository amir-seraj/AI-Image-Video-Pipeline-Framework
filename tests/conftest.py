"""Shared test fixtures."""


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        default="qwen_image_edit",
        help="Which Qwen model variant to test (default: qwen_image_edit)",
    )
