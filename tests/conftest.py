import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--reference-root", default=os.environ.get("PENTACERATOPS_REFERENCE"),
                     help="Preserved triceratops checkout for opt-in migration comparisons")


@pytest.fixture
def reference_root(request):
    value = request.config.getoption("--reference-root")
    if not value:
        pytest.skip("Pass --reference-root for preserved-engine comparisons")
    root = Path(value).resolve()
    if not (root / "triceratops" / "triceratops_new.py").is_file():
        pytest.fail(f"Not a research engine checkout: {root}")
    return root
