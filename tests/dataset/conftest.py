# conftest.py
from typing import Iterator
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_binaries() -> Iterator[None]:
    with patch("shutil.which", return_value="/usr/bin/true"):
        yield
