# conftest.py
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_binaries():
    with patch("shutil.which", return_value="/usr/bin/true"):
        yield
