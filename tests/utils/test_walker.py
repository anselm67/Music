import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from utils import Walker

# ---------------------------------------------------------------------------
# Testable subclass
# ---------------------------------------------------------------------------


class RecordingWalker(Walker):
    def __init__(self, root: Path):
        super().__init__(root)
        self.processed: list[Path] = []

    async def process(self, cmd_builder: Walker.CommandBuilder, file: Path):
        self.processed.append(file)

# Dummy command builder.


def command_builder(file: Path) -> tuple[str, list[str]]:
    return "/usr/bin/ls", list(file.as_posix())
# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_files_processed(tmp_path):
    (tmp_path / "a.mxl").touch()
    (tmp_path / "b.mxl").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mxl").touch()

    walker = RecordingWalker(tmp_path)
    await walker.run("*.mxl", command_builder)

    assert len(walker.processed) == 3


@pytest.mark.asyncio
async def test_only_matching_glob(tmp_path):
    (tmp_path / "a.mxl").touch()
    (tmp_path / "b.txt").touch()

    walker = RecordingWalker(tmp_path)
    await walker.run("*.mxl", command_builder)

    assert len(walker.processed) == 1
    assert walker.processed[0].suffix == ".mxl"


@pytest.mark.asyncio
async def test_empty_directory(tmp_path):
    walker = RecordingWalker(tmp_path)
    await walker.run("*.mxl", command_builder)

    assert walker.processed == []


@pytest.mark.asyncio
async def test_concurrency_limit(tmp_path):
    for i in range(20):
        (tmp_path / f"{i}.mxl").touch()

    active = 0
    peak = 0

    class PeakWalker(Walker):
        async def process(self, cmd_builder: Walker.CommandBuilder, file: Path):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)  # yield to let other tasks run
            active -= 1

    walker = PeakWalker(tmp_path)
    await walker.run("*.mxl", command_builder)

    assert peak <= walker.limit


@pytest.mark.asyncio
async def test_process_called_with_correct_paths(tmp_path):
    expected = {tmp_path / "a.mxl", tmp_path / "b.mxl"}
    for f in expected:
        f.touch()

    walker = RecordingWalker(tmp_path)
    await walker.run("*.mxl", command_builder)

    assert set(walker.processed) == expected
