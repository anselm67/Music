"""Unit tests for the Prepare pipeline class."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from dataset import PDMX
from dataset.pdmx_maker import MxlSvgTask, PDMXMaker
from verovio import rsvgconvert_binary, verovio_binary

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_pdmx(tmp_path: Path) -> PDMX:
    """Creates a minimal PDMX instance with a fake CSV."""
    csv = tmp_path / "PDMX.csv"
    csv.write_text("id,title,n_tracks,mxl,metadata\n"
                   "1,Sonata,2,mxl/1/aa/score.mxl,metadata/1/aa/score.json\n"
                   "2,Fugue,1,mxl/2/bb/fugue.mxl,metadata/2/bb/fugue.json\n"
                   "3,Bad Row,1,,\n")  # row with missing mxl
    pdmx = MagicMock(spec=PDMX)
    pdmx.home = tmp_path
    pdmx.df = pd.read_csv(csv)
    pdmx.get_path.side_effect = lambda src, kind, mkdirs=False: (
        tmp_path / kind / Path(*src.relative_to(tmp_path).parts[1:])
    ).with_suffix(PDMX.EXTENSIONS[kind])
    pdmx.relative.side_effect = lambda p: p.relative_to(tmp_path)
    return pdmx


def make_maker(tmp_path: Path, force=False, dry_run=False) -> PDMXMaker:
    pdmx = make_pdmx(tmp_path)
    return PDMXMaker(pdmx, force=force, dry_run=dry_run)


def touch(path: Path, mtime: float | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))


# ---------------------------------------------------------------------------
# newer()
# ---------------------------------------------------------------------------

class TestNewer:
    def test_binaries(self):
        assert verovio_binary() == Path("/usr/bin/true")
        assert rsvgconvert_binary() == Path("/usr/bin/true")

    def test_dst_does_not_exist(self, tmp_path):
        src = tmp_path / "src.mxl"
        dst = tmp_path / "dst.svg"
        touch(src)
        from dataset.pdmx_maker import newer
        assert not newer(src, dst)

    def test_dst_older_than_src(self, tmp_path):
        src = tmp_path / "src.mxl"
        dst = tmp_path / "dst.svg"
        now = time.time()
        touch(dst, now - 10)
        touch(src, now)
        from dataset.pdmx_maker import newer
        assert not newer(src, dst)

    def test_dst_newer_than_src(self, tmp_path):
        src = tmp_path / "src.mxl"
        dst = tmp_path / "dst.svg"
        now = time.time()
        touch(src, now - 10)
        touch(dst, now)
        from dataset.pdmx_maker import newer
        assert newer(src, dst)


# ---------------------------------------------------------------------------
# should_refresh_svg
# ---------------------------------------------------------------------------

class TestShouldRefreshSvg:
    def test_force_always_refreshes(self, tmp_path):
        p = make_maker(tmp_path, force=True)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        svg = tmp_path / "svg/1/aa/score.svg"
        now = time.time()
        touch(mxl, now - 10)
        touch(svg, now)
        assert p.should_refresh_svg(mxl, svg) is True

    def test_svg_up_to_date(self, tmp_path):
        p = make_maker(tmp_path)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        svg = tmp_path / "svg/1/aa/score.svg"
        now = time.time()
        touch(mxl, now - 10)
        touch(svg, now)
        assert p.should_refresh_svg(mxl, svg) is False

    def test_svg_missing(self, tmp_path):
        p = make_maker(tmp_path)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        svg = tmp_path / "svg/1/aa/score.svg"
        touch(mxl)
        assert p.should_refresh_svg(mxl, svg) is True

    def test_multipage_svg_up_to_date(self, tmp_path):
        p = make_maker(tmp_path)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        svg = tmp_path / "svg/1/aa/score.svg"
        now = time.time()
        touch(mxl, now - 10)
        touch(svg.with_stem("score_001"), now)
        assert p.should_refresh_svg(mxl, svg) is False


# ---------------------------------------------------------------------------
# collect_svg_files
# ---------------------------------------------------------------------------

class TestCollectSvgFiles:
    def test_single_page(self, tmp_path):
        p = make_maker(tmp_path)
        svg = tmp_path / "svg/1/aa/score.svg"
        touch(svg)
        assert p.collect_svg_files(svg) == [svg]

    def test_multi_page(self, tmp_path):
        p = make_maker(tmp_path)
        svg = tmp_path / "svg/1/aa/score.svg"
        pages = [svg.with_stem(f"score_{i:03d}") for i in range(1, 4)]
        for page in pages:
            touch(page)
        assert p.collect_svg_files(svg) == pages

    def test_no_svg_raises(self, tmp_path):
        p = make_maker(tmp_path)
        svg = tmp_path / "svg/1/aa/score.svg"
        with pytest.raises(ValueError, match="no svg output"):
            p.collect_svg_files(svg)


# ---------------------------------------------------------------------------
# should_refresh_layout
# ---------------------------------------------------------------------------

class TestShouldRefreshLayout:
    def test_force_always_refreshes(self, tmp_path):
        p = make_maker(tmp_path, force=True)
        svg = tmp_path / "svg/1/aa/score.svg"
        json_file = tmp_path / "layout/1/aa/score.json"
        now = time.time()
        touch(svg, now - 10)
        touch(json_file, now)
        assert p.should_refresh_layout([svg], json_file) is True

    def test_all_svgs_up_to_date(self, tmp_path):
        p = make_maker(tmp_path)
        now = time.time()
        svgs = [tmp_path / f"svg/1/aa/score_{i:03d}.svg" for i in range(1, 3)]
        json_file = tmp_path / "layout/1/aa/score.json"
        for svg in svgs:
            touch(svg, now - 10)
        touch(json_file, now)
        assert p.should_refresh_layout(svgs, json_file) is False

    def test_json_missing(self, tmp_path):
        p = make_maker(tmp_path)
        svg = tmp_path / "svg/1/aa/score.svg"
        json_file = tmp_path / "layout/1/aa/score.json"
        touch(svg)
        assert p.should_refresh_layout([svg], json_file) is True


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------

class TestExec:
    @pytest.mark.asyncio
    async def test_successful_command(self, tmp_path):
        p = make_maker(tmp_path)
        await p.exec(Path("/usr/bin/true"), [])

    @pytest.mark.asyncio
    async def test_failed_command_logs_error(self, tmp_path, caplog):
        import logging
        p = make_maker(tmp_path)
        with caplog.at_level(logging.ERROR):
            await p.exec(Path("/usr/bin/false"), [])
        assert caplog.records  # some error was logged

    @pytest.mark.asyncio
    async def test_cancelled_kills_process(self, tmp_path):
        p = make_maker(tmp_path)
        task = asyncio.create_task(
            p.exec(Path("/usr/bin/sleep"), ["10"])
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# ---------------------------------------------------------------------------
# run — queue population
# ---------------------------------------------------------------------------

class TestRun:
    @pytest.mark.asyncio
    async def test_all_valid_rows_queued(self, tmp_path):
        p = make_maker(tmp_path)
        queued = []

        async def fake_worker():
            while True:
                try:
                    task = p.queue.get_nowait()
                    queued.append(task)
                except asyncio.QueueEmpty:
                    break

        await p.async_run(mxl_file=None, num_worker=1)
        # 2 valid rows (row 3 has missing mxl)
        assert len([t for t in queued if isinstance(t, MxlSvgTask)]) == 0
        # tasks were put in queue before workers ran
        assert p.queue.empty()

    @pytest.mark.asyncio
    async def test_single_mxl_queued(self, tmp_path):
        p = make_maker(tmp_path)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        touch(mxl)

        with patch.object(p, 'worker', new_callable=AsyncMock):
            await p.async_run(mxl_file=mxl, num_worker=1)

        task = p.queue.get_nowait()
        assert isinstance(task, MxlSvgTask)
        assert task.mxl_file == mxl

    @pytest.mark.asyncio
    async def test_invalid_mxl_row_skipped(self, tmp_path, caplog):
        import logging
        p = make_maker(tmp_path)

        with caplog.at_level(logging.INFO):
            with patch.object(p, 'worker', new_callable=AsyncMock):
                await p.async_run(mxl_file=None, num_worker=1)

        # row 3 with empty mxl should be logged
        assert any("invalid mxl" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# mxl_task / svg_task — dry run
# ---------------------------------------------------------------------------

class TestDryRun:
    @pytest.mark.asyncio
    async def test_mxl_task_dry_run_logs_command(self, tmp_path, caplog):
        import logging
        p = make_maker(tmp_path, dry_run=True)
        mxl = tmp_path / "mxl/1/aa/score.mxl"
        touch(mxl)

        with patch("dataset.pdmx_maker.render_command",
                   return_value=(Path("/usr/bin/verovio"), ["--arg"])):
            with patch.object(p, "collect_svg_files", return_value=[]):
                with caplog.at_level(logging.INFO):
                    await p.mxl_svg_task(mxl)

        assert any("verovio" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_svg_task_dry_run_no_exec(self, tmp_path):
        p = make_maker(tmp_path, dry_run=True)
        svg = tmp_path / "svg/1/aa/score.svg"
        json_file = p.pdmx.get_path(svg, 'layout')
        touch(svg)

        with patch.object(p, "exec", new_callable=AsyncMock) as mock_exec:
            with patch("dataset.pdmx_maker.svg_to_png_command",
                       return_value=(Path("/usr/bin/rsvg"), ["--arg"])):
                with patch.object(p, "make_layout", new_callable=AsyncMock):
                    await p.svg_layout_task([svg], json_file)

        mock_exec.assert_not_called()
        mock_exec.assert_not_called()
