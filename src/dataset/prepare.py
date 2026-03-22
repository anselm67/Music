import json
import logging
from asyncio import (
    CancelledError,
    Queue,
    QueueEmpty,
    TaskGroup,
    create_subprocess_exec,
    run,
    subprocess,
)
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import aiofiles

from dataset import Page, Score
from verovio import LayoutExtractor, render_command, svg_to_png_command

from .pdmx import PDMX


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.exists() and dst_file.stat().st_mtime >= src_file.stat().st_mtime


@dataclass(frozen=True)
class Task:
    pass


@dataclass(frozen=True)
class MxlTask(Task):
    mxl_file: Path


@dataclass(frozen=True)
class SvgTask(Task):
    svg_files: list[Path]


class Prepare:
    pdmx: PDMX

    queue: Queue[Task]
    force: bool
    dry_run: bool

    def __init__(self, pdmx, force: bool = False, dry_run: bool = False):
        self.pdmx = pdmx
        self.queue = Queue()
        self.force = force
        self.dry_run = dry_run

    async def exec(self, binary: Path, args: list[str]):
        proc = None
        try:
            proc = await create_subprocess_exec(
                binary, *args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return_code = await proc.wait()
            if return_code != 0:
                logging.error(f"{binary} {' '.join(args)}: {return_code}")
        except CancelledError:
            if proc is not None:
                proc.kill()
            raise

    def should_refresh_svg(self, mxl_file: Path, svg_file: Path) -> bool:
        if self.force:
            return True
        # Checks against a one pager svg target.
        if newer(mxl_file, svg_file):
            return False
        # Checks against the first page of a multi-page svg target.
        stem = f"{svg_file.stem}_001"
        tst_file = svg_file.with_stem(stem)
        if newer(mxl_file, tst_file):
            return False
        return True

    def collect_svg_files(self, svg_file: Path) -> list[Path]:
        if svg_file.exists():
            return [svg_file]
        else:
            files = list()
            for i in range(1, 999):
                stem = f"{svg_file.stem}_{i:03d}"
                file = svg_file.with_stem(stem)
                if file.exists():
                    files.append(file)
                else:
                    if not files:
                        raise ValueError(f"{svg_file}: no svg output.")
                    return files
            raise ValueError("Too many pages in score!")

    async def mxl_task(self, mxl_file: Path):
        svg_file = self.pdmx.get_path(mxl_file, 'svg')
        if not self.should_refresh_svg(mxl_file, svg_file):
            logging.debug(f"-> {svg_file}")
        else:
            (binary, args) = render_command(mxl_file, svg_file)
            if self.dry_run:
                logging.info(f"{binary} {' '.join(args)}")
            else:
                logging.info(f"=> {svg_file}")
                await self.exec(binary, args)
        svg_files = self.collect_svg_files(svg_file)
        self.queue.put_nowait(SvgTask(svg_files))

    def should_refresh_layout(self, svg_files: list[Path], json_file: Path) -> bool:
        if self.force:
            return True
        # Checks against a one pager svg target.
        if all(lambda svg_file: newer(svg_file, json_file) for svg_file in svg_files):
            return False
        return True

    async def make_layout(self, svg_files: list[Path]):
        json_file = self.pdmx.get_path(svg_files[0], 'layout')
        if not self.should_refresh_layout(svg_files, json_file):
            logging.debug(f"-> {json_file}")
        elif self.dry_run:
            logging.info(f"layout {svg_files}")
        else:
            logging.info(f"=> {json_file}")
            pages: list[Page] = list()
            page_number = 1
            bar_number = 1
            try:
                for svg_file in svg_files:
                    page = LayoutExtractor(
                        svg_file).parse(page_number, bar_number)
                    pages.append(page)
                    bar_number += page.bar_count
                    page_number += 1
                # Saves the final json file.
                score = Score(
                    str(self.pdmx.relative(json_file).with_suffix('')), pages)
                async with aiofiles.open(json_file, 'w') as f:
                    await f.write(json.dumps(score.asdict(), indent=2))
            except Exception as e:
                json_file.unlink(missing_ok=True)
                logging.error(f"make_layout {svg_file}: {e}")

    async def make_png(self, svg_file: Path):
        png_file = self.pdmx.get_path(svg_file, 'png')
        if not self.force and newer(svg_file, png_file):
            logging.debug(f"-> {png_file}")
        else:
            (binary, args) = svg_to_png_command(svg_file, png_file)
            if self.dry_run:
                logging.info(f"{binary} {' '.join(args)}")
            else:
                logging.info(f"=> {png_file}")
                await self.exec(binary, args)

    async def svg_task(self, svg_files: list[Path]):
        await self.make_layout(svg_files)
        for svg_file in svg_files:
            await self.make_png(svg_file)

    async def worker(self):
        while True:
            try:
                task = self.queue.get_nowait()
            except QueueEmpty:
                break
            try:
                match task:
                    case MxlTask():
                        await self.mxl_task(cast(MxlTask, task).mxl_file)
                    case SvgTask():
                        await self.svg_task(cast(SvgTask, task).svg_files)
            except Exception as e:
                logging.error(e)

    async def run(self, xml_path: Path | None = None, num_worker: int = 5):
        if xml_path is None:
            for index, row in self.pdmx.df.iterrows():
                mxl_str = row['mxl']
                if not isinstance(mxl_str, str):
                    logging.info(
                        f"PDMX.csv@{index}: invalid mxl path {mxl_str}")
                else:
                    self.queue.put_nowait(MxlTask(self.pdmx.home / mxl_str))
        else:
            self.queue.put_nowait(MxlTask(xml_path))

        async with TaskGroup() as tg:
            for _ in range(num_worker):
                tg.create_task(self.worker())

    def prepare(self, xml_path, num_worker: int = 5):
        return run(self.run(xml_path, num_worker))
