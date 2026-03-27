import json
import logging
from asyncio import Queue, QueueEmpty, TaskGroup, run
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import torch
from torch import Tensor
from torchvision.io import decode_image

from dataset import Score

from .pdmx import PDMX


@dataclass(frozen=True)
class Task:
    pass


@dataclass(frozen=True)
class LayoutTask(Task):
    json_file: Path


@dataclass(frozen=True)
class PNGTask(Task):
    png_file: Path


class PDMXStats:
    mxl_count: int

    # Global layout statistics.
    layout_count: int
    score_count: int
    page_count: int
    system_count: int
    staff_count: int
    bar_count: int

    # Par page layout statistics.
    system_histo: Counter = Counter()
    staff_histo: Counter = Counter()
    width100_histo: Counter = Counter()
    height100_histo: Counter = Counter()

    # PNG Stats:
    png_count: int
    pixel_count: int
    pixel_sum: Tensor
    pixel_sum2: Tensor

    def __init__(self):
        self.mxl_count = 0
        self.layout_count = 0
        self.score_count = 0
        self.page_count = 0
        self.system_count = 0
        self.staff_count = 0
        self.bar_count = 0
        self.system_histo = Counter()
        self.staff_histo = Counter()
        self.png_count = 0
        self.pixel_count = 0
        self.pixel_sum = torch.zeros(3)
        self.pixel_sum2 = torch.zeros(3)

    def aggregate(self, score: Score):
        self.score_count += 1
        self.page_count += score.page_count
        self.system_count += score.system_count
        self.staff_count += score.staff_count
        self.bar_count += score.bar_count
        for p in score.pages:
            self.system_histo[p.system_count] += 1
            self.staff_histo[p.staff_count] += 1
            self.width100_histo[p.image_width // 100] += 1
            self.height100_histo[p.image_height // 100] += 1

    def collect(self, other: 'PDMXStats'):
        self.layout_count += other.layout_count
        self.score_count += other.score_count
        self.page_count += other.page_count
        self.system_count += other.system_count
        self.staff_count += other.staff_count
        self.bar_count += other.bar_count
        self.system_histo += other.system_histo
        self.staff_histo += other.staff_histo
        self.png_count += other.png_count
        self.pixel_count += other.pixel_count
        self.pixel_sum += other.pixel_sum
        self.pixel_sum2 += other.pixel_sum2


class PDMXStater:
    pdmx: PDMX
    queue: Queue[Task]

    def __init__(self, pdmx: PDMX):
        self.pdmx = pdmx
        self.queue = Queue()

    async def png_stats(self, stats: PDMXStats, png_file: Path):
        img = decode_image(png_file.as_posix()) / 255.0
        c, h, w = img.shape
        stats.png_count += 1
        stats.pixel_count += (c * h * w)
        stats.pixel_sum += torch.sum(img, dim=[1, 2])
        stats.pixel_sum2 += torch.sum(img ** 2, dim=[1, 2])

    async def layout_stats(self, stats: PDMXStats, json_file: Path):
        try:
            async with aiofiles.open(json_file, 'r') as f:
                text = await f.read()
            stats.layout_count += 1
            score = Score.from_json(json.loads(text))
            stats.aggregate(score)
            # Queue the corresponding PNG tasks:
            if score.page_count <= 1:
                self.queue.put_nowait(
                    PNGTask(self.pdmx.get_path(json_file, 'png')))
            else:
                for p in score.pages:
                    png_file = self.pdmx.get_page_path(
                        json_file, 'png', p.page_number)
                    self.queue.put_nowait(PNGTask(png_file))
            logging.debug(f"+ {json_file}")
        except FileNotFoundError:
            logging.info(f"- {json_file}")

    async def worker(self) -> PDMXStats:
        stats = PDMXStats()
        while True:
            try:
                task = self.queue.get_nowait()
                stats.mxl_count += 1
            except QueueEmpty:
                break
            match task:
                case LayoutTask():
                    await self.layout_stats(stats, task.json_file)
                case PNGTask():
                    await self.png_stats(stats, task.png_file)
        return stats

    async def async_run(self, num_worker: int) -> PDMXStats:
        for index, row in self.pdmx.df.iterrows():
            mxl_str = row['mxl']
            if not isinstance(mxl_str, str):
                logging.info(
                    f"PDMX.csv@{index}: invalid mxl path {mxl_str}")
            else:
                mxl_file = (self.pdmx.home / mxl_str)
                self.queue.put_nowait(LayoutTask(
                    self.pdmx.get_path(mxl_file, 'layout')))

        async with TaskGroup() as tg:
            tasks = [tg.create_task(self.worker()) for _ in range(num_worker)]

        (first, *rest) = [t.result() for t in tasks]
        for stats in rest:
            first.collect(stats)
        return first

    def run(self, num_worker: int) -> PDMXStats:
        return run(self.async_run(num_worker))
