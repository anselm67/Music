import json
import logging
from asyncio import Queue, QueueEmpty, TaskGroup, run
from collections import Counter
from pathlib import Path

import aiofiles

from dataset import Score

from .pdmx import PDMX


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
        self.score_count += other.score_count
        self.page_count += other.page_count
        self.system_count += other.system_count
        self.staff_count += other.staff_count
        self.bar_count += other.bar_count
        self.system_histo += other.system_histo
        self.staff_histo += other.staff_histo


class PDMXStater:
    pdmx: PDMX
    queue: Queue[Path]

    def __init__(self, pdmx: PDMX):
        self.pdmx = pdmx
        self.queue = Queue()

    async def worker(self) -> PDMXStats:
        stats = PDMXStats()
        while True:
            try:
                mxl_file = self.queue.get_nowait()
                stats.mxl_count += 1
            except QueueEmpty:
                break
            try:
                json_file = self.pdmx.get_path(mxl_file, 'layout')
                async with aiofiles.open(json_file, 'r') as f:
                    text = await f.read()
                stats.layout_count += 1
                layout = Score.from_json(json.loads(text))
                stats.aggregate(layout)
                logging.debug(f"+ {json_file}")
            except FileNotFoundError:
                logging.info(f"- {mxl_file}")
        return stats

    async def async_run(self, num_worker: int) -> PDMXStats:
        for index, row in self.pdmx.df.iterrows():
            mxl_str = row['mxl']
            if not isinstance(mxl_str, str):
                logging.info(
                    f"PDMX.csv@{index}: invalid mxl path {mxl_str}")
            else:
                self.queue.put_nowait(self.pdmx.home / mxl_str)

        async with TaskGroup() as tg:
            tasks = [tg.create_task(self.worker()) for _ in range(num_worker)]

        (first, *rest) = [t.result() for t in tasks]
        for stats in rest:
            first.collect(stats)
        return first

    def run(self, num_worker: int) -> PDMXStats:
        return run(self.async_run(num_worker))
