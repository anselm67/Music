import logging
import os
from asyncio import (
    CancelledError,
    Queue,
    QueueEmpty,
    TaskGroup,
    create_subprocess_exec,
    subprocess,
    to_thread,
)
from pathlib import Path


class Walker:
    root: Path
    limit: int

    def __init__(self, root: Path, limit: int = os.cpu_count() or 4):
        self.root = root
        self.limit = limit

    async def process(self, file: Path):
        proc = None
        try:
            logging.info(f"Processing file: {file}")
            proc = await create_subprocess_exec(
                "/usr/bin/ls", file.as_posix(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            returncode = await proc.wait()
            if returncode != 0:
                logging.info(f"Failed to process {file}.")
        except CancelledError:
            if proc is not None:
                proc.kill()
            raise

    async def worker(self, queue: Queue[Path]):
        while True:
            try:
                file = queue.get_nowait()
            except QueueEmpty:
                break
            await self.process(file)

    async def run(self, glob: str):
        # Queues all files to be processed.
        files = await to_thread(lambda: list(self.root.rglob(glob)))
        queue: Queue[Path] = Queue()
        for file in files:
            queue.put_nowait(file)

        # Processes all files within limit.
        async with TaskGroup() as tg:
            for _ in range(self.limit):
                tg.create_task(self.worker(queue))

        # Processes all files within limit.
        async with TaskGroup() as tg:
            for _ in range(self.limit):
                tg.create_task(self.worker(queue))
                tg.create_task(self.worker(queue))
