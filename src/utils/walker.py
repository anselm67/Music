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
from typing import Callable


# TODO Integrate with option tqdm progress report.
class Walker:
    type CommandBuilder = Callable[
        [Path],
        None | tuple[str | bytes | Path, list[str]]
    ]
    root: Path
    limit: int
    total_count: int
    failed_count: int

    def __init__(self, root: Path, limit: int = os.cpu_count() or 4):
        self.root = root
        self.limit = limit

    async def process(self, cmd_builder: CommandBuilder, file: Path):
        proc = None
        self.total_count += 1
        try:
            if (command := cmd_builder(file)) is None:
                return
            (binary, args) = command
            proc = await create_subprocess_exec(
                binary, *args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            returncode = await proc.wait()
            if returncode != 0:
                self.failed_count += 1
                logging.error(f"Failed to process {file}.")
        except CancelledError:
            if proc is not None:
                proc.kill()
            raise

    async def worker(self, queue: Queue[Path], cmd_builder: CommandBuilder):
        while True:
            try:
                file = queue.get_nowait()
            except QueueEmpty:
                break
            await self.process(cmd_builder, file)

    async def run(self, glob: str, cmd_builder: CommandBuilder) -> tuple[int, int]:
        """Runs a shell command on all files matching the glob pattern in this walker's root directory.

        Args:
            glob (str): The shell-like filename filter, e,g "*.krn"
            cmd_builder (CommandBuilder): Function that returns the command to run for a given matching file.

        Returns:
            int: _description_
        """
        self.total_count = 0
        self.failed_count = 0
        # Queues all files to be processed.
        files = await to_thread(lambda: list(self.root.rglob(glob)))
        queue: Queue[Path] = Queue()
        for file in files:
            queue.put_nowait(file)

        # Processes all files within limit.
        async with TaskGroup() as tg:
            for _ in range(self.limit):
                tg.create_task(self.worker(queue, cmd_builder))

        return self.total_count, self.failed_count
