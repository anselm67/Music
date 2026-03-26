import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd

from utils import compile_filter

from .layout import Score


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.exists() and dst_file.stat().st_mtime >= src_file.stat().st_mtime


type DirClass = Literal[
    'data', 'krn',
    'layout', 'metadata', 'mxl', 'pdf', 'svg', 'png'
]


class PDMX:
    EXTENSIONS: dict[DirClass, str] = {
        'data': '.json',
        'krn': '.krn',
        'layout': '.json',
        'metadata': '.json',
        'mxl': '.mxl',
        'pdf': '.pdf',
        'svg': '.svg',
        'png': '.png'
    }
    home: Path

    def __init__(self, home, name: str = "PDMX.csv"):
        self.home = home
        self.df = pd.read_csv(home / name)

    def relative(self, path) -> Path:
        return path.relative_to(self.home)

    def get_path(self, some: Path, dir_class: DirClass, mkdirs: bool = False) -> Path:
        relative = self.relative(some) if some.is_absolute() else some
        if len(relative.parts) <= 1:
            raise ValueError(f"Unexpected path structure: {some}")
        relative = Path(*relative.parts[1:])
        path = (self.home / dir_class /
                relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        if mkdirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_page_path(self, some: Path, dir_class: DirClass, page_number: int) -> Path:
        relative = self.relative(some) if some.is_absolute() else some
        if len(relative.parts) <= 1:
            raise ValueError(f"Unexpected path structure: {some}")
        relative = Path(*relative.parts[1:])
        path = (self.home / dir_class /
                relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        stem = f"{path.stem}_{page_number:03d}"
        return path.with_stem(stem)

    def get_err_path(self, path: Path) -> Path:
        return path.with_suffix('.err')

    def touch_err_path(self, path: Path):
        err_path = self.get_err_path(path)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.touch()

    def query(self, query_string, metadata: str | None, score: str | None) -> pd.DataFrame:
        metadata_filter, score_filter = (
            compile_filter(metadata) if metadata else None,
            compile_filter(score) if score else None,
        )

        df = self.df.query(query_string)
        if metadata_filter is not None or score_filter is not None:
            def filter_row(row) -> bool:
                if not isinstance(row['mxl'], str) or not isinstance(row['metadata'], str):
                    return False
                try:
                    if metadata_filter is not None:
                        metadata_file = (self.home / row['metadata'])
                        obj = json.loads(metadata_file.read_text())
                        if not metadata_filter(obj):
                            return False
                    if score_filter is not None:
                        layout_file = self.get_path(
                            (self.home / row['mxl']), 'layout')
                        obj = Score.from_json(
                            json.loads(layout_file.read_text()))
                        if not score_filter(obj):
                            return False
                    return True
                except (FileNotFoundError, json.JSONDecodeError):
                    return False
            return df[df.apply(filter_row, axis=1)]
        else:
            return df

    def make(self, mxl_file: Path | None = None, num_workers: int = os.cpu_count() or 4, force: bool = False, dry_run: bool = False):
        from .pdmx_maker import PDMXMaker
        maker = PDMXMaker(self, force=force, dry_run=dry_run)
        maker.run(mxl_file, num_workers)

    def stats(self, num_worker: int = os.cpu_count() or 4):
        from .pdmx_stater import PDMXStater
        stater = PDMXStater(self)
        return stater.run(num_worker)
