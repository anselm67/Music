import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd


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

    def __init__(self, home):
        self.home = home
        self.df = pd.read_csv(home / "PDMX.csv")

    def relative(self, path) -> Path:
        return path.relative_to(self.home)

    def get_path(self, some: Path, dir_class: DirClass, mkdirs: bool = False) -> Path:
        relative = self.relative(some)
        if len(relative.parts) <= 1:
            raise ValueError(f"Unexpected path structure: {some}")
        relative = Path(*relative.parts[1:])
        path = (self.home / dir_class /
                relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        if mkdirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_page_path(self, some: Path, dir_class: DirClass, page_number: int) -> Path:
        relative = self.relative(some)
        if len(relative.parts) <= 1:
            raise ValueError(f"Unexpected path structure: {some}")
        relative = Path(*relative.parts[1:])
        path = (self.home / dir_class /
                relative).with_suffix(PDMX.EXTENSIONS[dir_class])
        stem = f"{path.stem}_{page_number:03d}"
        return path.with_stem(stem)

    def query(self, query_string, parts_count: int = -1) -> pd.DataFrame:
        df = self.df.query(query_string)
        if parts_count > 0:

            def filter_part_count(row):
                try:
                    meta = json.loads(
                        (self.home / row["metadata"]).read_text())
                    return meta.get("data", {}).get("score", {}).get("parts") == parts_count
                except (FileNotFoundError, json.JSONDecodeError):
                    return False

            df = df[df.apply(filter_part_count, axis=1)]
        return df

    def make(self, mxl_file: Path | None = None, num_workers: int = os.cpu_count() or 4, force: bool = False, dry_run: bool = False):
        from .pdmx_maker import PDMXMaker
        maker = PDMXMaker(self, force=force, dry_run=dry_run)
        maker.run(mxl_file, num_workers)
