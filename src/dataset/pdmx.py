import asyncio
from pathlib import Path

import pandas as pd

from utils import Walker


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.stat().st_mtime >= src_file.stat().st_mtime


class PDMX:
    home: Path

    def __init__(self, home):
        self.home = home
        self.df = pd.read_csv(home / "PDMX.csv")

    def query(self, query_string) -> pd.DataFrame:
        return self.df.query(query_string)

    def verovio_mxl_to_svg(self, mxl_file: Path, force: bool) -> None | tuple[Path, list[str]]:
        relative = mxl_file.relative_to(self.home / "mxl")
        svg_file = (self.home / "svg" / relative).with_suffix(".svg")
        if not force and svg_file.exists() and newer(svg_file, mxl_file):
            return None
        print(f"{mxl_file} -> {svg_file}")
        return None

    def to_svg(self, force: bool = False):
        walker = Walker(self.home / "mxl")

        def builder(file: Path) -> None | tuple[Path, list[str]]:
            return self.verovio_mxl_to_svg(file, force)

        asyncio.run(walker.run("*.mxl", builder))
        asyncio.run(walker.run("*.mxl", builder))
