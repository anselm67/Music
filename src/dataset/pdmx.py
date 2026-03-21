import asyncio
import logging
from pathlib import Path

import pandas as pd

from utils import Walker
from verovio import render_command


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.stat().st_mtime >= src_file.stat().st_mtime


class PDMX:
    home: Path

    def __init__(self, home):
        self.home = home
        self.df = pd.read_csv(home / "PDMX.csv")

    def query(self, query_string) -> pd.DataFrame:
        return self.df.query(query_string)

    def verovio_mxl_to_svg(self, mxl_file: Path, force: bool, dry_run: bool) -> None | tuple[Path, list[str]]:
        relative = mxl_file.relative_to(self.home / "mxl")
        svg_file = (self.home / "svg" / relative).with_suffix(".svg")
        # Do we need to do the work?
        if not force:
            # Checks against a one pager svg target.
            if svg_file.exists() and newer(mxl_file, svg_file):
                logging.info(f"Ok: {mxl_file}")
                return None
            # Checks against the first page of a multi-page svg target.
            stem = f"{svg_file.stem}_001"
            tst_file = svg_file.with_stem(stem)
            if tst_file.exists() and newer(mxl_file, tst_file):
                logging.info(f"Ok: {mxl_file}")
                return None
        (binary, args) = render_command(mxl_file, svg_file)
        if dry_run:
            logging.info(f"{binary} {' '.join(args)}")
            return None
        logging.info(f"Do: {mxl_file}")
        svg_file.parent.mkdir(parents=True, exist_ok=True)
        return (binary, args)

    def to_svg(self, force: bool = False, dry_run: bool = False) -> int:
        """Renders PDMX mxl files into svg files using verovio.

        Args:
            force (bool, optional): Always re-render mxl file even if a newer svg file exists. Defaults to False.
            dry_run (bool, optional): Say what you'd do, but don't do it. Defaults to False.

        Returns:
            int: Total count of files processed.
        """
        walker = Walker(self.home / "mxl")
        (self.home / "svg").mkdir(exist_ok=True)

        def builder(file: Path) -> None | tuple[Path, list[str]]:
            return self.verovio_mxl_to_svg(file, force, dry_run)

        return asyncio.run(walker.run("*.mxl", builder))
