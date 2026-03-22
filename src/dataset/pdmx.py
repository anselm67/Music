import asyncio
import json
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from utils import Walker
from verovio import LayoutExtractor, mxl_to_kern_command, render_command

from .layout import Page, Score


def newer(src_file: Path, dst_file: Path) -> bool:
    return dst_file.exists() and dst_file.stat().st_mtime >= src_file.stat().st_mtime


type DirClass = Literal[
    'data', 'krn',
    'layout', 'metadata', 'mxl', 'pdf', 'svg'
]


class PDMX:
    EXTENSIONS: dict[DirClass, str] = {
        'data': '.json',
        'krn': '.krn',
        'layout': '.json',
        'metadata': '.json',
        'mxl': '.mxl',
        'pdf': '.pdf',
        'svg': '.svg'
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

    def verovio_mxl_to_svg(self, mxl_file: Path, force: bool, dry_run: bool) -> None | tuple[Path, list[str]]:
        svg_file = self.get_path(mxl_file, 'svg', mkdirs=True)
        # Do we need to do the work?
        if not force:
            # Checks against a one pager svg target.
            if newer(mxl_file, svg_file):
                logging.info(f"Ok: {mxl_file}")
                return None
            # Checks against the first page of a multi-page svg target.
            stem = f"{svg_file.stem}_001"
            tst_file = svg_file.with_stem(stem)
            if newer(mxl_file, tst_file):
                logging.info(f"Ok: {mxl_file}")
                return None
        (binary, args) = render_command(mxl_file, svg_file)
        if dry_run:
            logging.info(f"{binary} {' '.join(args)}")
            return None
        logging.info(f"Do: {mxl_file}")
        return (binary, args)

    def to_svg(self, force: bool = False, dry_run: bool = False) -> tuple[int, int]:
        """Renders PDMX mxl files into svg files using verovio.

        Args:
            force (bool, optional): Always re-render mxl file even if a newer svg file exists. Defaults to False.
            dry_run (bool, optional): Say what you'd do, but don't do it. Defaults to False.

        Returns:
            int, int: Total and failed count of files processed.
        """
        walker = Walker(self.home / "mxl")

        def builder(file: Path) -> None | tuple[Path, list[str]]:
            return self.verovio_mxl_to_svg(file, force, dry_run)

        return asyncio.run(walker.run("*.mxl", builder))

    def verovio_mxl_to_krn(self, mxl_file: Path, force: bool, dry_run: bool) -> None | tuple[Path, list[str]]:
        krn_file = self.get_path(mxl_file, 'krn', mkdirs=True)
        # Do we need to do the work?
        if not force:
            # Checks against a one pager svg target.
            if newer(mxl_file, krn_file):
                logging.info(f"Ok: {mxl_file}")
                return None
        (binary, args) = mxl_to_kern_command(mxl_file, krn_file)
        if dry_run:
            logging.info(f"{binary} {' '.join(args)}")
            return None
        logging.info(f"Do: {mxl_file}")
        return (binary, args)

    def to_kern(self, force: bool = False, dry_run: bool = False) -> tuple[int, int]:
        """Converts PDMX mxl files into krn humdrum files using verovio.

        Args:
            force (bool, optional): Always convert mxl file even if a newer svg file exists. Defaults to False.
            dry_run (bool, optional): Say what you'd do, but don't do it. Defaults to False.

        Returns:
            int, int: Total and failed count of files processed.
        """
        walker = Walker(self.home / "mxl")

        def builder(file: Path) -> None | tuple[Path, list[str]]:
            return self.verovio_mxl_to_krn(file, force, dry_run)

        return asyncio.run(walker.run("*.mxl", builder))

    def svg_files(self, mxl_file: Path) -> None | list[Path]:
        svg_file = self.get_path(mxl_file, 'svg')
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
                    return files
            raise ValueError("Too many pages in score!")

    def to_layout(self, force: bool = False) -> tuple[int, int]:
        """Extracts layout infos from the rendering svg files of mxl files.

        Args:
            force (bool, optional): Always extract layout even if a newer json file exists. Defaults to False.
            dry_run (bool, optional): Say what you'd do, but don't do it. Defaults to False.

        Returns:
            int, int: Total and failed count of files processed.
        """
        total = 0
        failed = 0
        for index, row in self.df.iterrows():
            total += 1
            mxl_path = row["mxl"]
            if not isinstance(mxl_path, str):
                logging.info(
                    f"{self.home}/PDMX.csv@{index} invalid mxl path {mxl_path}")
                continue
            mxl_file = self.home / mxl_path

            # Collects all svg files (one per page) for this mxl file.
            if (svg_files := self.svg_files(mxl_file)) is None:
                logging.error(f"{mxl_file}: missing .svg files.")
                failed += 1
                continue

            # Checks if the target file exists and is recent.
            json_file = self.get_path(mxl_file, 'layout', mkdirs=True)
            if not force and all(newer(svg_file, json_file) for svg_file in svg_files):
                logging.info(f"Ok: {mxl_file}")
                continue

            # Processes all pages of the score, adjusting page and bar numbers.
            logging.info(f"Do: {mxl_file}")
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
                    str(self.relative(mxl_file).with_suffix('')), pages)
                with open(json_file, 'w') as f:
                    json.dump(score.asdict(), f, indent=2)
            except Exception as e:
                json_file.unlink(missing_ok=True)
                logging.error(f"{svg_file}: {e}")
        return total, failed
