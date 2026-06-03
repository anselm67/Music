"""KernSheet dataset management: catalog I/O, path resolution, and health checks."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, cast

import cv2
import numpy as np
from cv2.typing import MatLike
from pdf2image import convert_from_path

from kern import KernReader, tokenize
from sheetmusic import Score
from utils import from_json


@dataclass
class KernScore:
    pdf_path: str
    pdf_url: str
    json_path: str
    id: str = ""


@dataclass
class KernEntry:
    source: str
    imslp_query: str
    imslp_url: str
    scores: list[KernScore] = field(default_factory=list)


@dataclass
class Catalog:
    version: int = 2
    entries: dict[str, KernEntry] = field(default_factory=dict)


class KernSheet:
    CATALOG_NAME = "catalog.json"
    catalog: Catalog
    id2score: dict[str, KernScore]
    id2key: dict[str, str]

    def __init__(self, home: Path) -> None:
        self.home = home
        self.id2score = {}
        self.id2key = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        text = (self.home / self.CATALOG_NAME).read_text()
        self.catalog = cast(Catalog, from_json(Catalog, json.loads(text)))
        for key, entry in self.catalog.entries.items():
            for score in entry.scores:
                # Canonical id = the layout-json stem, matching migrate.build_score
                # and Score.id. (The catalog's own `id` field is a stale `#N` encoding
                # nothing else speaks; derive here so every consumer keys the same way.)
                score.id = (
                    str(Path(score.json_path).with_suffix(""))
                    if score.json_path
                    else key
                )
                self.id2score[score.id] = score
                self.id2key[score.id] = key

    def save_catalog(self) -> None:
        (self.home / self.CATALOG_NAME).write_text(
            json.dumps(asdict(self.catalog), indent=4)
        )

    def kern_path(self, key: str) -> Path:
        return (self.home / key).with_suffix(".krn")

    def tokens_path(self, key: str) -> Path:
        return self.home / f"build/tokens/{key}.tokens"

    def png_path(self, id: str, page_number: int) -> Path:
        score = self.id2score[id]
        pdf_path = self.pdf_path(score)
        path_str = self.relative(pdf_path)
        stem = f"{pdf_path.stem}-{page_number:03d}"
        return (self.home / "build/png" / path_str).with_stem(stem).with_suffix(".png")

    def pdf_path(self, score: KernScore) -> Path:
        return self.home / score.pdf_path

    def layout_path(self, score: KernScore) -> Path:
        return self.home / "layout" / score.json_path

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.home))

    def items(self) -> Iterator[tuple[str, KernScore]]:
        for key, entry in self.catalog.entries.items():
            for score in entry.scores:
                yield key, score

    def get_kern_scores(self, key: str) -> list[KernScore]:
        return self.catalog.entries[key].scores

    def load_score(self, id: str) -> Score:
        score = self.id2score[id]
        if not score.json_path:
            raise ValueError(f"{id}: unmigrated score has no layout")
        return Score.from_json(json.loads(self.layout_path(score).read_text()))

    def save_score(self, id: str, score: Score) -> None:
        kern_score = self.id2score[id]
        path = self.layout_path(kern_score)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(score.asdict(), indent=2))

    def load_tokens(self, id: str) -> KernReader:
        return KernReader(self.tokens_path(self.id2key[id]))

    def delete_score(self, key: str, score: KernScore) -> None:
        entry = self.catalog.entries.get(key)
        if entry:
            entry.scores = [s for s in entry.scores if s != score]
            self.save_catalog()
        # TODO Delete removed score associated files.

    def check(self, verbose: bool = False) -> None:
        """Validate catalog integrity against the filesystem."""

        def v(msg: str) -> None:
            if verbose:
                print(msg)

        file_count, noent_count = 0, 0
        kern_seen: set[str] = set()
        for root, _, filenames in os.walk(self.home):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix != ".krn":
                    continue
                file_count += 1
                key = str(file.with_suffix("").relative_to(self.home))
                kern_seen.add(key)
                if key not in self.catalog.entries:
                    noent_count += 1
                    v(f"orphan .krn (no catalog entry): {key}")

        nokern_count, noscore_count = 0, 0
        score_count = 0
        score_nopdf, score_nolayout, broken_pdf, broken_layout = (
            0,
            0,
            0,
            0,
        )

        for key, entry in self.catalog.entries.items():
            if key not in kern_seen:
                nokern_count += 1
                v(f"{key}: no .krn file")
            if not entry.scores:
                noscore_count += 1
                v(f"{key}: no scores")
                continue
            score_count += len(entry.scores)
            for s in entry.scores:
                if not s.pdf_path:
                    score_nopdf += 1
                    v(f"{key}: score has no pdf_path")
                elif not self.pdf_path(s).exists():
                    broken_pdf += 1
                    v(f"{key}: pdf missing: {s.pdf_path}")
                if not s.json_path:
                    score_nolayout += 1
                    v(f"{key}: score has no json_path")
                elif not self.layout_path(s).exists():
                    broken_layout += 1
                    v(f"{key}: legacy json missing: {s.json_path}")

        print(
            f"{file_count} kern files:\n"
            f"  without entries: {noent_count}\n"
            f"{len(self.catalog.entries)} catalog entries:\n"
            f"  without .krn file: {nokern_count}\n"
            f"  without scores:    {noscore_count}\n"
            f"  score count:       {score_count}\n"
            f"    without pdf:     {score_nopdf}\n"
            f"    broken pdf:      {broken_pdf}\n"
            f"    without json:    {score_nolayout}\n"
            f"    broken json:     {broken_layout}\n"
        )

    def _transform(self, image: MatLike, width: int, angle: float) -> MatLike:
        h, w = image.shape[:2]
        scale = width / w
        image = cv2.resize(image, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
        if abs(angle) > 0:
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (width, height))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def _rebuild_images(self, kern_score: KernScore, score: Score) -> None:
        pdf_path = self.pdf_path(kern_score)
        images = convert_from_path(pdf_path)
        for page in score.pages:
            if page.page_number > len(images):  # page_number is 1-based
                logging.warning(
                    f"{kern_score.id}: score references page {page.page_number} "
                    f"but pdf has only {len(images)} page(s)"
                )
                continue
            image = self._transform(
                np.array(images[page.page_number - 1]),
                page.image_width,
                page.image_rotation,
            )
            png_path = self.png_path(kern_score.id, page.page_number)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(png_path.as_posix(), image)

    def make(self) -> None:
        """Ensures that for each entry, we have a tokens file, and for each pdf page of
        the score, we have a corresponding png image."""
        failed = 0
        for key, entry in self.catalog.entries.items():
            tok_path = self.tokens_path(key)
            if not tok_path.exists():
                try:
                    tok_path.parent.mkdir(parents=True, exist_ok=True)
                    tokenize(self.kern_path(key), tok_path)
                except Exception as e:
                    failed += 1
                    logging.error(f"tokenize {key}: {e}")
            for kern_score in entry.scores:
                try:
                    score = self.load_score(kern_score.id)
                    rebuild_images = False
                    for page in score.pages:
                        png_path = self.png_path(kern_score.id, page.page_number)
                        if not png_path.exists():
                            rebuild_images = True
                            break
                    if rebuild_images:
                        self._rebuild_images(kern_score, score)
                except Exception as e:
                    failed += 1
                    logging.error(f"make {kern_score.id}: {e}")
        if failed:
            logging.warning(f"make: {failed} item(s) failed")
