"""``Source`` implementation backed by the migrated **KernSheet** dataset.

KernSheet holds real scanned/published piano scores with manually-reviewed layout.
A one-off migration (``kernsheet migrate``) rewrote every legacy envelope layout into
a native :class:`sheetmusic.Score` under ``<home>/layout/<id>.json``, so
:meth:`KernSheetSource.score` just reloads it — no in-memory translation.

The score key (``Score.id``) is the layout-json stem (e.g.
``bach/fugue/bwv_856/bwv_856-0``): globally unique, so neither shared
"Prelude & Fugue"/"-all" PDFs nor ``-0``/``-1`` editions collide. The ``.krn`` / tokens
live under the entry *key* (shared across a work's editions), which
:class:`KernSheetSource` recovers from ``catalog.json``.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from torch import Tensor
from torchvision.io import decode_image

from kern import KernReader
from sheetmusic import Score


class KernSheetSource:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.layout_dir = home / "layout"
        self.tokens_dir = home / "build" / "tokens"
        self.png_dir = home / "build" / "png"
        # Map each score id -> (entry key for krn/tokens, pdf path for the page image).
        # id is the layout-json stem; the pdf may live under a different work's dir
        # (shared "Prelude & Fugue" scans), hence the explicit catalog lookup.
        catalog = json.loads((home / "catalog.json").read_text())["entries"]
        self._key: dict[str, str] = {}
        self._pdf: dict[str, str] = {}
        for key, entry in catalog.items():
            for score in entry["scores"]:
                jp = score.get("json_path", "")
                if not jp and score.get("pdf_path"):
                    jp = str(Path(score["pdf_path"]).with_suffix(".json"))
                if not jp:
                    continue
                id = str(Path(jp).with_suffix(""))
                self._key[id] = key
                self._pdf[id] = score.get("pdf_path", "")

    def scores(self) -> Iterator[Score]:
        for id in self._key:
            layout = self.layout_dir / f"{id}.json"
            if layout.exists():
                yield Score.from_json(json.loads(layout.read_text()))

    def score(self, id: str) -> Score:
        layout = self.layout_dir / f"{id}.json"
        return Score.from_json(json.loads(layout.read_text()))

    def image(self, id: str, page_number: int) -> Tensor:
        cached = self.png_dir / f"{id}-{page_number:03d}.png"
        if not cached.exists():
            self._render_pages(id)
        return decode_image(cached.as_posix())

    def records(self, id: str, first_bar: int, last_bar: int) -> list[str] | None:
        if id not in self._key:
            return None
        tokens_file = self.tokens_dir / f"{self._key[id]}.tokens"
        return KernReader(tokens_file).get_text(first_bar, last_bar)

    def _render_pages(self, id: str) -> None:
        """Render every page of the score's PDF into the annotation pixel space, cached.

        Reproduces OMR's ``apply_page_transforms``: scale each raw page so its width
        equals ``page.image_width`` (the space the boxes were drawn in), then rotate.
        """
        from pdf2image import convert_from_path

        pdf = self._pdf.get(id, "")
        if not pdf:
            raise ValueError(f"no PDF registered for score id {id!r}")
        score = self.score(id)
        images = convert_from_path(self.home / pdf)
        self.png_dir.mkdir(parents=True, exist_ok=True)
        for page in score.pages:
            if page.page_number >= len(images):
                continue
            rgb = np.array(images[page.page_number])
            h, w = rgb.shape[:2]
            scale = page.image_width / w
            rgb = cv2.resize(
                rgb, (page.image_width, int(h * scale)), interpolation=cv2.INTER_AREA
            )
            if abs(page.image_rotation) > 0:
                height, width = rgb.shape[:2]
                matrix = cv2.getRotationMatrix2D(
                    (width // 2, height // 2), page.image_rotation, 1
                )
                rgb = cv2.warpAffine(rgb, matrix, (width, height))
            dst = self.png_dir / f"{id}-{page.page_number:03d}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(dst.as_posix(), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
