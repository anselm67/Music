"""Book-keeping backend for the KernSheet staff editor.

This is the half of OMR's old ``Staffer`` we keep: load / save / delete / paths.
The other half — the heuristic *detector* that suggested a layout from the page
image — is dropped; :meth:`EditorBackend.find_bars` is a stub that will eventually
call the trained Staffer model.

The editor works on a flat *envelope* view (one ``EditorStaff`` per system, storing
the grand-staff ``rh_top``/``lh_bot`` and the barline x-positions). This backend
converts native :class:`sheetmusic.Score` <-> that view, re-applying the thirds
stave-split and recomputing absolute ``bar_numbers`` (via the kern bar walk) on save.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
from cv2.typing import MatLike

from kern import KernReader
from sheetmusic import Box, Page, Score, Staff, System

from .kernsheet_source import KernSheetSource


@dataclass
class EditorStaff:
    """One grand-staff *system* as the editor sees it: top/bottom + barline xs."""

    rh_top: int
    lh_bot: int
    bars: list[int]


@dataclass
class EditorPage:
    page_number: int
    image_width: int
    image_height: int
    staves: list[EditorStaff]
    validated: bool
    image_rotation: float = 0.0


def _to_editor_page(page: Page) -> EditorPage:
    # Both staves of a system share the same barline x-positions (see _split_envelope),
    # so the treble staff's bars are representative of the whole system.
    staves = [
        EditorStaff(
            rh_top=system.staves[0].box.top,
            lh_bot=system.staves[-1].box.bottom,
            bars=list(system.staves[0].bars),
        )
        for system in page.systems
    ]
    return EditorPage(
        page_number=page.page_number,
        image_width=page.image_width,
        image_height=page.image_height,
        staves=staves,
        validated=page.validated,
        image_rotation=page.image_rotation,
    )


def _split_envelope(staff: EditorStaff, image_width: int) -> list[Staff]:
    """Thirds split of a system envelope into [treble, bass] Staff boxes."""
    delta = (staff.lh_bot - staff.rh_top) // 3
    left = staff.bars[0] if staff.bars else 0
    right = staff.bars[-1] if staff.bars else image_width
    treble = Box((left, staff.rh_top), (right, staff.rh_top + delta))
    bass = Box((left, staff.lh_bot - delta), (right, staff.lh_bot))
    return [Staff(box=treble, bars=staff.bars), Staff(box=bass, bars=staff.bars)]


class EditorBackend:
    def __init__(self, home: Path, id: str) -> None:
        self.home = home
        self.id = id
        self.source = KernSheetSource(home)
        if id not in self.source._key:
            raise KeyError(f"unknown score id {id!r}")
        self.key = self.source._key[id]
        # Display-only paths shown in the editor header; the authoritative layout
        # path for I/O is self.layout_path (absolute).
        self.score = SimpleNamespace(
            pdf_path=self.source._pdf[id], json_path=f"layout/{id}.json"
        )
        self.data: list[tuple[MatLike, EditorPage]] | None = None

    @property
    def kern_path(self) -> Path:
        return (self.home / self.key).with_suffix(".krn")

    @property
    def tokens_path(self) -> Path:
        return self.home / "build" / "tokens" / f"{self.key}.tokens"

    @property
    def layout_path(self) -> Path:
        return self.home / "layout" / f"{self.id}.json"

    def staff(self) -> list[tuple[MatLike, EditorPage]]:
        if self.data is None:
            score = self.source.score(self.id)
            data = []
            for page in score.pages:
                tensor = self.source.image(self.id, page.page_number)
                bgr = cv2.cvtColor(tensor.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                data.append((bgr, _to_editor_page(page)))
            self.data = data
        return self.data

    def save(self, pages: tuple[EditorPage, ...]) -> None:
        assert self.data is not None and len(self.data) == len(pages)
        self.data = [(image, page) for (image, _), page in zip(self.data, pages)]
        score = self._to_score(pages)
        self.layout_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout_path.write_text(json.dumps(score.asdict(), indent=2))

    def _to_score(self, pages: tuple[EditorPage, ...]) -> Score:
        kr = KernReader(self.tokens_path)
        if kr.first_bar < 0:
            raise ValueError(f"{self.tokens_path}: no numbered bars in tokens")
        cursor = (0 if kr.has_bar_zero() else 1) + (kr.first_bar - 1)
        out_pages: list[Page] = []
        for ep in pages:
            systems: list[System] = []
            for es in ep.staves:
                n = len(es.bars) - 1
                bar_numbers = list(range(cursor, cursor + n)) if n > 0 else []
                systems.append(
                    System(
                        bar_numbers=bar_numbers,
                        staves=_split_envelope(es, ep.image_width),
                    )
                )
                cursor += max(n, 0)
            out_pages.append(
                Page(
                    page_number=ep.page_number,
                    image_width=ep.image_width,
                    image_height=ep.image_height,
                    systems=systems,
                    validated=ep.validated,
                    image_rotation=ep.image_rotation,
                )
            )
        return Score(id=self.id, pages=out_pages)

    def delete_score(self) -> None:
        catalog_path = self.home / "catalog.json"
        catalog = json.loads(catalog_path.read_text())
        scores = catalog["entries"].get(self.key, {}).get("scores", [])
        catalog["entries"][self.key]["scores"] = [
            s
            for s in scores
            if str(Path(s.get("json_path", "")).with_suffix("")) != self.id
        ]
        catalog_path.write_text(json.dumps(catalog, indent=2))
        self.layout_path.unlink(missing_ok=True)
        self.data = None

    def find_bars(self, image: MatLike) -> list[int]:
        """Detector stub — the heuristic bar finder is gone; the Staffer model
        will eventually fill this in."""
        raise NotImplementedError("staffer detector not wired yet")
