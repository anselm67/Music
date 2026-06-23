"""Defines the dataclass hierarchy for encoding score layout information."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, cast

from utils import from_json

# center-{x, y}, width, height
type CenteredBox = tuple[float, float, float, float]


class Status(str, Enum):
    """A page's review lifecycle. Only ``VALIDATED`` pages feed training/eval;
    ``PENDING`` is the un-reviewed output of ``kernsheet detect``, ``REJECTED`` is
    a page a human judged unusable (e.g. an unfixable scan)."""

    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def top_left(self) -> tuple[int, int]:
        """Top-left corner as an (x, y) point (e.g. for cv2.rectangle)."""
        return (self.left, self.top)

    @property
    def bot_right(self) -> tuple[int, int]:
        """Bottom-right corner as an (x, y) point (e.g. for cv2.rectangle)."""
        return (self.right, self.bottom)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def up(self, delta: int) -> "Box":
        return Box(self.left, self.top - delta, self.right, self.bottom - delta)

    def to_cxcywh(self, image_width: int, image_height: int) -> CenteredBox:
        cx = (self.left + self.right) / 2 / image_width
        cy = (self.top + self.bottom) / 2 / image_height
        w = (self.right - self.left) / image_width
        h = (self.bottom - self.top) / image_height
        return (cx, cy, w, h)

    def scale(self, w_scale: float, h_scale: float) -> "Box":
        return Box(
            int(self.left * w_scale),
            int(self.top * h_scale),
            int(self.right * w_scale),
            int(self.bottom * h_scale),
        )

    def contains(self, xy: tuple[int, int]) -> bool:
        x, y = xy
        return x >= self.left and x <= self.right and y <= self.bottom and y >= self.top

    @staticmethod
    def from_cxcywh(
        size: tuple[int, int], cx: float, cy: float, w: float, h: float
    ) -> "Box":
        left = int((cx - w / 2) * size[0])
        top = int((cy - h / 2) * size[1])
        right = int((cx + w / 2) * size[0])
        bot = int((cy + h / 2) * size[1])
        return Box(left, top, right, bot)


@dataclass(frozen=True)
class Staff:
    """A staff is a vertical band: it owns only ``top``/``bottom``. Its horizontal
    extent is the parent ``System``'s (its ``bars``), so a staff carries no x — read
    a staff's full box from ``System.staff_boxes``."""

    top: int
    bottom: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def scale(self, h_scale: float) -> "Staff":
        return Staff(top=int(self.top * h_scale), bottom=int(self.bottom * h_scale))


@dataclass(frozen=True)
class System:
    """Describes a system - or group of staves - layout.

    All staves in a System share the same horizontal extent — the barline span
    ``bars[0]..bars[-1]`` — so x lives only here, not per-staff. A staff carries just
    its vertical band; ``staff_boxes`` reconstructs full staff boxes on demand.
    """

    bar_numbers: list[int]
    bars: list[int]
    staves: list[Staff]
    # PDMX/Verovio-only: bar numbers scraped from the SVG, used to validate the
    # render against the computed numbering. KernSheet has no SVG, so it omits this.
    svg_bar_numbers: list[int | None] = field(default_factory=list)
    box: Box = field(init=False)

    def __post_init__(self) -> None:
        # The system's barlines are the single source of horizontal extent (x runs
        # from the first to the last barline); the staff hull gives the vertical.
        left = self.bars[0] if self.bars else 0
        right = self.bars[-1] if self.bars else 0
        top = min((s.top for s in self.staves), default=0)
        bottom = max((s.bottom for s in self.staves), default=0)
        object.__setattr__(self, "box", Box(left, top, right, bottom))

    @property
    def staff_boxes(self) -> list["Box"]:
        """Each staff's full box: the system's horizontal span (from ``bars``) with
        the staff's own top/bottom. Staves don't store x — it is the system's."""
        left, right = self.box.left, self.box.right
        return [Box(left, s.top, right, s.bottom) for s in self.staves]

    @property
    def top(self) -> int:
        return self.box.top

    @property
    def bottom(self) -> int:
        return self.box.bottom

    @property
    def left(self) -> int:
        return self.box.left

    @property
    def right(self) -> int:
        return self.box.right

    @property
    def staff_count(self) -> int:
        return len(self.staves)

    @property
    def bar_count(self) -> int:
        return max(len(self.bars) - 1, 0)

    @property
    def first_bar_number(self) -> int:
        return self.bar_numbers[0]

    @property
    def last_bar_number(self) -> int:
        return self.first_bar_number + self.bar_count

    def asdict(self) -> dict[str, object]:
        obj = asdict(self)
        obj.pop("box", None)
        return obj

    def scale(self, w_scale: float, h_scale: float) -> "System":
        return System(
            bar_numbers=self.bar_numbers,
            bars=[int(b * w_scale) for b in self.bars],
            staves=[s.scale(h_scale) for s in self.staves],
            svg_bar_numbers=self.svg_bar_numbers,
        )


@dataclass(frozen=True)
class Page:
    # Page number in the pdf (counting from 1)
    page_number: int

    # Image size for the coordinates in this Page
    image_width: int
    image_height: int

    # Staves and review state.
    systems: list[System]
    status: Status

    # Names of the automated reviews a human has acknowledged ("ok") on this page,
    # suppressing their findings. See sheetmusic.reviews / kernsheet.reviews.
    reviewed: list[str] = field(default_factory=list)

    image_rotation: float = 0.0

    @property
    def validated(self) -> bool:
        return self.status == Status.VALIDATED

    @property
    def system_count(self) -> int:
        return len(self.systems)

    @property
    def staff_count(self) -> int:
        return sum(s.staff_count for s in self.systems)

    @property
    def bar_count(self) -> int:
        return sum(x.bar_count for x in self.systems)

    @property
    def first_bar_number(self) -> int:
        return self.systems[0].first_bar_number

    @property
    def next_bar_number(self) -> int:
        last_system = self.systems[-1]
        return last_system.first_bar_number + last_system.bar_count

    def resize(self, width: int, height: int) -> "Page":
        w_scale = width / self.image_width
        h_scale = height / self.image_height
        return Page(
            page_number=self.page_number,
            image_width=width,
            image_height=height,
            systems=[s.scale(w_scale, h_scale) for s in self.systems],
            status=self.status,
            reviewed=self.reviewed,
            image_rotation=self.image_rotation,
        )


@dataclass(frozen=True)
class Score:
    id: str
    pages: list[Page]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def system_count(self) -> int:
        return sum(p.system_count for p in self.pages)

    @property
    def staff_count(self) -> int:
        return sum(p.staff_count for p in self.pages)

    @property
    def bar_count(self) -> int:
        return sum(p.bar_count for p in self.pages)

    def asdict(self) -> dict[str, object]:
        obj = asdict(self)
        # Hack out the derived 'box' attribute from all systems.
        for page in obj["pages"]:
            for system in page["systems"]:
                system.pop("box", None)
        return obj

    def resize(self, width: int, height: int) -> "Score":
        return Score(self.id, [p.resize(width, height) for p in self.pages])

    @staticmethod
    def from_json(obj: Any) -> "Score":
        return cast(Score, from_json(Score, obj))
