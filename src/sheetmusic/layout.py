"""Defines the dataclass hierarchy for encoding score layout information."""

from dataclasses import asdict, dataclass, field
from typing import Any, cast

from utils import from_json

# center-{x, y}, width, height
type CenteredBox = tuple[float, float, float, float]


def _box_to_disk(box: dict[str, Any]) -> dict[str, Any]:
    """Serialise a Box dict to the on-disk (top_left, bot_right) point format."""
    return {
        "top_left": [box["left"], box["top"]],
        "bot_right": [box["right"], box["bottom"]],
    }


def _box_from_disk(box: dict[str, Any]) -> dict[str, Any]:
    """Parse the on-disk (top_left, bot_right) point format into Box fields."""
    left, top = box["top_left"]
    right, bottom = box["bot_right"]
    return {"left": left, "top": top, "right": right, "bottom": bottom}


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
    box: Box

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

    def scale(self, w_scale: float, h_scale: float) -> "Staff":
        return Staff(box=self.box.scale(w_scale, h_scale))


@dataclass(frozen=True)
class System:
    """Describes a system - or group of staves - layout.

    Some constraints:
    - All staves in a System have the same number of bars,
    - All staves in a System have the same left and right coordinates,

    Returns:
        _type_: A frozen dataclass describing the system layout.
    """

    bar_numbers: list[int]
    bars: list[int]
    staves: list[Staff]
    # PDMX/Verovio-only: bar numbers scraped from the SVG, used to validate the
    # render against the computed numbering. KernSheet has no SVG, so it omits this.
    svg_bar_numbers: list[int | None] = field(default_factory=list)
    box: Box = field(init=False)

    def __post_init__(self) -> None:
        first, last = self.staves[0].box, self.staves[-1].box
        object.__setattr__(
            self, "box", Box(first.left, first.top, last.right, last.bottom)
        )

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
        for staff in obj["staves"]:
            staff["box"] = _box_to_disk(staff["box"])
        return obj

    def scale(self, w_scale: float, h_scale: float) -> "System":
        return System(
            bar_numbers=self.bar_numbers,
            bars=[int(b * w_scale) for b in self.bars],
            staves=[s.scale(w_scale, h_scale) for s in self.staves],
            svg_bar_numbers=self.svg_bar_numbers,
        )


@dataclass(frozen=True)
class Page:
    # Page number in the pdf (counting from 1)
    page_number: int

    # Image size for the coordinates in this Page
    image_width: int
    image_height: int

    # Staves and validation.
    systems: list[System]
    validated: bool

    image_rotation: float = 0.0

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
            validated=self.validated,
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
        # Hack out the 'box' attribute from all systems; serialise staff boxes in
        # the on-disk (top_left, bot_right) point format.
        for page in obj["pages"]:
            for system in page["systems"]:
                system.pop("box", None)
                for staff in system["staves"]:
                    staff["box"] = _box_to_disk(staff["box"])
        return obj

    def resize(self, width: int, height: int) -> "Score":
        return Score(self.id, [p.resize(width, height) for p in self.pages])

    @staticmethod
    def from_json(obj: Any) -> "Score":
        # On-disk staff boxes use the (top_left, bot_right) point format; rewrite
        # them into Box fields before generic dataclass deserialisation.
        for page in obj["pages"]:
            for system in page["systems"]:
                for staff in system["staves"]:
                    staff["box"] = _box_from_disk(staff["box"])
        return cast(Score, from_json(Score, obj))
