from dataclasses import asdict, dataclass, field
from typing import Any, cast

from utils import from_json


@dataclass(frozen=True)
class NormBox:
    top_left: tuple[float, float]
    bot_right: tuple[float, float]


@dataclass(frozen=True)
class Box:
    top_left: tuple[int, int]
    bot_right: tuple[int, int]

    @property
    def top(self) -> int:
        return self.top_left[1]

    @property
    def bottom(self) -> int:
        return self.bot_right[1]

    @property
    def left(self) -> int:
        return self.top_left[0]

    @property
    def right(self) -> int:
        return self.bot_right[0]

    def to_cxcywh(self, image_width: int, image_height: int) -> list[float]:
        cx = (self.left + self.right) / 2 / image_width
        cy = (self.top + self.bottom) / 2 / image_height
        w = (self.right - self.left) / image_width
        h = (self.bottom - self.top) / image_height
        return [cx, cy, w, h]

    def scale(self, w_scale: float, h_scale: float) -> 'Box':
        return Box(
            top_left=(int(self.top_left[0] * w_scale),
                      int(self.top_left[1] * h_scale)),
            bot_right=(int(self.bot_right[0] * w_scale),
                       int(self.bot_right[1] * h_scale)),
        )


@dataclass(frozen=True)
class Staff:
    box: Box
    bars: list[int]

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

    def scale(self, w_scale: float, h_scale: float) -> 'Staff':
        return Staff(
            box=self.box.scale(w_scale, h_scale),
            bars=[int(b * w_scale) for b in self.bars]
        )


@dataclass(frozen=True)
class System:
    """Describes a system - or group of staves - layout.

    Some constraints:
    - All staves in a System have the same number of bars,
    - All staves in a System have the same left and right coordinates,

    Returns:
        _type_: A frozen dataclass describing the system layout.
    """
    bar_number: int
    staves: list[Staff]
    box: Box = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "box", Box(
            self.staves[0].box.top_left, self.staves[-1].box.bot_right))

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
    def bar_count(self):
        return len(self.staves[0].bars) - 1

    def asdict(self) -> dict:
        obj = asdict(self)
        obj.pop("box", None)
        return obj

    def scale(self, w_scale: float, h_scale: float) -> 'System':
        return System(
            bar_number=self.bar_number,
            staves=[s.scale(w_scale, h_scale) for s in self.staves]
        )


@dataclass(frozen=True)
class Page:
    # Page number in the pdf (counting from 0)
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
    def bar_count(self):
        return sum(x.bar_count for x in self.systems)

    def resize(self, width: int, height: int) -> 'Page':
        w_scale = width / self.image_width
        h_scale = height / self.image_height
        return Page(
            page_number=self.page_number,
            image_width=width,
            image_height=height,
            systems=[s.scale(w_scale, h_scale) for s in self.systems],
            validated=self.validated,
            image_rotation=self.image_rotation
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

    def asdict(self):
        obj = asdict(self)
        # Hack out the 'box' attribute from all systems.
        for page in obj['pages']:
            for system in page['systems']:
                system.pop('box', None)
        return obj

    def resize(self, width: int, height: int) -> 'Score':
        return Score(self.id, [p.resize(width, height) for p in self.pages])

    @staticmethod
    def from_json(obj: Any) -> 'Score':
        return cast(Score, from_json(Score, obj))
