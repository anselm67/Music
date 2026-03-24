from dataclasses import asdict, dataclass, field
from typing import Any, cast

from utils import from_json


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

    @staticmethod
    def from_json(obj: Any) -> 'Score':
        return cast(Score, from_json(Score, obj))
