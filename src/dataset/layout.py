from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


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
    """Describes a system - or gropu of staves - layout.

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
    def bar_count(self):
        return len(self.staves[0].bars) - 1


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
    def bar_count(self):
        return sum([x.bar_count for x in self.systems])

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True)
class Score:
    id: str
    pages: list[Page]

    @staticmethod
    def from_dict(obj: Any):
        return replace(
            Page(**obj),
            staves=[Staff(**x) for x in obj["staves"]]
        )

    def asdict(self):
        return asdict(self)
