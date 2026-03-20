from dataclasses import dataclass, replace, asdict
from typing import Any


@dataclass(frozen=True)
class Staff:
    rh_top: int
    lh_bot: int
    bars: list[int]

    def box(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (self.bars[0], self.rh_top),
            (self.bars[-1], self.lh_bot)
        )


@dataclass(frozen=True)
class Page:
    # Page number in the pdf (counting from 0)
    page_number: int

    # Image size for the coordinates in this Page
    image_width: int
    image_height: int

    # Staves and validation.
    staves: list[Staff]
    validated: bool

    image_rotation: float = 0.0

    @staticmethod
    def from_dict(obj: Any):
        return replace(
            Page(**obj),
            staves=[Staff(**x) for x in obj["staves"]]
        )

    def asdict(self):
        return asdict(self)
