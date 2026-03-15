import re
from dataclasses import dataclass
from enum import Enum


class Pitch(Enum):
    CCCC = (0, 1)
    DDDD = (0, 2)
    EEEE = (0, 3)
    FFFF = (0, 4)
    GGGG = (0, 5)
    AAAA = (0, 6)
    BBBB = (0, 7)

    CCC = (1, 1)
    DDD = (1, 2)
    EEE = (1, 3)
    FFF = (1, 4)
    GGG = (1, 5)
    AAA = (1, 6)
    BBB = (1, 7)

    CC = (2, 1)
    DD = (2, 2)
    EE = (2, 3)
    FF = (2, 4)
    GG = (2, 5)
    AA = (2, 6)
    BB = (2, 7)

    C = (3, 1)
    D = (3, 2)
    E = (3, 3)
    F = (3, 4)
    G = (3, 5)
    A = (3, 6)
    B = (3, 7)

    c = (4, 1)
    d = (4, 2)
    e = (4, 3)
    f = (4, 4)
    g = (4, 5)
    a = (4, 6)
    b = (4, 7)

    cc = (5, 1)
    dd = (5, 2)
    ee = (5, 3)
    ff = (5, 4)
    gg = (5, 5)
    aa = (5, 6)
    bb = (5, 7)

    ccc = (6, 1)
    ddd = (6, 2)
    eee = (6, 3)
    fff = (6, 4)
    ggg = (6, 5)
    aaa = (6, 6)
    bbb = (6, 7)

    cccc = (7, 1)
    dddd = (7, 2)
    eeee = (7, 3)
    ffff = (7, 4)
    gggg = (7, 5)
    aaaa = (7, 6)
    bbbb = (7, 7)

    ccccc = (8, 1)
    ddddd = (8, 2)
    eeeee = (8, 3)
    fffff = (8, 4)
    ggggg = (8, 5)
    aaaaa = (8, 6)
    bbbbb = (8, 7)

    def order(self) -> int:
        return self.value[0] * 8 + self.value[1]

    def __lt__(self, other):
        if isinstance(other, Pitch):
            return self.order() < other.order()
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Pitch):
            return self.order() <= other.order()
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Pitch):
            return self.order() > other.order()
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Pitch):
            return self.order() >= other.order()
        return NotImplemented


def pitch_from_note_and_octave(note: str, octave: int) -> Pitch:
    index = ['c', 'd', 'e', 'f', 'g', 'a', 'b'].index(note.lower())
    assert index >= 0, f"Invalid note name: {note}, expected [A-Za-z]."
    return Pitch((octave, 1+index))


CLEF_RE = re.compile(r'^\*clef([a-zA-Z])([0-9])$')


def pitch_from_clef(clef: str) -> Pitch:
    m = CLEF_RE.match(clef)
    assert m is not None, "Invalid clef specification."
    name = m.group(1)
    octave = int(m.group(2))
    return pitch_from_note_and_octave(name, octave)


@dataclass(frozen=True)
class Duration:
    duration: int
    dots: int = 0

    def __lt__(self, other) -> bool:
        if isinstance(other, Duration):
            return self.length < other.length
        return NotImplemented

    @property
    def length(self) -> float:
        return (1 / self.duration) * sum(1 / (2**i) for i in range(self.dots + 1))

    def __add__(self, other) -> 'Duration':
        if isinstance(other, Duration):
            return Duration.from_length(self.length + other.length)
        return NotImplemented

    def __sub__(self, other) -> 'Duration':
        if isinstance(other, Duration):
            return Duration.from_length(self.length - other.length)
        return NotImplemented

    @classmethod
    def from_length(cls, length: float) -> 'Duration':
        duration = 1
        while duration > length:
            duration /= 2
        remaining = length - duration
        dot_length = duration / 2
        dot_count = 0
        while remaining > 0:
            remaining -= dot_length
            dot_length /= 2
            dot_count += 1
        return cls(int(1 / duration), dot_count)


@dataclass(frozen=True)
class Token:
    pass

    def __lt__(self, other) -> bool:
        return NotImplemented


@dataclass(frozen=True)
class Clef(Token):
    pitch: Pitch

    def __lt__(self, other) -> bool:
        if isinstance(other, Clef):
            return self.pitch < other.pitch
        return NotImplemented


@dataclass(frozen=True)
class Key(Token):
    is_flats: bool
    count: int


@dataclass(frozen=True)
class Meter(Token):
    numerator: int
    denominator: int


@dataclass(frozen=True)
class Bar(Token):
    text: str
    barno: int
    is_final: bool
    is_repeat_start: bool
    is_repeat_end: bool
    is_invisible: bool

    def requires_valid_bar_number(self):
        return not (
            self.is_final or
            self.is_repeat_start or
            self.is_repeat_end or
            self.is_invisible
        )


@dataclass(frozen=True)
class Continue(Token):
    pass


@dataclass(frozen=True)
class Comment(Token):
    text: str


@dataclass(frozen=True)
class SpinePath(Token):
    indicator: str


@dataclass(frozen=True)
class DurationToken(Token):
    duration: Duration | None

    def __lt__(self, other) -> bool:
        if isinstance(other, DurationToken):
            return self.duration is None or self.duration < other.duration
        return True


@dataclass(frozen=True)
class Rest(DurationToken):
    pass


@dataclass(frozen=True)
class Note(DurationToken):
    pitch: Pitch
    flats: int = 0
    sharps: int = 0
    starts_tie: bool = False
    ends_tie: bool = False
    starts_slur: bool = False
    ends_slur: bool = False
    starts_phrase: bool = False
    ends_phrase: bool = False
    starts_beam: int = 0
    ends_beam: int = 0
    is_gracenote: bool = False
    is_groupetto: bool = False
    has_left_beam: bool = False
    has_right_beam: bool = False
    is_upper_thrill: bool = False
    is_lower_thrill: bool = False

    def __lt__(self, other) -> bool:
        if isinstance(other, Note):
            return self.pitch < other.pitch
        return False


@dataclass(frozen=True)
class Chord(Token):
    notes: list[Note]
