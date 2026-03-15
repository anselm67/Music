# https://www.humdrum.org/guide/
# Formal syntax: https://www.humdrum.org/guide/ch05/
# Note tokens: https://www.humdrum.org/rep/kern/
# Note tokens: https://www.humdrum.org/Humdrum/representations/kern.html#Note%20Tokens
# TODO Handle < and > as in:
# /home/anselm/Downloads/GrandPiano/mozart/piano-sonatas/sonata01-1/min3_up_m-97-100.krn
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from kern.typing import (
    Bar,
    Chord,
    Clef,
    Comment,
    Continue,
    Duration,
    Key,
    Meter,
    Note,
    Pitch,
    Rest,
    SpinePath,
    Token,
    pitch_from_note_and_octave,
)
from utils import iterable_from_file

T = TypeVar("T")


class SpineHolder[T](ABC):

    spine: T
    enable_warnings: bool = True

    def __init__(self, spine: T, enable_warnings: bool = False):
        self.spine = spine
        self.enable_warnings = enable_warnings

    @abstractmethod
    def parse_token(self, text: str) -> Token:
        pass


class KernSpineHolder[T](SpineHolder):

    OPEN_NOTE_RE = re.compile(r'^\.*([ZN\&<>\{\[\(\)\]\}\\/yqP]+)(.*)$')
    NOTE_RE = re.compile(r'^(\d+%)?([\d]+)?(\.*)?([PQq]*)([a-gA-G]+)(.*)$')

    def error(self, msg: str):
        raise SyntaxError(msg)

    def parse_note(self, token, suggested_duration: Optional[Duration] = None) -> Note:
        orig_token = token
        # When an opening (tie, slur, and phrase) starts, keep it for the end.
        additional = ""
        if (m := self.OPEN_NOTE_RE.match(token)):
            additional += m.group(1)
            token = m.group(2)
        if not (m := self.NOTE_RE.match(token)):
            self.error(f"Invalid note token '{orig_token}'")
        ritardendo_text, duration_int, dots, q, pitch, marks = m.groups()
        if marks:
            additional += marks
        if q:
            additional += q
        # Checks for a valid pitch:
        if pitch not in Pitch.__members__:
            self.error(f"Unknown pitch '{pitch}'.")
        # Computes duration with optional dots
        duration = None
        if duration_int:
            duration = Duration(int(duration_int), len(dots))
        elif suggested_duration is not None:
            duration = suggested_duration
        elif "q" in additional or "Q" in additional:
            # Gracenotes withut duration are fine.
            pass
        else:
            self.error(f"Can't parse not '{orig_token}'.")

        # https://www.humdrum.org/Humdrum/representations/kern.html
        # 3.5 Editorial signifiers: XxYy not handled.
        if self.enable_warnings:
            if ritardendo_text:
                print(f"Un-handled ritardendo(?) duration in {orig_token}.")
            for x in r'MmWwsS$R\'/\\Q"`~^':
                if x in additional:
                    print(f"Warning: flags {additional} in {
                          orig_token} not handled.")
                    break
        # TODO Handle 'n' as neither sharp nor flat.
        return Note(
            pitch=Pitch[pitch],
            duration=duration,
            flats=additional.count("-"),
            sharps=additional.count("#"),
            starts_tie="[" in additional,
            ends_tie="]" in additional,
            starts_slur="(" in additional,
            ends_slur=")" in additional,
            starts_phrase="{" in additional,
            ends_phrase="}" in additional,
            starts_beam=additional.count("L"),
            ends_beam=additional.count("J"),
            is_gracenote="q" in additional,
            is_groupetto="Q" in additional,
            has_left_beam="k" in additional,
            has_right_beam="K" in additional,
            is_upper_thrill="T" in additional,
            is_lower_thrill="t" in additional,
        )

    CLEF_RE = re.compile(r'^\*clef([a-zA-Z])([0-9])$')
    SIGNATURE_RE = re.compile(r'\*k\[(([a-z][#-])*)\]')
    METER_RE = re.compile(r'^\*M(\d+)/(\d+)$')
    METRICAL_RE = re.compile(r'^\*met\(([cC]\|?)\)$')

    REST_RE = re.compile(
        r'^\.*Z*(\d+%)?-?([qN\&<>\{\[\(\)\]\}\\/y]*)([\d]+)?(\.*)r(.*)$')
    BAR_RE = re.compile(r'^=(=?)\s*(\d+)?(.*)$')

    def parse_event(self, text: str) -> Token:
        if (m := self.BAR_RE.match(text)):
            is_final, barno, additional = m.group(1), m.group(2), m.group(3)
            barno = int(barno) if barno else -1
            return Bar(
                text,
                barno=barno,
                is_final=(is_final == '=' or additional == "||"),
                is_repeat_start=(additional.endswith(":")),
                is_repeat_end=(additional.startswith(":")),
                is_invisible=(additional == '-' and barno < 0)
            )
        elif text == '.':
            return Continue()
        elif text.startswith("!"):
            # A comment, we don't .
            return Comment(text)
        elif (m := self.REST_RE.match(text)):
            percent, opening, duration_int, dots, left_over = m.groups()
            if not duration_int:
                duration_int = 0
            return Rest(Duration(int(duration_int), len(dots)))
        else:
            notes = list([])
            suggested_duration = None
            for note in text.split():
                token = self.parse_note(note, suggested_duration)
                notes.append(token)
                suggested_duration = token.duration
            if len(notes) == 1:
                return notes[0]
            else:
                return Chord(notes)

    def parse_token(self, text: str) -> Optional[Token]:
        """Parses the given text as a **kern token.

            Returns: The parsed token, or None if the text should be parsed
                by the parser as a Spine path indicator.
        """
        if (m := self.CLEF_RE.match(text)):
            return Clef(pitch_from_note_and_octave(m.group(1), int(m.group(2))))
        elif (m := self.SIGNATURE_RE.match(text)):
            # Empty key signature is allowed.
            if (accidental := m.group(1)):
                # TODO Check that accidental is really valid as the RE isn't prefect.
                return Key(
                    is_flats=(accidental[-1] == '-'),
                    count=len(accidental) // 2
                )
            else:
                # Empty key signatures are ok.
                return Key(False, 0)
        elif (m := self.METER_RE.match(text)):
            return Meter(
                int(m.group(1)),
                int(m.group(2))
            )
        elif (m := self.METRICAL_RE.match(text)):
            metric = m.group(1).upper()
            if metric == 'C':
                return Meter(4, 4)
            elif metric == "C|":
                return Meter(2, 2)
            else:
                self.error(f"Invalid metric '{metric}'.")
        elif text.startswith("*"):
            # This asks the parser to parse the text as a path indicator
            return None
        else:
            return self.parse_event(text)


class DynamSpineHolder(SpineHolder):

    def parse_token(self, text: str) -> Optional[Token]:
        if text.startswith("*"):
            # This still has to return None if a spine path is to be parsed.
            return None
        elif text.startswith("!"):
            # Comment needs to be returned because they're record level.
            return Comment(text)
        else:
            return Continue()


class Parser(Generic[T]):

    class Handler(ABC):

        @abstractmethod
        def open_spine(self,
                       spine_type: Optional[str] = None,
                       parent: Optional[T] = None
                       ) -> T:
            """A new spine is opened.

            This can happen either form the header of the score, in which case a spine_type
            will be provided (e.g. '**kern' or '**dynam') or as a spine is splitted in which
            case the parent spine is provided."""
            pass

        @abstractmethod
        def close_spine(self, spine: T):
            pass

        @abstractmethod
        def branch_spine(self, source: T) -> T:
            pass

        @abstractmethod
        def merge_spines(self, source: T, into: T):
            pass

        @abstractmethod
        def rename_spine(self, spine: T, name: str):
            pass

        @abstractmethod
        def append(self, tokens: List[Tuple[T, Token]]):
            pass

        @abstractmethod
        def done(self):
            pass

    path: Union[str, Path]
    records: Iterator[str]
    lineno: int = 0
    verbose: bool = False

    spines: List[SpineHolder]
    handler: Handler

    enable_warnings: bool

    def __init__(
        self,
        path: Union[str, Path],
        records: Iterable[str],
        handler: Handler,
        enable_warnings: bool = False
    ):
        self.path = path
        self.records = iter(records)
        self.handler = handler
        self.spines = list([])
        self.enable_warnings = enable_warnings

    @staticmethod
    def from_file(path: Union[str, Path], handler: Handler) -> 'Parser':
        return Parser(path, iterable_from_file(path), handler)

    @staticmethod
    def from_text(text: str, handler: Handler) -> 'Parser':
        return Parser("text", iter(text.split("\n")), handler)

    @staticmethod
    def from_iterator(iterator: Iterable[str], handler: Handler) -> 'Parser':
        return Parser("iterator", iterator, handler)

    def error(self, msg: str):
        raise ValueError(f"{self.path}, {self.lineno}: {msg}")

    COMMENT_RE = re.compile(r'^!!.*$')

    def next(self, throw_on_end: bool = False) -> Optional[str]:
        while True:
            line = next(self.records, None)
            self.lineno += 1
            if not line:
                if throw_on_end:
                    self.error("Unexpected end of file.")
                return None
            line = line.strip()
            if not self.COMMENT_RE.match(line):
                return line

    def position(self, spine_holder: SpineHolder) -> int:
        if (pos := self.spines.index(spine_holder)) < 0:
            self.error(f"Spine {spine_holder} missing.")
        return pos

    def insert_spine(self, at: int, spine_holder: SpineHolder):
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.insert(at, spine_holder)
        self.spines = spines

    def open_spine(self, at: int) -> SpineHolder:
        spine_holder = KernSpineHolder(
            self.handler.open_spine("*+"), self.enable_warnings)
        self.insert_spine(at, spine_holder)
        return spine_holder

    def close_spine(self, spine_holder: SpineHolder):
        self.handler.close_spine(spine_holder.spine)
        # Copying is required as these are called from within self.spines iterators.
        spines = list(self.spines)
        spines.remove(spine_holder)
        self.spines = spines

    def branch_spine(self, source_holder: SpineHolder) -> T:
        branch = self.handler.branch_spine(source_holder.spine)
        holder = type(source_holder)(branch)
        holder.enable_warnings = self.enable_warnings
        self.insert_spine(self.position(source_holder), holder)
        return branch

    def merge_spines(self, source_holder: SpineHolder, into_holder: SpineHolder):
        self.handler.merge_spines(source_holder.spine, into_holder.spine)
        self.close_spine(source_holder)

    INDICATOR_RE = re.compile(r'^\*([#:/\w+-]*)$')
    SECTION_LABEL_RE = re.compile(r'^\*>.*$')

    def parse_spine_indicator(
        self,
        spine_holder: SpineHolder,
        indicator: str,
        tokens_iterator: Iterator[Tuple[SpineHolder, str]]
    ) -> Token:
        match indicator:
            case '*-':
                self.close_spine(spine_holder)
            case '*+':
                self.open_spine(self.position(spine_holder))
            case '*^':
                # Branch off into a new spine.
                self.branch_spine(spine_holder)
            case '*v':
                holder = spine_holder
                for next_spine, next_token in tokens_iterator:
                    if next_token == "*v":
                        if holder:
                            self.merge_spines(next_spine, spine_holder)
                        else:
                            holder = next_spine
                    elif next_token == "*":
                        holder = None
                    else:
                        self.parse_spine_indicator(
                            next_spine, next_token, tokens_iterator)
            case '*x':
                self.error("Spine exchange not implemented.")
            case _ if self.SECTION_LABEL_RE.match(indicator):
                # Section labels https://www.humdrum.org/guide/ch20/ skip for now.
                for next_spine, next_token in tokens_iterator:
                    if next_token != '*' and not next_token.startswith('*>'):
                        self.error(f"Unexpected token {
                                   next_token} within section labels.")
            case _ if (m := self.INDICATOR_RE.match(indicator)):
                # Noop spine indicator.
                if (indicator := m.group(1)):
                    self.handler.rename_spine(spine_holder.spine, indicator)
            case _:
                if self.enable_warnings:
                    self.error(f"Unknown spine indicator '{indicator}'.")
        return SpinePath(indicator)

    def parse_token(self, holder: SpineHolder, text: str, tokens_iterator) -> Token:
        try:
            token = holder.parse_token(text)
            if token is None:
                return self.parse_spine_indicator(holder, text, tokens_iterator)
            return token
        except SyntaxError as e:
            raise SyntaxError(f"{self.path}, {self.lineno}: {e}") from e

    def parse(self):
        self.header()
        while True:
            line = cast(str, self.next())
            if not line:
                self.handler.done()
                return
            text_tokens = line.split("\t")
            if len(text_tokens) != len(self.spines):
                self.error(f"Got {len(text_tokens)} tokens for {
                           len(self.spines)} spines.")
            tokens_iterator = zip(self.spines, text_tokens)

            tokens = []
            for holder, text in tokens_iterator:
                tokens.append((
                    holder.spine,
                    self.parse_token(holder, text, tokens_iterator)
                ))
            try:
                self.handler.append(tokens)
            except Exception as e:
                raise SyntaxError(f"{self.path}, {self.lineno}: {e}")

    def header(self):
        kerns = self.next(throw_on_end=True).split()    # type: ignore
        for kern in kerns:
            holder = None
            match kern:
                case "**kern":
                    holder = KernSpineHolder(
                        self.handler.open_spine(kern), self.enable_warnings
                    )
                case "**dynam" | "**dynam/2" | "**mxhm" | "**recip" | "**fb":
                    holder = DynamSpineHolder(
                        self.handler.open_spine(kern), self.enable_warnings)
                case "**text":
                    holder = DynamSpineHolder(
                        self.handler.open_spine(kern), self.enable_warnings)
                case _:
                    self.error(f"Expected a **kern symbol, got '{kern}'.")
            self.spines.append(holder)
