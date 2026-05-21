#!/usr/bin/env python3

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, TextIO, Type, cast

from kern.parser import Handler, Parser
from kern.typing import (
    Bar,
    Chord,
    Clef,
    Comment,
    Continue,
    Duration,
    Instrument,
    Key,
    Meter,
    Note,
    Pitch,
    Rest,
    SpinePath,
    Token,
)


class Spine:
    pass


class IgnoredSpine(Spine):
    pass


class MergeSpine(Spine):
    into: Spine

    def __init__(self, into: Spine):
        self.into = into


class TokenFormatter:
    formatters: dict[Type, Callable[[Token], str]]
    barno: int
    display_bar_number: bool

    def __init__(self) -> None:
        self.barno = 1
        self.display_bar_number = True
        self.formatters = {
            Bar: self.format_bar,
            Rest: self.format_rest,
            Clef: self.format_clef,
            Key: self.format_key,
            Meter: self.format_meter,
            Continue: self.format_continue,
            Note: self.format_note,
            Chord: self.format_chord,
            Instrument: self.format_instrument,
            SpinePath: self.format_spine_path,
        }

    def format_unknown(self, token: Token) -> str:
        raise ValueError(f"No format for token {token}")

    def format_duration(self, duration: Duration) -> str:
        match duration:
            case Duration(duration=d, dots=0):
                return str(d)
            case _:
                return f"{duration.duration}:{duration.dots}"

    def format_pitch(self, pitch: Pitch) -> str:
        return pitch.name

    def format_bar(self, token: Token) -> str:
        bar = cast(Bar, token)
        if self.display_bar_number:
            barno_str = str(bar.barno) if bar.barno >= 0 else ""
        else:
            barno_str = ""
        if bar.is_final:
            return f"=={barno_str}"
        else:
            return f"={barno_str}"

    def format_rest(self, token: Token) -> str:
        rest = cast(Rest, token)
        assert rest.duration
        return f"rest/{self.format_duration(rest.duration)}"

    def format_clef(self, token: Token) -> str:
        clef = cast(Clef, token)
        return f"clef-{self.format_pitch(clef.pitch)}"

    def format_key(self, token: Token) -> str:
        key = cast(Key, token)
        return f"key{('-' if key.is_flats else '#') * key.count}"

    def format_meter(self, token: Token) -> str:
        meter = cast(Meter, token)
        return f"{meter.numerator}/{meter.denominator}"

    def format_continue(self, _: Token) -> str:
        return "."

    def format_note(self, token: Token) -> str:
        note = cast(Note, token)
        accidentals = ("#" * note.sharps) or ("-" * note.flats)
        duration_text = ""
        if (duration := note.duration) is None:
            assert note.is_gracenote or note.is_groupetto, (
                "Only gracenotes don't have duration."
            )
            duration_text = "/q"
        else:
            duration_text = f"/{self.format_duration(duration)}"
        text = (
            self.format_pitch(note.pitch)
            + (":x" if note.is_drum else "")
            + accidentals
            + duration_text
        )
        return text

    def format_chord(self, token: Token) -> str:
        chord = cast(Chord, token)
        text = " ".join([self.format_note(note) for note in chord.notes])
        return text

    def format_instrument(self, token: Token) -> str:
        instrument = cast(Instrument, token)
        return f"Instr: {instrument.literal}"

    def format_spine_path(self, _: Token) -> str:
        return self.format_continue(Continue())

    def format(self, token: Token) -> str:
        text = self.formatters.get(token.__class__, self.format_unknown)(token)
        self.last_token = token
        return text


class BaseHandler(Handler[Spine]):
    spines: list[Spine]

    def __init__(self) -> None:
        super(BaseHandler, self).__init__()
        self.spines = list([])

    def position(self, spine: Spine) -> int:
        return self.spines.index(spine)

    def output_position(self, spine: Spine) -> int:
        pos = 0
        for s in self.spines:
            if s == spine:
                return pos
            if not isinstance(s, IgnoredSpine):
                pos += 1
        raise ValueError("spine not found.")

    def open_spine(
        self, spine_type: str | None = None, parent: Spine | None = None
    ) -> Spine:
        match spine_type:
            case "**dynam" | "**dynam/2" | "**mxhm" | "**recip" | "**fb" | "**text":
                spine: Spine = IgnoredSpine()
            case _:
                spine = Spine()
        self.spines.append(spine)
        return spine

    def debug_spine(self) -> None:
        print("\t".join(f"{id(spine):#x}" for spine in self.spines))

    def close_spine(self, spine: Spine) -> None:
        # print(f"close: {id(spine):#x}")
        self.spines.remove(spine)
        # self.debug_spine()

    def branch_spine(self, source: Spine) -> Spine:
        branch = MergeSpine(source)
        self.spines.insert(self.position(source) + 1, branch)
        # print(f"branch: {id(source):#x} => {id(branch):#x}")
        # self.debug_spine()
        return branch

    def merge_spines(self, source: Spine, into: Spine) -> None:
        # The source will be close_spine() by the parser.
        # print(f"merge: {id(source):#x} => {id(into):#x}")
        pass

    def rename_spine(self, spine: Spine, name: str) -> None:
        pass


class NormHandler(BaseHandler):
    formatter: TokenFormatter
    output: TextIO | None

    # The current bar number, when none provided.
    bar_numbering: bool
    bar_number: int
    bar_seen: bool
    bar_zero: bool

    def __init__(self, output_path: Path | None):
        super(NormHandler, self).__init__()
        self.output = open(output_path, "w+") if output_path else None
        self.formatter = TokenFormatter()
        self.bar_numbering = False
        self.bar_number = 1
        self.bar_seen = False
        self.bar_zero = False

    last_metric: Meter | None = None

    def check_type(self, tokens: Iterable[Token], t: type | tuple[type, ...]) -> bool:
        return all(isinstance(token, t) for token in tokens)

    def should_skip(self, tokens: list[tuple[Spine, Token]]) -> bool:
        # Sometimes we get both a 4/4 and C meter, skip.
        token = tokens[0][1]
        if isinstance(token, Meter) and all(t == token for _, t in tokens):
            if token == self.last_metric:
                return True
            self.last_metric = token
        else:
            self.last_metric = None
        # Skip metadata annotations — not musical content.
        if self.check_type((t for _, t in tokens), (Comment, Instrument)):
            return True
        # Pure spine paths aren't interesting to us.
        if self.check_type((t for _, t in tokens), SpinePath):
            return True
        # Empty keys aren't interesting either:
        if self.check_type((t for _, t in tokens), Key):
            if all(cast(Key, t).count == 0 for _, t in tokens):
                return True
        return False

    def fix_bar(
        self, tokens: list[tuple[Spine, Token]]
    ) -> list[tuple[Spine, Token]] | None:

        def requires_bar(t: Token) -> bool:
            if isinstance(t, (Note, Chord, Rest)):
                return True
            # A non numbered repeat bar also requires a preceeding bar zero.
            if isinstance(t, Bar) and not cast(Bar, t).requires_valid_bar_number():
                return True
            return False

        # If we see a note or chord before any bar, emit a fake bar 0.
        if not self.bar_zero:
            if any(requires_bar(t) for _, t in tokens):
                bar = Bar("*fake*", 0, False, False, False, False)
                if self.output:
                    self.output.write(
                        "\t".join([self.formatter.format(bar) for _, _ in tokens])
                        + "\n"
                    )
                self.bar_zero = True

        # Adjusts the bar number when none provided.
        if self.check_type((t for _, t in tokens), Bar):
            bars = [cast(Bar, token) for _, token in tokens]
            if self.bar_number <= 2:
                if all(
                    [bar.barno < 0 and bar.requires_valid_bar_number() for bar in bars]
                ):
                    self.bar_numbering = True
                self.bar_zero = True

            if self.bar_numbering:
                bars = [replace(bar, barno=self.bar_number) for bar in bars]
                self.bar_number += 1
            elif (barno := max((bar.barno for bar in bars))) >= 0:
                self.bar_number = barno + 1
            elif any(
                (bar.requires_valid_bar_number() for bar in bars if bar.barno < 0)
            ):
                self.bar_number += 1

            bars = [
                replace(bar, barno=self.bar_number)
                if bar.is_final and bar.barno < 0
                else bar
                for bar in bars
            ]
            # TODO We're not supposed to see any more bars, so it's ok
            # not to incrememt self.bar_number

            if any((bar.barno >= 0 for bar in bars)):
                return list(zip([spine for spine, _ in tokens], bars))
            else:
                return None

        return tokens

    def merge(self, toks: list[Token]) -> list[Token]:
        # We might have merge multiple bar tokens, keep only one.
        if self.check_type(toks, Bar):
            return [toks[0]]

        # Continue tokens are redundant with others non-Continue.
        if self.check_type(toks, Continue):
            return [toks[0]]
        else:
            toks = [tok for tok in toks if not isinstance(tok, Continue)]

        # Rest tokens should have matching length and are redundant.
        if self.check_type(toks, Rest):
            return [toks[0]]
        return toks

    def merge_tokens(self, tokens: list[tuple[Spine, Token]]) -> list[str]:
        output: list[list[Token]] = [[] for _ in range(len(self.spines))]
        for idx, (spine, tok) in enumerate(tokens):
            if isinstance(spine, MergeSpine):
                dst_index = self.output_position(cast(MergeSpine, spine).into)
                output[dst_index].append(tok)
            else:
                output[idx].append(tok)
        # Remove redundant tokens from each remaining spine.
        output = [self.merge(toks) for toks in output if toks]
        # Space join tokens that belong to the same spine.
        formatted_output: list[list[str]] = [
            [self.formatter.format(tok) for tok in toks] for toks in output
        ]
        return [" ".join(toks) for toks in formatted_output]

    def append(self, tokens: list[tuple[Spine, Token]]) -> None:
        tokens = [
            (spine, token)
            for spine, token in tokens
            if not isinstance(spine, IgnoredSpine)
        ]
        if self.should_skip(tokens):
            return
        if not (fixed_bars := self.fix_bar(tokens)):
            return
        tokens = fixed_bars
        output = self.merge_tokens(tokens)
        if self.output:
            self.output.write("\t".join(tok for tok in output if tok) + "\n")

    def done(self) -> None:
        if self.output:
            self.output.close()


def tokenize(
    src_file: Path,
    dst_file: Path | None,
    enable_warnings: bool = False,
) -> bool:
    """Tokenizes a krn file into a normalized form."""
    handler = NormHandler(dst_file)
    parser = Parser.from_file(src_file, handler)
    parser.enable_warnings = enable_warnings
    parser.parse()
    return True
