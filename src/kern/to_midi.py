from pathlib import Path
from typing import cast

from kern.parser import Parser
from kern.typing import Chord, Duration, Meter, Note, Rest, Token
from midi import Channel, MidiOutput
from midi import Pitch as MidiPitch
from midi import Velocity

# Convert
# C D E F G A B                     from 1 to 7
# C C# D D# E F F# G G# A A# B      into 1 to 12
TO_TWELVE = {
    1: 1,
    2: 3,
    3: 5,
    4: 6,
    5: 8,
    6: 10,
    7: 12,
}


def note_to_midi(note: Note) -> MidiPitch:
    (octave, number) = note.pitch.value
    number = TO_TWELVE[number]
    number += note.sharps
    number -= note.flats
    if number < 1:
        number = 12
        octave -= 1
    elif number > 12:
        number -= 12
        octave += 1
    return MidiPitch(octave * 12 + number - 1)


class Spine:
    notes: list[Note | Rest | Chord]

    def __init__(self):
        self.notes = list([])
        self.ticks = 0

    def append(self, token: Token):
        match token:
            case Note() | Chord() | Rest():
                self.notes.append(token)
            case Meter():
                # TODO Implement!
                pass


class MidiHandler(Parser[Spine].Handler):
    spines: list[Spine]

    def __init__(self):
        super(MidiHandler, self).__init__()
        self.spines = list([])

    def position(self, spine) -> int:
        return self.spines.index(spine)

    def open_spine(self, spine_type: str | None = None, parent: Spine | None = None) -> Spine:
        match spine_type:
            case "**dynam" | "**dynam/2" | "**mxhm" | "**recip" | "**fb":
                spine = Spine()
            case _:
                spine = Spine()
        self.spines.append(spine)
        return spine

    def close_spine(self, spine: Spine):
        # TODO Removing spines is the way to go eventually.
        # self.spines.remove(spine)
        pass

    def branch_spine(self, source: Spine) -> Spine:
        branch = type(source)()
        self.spines.insert(self.position(source), branch)
        return branch

    def merge_spines(self, source: Spine, into: Spine):
        # The source will be close_spine() by the parser.
        pass

    def rename_spine(self, spine: Spine, name: str):
        pass

    def append(self, tokens: list[tuple[Spine, Token]]):
        for (spine, token) in tokens:
            spine.append(token)

    def done(self):
        pass


class Emitter:
    output: MidiOutput
    ticks_per_quarter: int

    def __init__(self):
        self.output = MidiOutput()
        self.ticks_per_quarter = 480

    def duration_to_ticks(self, duration: Duration) -> int:
        ticks = 4 * self.ticks_per_quarter // duration.duration
        if duration.dots > 0:
            for _ in range(duration.dots):
                dot_ticks = ticks // 2
                ticks += dot_ticks
        return ticks

    def emit_note(self, note: Note, delta_time: int, into: Channel = Channel.Chan1):
        assert note.duration is not None, "Note must have a duration."
        ticks = self.duration_to_ticks(note.duration)
        midi_note = note_to_midi(note).value
        self.output.note_on(into, midi_note, Velocity.Forte, delta_time)
        self.output.note_off(into, midi_note, Velocity.Forte, ticks)

    def emit_track(self, notes: list[Note | Rest | Chord], into: Channel):
        track = self.output.open_chunk('MTrk')
        self.output.time_signature(4, 4)
        self.output.tempo(60)

        pending_delta = 0
        for token in notes:
            match token:
                case Note():
                    note = cast(Note, token)
                    self.emit_note(note, pending_delta, into)
                    pending_delta = 0
                case Chord():
                    chord = cast(Chord, token)
                    for note in chord.notes:
                        self.emit_note(note, pending_delta, into)
                    pending_delta = 0
                case Rest():
                    rest = cast(Rest, token)
                    assert rest.duration is not None, "Rest must have a duration."
                    pending_delta += self.duration_to_ticks(rest.duration)

        self.output.track_end(0)
        self.output.close_chunk(track)

    def emit(self, all_notes: list[list[Note | Rest | Chord]]):
        # Header: Format 1, 2 Tracks
        hd = self.output.open_chunk('MThd')
        self.output.write_u16(1)
        self.output.write_u16(len(all_notes))
        self.output.write_u16(self.ticks_per_quarter)
        self.output.close_chunk(hd)

        # TODO Check we have enough channels.
        all_channels = [
            Channel.Chan0, Channel.Chan1, Channel.Chan2, Channel.Chan3,
            Channel.Chan4, Channel.Chan5, Channel.Chan6, Channel.Chan7,
            Channel.Chan8, Channel.Chan9, Channel.Chan10, Channel.Chan11,
            Channel.Chan12, Channel.Chan13, Channel.Chan14, Channel.Chan15
        ]
        for (notes, channel) in zip(all_notes, all_channels[0:len(all_notes)]):
            self.emit_track(notes, channel)

    def save(self, file: Path):
        self.output.save(file)


def to_midi(kern_file: Path, midi_file: Path):
    handler = MidiHandler()
    parser = Parser.from_file(kern_file, handler)
    parser.parse()
    emitter = Emitter()
    emitter.emit([spine.notes for spine in handler.spines])
    emitter.save(midi_file)
    emitter.emit([spine.notes for spine in handler.spines])
    emitter.save(midi_file)
