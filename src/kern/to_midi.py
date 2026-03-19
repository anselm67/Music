from abc import ABC, abstractmethod
from collections import deque
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


class Spine():

    def append(self, token: Token):
        pass

    def close(self) -> MidiOutput | None:
        pass

    def branch(self, channel: Channel) -> 'Spine':
        raise ValueError("Can't clone a basic Spine.")


class MidiSpine(Spine):
    channel: Channel
    clock: int
    pending_delta: int
    ticks_per_quarter: int
    track: MidiOutput
    header_offset: int

    def __init__(self,
                 channel: Channel,
                 ticks_per_quarter: int,
                 time_signature: tuple[int, int],
                 tempo: int,
                 pending_delta: int = 0):
        self.channel = channel
        self.clock = 0
        self.ticks_per_quarter = ticks_per_quarter
        self.time_signature = time_signature
        self.tempo = tempo
        self.pending_delta = pending_delta
        self.track = MidiOutput()
        self.header_offset = self.track.open_chunk('MTrk')
        self.track.time_signature(time_signature)
        self.track.tempo(tempo)

    def branch(self, channel: Channel) -> 'MidiSpine':
        return MidiSpine(channel, self.ticks_per_quarter,
                         self.time_signature, self.tempo,
                         self.clock + self.pending_delta)

    def close(self) -> MidiOutput:
        self.track.track_end()
        self.track.close_chunk(self.header_offset)
        return self.track

    def duration_to_ticks(self, duration: Duration) -> int:
        ticks = 4 * self.ticks_per_quarter // duration.duration
        if duration.dots > 0:
            for _ in range(duration.dots):
                dot_ticks = ticks // 2
                ticks += dot_ticks
        return ticks

    def note_duration_to_ticks(self, note: Note) -> int:
        if note.is_gracenote:
            # TODO Make this parametrable.
            return 4 * self.ticks_per_quarter // 32
        else:
            assert note.duration is not None, "Note must have a duration."
            return self.duration_to_ticks(note.duration)

    def emit_note(self, note: Note, delta_time: int) -> int:
        ticks = self.note_duration_to_ticks(note)
        midi_note = note_to_midi(note).value
        self.track.note_on(self.channel, midi_note, Velocity.Forte, delta_time)
        self.track.note_off(self.channel, midi_note, Velocity.Forte, ticks)
        return delta_time + ticks

    def emit_chord(self, chord: list[Note], delta_time: int) -> int:
        first, *rest = chord
        # Emits the first note on, then the rest.
        midi_note = note_to_midi(first).value
        self.track.note_on(self.channel, midi_note,
                           Velocity.Forte, delta_time)
        for note in rest:
            midi_note = note_to_midi(note).value
            self.track.note_on(self.channel, midi_note, Velocity.Forte, 0)
        # Emits the first note off, then the rest.
        assert first.duration is not None, "First note of Chord must have a duration."
        ticks = self.note_duration_to_ticks(first)
        self.track.note_off(self.channel, midi_note, Velocity.Forte, ticks)
        for note in rest:
            self.track.note_off(self.channel, midi_note, Velocity.Forte, 0)
        return delta_time + ticks

    def append(self, token: Token):
        match token:
            case Note():
                note = cast(Note, token)
                self.clock += self.emit_note(note, self.pending_delta)
                self.pending_delta = 0
            case Chord():
                chord = cast(Chord, token)
                self.clock += self.emit_chord(chord.notes, self.pending_delta)
                self.pending_delta = 0
            case Rest():
                rest = cast(Rest, token)
                assert rest.duration is not None, "Rest must have a duration."
                self.pending_delta += self.duration_to_ticks(rest.duration)


class MidiHandler(Parser[Spine].Handler):
    spines: list[Spine]
    time_signature: tuple[int, int]
    tempo: int
    channels: deque[Channel]
    tracks: list[MidiOutput]

    def __init__(self, ticks_per_quarter: int, tempo):
        super(MidiHandler, self).__init__()
        self.ticks_per_quarter = ticks_per_quarter
        self.tracks = list([])
        self.channels = deque(Channel)
        self.spines = list([])
        self.time_signature = (4, 4)
        self.tempo = tempo

    def position(self, spine) -> int:
        return self.spines.index(spine)

    def allocate_channel(self) -> Channel:
        return self.channels.popleft()

    def free_channel(self, channel: Channel):
        self.channels.appendleft(channel)

    def open_spine(self, spine_type: str | None = None, parent: Spine | None = None) -> Spine:
        match spine_type:
            case "**dynam" | "**dynam/2" | "**mxhm" | "**recip" | "**fb":
                spine = Spine()
            case _:
                channel = self.allocate_channel()
                spine = MidiSpine(channel, self.ticks_per_quarter,
                                  self.time_signature, self.tempo)
                print(f"midi-spine {id(spine):#x} on {channel}")
        self.spines.append(spine)
        return spine

    def close_spine(self, spine: Spine):
        match spine:
            case MidiSpine():
                self.free_channel(spine.channel)
                if track := spine.close():
                    print(
                        f"close midi-spine {id(spine):#x} on {spine.channel}")
                    self.tracks.append(track)
        self.spines.remove(spine)

    def branch_spine(self, source: Spine) -> Spine:
        branch = source.branch(self.allocate_channel())
        self.spines.insert(self.position(source), branch)
        return branch

    def merge_spines(self, source: Spine, into: Spine):
        # The source will be close_spine() by the parser.
        pass

    def rename_spine(self, spine: Spine, name: str):
        pass

    def append(self, tokens: list[tuple[Spine, Token]]):
        pass
        for (spine, token) in tokens:
            spine.append(token)

    def done(self):
        pass


def to_midi(kern_file: Path, midi_file: Path, tempo=60):
    ticks_per_quarter = 480
    handler = MidiHandler(ticks_per_quarter, tempo)
    parser = Parser.from_file(kern_file, handler)
    parser.parse()
    # Generates the final midi, wrapping all tracks.
    output = MidiOutput()
    # Header: Format 1, 2 Tracks
    tracks = handler.tracks
    hd = output.open_chunk('MThd')
    output.write_u16(1)
    output.write_u16(len(tracks))
    output.write_u16(ticks_per_quarter)
    output.close_chunk(hd)
    for track in tracks:
        output.append_track(track)
    output.save(midi_file)
