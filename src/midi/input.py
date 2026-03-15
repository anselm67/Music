import array
from abc import ABC, abstractmethod
from typing import Optional

from midi.typing import (
    Channel,
    CloseTrackEvent,
    ControlChangeEvent,
    DataEvent,
    Event,
    EventType,
    Format,
    HeaderDataEvent,
    Instrument,
    KeySignatureEvent,
    NoteOffEvent,
    NoteOnEvent,
    Notes,
    OpenTrackEvent,
    ProgramChangeEvent,
    SequenceNumberEvent,
    TempoEvent,
    TextEvent,
    TimeSignatureEvent,
)


class MidiInput(ABC):

    buf: array.array
    pos: int = 0

    def __init__(self, buf: array.array):
        self.buf = buf

    def debug(self, start_off: int = 5, end_off: int = 5):
        start = max(0, self.pos - start_off)
        end = min(self.pos + end_off, len(self.buf))
        print(''.join([f"{
            hex(self.buf[pos])} " for pos in range(start, end)]))

    def parse(self):
        self.decode_header()

    def next(self):
        value = self.buf[self.pos]
        self.pos += 1
        return value

    def skip(self, count: int):
        self.pos += count

    def done(self):
        return self.pos >= len(self.buf)

    def read_varlen(self):
        byte = self.next()
        value = (byte & 0x7F)
        while (byte & 0x80) != 0:
            byte = self.next()
            value = (value << 7) + (byte & 0x7F)
        return value

    def read_u16(self):
        return (
            ((self.next() & 0xFF) << 8) +
            (self.next() & 0xFF)
        )

    def read_u24(self):
        return (
            ((self.next() & 0xFF) << 16) +
            ((self.next() & 0xFF) << 8) +
            (self.next() & 0xFF)
        )

    def read_u32(self):
        return (
            ((self.next() & 0xFF) << 24) +
            ((self.next() & 0xFF) << 16) +
            ((self.next() & 0xFF) << 8) +
            (self.next() & 0xFF)
        )

    def read_text(self, length: int) -> str:
        ascii = [self.next() for _ in range(length)]
        return ''.join([chr(ch) for ch in ascii])

    def decode_header(self):
        while not self.done():
            chunk_type = ''.join([chr(self.next()) for _ in range(4)])
            if chunk_type == 'MThd':
                self.parse_mthd()
            elif chunk_type == 'MTrk':
                self.parse_mtrk()
            else:
                raise ValueError(f"Invalid chunk type {chunk_type}")
        print(f"{chunk_type}")

    def parse_mthd(self):
        length = self.read_u32()
        assert length == 6, f"MThd length of {length}, expected 6."
        format = Format(self.read_u16())
        number_of_tracks = self.read_u16()
        divisions = self.read_u16()
        self.handle(HeaderDataEvent(format, number_of_tracks, divisions))

    def parse_meta_event(self, dt: int):
        meta_type = self.next()
        if meta_type == EventType.SequenceNumber.code():
            assert self.next() == 2, "Expecting sequence number meta-event of length 2."
            sequence_number = self.next()
            self.handle(SequenceNumberEvent(dt, sequence_number))
        elif meta_type == EventType.Text.code():
            length = self.next()
            text = self.read_text(length)
            self.handle(TextEvent(dt, EventType.Text, text))
        elif meta_type == EventType.Copyright.code():
            # Copyright notice.
            length = self.next()
            text = self.read_text(length)
            self.handle(TextEvent(dt, EventType.Copyright, text))
        elif meta_type == EventType.TrackName.code():
            # Copyright notice.
            length = self.next()
            text = self.read_text(length)
            self.handle(TextEvent(dt, EventType.TrackName, text))
        elif meta_type == EventType.EndTrack.code():
            # End of track.
            assert self.next() == 0, "Expecting an end-of-track event of zero length."
        elif meta_type == EventType.Tempo.code():
            # Set tempo.
            assert self.next() == 3, "Expecting tempo meta event of length 3."
            tempo = self.read_u24()
            self.handle(TempoEvent(dt, 60 * 1_000_000 / tempo))
        elif meta_type == EventType.TimeSignature.code():
            assert self.next() == 4, "Expecting time-signature meta event of length 4."
            nn, dd, cc, bb = self.next(), self.next(), self.next(), self.next()
            self.handle(TimeSignatureEvent(dt, nn, dd, cc, bb))
        elif meta_type == EventType.KeySignature.code():
            # Parse key signature (two bytes).
            assert self.next() == 2, "Expecting key-signature meta event of length 2."
            sf = self.next()
            mi = 'Minor' if self.next() == 1 else 'Major'
            self.handle(KeySignatureEvent(dt, sf, mi))
        elif meta_type == EventType.Sequencer.code():
            length = self.next()
            self.handle(DataEvent(dt, EventType.Sequencer,
                        self.buf[self.pos:self.pos+length]))
            self.skip(length)
        else:
            raise ValueError(
                f"Unnown meta-event type {hex(meta_type)}")

    last_status: Optional[int] = None

    def parse_channel_message(self, dt: int, event_type: int):
        self.last_status = None
        channel = (event_type & 0x7)
        message_type = (event_type & 0xF0)
        if message_type == EventType.ProgramChange.code():
            # Channel program change, supports running status.
            channel = (event_type & 0x7)
            program = self.next() & 0x7F
            self.handle(ProgramChangeEvent(
                dt, Channel(channel), Instrument(program)))
        elif (event_type & 0xF0) == EventType.NoteOn.code():
            # Note on event, supports running status.
            self.last_status = event_type
            channel = (event_type & 0x7)
            key = self.next()
            vel = self.next()
            # Converts 0-velocity NoteOn into NoteOff.
            self.handle(
                NoteOnEvent(
                    dt,
                    Channel(channel),
                    Notes(key), vel) if vel > 0 else NoteOffEvent
                (dt, Channel(channel), Notes(key), vel))
        elif (event_type & 0xF0) == EventType.NoteOff.code():
            # Note on event, supports running status.
            self.last_status = event_type
            channel = (event_type & 0x7)
            key = self.next()
            vel = self.next()
            self.handle(NoteOffEvent(dt, Channel(channel), Notes(key), vel))
        elif (event_type & 0xF0) == EventType.ControlChange.code():
            self.last_status = event_type
            # Todo this includes pedal settings (controller number 64 or 91)
            channel = (event_type & 0x07)
            controller_number = self.next()
            value = self.next()
            self.handle(ControlChangeEvent(
                dt, Channel(channel), controller_number, value))
        else:
            raise ValueError(f"[{Channel(channel).name}] Unknown channel message type {
                hex(message_type)}.")

    def parse_running_status(self, dt: int) -> bool:
        if self.last_status:
            self.pos -= 1
            self.parse_channel_message(dt, self.last_status)
            return True
        return False

    def parse_event(self):
        dt = self.read_varlen()
        event_type = self.next()
        if EventType.is_sysex_code(event_type):
            # System exclusive message.
            length = self.read_varlen()
            self.handle(DataEvent(
                dt, EventType(event_type),
                data=self.buf[self.pos:self.pos+length]
            ))
            self.skip(length)
        elif EventType.is_meta_code(event_type):
            self.parse_meta_event(dt)
        elif EventType.is_channel_code(event_type):
            self.parse_channel_message(dt, event_type)
        elif not self.parse_running_status(dt):
            raise ValueError(f"Unknown event type {hex(event_type)}")

    def parse_mtrk(self):
        self.handle(OpenTrackEvent())
        length = self.read_u32()
        start = self.pos
        while (self.pos - start < length):
            self.parse_event()
        self.handle(CloseTrackEvent())

    @abstractmethod
    def handle(self, event: Event):
        pass
