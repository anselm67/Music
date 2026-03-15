import array
from math import log2
from typing import Iterable, List, Literal

from midi.typing import Channel, Velocity


class MidiOutput():

    buf: array.array

    def __init__(self):
        self.buf = array.array('B')

    def append(self, bytes: Iterable[int]):
        self.buf.extend(bytes)

    def varlen(self, value: int):
        buffer: List[int] = [value & 0x7f]
        value >>= 7
        while value > 0:
            buffer.append(0x80 | (value & 0x7f))
            value >>= 7
        buffer.reverse()
        self.append(buffer)

    def open_chunk(self, type: Literal['MThd', 'MTrk']):
        self.append([ord(b) for b in type])
        # Reserve 4 bytes for chunk length to be filled on completion of chunk.
        _, off = self.buf.buffer_info()
        self.append([0, 0, 0, 0])
        return off

    def write_u32(self, value: int, off: int = -1):
        if off < 0:
            _, off = self.buf.buffer_info()
            self.buf.extend([0, 0, 0, 0])
        for idx in range(3, -1, -1):
            self.buf[off+idx] = (value & 0xFF)
            value >>= 8

    def write_u24(self, value: int, off: int = -1):
        if off < 0:
            _, off = self.buf.buffer_info()
            self.buf.extend([0, 0, 0])
        for idx in range(2, -1, -1):
            self.buf[off+idx] = (value & 0xFF)
            value >>= 8

    def write_u16(self, value: int, off: int = -1):
        if off < 0:
            _, off = self.buf.buffer_info()
            self.buf.extend([0, 0])
        for idx in range(1, -1, -1):
            self.buf[off+idx] = (value & 0xFF)
            value >>= 8

    def close_chunk(self, off: int):
        _, len = self.buf.buffer_info()
        self.write_u32(len - (off + 4), off)

    def format(self, number: int):
        assert number == 0, "MidiOutput only supports format 0."
        self.write_u16(number)

    def number_of_tracks(self, count: int):
        self.write_u16(count)

    def ticks_per_quarter_notes(self, div: int):
        assert div & 0x80 == 0, "MidiOutput only supports ticks per quarter notes."
        self.write_u16(div)

    def delta_time(self, duration: int = 0):
        self.varlen(duration)

    def time_signature(self, num: int, den: int, dt: int = 0):
        self.delta_time(dt)
        # Derive cc / bb from the time signature.
        # cc is the number of midi-clocks per metronome tick, 24 per quarter => 36 per quarter dot.
        # bb is the number of notated 32nd-notes in a MIDI quarter-note (24 MIDI clocks).
        if num == 6 and den == 8:
            nn, dd, cc, bb = num, int(log2(8)), 36, 8
        elif num == 4 and den == 4:
            nn, dd, cc, bb = 4, int(log2(4)), 24, 8
        elif num == 2 and den == 4:
            nn, dd, cc, bb = 2, int(log2(4)), 48, 8
        else:
            assert False, "Unsupported time signature {num} / {den}"
        self.append([0xFF, 0x58, 0x04, nn, dd, cc, bb])

    def tempo(self, bpm: int, dt: int = 0):
        self.delta_time(dt)
        usec = int((60.0 / float(bpm)) * 1000000)
        self.append([0xFF, 0x51, 0x3])
        self.write_u24(usec)

    def program_change(self, chan: Channel, prog_number: int, dt: int = 0):
        self.delta_time(dt)
        assert prog_number < 128, f"Invalid program number {
            prog_number} must be < 128."
        self.append([0xC0 | chan.value, prog_number])

    def note_on(self, chan: Channel, note: int, v: Velocity = Velocity.Standard, dt: int = 0):
        self.delta_time(dt)
        assert note >= 0 and note < 128, f"Invalid note {
            note} must be >= 0 amd < 128."
        assert v.value >= 0 and v.value < 128, f"Invalid velocity {
            v} must be >= 0 and < 128."
        self.append([0x90 | chan.value, note, v.value])

    def note_off(self, chan: Channel, note: int, v: Velocity = Velocity.Standard, dt: int = 0):
        assert note >= 0 and note < 128, f"Invalid note {
            note} must be >= 0 amd < 128."
        self.delta_time(dt)
        self.append([0x80 | chan.value, note, v.value])

    def track_end(self, dt: int = 0):
        self.delta_time(dt)
        self.append([0xFF, 0x2F, 0x00])

    def save(self, filename: str):
        with open(filename, "wb+") as f:
            f.write(self.buf)
