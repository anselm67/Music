import os
import tempfile
import unittest
from pathlib import Path

from midi.output import MidiOutput
from midi.typing import Channel, Velocity


class TestMidiOutput(unittest.TestCase):

    def test_append(self):
        output = MidiOutput()
        output.append([0x01, 0x02, 0x03])
        self.assertEqual(output.buf.tolist(), [1, 2, 3])

    def test_varlen(self):
        # Test cases for variable length encoding (value -> expected bytes)
        cases = [
            (0, [0x00]),
            (127, [0x7F]),
            (128, [0x81, 0x00]),
            (1000, [0x87, 0x68]),
            (0x1FFFFF, [0xFF, 0xFF, 0x7F]),
        ]
        for val, expected in cases:
            output = MidiOutput()
            output.varlen(val)
            self.assertEqual(
                output.buf.tolist(),
                expected,
                f"Failed for value {val}"
            )

    def test_write_ints(self):
        output = MidiOutput()
        output.write_u32(0x12345678)
        self.assertEqual(output.buf.tolist(), [0x12, 0x34, 0x56, 0x78])

        output = MidiOutput()
        output.write_u24(0x123456)
        self.assertEqual(output.buf.tolist(), [0x12, 0x34, 0x56])

        output = MidiOutput()
        output.write_u16(0x1234)
        self.assertEqual(output.buf.tolist(), [0x12, 0x34])

    def test_chunks(self):
        output = MidiOutput()
        # Start a header chunk
        off = output.open_chunk('MThd')
        # Write format (2 bytes)
        output.write_u16(0x0001)
        output.close_chunk(off)

        expected = [
            ord('M'), ord('T'), ord('h'), ord('d'),
            0, 0, 0, 2,  # Length of the chunk content (2 bytes written)
            0, 1         # Payload
        ]
        self.assertEqual(output.buf.tolist(), expected)

    def test_time_signature(self):
        output = MidiOutput()
        output.time_signature(4, 4)
        # Expect: delta_time(0) + Meta Event (FF 58 04) + 4/4 params
        # 4/4: nn=4, dd=2 (2^2=4), cc=24 clocks/beat, bb=8 32nd/quarter
        expected_4_4 = [0x00, 0xFF, 0x58, 0x04, 0x04, 0x02, 24, 8]
        self.assertEqual(output.buf.tolist(), expected_4_4)

        output = MidiOutput()
        output.time_signature(6, 8)
        # 6/8: nn=6, dd=3 (2^3=8), cc=36 clocks/beat (1.5 * 24), bb=8
        expected_6_8 = [0x00, 0xFF, 0x58, 0x04, 0x06, 0x03, 36, 8]
        self.assertEqual(output.buf.tolist(), expected_6_8)

    def test_note_events(self):
        output = MidiOutput()
        output.note_on(Channel.Chan1, 60, Velocity.Standard)
        # dt=0 (0x00), NoteOn Ch1 (0x90 | 1 = 0x91), Note 60, Vel 64
        self.assertEqual(output.buf.tolist(), [0x00, 0x91, 60, 64])

        output = MidiOutput()
        output.note_off(Channel.Chan1, 60)
        # dt=0 (0x00), NoteOff Ch1 (0x80 | 1 = 0x81), Note 60, Vel 64 (default)
        self.assertEqual(output.buf.tolist(), [0x00, 0x81, 60, 64])

    def test_tempo(self):
        output = MidiOutput()
        output.tempo(120)
        # 120 bpm = 500,000 microseconds per quarter note (0x07A120)
        # dt=0 (0x00), Meta Tempo (0xFF 0x51 0x03), Value
        expected = [0x00, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]
        self.assertEqual(output.buf.tolist(), expected)

    def test_save(self):
        output = MidiOutput()
        output.append([0x90, 60, 64])
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.close()
            try:
                output.save(Path(tmp.name))
                with open(tmp.name, 'rb') as f:
                    content = f.read()
                self.assertEqual(list(content), [0x90, 60, 64])
            finally:
                os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
