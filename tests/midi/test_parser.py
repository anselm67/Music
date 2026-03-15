import array
import shutil
import unittest
from pathlib import Path

from midi.input import MidiInput
from midi.output import MidiOutput
from midi.typing import Channel, Event, HeaderDataEvent, NoteOnEvent, Velocity

FIXTURES = Path(__file__).parent / "fixtures"
TEST_IODIR = FIXTURES / "midi"


class CapturingMidiInput(MidiInput):
    events: list[Event]

    def __init__(self, buf: array.array):
        super().__init__(buf)
        self.events = []

    def handle(self, event: Event):
        self.events.append(event)


class TestMidiParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_IODIR.mkdir(parents=True, exist_ok=True)
        cls.create_mono_file()
        cls.create_multi_file()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TEST_IODIR)

    @classmethod
    def create_mono_file(cls):
        output = MidiOutput()
        # Header: Format 0, 1 Track
        hd = output.open_chunk('MThd')
        output.write_u16(0)
        output.write_u16(1)
        output.write_u16(480)
        output.close_chunk(hd)

        # Track
        trk = output.open_chunk('MTrk')
        output.time_signature(4, 4)
        output.tempo(120)
        output.note_on(Channel.Chan1, 60, Velocity.Standard, 0)
        output.note_off(Channel.Chan1, 60, Velocity.Standard, 480)
        output.track_end(0)
        output.close_chunk(trk)

        output.save(TEST_IODIR / "mono.mid")

    @classmethod
    def create_multi_file(cls):
        output = MidiOutput()
        # Header: Format 1, 2 Tracks
        hd = output.open_chunk('MThd')
        output.write_u16(1)
        output.write_u16(2)
        output.write_u16(480)
        output.close_chunk(hd)

        # Track 1: Meta
        trk1 = output.open_chunk('MTrk')
        output.time_signature(4, 4)
        output.tempo(120)
        output.track_end(0)
        output.close_chunk(trk1)

        # Track 2: Notes
        trk2 = output.open_chunk('MTrk')
        # Channel 1 Note
        output.note_on(Channel.Chan1, 60, Velocity.Standard, 0)
        # Channel 2 Note (Simultaneous start, dt=0)
        output.note_on(Channel.Chan2, 64, Velocity.Standard, 0)

        # Note Offs
        output.note_off(Channel.Chan1, 60, Velocity.Standard, 480)
        output.note_off(Channel.Chan2, 64, Velocity.Standard, 0)

        output.track_end(0)
        output.close_chunk(trk2)

        output.save(TEST_IODIR / "multi.mid")

    def test_parse_mono(self):
        path = TEST_IODIR / "mono.mid"
        with open(path, "rb") as f:
            buf = array.array('B', f.read())

        parser = CapturingMidiInput(buf)
        parser.parse()

        events = parser.events
        # Check Header
        self.assertIsInstance(events[0], HeaderDataEvent)
        # self.assertEqual(events[0].format.value, 0)

        # Find NoteOn
        note_ons = [e for e in events if isinstance(e, NoteOnEvent)]
        self.assertEqual(len(note_ons), 1)
        self.assertEqual(note_ons[0].channel, Channel.Chan1)
        self.assertEqual(note_ons[0].note.value, 60)

    def test_parse_multi(self):
        path = TEST_IODIR / "multi.mid"
        with open(path, "rb") as f:
            buf = array.array('B', f.read())

        parser = CapturingMidiInput(buf)
        parser.parse()

        events = parser.events
        # Check Header
        self.assertIsInstance(events[0], HeaderDataEvent)
        # self.assertEqual(events[0].value, 1)

        # Find NoteOns
        note_ons = [e for e in events if isinstance(e, NoteOnEvent)]
        self.assertEqual(len(note_ons), 2)

        channels = {e.channel for e in note_ons}
        self.assertEqual(channels, {Channel.Chan1, Channel.Chan2})


if __name__ == '__main__':
    unittest.main()
