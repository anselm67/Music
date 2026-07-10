import unittest
from unittest.mock import MagicMock, call, patch

from kern import Duration, Note, Parser
from kern import Pitch as KernPitch
from kern.to_midi import MidiHandler, note_to_midi
from midi import Channel, Velocity
from midi import Pitch as MidiPitch


class TestToMidi(unittest.TestCase):
    def test_note_to_midi_octave(self) -> None:
        # Kern middle C `c` is MIDI 60; `C` is C3=48, `CCCC` is C0=12, `ccccc` C8=108.
        note = Note(Duration(4, 0), KernPitch.CCCC)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C0)

        note = Note(Duration(4, 0), KernPitch.C)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C3)

        note = Note(Duration(4, 0), KernPitch.ccccc)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C8)

    def test_note_to_midi_sharp(self) -> None:
        note = Note(Duration(4, 0), KernPitch.f, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.F4Sharp)

        note = Note(Duration(4, 0), KernPitch.ee, flats=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.D5Sharp)

        note = Note(Duration(4, 0), KernPitch.ee, flats=2)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.D5)

        note = Note(Duration(4, 0), KernPitch.bb, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C6)

    def test_handler_channel(self) -> None:
        handler = MidiHandler(480, 120)
        self.assertEqual(handler.allocate_channel(), Channel.Chan0)
        self.assertEqual(handler.allocate_channel(), Channel.Chan1)
        self.assertEqual(handler.allocate_channel(), Channel.Chan2)
        self.assertEqual(handler.allocate_channel(), Channel.Chan3)
        self.assertEqual(handler.allocate_channel(), Channel.Chan4)
        handler.free_channel(Channel.Chan4)
        self.assertEqual(handler.allocate_channel(), Channel.Chan4)
        handler.free_channel(Channel.Chan4)
        handler.free_channel(Channel.Chan3)
        handler.free_channel(Channel.Chan3)
        handler.free_channel(Channel.Chan1)
        handler.free_channel(Channel.Chan0)
        self.assertEqual(handler.allocate_channel(), Channel.Chan0)

    @patch("kern.to_midi.MidiOutput")
    def test_generating_bar_single_staff(self, mock_midi_output: MagicMock) -> None:
        handler = MidiHandler(480, 120)
        # A simple bar with two quarter notes: C and D
        parser = Parser.from_text("**kern\n4c\n4d\n*-", handler)
        parser.parse()

        self.assertEqual(len(handler.tracks), 1)

        # 4c is Middle C (C4 = MIDI 60), 4d is D4 (62); duration 4 -> 480 ticks
        expected_calls = [
            call.note_on(Channel.Chan0, MidiPitch.C4, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.C4, Velocity.Forte, 480),
            call.note_on(Channel.Chan0, MidiPitch.D4, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.D4, Velocity.Forte, 480),
        ]
        mock_midi_output.return_value.assert_has_calls(expected_calls, any_order=False)

    @patch("kern.to_midi.MidiOutput")
    def test_bare_rest_consumes_no_time(self, mock_midi_output: MagicMock) -> None:
        handler = MidiHandler(480, 120)
        # A bare rest `r` parses to Duration(0); duration_to_ticks would divide by
        # zero. It degrades to 0 ticks, so the following D still onsets at delta 0.
        parser = Parser.from_text("**kern\n4c\nr\n4d\n*-", handler)
        parser.parse()
        expected_calls = [
            call.note_off(Channel.Chan0, MidiPitch.C4, Velocity.Forte, 480),
            call.note_on(Channel.Chan0, MidiPitch.D4, Velocity.Forte, 0),
        ]
        mock_midi_output.return_value.assert_has_calls(expected_calls, any_order=False)

    @patch("kern.to_midi.MidiOutput")
    def test_chord_conversion(self, mock_midi_output: MagicMock) -> None:
        handler = MidiHandler(480, 120)
        # Chord C E G
        parser = Parser.from_text("**kern\n4c 4e 4g\n*-", handler)
        parser.parse()
        expected_calls = [
            call.note_on(Channel.Chan0, MidiPitch.C4, Velocity.Forte, 0),
            call.note_on(Channel.Chan0, MidiPitch.E4, Velocity.Forte, 0),
            call.note_on(Channel.Chan0, MidiPitch.G4, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.C4, Velocity.Forte, 480),
            call.note_off(Channel.Chan0, MidiPitch.E4, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.G4, Velocity.Forte, 0),
        ]
        mock_midi_output.return_value.assert_has_calls(expected_calls, any_order=False)
