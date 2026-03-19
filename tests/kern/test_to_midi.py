import unittest
from unittest.mock import call, patch

from kern import Duration, Note, Parser
from kern import Pitch as KernPitch
from kern.to_midi import MidiHandler, note_to_midi
from midi import Channel
from midi import Pitch as MidiPitch
from midi import Velocity


class TestToMidi(unittest.TestCase):

    def test_note_to_midi_octave(self):
        note = Note(Duration(4, 0), KernPitch.CCCC)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.CX)

        note = Note(Duration(4, 0), KernPitch.C)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C2)

        note = Note(Duration(4, 0), KernPitch.ccccc)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C7)

    def test_note_to_midi_sharp(self):
        note = Note(Duration(4, 0), KernPitch.f, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.F3Sharp)

        note = Note(Duration(4, 0), KernPitch.ee, flats=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.D4Sharp)

        note = Note(Duration(4, 0), KernPitch.ee, flats=2)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.D4)

        note = Note(Duration(4, 0), KernPitch.bb, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, MidiPitch.C5)
        self.assertEqual(midi, MidiPitch.C5)

    def test_handler_channel(self):
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
    def test_generating_bar_single_staff(self, mock_midi_output):
        handler = MidiHandler(480, 120)
        # A simple bar with two quarter notes: C and D
        parser = Parser.from_text("**kern\n4c\n4d\n*-", handler)
        parser.parse()

        self.assertEqual(len(handler.tracks), 1)
        track = handler.tracks[0]

        # 4c is Middle C (C3 in this system per tests, 48)
        # 4d is D3 (50)
        # Duration 4 -> 480 ticks
        expected_calls = [
            call.note_on(Channel.Chan0, MidiPitch.C3, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.C3, Velocity.Forte, 480),
            call.note_on(Channel.Chan0, MidiPitch.D3, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.D3, Velocity.Forte, 480),
        ]
        mock_midi_output.return_value.assert_has_calls(
            expected_calls, any_order=False)

    @patch("kern.to_midi.MidiOutput")
    def test_chord_conversion(self, mock_midi_output):
        handler = MidiHandler(480, 120)
        # Chord C E G
        parser = Parser.from_text("**kern\n4c 4e 4g\n*-", handler)
        parser.parse()
        expected_calls = [
            call.note_on(Channel.Chan0, MidiPitch.C3, Velocity.Forte, 0),
            call.note_on(Channel.Chan0, MidiPitch.E3, Velocity.Forte, 0),
            call.note_on(Channel.Chan0, MidiPitch.G3, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.C3, Velocity.Forte, 480),
            call.note_off(Channel.Chan0, MidiPitch.E3, Velocity.Forte, 0),
            call.note_off(Channel.Chan0, MidiPitch.G3, Velocity.Forte, 0),
        ]
        mock_midi_output.return_value.assert_has_calls(
            expected_calls, any_order=False)
