import unittest

from kern import Duration, Note
from kern import Pitch as KernPitch
from kern.to_midi import note_to_midi
from midi import Pitch as MidiPitch


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
