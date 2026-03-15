import unittest

from kern import Duration, Note, Pitch
from kern.to_midi import note_to_midi
from midi import Notes


class TestToMidi(unittest.TestCase):

    def test_note_to_midi_octave(self):
        note = Note(Duration(4, 0), Pitch.CCCC)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.CX)

        note = Note(Duration(4, 0), Pitch.C)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.C2)

        note = Note(Duration(4, 0), Pitch.ccccc)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.C7)

    def test_note_to_midi_sharp(self):
        note = Note(Duration(4, 0), Pitch.f, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.F3Sharp)

        note = Note(Duration(4, 0), Pitch.ee, flats=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.D4Sharp)

        note = Note(Duration(4, 0), Pitch.ee, flats=2)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.D4)

        note = Note(Duration(4, 0), Pitch.bb, sharps=1)
        midi = note_to_midi(note)
        self.assertEqual(midi, Notes.C5)
