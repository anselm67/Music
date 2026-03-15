import unittest

# from midi.typing import EventType, Notes, Velocity
from midi.typing import EventType, Notes, Velocity


class TestMidiTyping(unittest.TestCase):

    def test_event_type_meta_helpers(self):
        # Test helper methods for Meta events
        self.assertTrue(EventType.Tempo.is_meta())
        self.assertEqual(EventType.Tempo.code(), 0x51)
        self.assertTrue(EventType.is_meta_code(0xFF))

    def test_event_type_channel_helpers(self):
        # Test helper methods for Channel events
        self.assertTrue(EventType.NoteOn.is_channel())
        self.assertEqual(EventType.NoteOn.code(), 0x90)
        self.assertTrue(EventType.is_channel_code(0x90))
        self.assertTrue(EventType.is_channel_code(0x80))
        # 0xFF is Meta, not Channel
        self.assertFalse(EventType.is_channel_code(0xFF))

    def test_event_type_sysex_helpers(self):
        self.assertTrue(EventType.is_sysex_code(0xF0))
        self.assertTrue(EventType.is_sysex_code(0xF7))
        self.assertFalse(EventType.is_sysex_code(0x90))

    def test_notes_enum(self):
        # Spot check note values
        self.assertEqual(Notes.C4.value, 60)
        self.assertEqual(Notes.A4.value, 69)

    def test_velocity_enum(self):
        self.assertEqual(Velocity.Standard.value, 64)


if __name__ == '__main__':
    unittest.main()
