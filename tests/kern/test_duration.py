import unittest

from kern.typing import Duration


class TestDuration(unittest.TestCase):

    def test_length(self):
        # Test length calculation for a Duration object
        d = Duration(4, 0)
        self.assertEqual(d.length, 0.25)
        d = Duration(4, 1)
        self.assertEqual(d.length, 0.25 + 0.125)
        d = Duration(4, 2)
        self.assertEqual(d.length, 0.25 + 0.125 + 0.0625)
        d = Duration(4, 3)
        self.assertEqual(d.length, 0.25 + 0.125 + 0.0625 + 0.03125)

    def test_from_length(self):
        # Test creation of Duration object from length
        d = Duration.from_length(0.25)
        self.assertEqual(d, Duration(4, 0))
        d = Duration.from_length(0.25 + 0.125)
        self.assertEqual(d, Duration(4, 1))
        d = Duration.from_length(0.25 + 0.125 + 0.0625)
        self.assertEqual(d, Duration(4, 2))
        d = Duration.from_length(0.25 + 0.125 + 0.0625 + 0.03125)
        self.assertEqual(d, Duration(4, 3))

    def test_substract(self):
        # Test subtraction of two Duration objects
        d1 = Duration(4, 0)
        d2 = Duration(8, 0)
        self.assertEqual(d1 - d2, Duration(8, 0))
