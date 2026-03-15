import unittest
from pathlib import Path

from kern.empty import EmptyHandler
from kern.parser import Parser

FIXTURES = Path(__file__).parent / "fixtures"


class TestHumdrumFixtures(unittest.TestCase):

    def test_fixtures(self):
        for file in FIXTURES.glob("*.krn"):
            try:
                parser = Parser.from_file(file, EmptyHandler())
                parser.parse()
            except Exception as e:
                self.fail(f"{e}")


if __name__ == '__main__':
    unittest.main()
