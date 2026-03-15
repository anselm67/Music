import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from utils import from_json, iterable_from_file, path_substract

FIXTURES = Path(__file__).parent / "fixtures"


class TestUtils(unittest.TestCase):

    def test_reads_lines(self):
        path = FIXTURES / "sample.txt"
        lines = list(iterable_from_file(path))
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Line 1")
        self.assertEqual(lines[1], "Line 2")
        self.assertEqual(lines[2], "Line 3")

    def test_path_substract(self):
        shorter = Path("/home/user/project")
        longer = Path("/home/user/project/src/main.py")
        self.assertEqual(path_substract(shorter, longer), Path("src/main.py"))

    def test_from_json_dataclass(self):

        @dataclass
        class Child:
            name: str

        @dataclass
        class Parent:
            id: int
            children: list[Child]

        data = {"id": 1, "children": [{"name": "A"}, {"name": "B"}]}
        result = cast(Parent, from_json(Parent, data))

        self.assertIsInstance(result, Parent)
        self.assertEqual(result.id, 1)
        self.assertEqual(len(result.children), 2)
        self.assertIsInstance(result.children[0], Child)
        self.assertEqual(result.children[0].name, "A")


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
