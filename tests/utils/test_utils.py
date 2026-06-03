import logging
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from utils import (
    from_json,
    iterable_from_file,
    log_uncaught_exceptions,
    path_substract,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestUtils(unittest.TestCase):
    def test_reads_lines(self) -> None:
        path = FIXTURES / "sample.txt"
        lines = list(iterable_from_file(path))
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], "Line 1")
        self.assertEqual(lines[1], "Line 2")
        self.assertEqual(lines[2], "Line 3")

    def test_path_substract(self) -> None:
        shorter = Path("/home/user/project")
        longer = Path("/home/user/project/src/main.py")
        self.assertEqual(path_substract(shorter, longer), Path("src/main.py"))

    def test_from_json_dataclass(self) -> None:

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


class _ListHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__()
        self.records = records

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestLogUncaughtExceptions(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_hook = sys.excepthook
        self.records: list[logging.LogRecord] = []
        self.handler = _ListHandler(self.records)
        logging.getLogger().addHandler(self.handler)

    def tearDown(self) -> None:
        sys.excepthook = self._saved_hook
        logging.getLogger().removeHandler(self.handler)

    def test_routes_uncaught_through_logging(self) -> None:
        log_uncaught_exceptions()
        try:
            raise ValueError("boom")
        except ValueError:
            sys.excepthook(*sys.exc_info())  # type: ignore[misc]
        self.assertEqual(len(self.records), 1)
        self.assertEqual(self.records[0].levelno, logging.CRITICAL)
        self.assertEqual(self.records[0].exc_info[0], ValueError)  # type: ignore[index]

    def test_keyboard_interrupt_is_delegated(self) -> None:
        delegated: list[type[BaseException]] = []
        original = sys.__excepthook__
        sys.__excepthook__ = lambda t, v, tb: delegated.append(t)  # type: ignore[assignment]
        try:
            log_uncaught_exceptions()
            try:
                raise KeyboardInterrupt()
            except KeyboardInterrupt:
                sys.excepthook(*sys.exc_info())  # type: ignore[misc]
        finally:
            sys.__excepthook__ = original  # type: ignore[assignment]
        self.assertEqual(delegated, [KeyboardInterrupt])
        self.assertEqual(self.records, [])


if __name__ == "__main__":
    unittest.main()
