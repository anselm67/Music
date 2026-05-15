import tempfile
import unittest
from pathlib import Path

from kern.kern_reader import KernReader


class TestKernReader(unittest.TestCase):
    def test_load_tokens_indexes_bar_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
!!!
= 0
some line
=1
line A
==2
lineB
= 3
END
""".strip()
            token_file = tmp_path / "example.tokens"
            token_file.write_text(content)

            reader = KernReader(token_file)

            self.assertEqual(reader.first_bar, 1)
            self.assertEqual(reader.bar_count, 4)
            self.assertTrue(reader.has_bar_zero())
            self.assertEqual(
                reader.bars,
                {
                    0: 1,
                    1: 3,
                    2: 5,
                    3: 7,
                },
            )
            self.assertEqual(reader.lines[0], "!!!")
            self.assertEqual(reader.lines[-1], "END")

    def test_get_text_includes_next_bar_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
= 0
intro
=1
foo
=2
bar
=3
baz
""".strip()
            token_file = tmp_path / "example.tokens"
            token_file.write_text(content)

            reader = KernReader(token_file)

            self.assertEqual(reader.get_text(1, 2), ["=1", "foo", "=2"])
            self.assertEqual(reader.get_text(2, 3), ["=2", "bar", "=3"])

    def test_get_text_returns_to_eof_when_end_bar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
= 0
intro
=1
foo
=2
bar
""".strip()
            token_file = tmp_path / "example.tokens"
            token_file.write_text(content)

            reader = KernReader(token_file)

            self.assertEqual(reader.get_text(2, 99), ["=2", "bar"])

    def test_start_before_first_bar_adjusts_to_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
= 0
intro
=1
foo
=2
bar
""".strip()
            token_file = tmp_path / "example.tokens"
            token_file.write_text(content)

            reader = KernReader(token_file)

            self.assertEqual(reader.first_bar, 1)
            self.assertEqual(reader.get_text(0, 1), ["= 0", "intro", "=1"])

    def test_get_text_missing_start_bar_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
= 1
foo
=2
bar
""".strip()
            token_file = tmp_path / "example.tokens"
            token_file.write_text(content)

            reader = KernReader(token_file)

            self.assertIsNone(reader.get_text(5, 6))

    def test_header_returns_first_ten_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content_lines = [f"line {i}" for i in range(12)]
            content_lines[0] = "= 0"
            content_lines[6] = "=1"
            token_file = tmp_path / "example.tokens"
            token_file.write_text("\n".join(content_lines))

            reader = KernReader(token_file)

            self.assertEqual(reader.header(), content_lines[:10])

    def test_can_open_tokens_file_from_non_tokens_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            content = """
= 0
intro
=1
foo
""".strip()
            krn_path = tmp_path / "song.krn"
            tokens_path = krn_path.with_suffix(".tokens")
            tokens_path.write_text(content)

            reader = KernReader(krn_path)

            self.assertEqual(reader.lines[0], "= 0")
            self.assertEqual(reader.bar_count, 2)
            self.assertEqual(reader.get_text(1), ["=1", "foo"])
