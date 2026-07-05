import tempfile
import unittest
from pathlib import Path

from kern.kern_reader import KernReader


def get_kern_reader(content: str) -> KernReader:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        token_file = tmp_path / "example.tokens"
        token_file.write_text(content)
        return KernReader(token_file)


class TestKernReader(unittest.TestCase):
    def test_load_tokens_indexes_bar_markers(self) -> None:
        reader = get_kern_reader(
            """
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
        )
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
        reader = get_kern_reader(
            """
= 0
intro
=1
foo
=2
bar
=3
baz
""".strip()
        )
        self.assertEqual(reader.get_text(1, 2), ["=1", "foo", "=2"])
        self.assertEqual(reader.get_text(2, 3), ["=2", "bar", "=3"])

    def test_get_text_returns_to_eof_when_end_bar_missing(self) -> None:
        reader = get_kern_reader(
            """
= 0
intro
=1
foo
=2
bar
""".strip()
        )
        self.assertEqual(reader.get_text(2, 99), ["=2", "bar"])

    def test_get_text_handles_non_consecutive_bars(self) -> None:
        # pickup =0 then jumps to =2 (no bar 1), like chopin/prelude/prelude28-02.
        # The editor's running number for the pickup comes out below first_bar and
        # is remapped to 0; end (1) is not a real bar, so the old code fell through
        # to EOF — it must return ONLY the pickup bar.
        reader = get_kern_reader(
            """
=0
pickup
=2
two
=3
three
""".strip()
        )
        self.assertEqual(reader.first_bar, 2)
        self.assertTrue(reader.has_bar_zero())
        self.assertEqual(reader.get_text(1), ["=0", "pickup", "=2"])
        self.assertEqual(reader.get_text(0), ["=0", "pickup", "=2"])
        self.assertEqual(reader.get_text(2), ["=2", "two", "=3"])

    def test_closing_barline_not_counted(self) -> None:
        # A bar marker that no music follows is the piece's closing barline, not a
        # measure: written ==N, =N, =|| or ==, it must not inflate the bar count.
        for terminal in ["==4", "=4", "=||", "=="]:
            reader = get_kern_reader(f"=1\na\n=2\nb\n=3\nc\n{terminal}")
            self.assertEqual(reader.bar_count, 3, terminal)
            self.assertEqual(sorted(reader.bars), [1, 2, 3], terminal)

    def test_content_ending_keeps_last_bar(self) -> None:
        # When music (not a barline) ends the file every bar marker has content
        # after it, so none is dropped and a real layout/kern off-by-one still shows.
        reader = get_kern_reader("=1\na\n=2\nb")
        self.assertEqual(reader.bar_count, 2)
        self.assertEqual(sorted(reader.bars), [1, 2])

    def test_only_the_final_barline_dropped_not_a_run(self) -> None:
        # A content-free final measure: `=3` then the closing `==4` with nothing
        # between. Only the closing barline (last line) is dropped; bar 3 survives.
        reader = get_kern_reader("=1\na\n=2\nb\n=3\n==4")
        self.assertEqual(reader.bar_count, 3)
        self.assertIn(3, reader.bars)

    def test_start_before_first_bar_adjusts_to_zero(self) -> None:
        reader = get_kern_reader(
            """
= 0
intro
=1
foo
=2
bar
""".strip()
        )
        self.assertEqual(reader.first_bar, 1)
        self.assertEqual(reader.get_text(0, 1), ["= 0", "intro", "=1"])

    def test_get_text_missing_start_bar_returns_none(self) -> None:
        reader = get_kern_reader(
            """
= 1
foo
=2
bar
""".strip()
        )
        self.assertIsNone(reader.get_text(5, 6))

    def test_header_returns_first_ten_lines(self) -> None:
        content_lines = [f"line {i}" for i in range(12)]
        content_lines[0] = "= 0"
        content_lines[6] = "=1"
        reader = get_kern_reader("\n".join(content_lines))
        self.assertEqual(reader.header(), content_lines[:10])

    def test_can_open_tokens_file_from_non_tokens_path(self) -> None:
        reader = get_kern_reader(
            """
= 0
intro
=1
foo
""".strip()
        )
        self.assertEqual(reader.lines[0], "= 0")
        self.assertEqual(reader.bar_count, 2)
        self.assertEqual(reader.get_text(1), ["=1", "foo"])

    def test_preamble(self) -> None:
        reader = get_kern_reader(
            """
clef-G
4/4
=1
foo
=2
bar
=3
""".strip()
        )
        self.assertEqual(reader.get_text(1, 2), ["clef-G", "4/4", "=1", "foo", "=2"])

    def test_preamble_multi_spine(self) -> None:
        reader = get_kern_reader(
            """
clef-G\tclef-f
4/4\t4/4
=1
foo1\tfoo2
=2
bar1\tbar2
=3
""".strip()
        )
        self.assertEqual(
            reader.get_text(1, 2),
            ["clef-G\tclef-f", "4/4\t4/4", "=1", "foo1\tfoo2", "=2"],
        )

    def test_preamble_multi_spine_key_change(self) -> None:
        reader = get_kern_reader(
            """
clef-G\tclef-f
4/4\t4/4
=1
foo1\tfoo2
=2
keys-\t.
bar1\tbar2
=3
""".strip()
        )
        self.assertEqual(
            reader.get_text(1, 2),
            ["clef-G\tclef-f", "4/4\t4/4", "=1", "foo1\tfoo2", "=2"],
        )
        self.assertEqual(
            reader.get_text(2, 3),
            ["clef-G\tclef-f", "=2", "keys-\t.", "bar1\tbar2", "=3"],
        )

    def test_staff_map_from_staff_row(self) -> None:
        reader = get_kern_reader(
            """
*staff2\t*staff1
clef-f\tclef-G
=1
foo1\tfoo2
=2
""".strip()
        )
        self.assertEqual(reader.staff_numbers, [2, 1])
        self.assertEqual(reader.staff_map(), [1, 0])
        # The `*staff` row is metadata: excluded from bars, preamble, and records.
        self.assertEqual(reader.first_bar, 1)
        self.assertEqual(
            reader.get_text(1, 2), ["clef-f\tclef-G", "=1", "foo1\tfoo2", "=2"]
        )

    def test_staff_map_treble_first(self) -> None:
        reader = get_kern_reader("*staff1\t*staff2\n=1\nfoo1\tfoo2\n=2")
        self.assertEqual(reader.staff_map(), [0, 1])

    def test_staff_map_positional_fallback_without_staff_row(self) -> None:
        # No `*staff` row -> reversed-column (bass-first) fallback: the old [1, 0].
        reader = get_kern_reader("clef-f\tclef-G\n=1\nfoo1\tfoo2\n=2")
        self.assertEqual(reader.staff_numbers, [])
        self.assertEqual(reader.staff_map(), [1, 0])

    def test_staff_map_noncontiguous_falls_back(self) -> None:
        # A gapped labelling (staff3 with no staff2) is not invertible -> fallback.
        reader = get_kern_reader("*staff3\t*staff1\n=1\nfoo1\tfoo2\n=2")
        self.assertEqual(reader.staff_map(), [1, 0])

    def test_staff_map_voiced_falls_back(self) -> None:
        # A voiced staff (staff2 on two columns) is not a one-per-staff bijection:
        # fall back to a full-width positional map, not a short "clean" list.
        reader = get_kern_reader("*staff2\t*staff2\t*staff1\n=1\na\tb\tc\n=2")
        self.assertEqual(reader.staff_map(), [2, 1, 0])
