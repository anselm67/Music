from pathlib import Path

import pytest

from kern import KernReader, tokenize


def check_tokenization(tmp_path: Path, kern_text: str) -> bool:
    input = tmp_path / "input.krn"
    output = tmp_path / "output.krn"
    input.write_text(kern_text)
    try:
        tokenize(input, output)
        KernReader(output)
    except Exception:
        return False
    return True


def tokenize_input(tmp_path: Path, kern_text: str) -> KernReader:
    input = tmp_path / "input.krn"
    output = tmp_path / "output.krn"
    input.write_text(kern_text)
    tokenize(input, output)
    return KernReader(output)


@pytest.mark.parametrize(
    "input",
    [
        """
!!!COM: William James Kirkpatrick
!!!OTL: I Shall Be No Stranger There
**kern	**text	**kern	**text	**text	**text
*part2	*part2	*part1	*part1	*part1	*part1
*staff2	*staff2	*staff1	*staff1	*staff1	*staff1
*clefF4	*	*clefG2	*	*	*
*k[f#c#g#d#]	*	*k[f#c#g#d#]	*	*	*
*E:	*	*E:	*	*	*
*M3/4	*	*M3/4	*	*	*
*MM100	*	*MM100	*	*	*
=1	=1	=1	=1	=1	=1
*	*	*^	*	*	*
!	!	!LO:TX:a:t=[quarter]=100	!	!	!	!
8.EL 8.G#	.	8.eL	8.e	1. When	2. Thro’	3. There
*	*	*v	*v	*	*	*
16EJk 16A	.	16eJk 16f#	the	time’s	my
=	=	=	=	=	=
""".strip()
    ],
)
def test_spine_merge_with_extras(tmp_path: Path, input: str) -> None:
    assert check_tokenization(tmp_path, input)


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            """
**kern
*clefG2
*k[]
*M6/8
=
""".strip(),
            ["clef-GG", "6/8", "=1"],
        )
    ],
)
def test_remove_unused_keys(tmp_path: Path, input: str, expected: list[str]) -> None:
    reader = tokenize_input(tmp_path, input)
    for line, check in zip(reader.lines, expected):
        assert line == check


def test_instrument_all_spines_skipped(tmp_path: Path) -> None:
    """Rows where every spine is an instrument annotation must not appear in output."""
    kern = """
**kern\t**kern
*clefG2\t*clefF4
*k[]\t*k[]
*M4/4\t*M4/4
*IPiano\t*IPiano
=1\t=1
4c\t4C
==2\t==2
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert not any("Instr:" in line for line in reader.lines)


def test_unnumbered_double_barline_is_a_glyph(tmp_path: Path) -> None:
    """An un-numbered `=||` is a distinct double-bar glyph: it neither inherits a
    bar number nor advances the count, so it must not collide with the next bar."""
    kern = """
**kern
*clefG2
*M4/4
=1
4r
=||
=2
4r
==
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert reader.lines == [
        "clef-GG",
        "4/4",
        "=1",
        "rest/4",
        "=||",
        "=2",
        "rest/4",
        "==3",
    ]


def test_numbered_double_barline_keeps_its_number(tmp_path: Path) -> None:
    """A numbered `=2||` is a real measure boundary: the number must survive (for
    score alignment) while still rendering as a double bar, never as final `==`."""
    kern = """
**kern
*clefG2
*M4/4
=1
4r
=2||
4r
=3
4r
==
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert reader.lines == [
        "clef-GG",
        "4/4",
        "=1",
        "rest/4",
        "=2||",
        "rest/4",
        "=3",
        "rest/4",
        "==4",
    ]


def test_repeat_barline_opening_is_bar_one_not_pickup(tmp_path: Path) -> None:
    """A piece that opens on a numbered repeat barline (`=||:1`, the measure number
    after the style marks, as in bach/inventions/inven06) starts *at* bar 1 with no
    pickup. The number must be captured and no spurious `=0` invented — otherwise the
    editor's bar offset slips by one and the layout no longer aligns to the kern."""
    kern = """
**kern
*clefG2
*M3/8
=||:1
4r
=2
4r
==
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert reader.lines == [
        "clef-GG",
        "3/8",
        "=1||",
        "rest/4",
        "=2",
        "rest/4",
        "==3",
    ]
    assert reader.first_bar == 1
    assert not reader.has_bar_zero()


def test_repeat_barline_opening_after_pickup_keeps_bar_zero(tmp_path: Path) -> None:
    """Contrast: when real pickup notes precede the opening barline, the bar zero is
    genuine and must still be emitted (the fix only suppresses the *spurious* one)."""
    kern = """
**kern
*clefG2
*M3/8
4r
=||:1
4r
=2
4r
==
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert reader.has_bar_zero()
    assert reader.first_bar == 1


def test_instrument_mixed_with_spine_path_skipped(tmp_path: Path) -> None:
    """Rows mixing instrument and spine-path tokens must not appear in output."""
    kern = """
**kern\t**kern
*clefG2\t*clefF4
*k[]\t*k[]
*M4/4\t*M4/4
*IPiano\t*
=1\t=1
4c\t4C
==2\t==2
""".strip()
    reader = tokenize_input(tmp_path, kern)
    assert not any("Instr:" in line for line in reader.lines)
