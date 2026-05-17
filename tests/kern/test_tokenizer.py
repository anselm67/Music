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
