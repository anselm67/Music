import torch

from utils.sequence import align_sequences, format_sequence_columns

_RED = "\033[31m"
_RESET = "\033[0m"


def lines(result: str) -> list[str]:
    return result.split("\n")


def _seq(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows)


def test_align_identical_sequences_all_substitutions() -> None:
    s = _seq([[1, 0], [2, 0], [3, 0]])
    assert align_sequences(s, s, pad_id=0) == [(0, 0), (1, 1), (2, 2)]


def test_align_insertion_in_pred() -> None:
    gt = _seq([[1, 0], [3, 0]])
    pred = _seq([[1, 0], [2, 0], [3, 0]])
    # The extra pred row 1 (token 2) has no GT counterpart.
    assert align_sequences(gt, pred, pad_id=0) == [(0, 0), (None, 1), (1, 2)]


def test_align_deletion_in_pred() -> None:
    gt = _seq([[1, 0], [2, 0], [3, 0]])
    pred = _seq([[1, 0], [3, 0]])
    assert align_sequences(gt, pred, pad_id=0) == [(0, 0), (1, None), (2, 1)]


def test_header_and_separator_always_present() -> None:
    rows = lines(format_sequence_columns([], [], highlight_mismatches=False))
    assert rows[0].startswith("GT")
    assert "Pred" in rows[0]
    assert rows[1].startswith("-")
    assert "+" in rows[1]


def test_equal_length_no_mismatch() -> None:
    left = ["4c", "4d", "EOS"]
    right = ["4c", "4d", "EOS"]
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    assert len(rows) == 2 + len(left)
    for i, tok in enumerate(left):
        assert tok in rows[i + 2]
        assert right[i] in rows[i + 2]


def test_left_longer_than_right() -> None:
    left = ["4c", "4d", "4e", "EOS"]
    right = ["4c", "EOS"]
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    assert len(rows) == 2 + len(left)
    assert rows[4].endswith("| ")


def test_right_longer_than_left() -> None:
    left = ["4c", "EOS"]
    right = ["4c", "4d", "4e", "EOS"]
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    assert len(rows) == 2 + len(right)


def test_chord_tokens_with_spaces() -> None:
    left = ["4c 4e 4g", "8d", "EOS"]
    right = ["4c 4e 4g", "8d", "EOS"]
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    assert "4c 4e 4g" in rows[2]


def test_column_width_based_on_widest_left_token() -> None:
    left = ["short", "a-much-longer-token", "EOS"]
    right = ["short", "x", "EOS"]
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    pipe_positions = [r.index("|") for r in rows[2:]]  # data rows only
    assert len(set(pipe_positions)) == 1


def test_left_header_longer_than_all_left_tokens() -> None:
    left = ["4c", "EOS"]
    right = ["4c", "EOS"]
    rows = lines(
        format_sequence_columns(
            left, right, left_header="Ground truth", highlight_mismatches=False
        )
    )
    pipe_positions = [r.index("|") for r in rows[2:]]
    assert len(set(pipe_positions)) == 1
    assert rows[0].startswith("Ground truth")


def test_separator_width_covers_long_right_tokens() -> None:
    left = ["4c"]
    right = ["4c 4e 4g 4B 4GG"]  # longer than "Pred"
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=False))
    separator = rows[1]
    data_row = rows[2]
    assert len(separator) >= len(data_row)


def test_mismatch_colors_right_token() -> None:
    left = ["4c", "4d"]
    right = ["4c", "8d"]  # second token differs
    rows = lines(format_sequence_columns(left, right, highlight_mismatches=True))
    assert _RED not in rows[2]
    assert _RED in rows[3]
    assert _RESET in rows[3]
    assert "8d" in rows[3]


def test_no_color_when_highlight_disabled() -> None:
    left = ["4c"]
    right = ["8d"]
    result = format_sequence_columns(left, right, highlight_mismatches=False)
    assert _RED not in result


def test_fill_slots_not_highlighted() -> None:
    # left longer than right — missing right slots should not be colored
    left = ["4c", "4d", "EOS"]
    right = ["4c"]
    result = format_sequence_columns(left, right, highlight_mismatches=True)
    assert _RED not in result


def test_empty_left_nonempty_right_no_highlight_on_fill() -> None:
    # right longer than left — fill slots on left side, no red coloring
    left: list[str] = []
    right = ["4c", "4d"]
    result = format_sequence_columns(left, right, highlight_mismatches=True)
    assert _RED not in result


def test_custom_headers() -> None:
    rows = lines(
        format_sequence_columns(
            ["a"],
            ["b"],
            left_header="Ground truth",
            right_header="Prediction",
            highlight_mismatches=False,
        )
    )
    assert "Ground truth" in rows[0]
    assert "Prediction" in rows[0]


def test_empty_sequences_produces_only_header_and_separator() -> None:
    rows = lines(format_sequence_columns([], [], highlight_mismatches=False))
    assert len(rows) == 2
