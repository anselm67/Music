from itertools import zip_longest

_RED = "\033[31m"
_RESET = "\033[0m"


def format_sequence_columns(
    left: list[str],
    right: list[str],
    left_header: str = "GT",
    right_header: str = "Pred",
    highlight_mismatches: bool = True,
) -> str:
    """Return a two-column string comparing left and right token sequences.

    Mismatching tokens in the right column are highlighted red when
    highlight_mismatches is True (default). Fill slots (when one sequence is
    shorter) are never highlighted.
    """
    left_width = max(max((len(s) for s in left), default=0), len(left_header)) + 2
    right_width = max(max((len(s) for s in right), default=0), len(right_header))

    rows: list[str] = []
    rows.append(f"{left_header:<{left_width}}| {right_header}")
    rows.append("-" * left_width + "+" + "-" * (right_width + 2))

    for l_tok, r_tok in zip_longest(left, right, fillvalue=""):
        is_fill = l_tok == "" or r_tok == ""
        mismatch = highlight_mismatches and not is_fill and l_tok != r_tok
        r_display = f"{_RED}{r_tok}{_RESET}" if mismatch else r_tok
        rows.append(f"{l_tok:<{left_width}}| {r_display}")

    return "\n".join(rows)
