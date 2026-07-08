from cli.play import select_staves


def test_select_staves_keeps_requested_slots() -> None:
    systems = [["a0", "a1", "a2"], ["b0", "b1"]]
    assert select_staves(systems, (0, 1)) == [["a0", "a1"], ["b0", "b1"]]


def test_select_staves_single_slot() -> None:
    systems = [["a0", "a1", "a2"], ["b0", "b1"]]
    assert select_staves(systems, (0,)) == [["a0"], ["b0"]]


def test_select_staves_ignores_out_of_range_per_system() -> None:
    # Slot 2 exists only in the first system; the second contributes nothing.
    systems = [["a0", "a1", "a2"], ["b0", "b1"]]
    assert select_staves(systems, (2,)) == [["a2"], []]


def test_select_staves_preserves_requested_order() -> None:
    systems = [["a0", "a1", "a2"]]
    assert select_staves(systems, (2, 0)) == [["a2", "a0"]]
