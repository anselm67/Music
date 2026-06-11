"""Unit tests for the interactive StaffEditor commands.

The editor's ``__init__`` opens a cv2 window and loads page images, so the tests
build an instance via ``object.__new__`` and populate only the state a given
command touches. This first suite covers the ``e``/``r`` resize commands; the
remaining commands get their own suite in a later commit.
"""

from kernsheet.editor import StaffEditor
from sheetmusic import Box, Page, Score, Staff, System


def _grand_staff() -> System:
    """A two-staff (treble/bass) system with a 50px inter-staff gap."""
    return System(
        bar_numbers=[1],
        bars=[10, 500],
        staves=[
            Staff(box=Box((10, 100), (500, 150))),
            Staff(box=Box((10, 200), (500, 250))),
        ],
    )


def _editor(systems: list[System]) -> StaffEditor:
    """A StaffEditor holding one page of ``systems``, bypassing the heavy
    (display-dependent) ``__init__``. Only the fields the staff/system edit
    commands read are set."""
    editor = object.__new__(StaffEditor)
    editor.score = Score(
        id="t",
        pages=[
            Page(
                page_number=1,
                image_width=600,
                image_height=800,
                systems=systems,
                validated=False,
            )
        ],
    )
    editor.page_index = 0
    editor.system_index = 0
    return editor


class TestResizeSystem:
    def test_extend_down_spreads_height_equally(self) -> None:
        editor = _editor([_grand_staff()])

        editor.resize_system(1)  # the 'e' command

        rh, lh = editor.system.staves
        # Every staff grows by delta, not just the last one.
        assert rh.box.height == 51
        assert lh.box.height == 51
        # The system top is fixed; the bottom moves by staff_count * delta.
        assert editor.system.top == 100
        assert editor.system.bottom == 252
        # The inter-staff gap is preserved (staves shift, they don't stretch).
        assert lh.box.top - rh.box.bottom == 50

    def test_spreads_across_all_staves(self) -> None:
        # A three-staff system shows the change is shared by *every* staff.
        editor = _editor(
            [
                System(
                    bar_numbers=[1],
                    bars=[10, 500],
                    staves=[
                        Staff(box=Box((10, 100), (500, 140))),
                        Staff(box=Box((10, 200), (500, 240))),
                        Staff(box=Box((10, 300), (500, 340))),
                    ],
                )
            ]
        )

        editor.resize_system(2)

        assert [s.box.height for s in editor.system.staves] == [42, 42, 42]
        assert editor.system.top == 100
        assert editor.system.bottom == 346  # 340 + 3 * 2

    def test_shrink_up_is_inverse_of_extend(self) -> None:
        editor = _editor([_grand_staff()])
        before = [(s.box.top, s.box.bottom) for s in editor.system.staves]

        editor.resize_system(1)  # 'e'
        editor.resize_system(-1)  # 'r'

        after = [(s.box.top, s.box.bottom) for s in editor.system.staves]
        assert after == before

    def test_noop_when_no_system_selected(self) -> None:
        editor = _editor([_grand_staff()])
        editor.system_index = -1

        editor.resize_system(1)  # must not raise

        assert editor.page.systems[0] == _grand_staff()
