"""Unit tests for the interactive StaffEditor commands.

The editor's ``__init__`` opens a cv2 window and loads page images from disk, so
these tests build an instance via ``object.__new__`` (``_editor``) and populate
only the state the commands read: an in-memory ``Score``, blank page images, a
fake kern reader, and a mocked ``KernSheet`` for the I/O commands. Each command
(keystroke) registered in ``init_commands`` plus the resize helper is covered.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from kernsheet.editor import StaffEditor
from sheetmusic import Box, Page, Score, Staff, Status, System


def _sys(top: int, bottom: int, bars: list[int]) -> System:
    """A two-staff (treble/bass) grand system spanning ``[top, bottom]`` with the
    inter-staff gap set to a third of the height."""
    third = (bottom - top) // 3
    left, right = (bars[0], bars[-1]) if bars else (10, 500)
    return System(
        bar_numbers=[1],
        bars=list(bars),
        staves=[
            Staff(box=Box(left, top, right, top + third)),
            Staff(box=Box(left, bottom - third, right, bottom)),
        ],
    )


def _page(
    systems: list[System],
    *,
    page_number: int = 1,
    validated: bool = False,
    width: int = 600,
    height: int = 800,
) -> Page:
    return Page(
        page_number=page_number,
        image_width=width,
        image_height=height,
        systems=list(systems),
        status=Status.VALIDATED if validated else Status.PENDING,
    )


class _FakeKern:
    """Minimal stand-in for KernReader exposing only what the editor reads."""

    def __init__(
        self,
        *,
        bar_count: int = 4,
        first_bar: int = 1,
        bar_zero: bool = False,
        text: list[str] | None = None,
    ) -> None:
        self.bar_count = bar_count
        self.first_bar = first_bar
        self._bar_zero = bar_zero
        self._text = text

    def has_bar_zero(self) -> bool:
        return self._bar_zero

    def header(self) -> list[str]:
        return ["**kern", "*-"]

    def get_text(self, bar: int) -> list[str] | None:
        return self._text


def _editor(
    pages: list[Page],
    *,
    page_index: int = 0,
    system_index: int = 0,
    bar_index: int = 0,
    bar_offset: int = 0,
    fast_mode: bool = False,
    kern: _FakeKern | None = None,
    kern_sheet: MagicMock | None = None,
) -> StaffEditor:
    """A StaffEditor holding ``pages``, bypassing the display-dependent
    ``__init__``; every field a command touches is set explicitly."""
    editor = object.__new__(StaffEditor)
    editor.score = Score(id="t", pages=list(pages))
    editor.images = [
        np.zeros((p.image_height, p.image_width, 3), np.uint8) for p in pages
    ]
    editor.kern_sheet = kern_sheet if kern_sheet is not None else MagicMock()
    editor.kern = kern if kern is not None else _FakeKern()  # type: ignore[assignment]
    editor.key = "composer/work"
    editor.id = "composer/work-0"
    editor.max_size = (992, 780)
    editor.fast_mode = fast_mode
    editor.page_index = page_index
    editor.system_index = system_index
    editor.staff_index = 0
    editor.bar_index = bar_index
    editor.bar_offset = bar_offset
    editor.init_commands()
    return editor


def _two_systems() -> list[System]:
    return [_sys(100, 250, [10, 200, 400]), _sys(300, 450, [10, 200, 400])]


class TestResizeSystem:
    def test_extend_down_scales_proportionally(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 500])])])

        editor.resize_system(1)  # the 'e' command

        rh, lh = editor.system.staves
        # The system top is anchored; the bottom moves by delta.
        assert editor.system.top == 100
        assert editor.system.bottom == 251
        # Every offset below the top scales by (height+delta)/height, so the top
        # staff stays put and the gap below it widens proportionally.
        assert rh.box.top == 100
        assert lh.box.top - rh.box.bottom == 51

    def test_spreads_across_all_staves(self) -> None:
        # A three-staff system: staff heights are preserved and the added height
        # is shared across the inter-staff gaps.
        editor = _editor(
            [
                _page(
                    [
                        System(
                            bar_numbers=[1],
                            bars=[10, 500],
                            staves=[
                                Staff(box=Box(10, 100, 500, 140)),
                                Staff(box=Box(10, 200, 500, 240)),
                                Staff(box=Box(10, 300, 500, 340)),
                            ],
                        )
                    ]
                )
            ]
        )

        editor.resize_system(2)

        assert [s.box.height for s in editor.system.staves] == [40, 40, 40]
        gaps = [
            editor.system.staves[i + 1].top - editor.system.staves[i].bottom
            for i in range(2)
        ]
        assert gaps == [61, 61]  # both gaps absorb the added height
        assert editor.system.top == 100
        assert editor.system.bottom == 342  # 340 + delta, top anchored

    def test_shrink_up_is_inverse_of_extend(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 500])])])
        before = [(s.box.top, s.box.bottom) for s in editor.system.staves]

        editor.resize_system(1)  # 'e'
        editor.resize_system(-1)  # 'r'

        after = [(s.box.top, s.box.bottom) for s in editor.system.staves]
        assert after == before

    def test_noop_when_no_system_selected(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 500])])], system_index=-1)
        before = [
            (s.box.top, s.box.bottom) for s in editor.score.pages[0].systems[0].staves
        ]

        editor.resize_system(1)  # must not raise

        after = [
            (s.box.top, s.box.bottom) for s in editor.score.pages[0].systems[0].staves
        ]
        assert after == before


class TestResizeStaves:
    @staticmethod
    def _uneven() -> System:
        # two staves of unequal height (40 and 60), tops at 100 and 200.
        return System(
            bar_numbers=[1],
            bars=[10, 500],
            staves=[
                Staff(box=Box(10, 100, 500, 140)),
                Staff(box=Box(10, 200, 500, 260)),
            ],
        )

    def test_equalises_to_common_height_keeping_tops(self) -> None:
        editor = _editor([_page([self._uneven()])])

        editor.resize_staves(2)  # 'g' — target max(40, 60) + 2 = 62

        s0, s1 = editor.system.staves
        assert (s0.box.top, s0.box.height) == (100, 62)
        assert (s1.box.top, s1.box.height) == (200, 62)

    def test_shrink(self) -> None:
        editor = _editor([_page([self._uneven()])])

        editor.resize_staves(-2)  # 'h' — target max(40, 60) - 2 = 58

        assert [s.box.height for s in editor.system.staves] == [58, 58]

    def test_noop_when_no_system_selected(self) -> None:
        editor = _editor([_page([self._uneven()])], system_index=-1)
        before = [s.box.height for s in editor.score.pages[0].systems[0].staves]

        editor.resize_staves(2)  # must not raise

        after = [s.box.height for s in editor.score.pages[0].systems[0].staves]
        assert after == before


class TestApplySystemRatio:
    def test_copies_height_and_gap_keeping_top_and_sides(self) -> None:
        # sys0 (selected): height 40, gap 20 (160-140). sys1: different shape,
        # top 300, sides 20..400 — must adopt 40/20 while keeping top and sides.
        sys0 = System(
            bar_numbers=[1],
            bars=[10, 500],
            staves=[
                Staff(box=Box(10, 100, 500, 140)),
                Staff(box=Box(10, 160, 500, 200)),
            ],
        )
        sys1 = System(
            bar_numbers=[1],
            bars=[20, 400],
            staves=[
                Staff(box=Box(20, 300, 400, 360)),
                Staff(box=Box(20, 400, 400, 470)),
            ],
        )
        editor = _editor([_page([sys0, sys1])], system_index=0)

        editor.apply_system_ratio()  # 'u'

        s0, s1 = editor.page.systems[1].staves
        # height 40, gap 20, top 300 and sides 20..400 preserved
        assert (s0.box.left, s0.box.top, s0.box.right, s0.box.bottom) == (
            20,
            300,
            400,
            340,
        )
        assert (s1.box.left, s1.box.top, s1.box.right, s1.box.bottom) == (
            20,
            360,
            400,
            400,
        )
        # the selected reference system is untouched
        assert [s.box.height for s in editor.page.systems[0].staves] == [40, 40]

    def test_single_staff_target_adopts_height(self) -> None:
        # reference is 2-staff (height 40); a 1-staff target adopts the height,
        # keeping its own top (300) and sides (20..400).
        sys0 = System(
            bar_numbers=[1],
            bars=[10, 500],
            staves=[
                Staff(box=Box(10, 100, 500, 140)),
                Staff(box=Box(10, 160, 500, 200)),
            ],
        )
        sys1 = System(
            bar_numbers=[1],
            bars=[20, 400],
            staves=[Staff(box=Box(20, 300, 400, 380))],
        )
        editor = _editor([_page([sys0, sys1])], system_index=0)

        editor.apply_system_ratio()  # 'u'

        (s,) = editor.page.systems[1].staves
        assert (s.box.left, s.box.top, s.box.right, s.box.bottom) == (20, 300, 400, 340)

    def test_noop_when_no_system_selected(self) -> None:
        page = _page(_two_systems())
        editor = _editor([page], system_index=-1)
        before = [
            [(s.box.top, s.box.bottom) for s in sys.staves] for sys in page.systems
        ]

        editor.apply_system_ratio()  # must not raise

        after = [
            [(s.box.top, s.box.bottom) for s in sys.staves]
            for sys in editor.page.systems
        ]
        assert after == before


class TestBarEditing:
    def test_add_bar_inserts_after_selected(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])], bar_index=1)

        editor.add_bar()  # 'a'

        assert editor.system.bars == [10, 200, 210, 400]
        assert editor.bar_index == 2  # now points at the inserted bar

    def test_add_bar_with_no_selection_appends(self) -> None:
        editor = _editor([_page([_sys(100, 250, [])])], bar_index=-1)

        editor.add_bar()

        assert editor.system.bars == [10]
        assert editor.bar_index == 0

    def test_delete_selected_bar(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])], bar_index=1)

        editor.delete_selected_bar()  # 'd'

        assert editor.system.bars == [10, 400]
        assert editor.bar_index == 0

    def test_delete_bar_noop_when_none_selected(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])], bar_index=-1)

        editor.delete_selected_bar()

        assert editor.system.bars == [10, 200, 400]

    def test_move_bar_shifts_only_selected(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])], bar_index=1)

        editor.move_bar(2)  # 'l'
        assert editor.system.bars == [10, 202, 400]
        editor.move_bar(-2)  # 'j'
        assert editor.system.bars == [10, 200, 400]

    def test_move_bar_noop_without_bars(self) -> None:
        editor = _editor([_page([_sys(100, 250, [])])], bar_index=-1)

        editor.move_bar(2)

        assert editor.system.bars == []


class TestSystemEditing:
    def test_add_system_inserts_after_current(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        editor.add_system()  # 'w'

        assert editor.page.system_count == 2
        assert editor.system_index == 1  # selection follows the new system
        new = editor.page.systems[1]
        assert new.staff_count == 2
        # The new system inherits the page's side bars (left/right margins).
        assert new.bars == [10, 400]

    def test_delete_selected_system(self) -> None:
        editor = _editor([_page(_two_systems())], system_index=0)

        editor.delete_selected_system()  # 'x'

        assert editor.page.system_count == 1
        assert editor.system_index == 0

    def test_delete_system_noop_when_none_selected(self) -> None:
        editor = _editor([_page(_two_systems())], system_index=-1)

        editor.delete_selected_system()

        assert editor.page.system_count == 2

    def test_move_system_up_and_down(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        editor.move_system(2)  # 'i' moves up (smaller y)
        assert editor.system.top == 98
        assert editor.system.bottom == 248
        editor.move_system(-2)  # 'm' moves down
        assert editor.system.top == 100
        assert editor.system.bottom == 250


class TestNavigation:
    def test_select_next_bar_within_system(self) -> None:
        editor = _editor([_page(_two_systems())], bar_index=0)

        editor.select_next_bar()  # Right

        assert editor.system_index == 0
        assert editor.bar_index == 1

    def test_select_next_bar_rolls_into_next_system(self) -> None:
        editor = _editor([_page(_two_systems())], system_index=0, bar_index=2)

        editor.select_next_bar()

        assert editor.system_index == 1
        assert editor.bar_index == 0

    def test_select_prev_bar_within_system(self) -> None:
        editor = _editor([_page(_two_systems())], bar_index=1)

        editor.select_prev_bar()  # Left

        assert editor.bar_index == 0

    def test_select_prev_bar_rolls_into_prev_system_last_bar(self) -> None:
        editor = _editor([_page(_two_systems())], system_index=1, bar_index=0)

        editor.select_prev_bar()

        assert editor.system_index == 0
        assert editor.bar_index == 2  # last bar of the previous system

    def test_select_next_and_prev_system(self) -> None:
        editor = _editor([_page(_two_systems())], system_index=0)

        editor.select_next_system()  # Down
        assert editor.system_index == 1
        assert editor.bar_index == 0
        editor.select_prev_system()  # Up
        assert editor.system_index == 0
        assert editor.bar_index == 0

    def test_next_and_prev_page(self) -> None:
        editor = _editor(
            [
                _page([_sys(100, 250, [10, 200, 400])], page_number=1),
                _page([_sys(100, 250, [10, 200, 400])], page_number=2),
            ]
        )

        editor.next()  # space
        assert editor.page_index == 1
        assert editor.system_index == 0
        editor.prev()  # 'p'
        assert editor.page_index == 0

    def test_page_navigation_respects_boundaries(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        editor.prev()  # already at first page
        assert editor.page_index == 0
        editor.next()  # already at last page
        assert editor.page_index == 0

    def test_next_in_fast_mode_validates_and_saves(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [
                _page([_sys(100, 250, [10, 200, 400])], page_number=1),
                _page([_sys(100, 250, [10, 200, 400])], page_number=2),
            ],
            fast_mode=True,
            kern_sheet=kern_sheet,
        )

        editor.next()

        assert editor.score.pages[0].validated is True
        kern_sheet.save_score.assert_called_once_with(editor.id, editor.score)


class TestPageCommands:
    def test_cycle_status(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])], validated=False)])

        assert editor.page.status is Status.PENDING
        editor.run_command(ord("v"))
        assert editor.page.status is Status.VALIDATED
        editor.run_command(ord("v"))
        assert editor.page.status is Status.REJECTED
        editor.run_command(ord("v"))
        assert editor.page.status is Status.PENDING

    def test_blank_pages_clears_and_validates_range(self) -> None:
        editor = _editor(
            [
                _page([_sys(100, 250, [10, 200, 400])], page_number=1),
                _page([_sys(100, 250, [10, 200, 400])], page_number=2),
                _page([_sys(100, 250, [10, 200, 400])], page_number=3),
            ],
        )

        editor.blank_pages(range(0, 2))

        assert editor.score.pages[0].systems == [] and editor.score.pages[0].validated
        assert editor.score.pages[1].systems == [] and editor.score.pages[1].validated
        assert editor.score.pages[2].systems != []  # untouched

    def test_acknowledge_reviews_records_firing_reviews(self) -> None:
        # A system with mismatched staff heights fires the staff_height review.
        bad = System(
            bar_numbers=[1],
            bars=[10, 400],
            staves=[
                Staff(box=Box(10, 0, 400, 40)),
                Staff(box=Box(10, 60, 400, 200)),
            ],
        )
        editor = _editor([_page([bad])])
        assert {f.review for f in editor.page_findings()} == {"staff_height"}

        editor.run_command(ord("o"))

        assert editor.page.reviewed == ["staff_height"]
        assert editor.page_findings() == []  # acknowledged -> no longer flagged

    def test_toggle_fast_mode(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])], fast_mode=False)

        editor.toggle_fast_mode()
        assert editor.fast_mode is True
        editor.toggle_fast_mode()
        assert editor.fast_mode is False

    def test_system_height_averages_staves(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        assert editor.system_height() == 50  # both staves are 50px tall


class TestFixSideBars:
    def test_extends_systems_to_shared_margins(self) -> None:
        page = _page(
            [_sys(100, 250, [100, 200, 300]), _sys(300, 450, [150, 250])],
            width=600,
        )
        editor = _editor([page])

        editor.fix_side_bars()  # 'z'

        # First system already spans the page-wide [100, 300] extent.
        assert editor.page.systems[0].bars == [100, 200, 300]
        # Second system gets the left (100) and right (300) edges added.
        assert editor.page.systems[1].bars == [100, 150, 250, 300]

    def test_noop_when_no_bars_anywhere(self) -> None:
        editor = _editor([_page([_sys(100, 250, []), _sys(300, 450, [])])])

        editor.fix_side_bars()

        assert all(s.bars == [] for s in editor.page.systems)


class TestBarCounting:
    def test_get_bar_offset(self) -> None:
        editor = _editor([_page(_two_systems())], kern=_FakeKern(first_bar=1))

        # 1 (no bar zero) + 2 + 2 bars across the two systems.
        assert editor.get_bar_offset() == 5

    def test_get_bar_offset_with_bar_zero_and_pickup(self) -> None:
        editor = _editor(
            [_page(_two_systems())], kern=_FakeKern(bar_zero=True, first_bar=3)
        )

        # start 0 (bar zero) + 4 bars + (first_bar - 1 = 2).
        assert editor.get_bar_offset() == 6

    def test_bar_number_property(self) -> None:
        editor = _editor(
            [_page(_two_systems())], system_index=1, bar_index=2, bar_offset=10
        )

        # bar_offset 10 + 2 (bars in system 0) + bar_index 2.
        assert editor.bar_number == 14

    def test_check_bar_count_match(self, capsys) -> None:  # type: ignore[no-untyped-def]
        editor = _editor([_page(_two_systems())], kern=_FakeKern(bar_count=5))

        editor.check_bar_count()  # '/'

        assert "victory" in capsys.readouterr().out

    def test_check_bar_count_mismatch(self, capsys) -> None:  # type: ignore[no-untyped-def]
        editor = _editor([_page(_two_systems())], kern=_FakeKern(bar_count=99))

        editor.check_bar_count()

        assert "Yuck" in capsys.readouterr().out

    def test_renumber_bars_is_sequential_across_pages(self) -> None:
        # _two_systems seeds both systems with the stale bar_numbers=[1]; each has
        # 2 bars (3 barlines). Renumber rebuilds them from geometry, continuous
        # across the page break.
        editor = _editor(
            [_page(_two_systems()), _page(_two_systems(), page_number=2)],
            kern=_FakeKern(first_bar=1),
        )

        editor.renumber_bars()

        numbers = [s.bar_numbers for p in editor.score.pages for s in p.systems]
        assert numbers == [[1, 2], [3, 4], [5, 6], [7, 8]]

    def test_renumber_bars_honours_kern_anchor(self) -> None:
        # bar-zero pickup + first_bar 3 => start at 0 + (3 - 1) = 2.
        editor = _editor(
            [_page(_two_systems())], kern=_FakeKern(bar_zero=True, first_bar=3)
        )

        editor.renumber_bars()

        assert [s.bar_numbers for s in editor.page.systems] == [[2, 3], [4, 5]]

    def test_renumber_bars_clears_barless_systems(self) -> None:
        # A system with staves but no barlines (count 0) carrying a stale [1] must
        # be cleared so it doesn't linger inconsistent and the bar_numbers review
        # can flag it; it also must not consume a bar number from the chain.
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400]), _sys(300, 450, [])])],
            kern=_FakeKern(first_bar=1),
        )

        editor.renumber_bars()

        assert [s.bar_numbers for s in editor.page.systems] == [[1, 2], []]


class TestIoCommands:
    def test_save_writes_score(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])], kern_sheet=kern_sheet
        )

        editor.save()  # 's'

        kern_sheet.save_score.assert_called_once_with(editor.id, editor.score)

    def test_save_renumbers_before_writing(self) -> None:
        # save must resync bar_numbers from geometry so the written score is never
        # left drifted (the noter/scorer slice the kern by first_bar_number).
        kern_sheet = MagicMock()
        editor = _editor(
            [_page(_two_systems())], kern=_FakeKern(first_bar=1), kern_sheet=kern_sheet
        )

        editor.save()

        saved = kern_sheet.save_score.call_args.args[1]
        assert [s.bar_numbers for s in saved.pages[0].systems] == [[1, 2], [3, 4]]

    def test_delete_score_removes_from_catalog(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])], kern_sheet=kern_sheet
        )

        editor.delete_score()  # '1'

        kern_sheet.delete_score.assert_called_once_with(
            editor.key, kern_sheet.get_score(editor.id)
        )

    def test_delete_entry_removes_entry(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])], kern_sheet=kern_sheet
        )

        editor.delete_entry()  # '2'

        kern_sheet.delete_entry.assert_called_once_with(editor.key)

    def test_confirm_true_only_on_y(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        with patch("kernsheet.editor.cv2.waitKey", return_value=ord("y")):
            assert editor.confirm("Delete?") is True
        with patch("kernsheet.editor.cv2.waitKey", return_value=ord("n")):
            assert editor.confirm("Delete?") is False

    def test_recompute_bars_is_a_harmless_stub(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        editor.recompute_bars()  # 'c' — not implemented yet, must not raise

    def test_edit_kern_and_tokens_launches_editor(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])], kern_sheet=kern_sheet
        )

        with patch("kernsheet.editor.subprocess.run") as run:
            editor.edit_kern_and_tokens()  # '!'

        run.assert_called_once()

    def test_help_and_title_and_print_kerns_run(self, capsys) -> None:  # type: ignore[no-untyped-def]
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])],
            kern=_FakeKern(text=["1\t4c"]),
        )

        editor.help()  # 'h' / '?'
        editor.title()  # 't'
        editor.print_kerns()  # 'k'

        assert capsys.readouterr().out  # produced some output, no exception


class TestRunCommand:
    def test_dispatches_registered_key(self) -> None:
        kern_sheet = MagicMock()
        editor = _editor(
            [_page([_sys(100, 250, [10, 200, 400])])], kern_sheet=kern_sheet
        )

        assert editor.run_command(ord("s")) is True
        kern_sheet.save_score.assert_called_once()

    def test_unknown_key_returns_false(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])

        assert editor.run_command(ord("Z")) is False

    def test_failing_command_is_swallowed(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 200, 400])])])
        editor.actions[ord("s")].func = MagicMock(side_effect=RuntimeError("boom"))

        # The exception is logged inside run_command, not propagated.
        assert editor.run_command(ord("s")) is True


class TestRecomputeBars:
    """The 'c' key replaces the current system's bars with detector output."""

    def test_recompute_replaces_bars_from_image(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 500])])])
        # White page with black barlines spanning the system band [100, 250].
        img = np.full((800, 600, 3), 255, np.uint8)
        for x in (40, 300, 560):
            img[100:250, x : x + 1] = 0
        editor.images[0] = img

        editor.recompute_bars()

        assert editor.system.bars == [40, 300, 560]

    def test_recompute_noop_without_system(self) -> None:
        editor = _editor([_page([_sys(100, 250, [10, 500])])], system_index=-1)
        editor.recompute_bars()  # must not raise
