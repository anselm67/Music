"""Interactive cv2 staff-layout editor, ported from OMR's ``staff_editor.py``.

Reviews/corrects a score's grand-staff layout one page at a time: move/extend
staves, add/move/delete barlines, validate pages, check the bar count against the
kern. It edits a flat *envelope* view (``rh_top``/``lh_bot``/``bars`` per system)
supplied by :class:`EditorBackend`, which persists it back as native ``Score`` JSON.
"""

import logging
import subprocess
from dataclasses import replace
from typing import Any, Callable, Tuple

import cv2
from cv2.typing import MatLike

from kern import KernReader
from kernsheet import ClassicalStaffer, Finding, KernSheet, score_findings
from sheetmusic import Box, Page, Score, Staff, Status, System
from utils import format_sequence_columns


class Action:
    key_code: int
    func: Callable[[], None]
    help: str

    def __init__(self, key: int | str, func: Callable[[], None], help: str):
        self.key_code = key if type(key) is int else ord(str(key))
        self.func = func
        self.help = help


class StaffEditor:
    STAFFER_WINDOW = "StaffEditor"

    # X11 keysym low-bytes for bare modifier presses (Shift/Ctrl/Caps/Alt/Super),
    # which cv2.waitKey reports on their own — ignored rather than warned about.
    MODIFIER_KEYS = frozenset(range(225, 237))

    kern_sheet: KernSheet

    # Data being edited.
    key: str
    id: str
    images: list[MatLike]
    score: Score
    kern: KernReader

    # Editor's config
    max_size: tuple[int, int]
    actions: dict[int, Action]
    fast_mode: bool

    # Current state of the editor.
    page_index: int = 0
    system_index: int = 0
    staff_index: int = 0
    bar_index: int = 0
    bar_offset: int = 0
    scale_ratio: float = 1.0

    @property
    def image(self) -> MatLike:
        assert 0 <= self.page_index < len(self.images)
        return self.images[self.page_index]

    @property
    def page(self) -> Page:
        assert 0 <= self.page_index < self.score.page_count
        return self.score.pages[self.page_index]

    def replace_page(self, **kwargs: Any) -> None:
        self.score.pages[self.page_index] = replace(self.page, **kwargs)

    def page_findings(self, page: Page | None = None) -> list[Finding]:
        """Un-suppressed review findings for ``page`` (the current page by default)."""
        page = page if page is not None else self.page
        return [
            f for f in score_findings(self.score) if f.page_number == page.page_number
        ]

    def cycle_status(self) -> None:
        """Advance the page through PENDING -> VALIDATED -> REJECTED -> PENDING."""
        order = [Status.PENDING, Status.VALIDATED, Status.REJECTED]
        nxt = order[(order.index(self.page.status) + 1) % len(order)]
        self.replace_page(status=nxt)
        print(f"Page {self.page.page_number} status: {nxt.value}")

    def acknowledge_reviews(self) -> None:
        """Mark every review currently firing on this page as "ok" — recording each
        in Page.reviewed so it stops being flagged (the layout itself is unchanged)."""
        firing = {f.review for f in self.page_findings()}
        if not firing:
            print("No outstanding reviews on this page.")
            return
        self.replace_page(reviewed=sorted(set(self.page.reviewed) | firing))
        print(f"Acknowledged: {', '.join(sorted(firing))}")

    @property
    def system(self) -> System:
        assert 0 <= self.system_index < self.page.system_count
        return self.page.systems[self.system_index]

    def replace_system(self, **kwargs: Any) -> None:
        self.page.systems[self.system_index] = replace(self.system, **kwargs)

    @property
    def staff(self) -> Staff:
        assert 0 <= self.staff_index < len(self.system.staves)
        return self.system.staves[self.staff_index]

    def move_bar(self, delta: int) -> None:
        if self.system.bar_count > 0:
            bars = self.system.bars.copy()
            assert 0 <= self.bar_index < len(bars)
            bars[self.bar_index] += delta
            self.replace_system(bars=bars)

    def move_system(self, delta: int) -> None:
        if self.system_index < 0:
            return
        new_staves = [
            replace(staff, box=staff.box.up(delta)) for staff in self.system.staves
        ]
        self.page.systems[self.system_index] = replace(self.system, staves=new_staves)

    def resize_system(self, delta: int) -> None:
        # Grows/shrinks the system from the bottom, keeping the top anchored and
        # the internal proportions (staff heights and gaps) constant. The system
        # top (staves[0].top) is fixed; the new height is the old height + delta;
        # every staff edge's offset below the top is scaled by the same factor, so
        # the bottom moves by delta and the staff/gap ratios are preserved.
        if self.system_index < 0 or self.system.staff_count == 0:
            return
        top = self.system.staves[0].top
        height = self.system.staves[-1].bottom - top
        if height <= 0 or height + delta <= 0:
            return
        factor = (height + delta) / height
        staves = []
        for staff in self.system.staves:
            box = staff.box
            new_top = top + round((box.top - top) * factor)
            new_bottom = top + round((box.bottom - top) * factor)
            staves.append(
                replace(staff, box=Box(box.left, new_top, box.right, new_bottom))
            )
        self.replace_system(staves=staves)

    def resize_staves(self, delta: int) -> None:
        # Sets every staff in the selected system to one common height, nudged by
        # ``delta`` — the grand-staff invariant the staff_height review checks. Each
        # staff keeps its top; its bottom becomes top + height. The target is the
        # current tallest staff (so nothing clips) plus delta, so the first press
        # also equalises staves that had drifted apart.
        if self.system_index < 0 or self.system.staff_count == 0:
            return
        height = max(staff.box.height for staff in self.system.staves) + delta
        if height <= 0:
            return
        staves = [
            replace(
                staff,
                box=Box(
                    staff.box.left,
                    staff.box.top,
                    staff.box.right,
                    staff.box.top + height,
                ),
            )
            for staff in self.system.staves
        ]
        self.replace_system(staves=staves)

    def apply_system_ratio(self) -> None:
        # Copies the selected system's internal stave geometry — stave height and
        # inter-staff gap — onto every other system on the page, so the whole page
        # shares one grand-staff shape (what the staff_height review checks). Each
        # target system keeps its own vertical position (top) and horizontal extent
        # (bars / left-right); its staves are re-laid from its top with the reference
        # height and gap. Height is the tallest reference stave (matching resize_staves,
        # so it's well-defined even if the reference isn't equalised yet); the gap is
        # the reference's first inter-staff gap, applied uniformly to every target gap.
        if self.system_index < 0 or self.system.staff_count == 0:
            return
        ref = self.system.staves
        height = max(staff.box.height for staff in ref)
        gap = ref[1].box.top - ref[0].box.bottom if len(ref) > 1 else 0
        for i, system in enumerate(self.page.systems):
            if i == self.system_index or system.staff_count == 0:
                continue
            top = system.staves[0].box.top
            staves = [
                replace(
                    staff,
                    box=Box(
                        staff.box.left,
                        top + j * (height + gap),
                        staff.box.right,
                        top + j * (height + gap) + height,
                    ),
                )
                for j, staff in enumerate(system.staves)
            ]
            self.page.systems[i] = replace(system, staves=staves)

    @property
    def bar_number(self) -> int:
        bar_number = self.bar_offset
        for system in self.page.systems[: self.system_index]:
            bar_number += max(len(system.bars) - 1, 0)
        return bar_number + self.bar_index

    def __init__(
        self,
        kern_sheet: KernSheet,
        key: str,
        id: str,
        max_size: tuple[int, int] = (992, 780),
    ):
        self.kern_sheet = kern_sheet
        self.key = key
        self.id = id
        self._load_score(id)
        self.kern = KernReader(kern_sheet.tokens_path(key))
        self.max_size = max_size
        self.page_index = 0
        self.system_index = 0
        self.staff_index = 0
        self.bar_index = 0
        self.bar_offset = 0
        self.fast_mode = False
        self._reset_selection()
        self.update_bar_offset()
        cv2.namedWindow(self.STAFFER_WINDOW)
        self.init_commands()

    def _reset_selection(self) -> None:
        """Point system/staff/bar selection at the first selectable item on the
        current page (or -1 where the page/system has nothing to select)."""
        self.system_index = 0
        self.staff_index = 0
        self.bar_index = 0
        if self.page.system_count <= 0:
            self.system_index = -1
            self.staff_index = -1
            self.bar_index = -1
        elif self.system.staff_count <= 0:
            self.staff_index = -1
            self.bar_index = -1
        elif len(self.system.bars) <= 0:
            self.bar_index = -1

    def _load_score(self, id: str) -> None:
        self.score = self.kern_sheet.load_score(id)
        self.images = []
        for page in self.score.pages:
            png_path = self.kern_sheet.png_path(id, page.page_number)
            image = cv2.imread(png_path.as_posix())  # already BGR, ready for imshow
            if image is None:
                raise FileNotFoundError(
                    f"{id}: missing page image {png_path}; run `kernsheet make`"
                )
            self.images.append(image)
        self.id = id

    def draw_page(
        self,
        image: MatLike,
        page: Page,
        bar_offset: int,
        selected_system: int = -1,
        selected_bar: int = -1,
        thickness: int = 2,
    ) -> MatLike:
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        GREEN = (2, 7 * 16 + 1, 4 * 16 + 8)
        PURPLE = (160, 32, 160)
        STATUS_COLOR = {
            Status.PENDING: BLUE,
            Status.VALIDATED: GREEN,
            Status.REJECTED: PURPLE,
        }
        style, selected_style = (STATUS_COLOR[page.status], 2), (RED, 4)
        rgb_image = image.copy()
        if page.system_count == 0:
            # That's fine let the user know.
            cv2.putText(
                rgb_image,
                page.status.value.capitalize(),
                (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=style[0],
                thickness=style[1],
            )

        for system_index, system in enumerate(page.systems):
            if len(system.bars) == 0:
                width = rgb_image.shape[1]
                color, thickness = (
                    selected_style if (system_index == selected_system) else style
                )
                cv2.line(
                    rgb_image,
                    (0, system.top),
                    (width, system.top),
                    color,
                    thickness,
                )
                # Left hand staff
                cv2.line(
                    rgb_image,
                    (0, system.bottom),
                    (width, system.bottom),
                    color,
                    thickness,
                )
                continue
            bars = system.bars
            for barno, bar in enumerate(bars):
                color, thickness = (
                    selected_style
                    if (system_index == selected_system) and (barno == selected_bar)
                    else style
                )
                cv2.line(
                    rgb_image,
                    (bar, system.top),
                    (bar, system.bottom),
                    color,
                    thickness,
                )
                # Renders the bar number only if not last of staff.
                if barno != len(bars) - 1:
                    cv2.putText(
                        rgb_image,
                        str(bar_offset + barno),
                        (bar + 3, system.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=thickness,
                    )
            bar_offset += len(bars) - 1
            # Draws the top and bottom staves:
            color, thickness = (
                selected_style if (system_index == selected_system) else style
            )
            cv2.line(  # Right hand.
                rgb_image,
                (bars[0], system.top),
                (bars[-1], system.top),
                color,
                thickness,
            )
            cv2.line(
                rgb_image,
                (bars[0], system.staves[0].bottom),
                (bars[-1], system.staves[0].bottom),
                color,
                thickness,
            )
            cv2.line(  # Left hand.
                rgb_image,
                (bars[0], system.bottom),
                (bars[-1], system.bottom),
                color,
                thickness,
            )
            cv2.line(  # Left hand.
                rgb_image,
                (bars[0], system.staves[-1].top),
                (bars[-1], system.staves[-1].top),
                color,
                thickness,
            )
        return rgb_image

    def get_bar_offset(self, page_index: int = -1) -> int:
        if page_index < 0:
            page_index = self.score.page_count
        bar_number = 0 if self.kern.has_bar_zero() else 1
        for page in self.score.pages[:page_index]:
            for system in page.systems:
                if system.staff_count > 0:
                    bar_number += max(system.bar_count, 0)
        return bar_number + (self.kern.first_bar - 1)

    def update_bar_offset(self) -> None:
        self.bar_offset = self.get_bar_offset(self.page_index)

    def get(self) -> Tuple[MatLike, Page]:
        return self.images[self.page_index], self.page

    def print_page_status(self) -> None:
        """One-liner shown on page entry: status, flagged reviews, acknowledged ones."""
        parts = [
            f"Page {self.page.page_number}/{self.score.page_count}: "
            f"{self.page.status.value}"
        ]
        if firing := sorted(f.review for f in self.page_findings()):
            parts.append(f"flagged: {', '.join(firing)}")
        if self.page.reviewed:
            parts.append(f"oked: {', '.join(self.page.reviewed)}")
        print(" | ".join(parts))

    def next(self) -> None:
        if self.page_index + 1 >= self.score.page_count:
            print("End of score.")
            return
        if self.fast_mode:
            self.replace_page(status=Status.VALIDATED)
            self.save()
        self.page_index += 1
        self.system_index = 0
        if self.page.system_count == 0:
            self.system_index = -1
            self.staff_index = -1
            self.bar_index = -1
        elif self.system.staff_count == 0:
            self.staff_index = -1
            self.bar_index = -1
        elif self.system.bar_count <= 0:
            self.bar_index = -1
        else:
            self.bar_index = 0
        self.print_page_status()

    def prev(self, select_last: bool = False) -> None:
        if self.page_index - 1 < 0:
            print("Beginning of score.")
            return
        self.page_index -= 1
        self.print_page_status()
        system_count = self.page.system_count
        if system_count == 0:
            self.system_index = -1
            self.staff_index = -1
            self.bar_index = -1
            bars_len = 0
        elif self.system.staff_count == 0:
            self.staff_index = -1
            self.bar_index = -1
            bars_len = 0
        elif select_last:
            self.system_index = system_count - 1
            bars_len = len(self.system.bars)
        else:
            self.system_index = 0
            bars_len = len(self.system.bars)
        if self.system_index < 0 or bars_len == 0:
            self.bar_index = -1
        else:
            self.bar_index = bars_len - 1 if select_last else 0

    def select_prev_system(self, select_last: bool = False) -> None:
        if self.system_index - 1 < 0:
            self.prev(select_last=True)
        else:
            self.system_index = self.system_index - 1
            self.bar_index = 0
            bar_count = len(self.system.bars)
            if bar_count <= 0:
                self.bar_index = -1
            elif select_last:
                self.bar_index = bar_count - 1

    def select_next_system(self) -> None:
        if self.system_index + 1 >= self.page.system_count:
            self.next()
        else:
            self.system_index = self.system_index + 1
            self.bar_index = 0
            if len(self.system.bars) <= 0:
                self.bar_index = -1

    def select_next_bar(self) -> None:
        bar_count = 0
        if self.system_index >= 0 and self.system.staff_count >= 0:
            bar_count = len(self.system.bars)
        if self.bar_index + 1 >= bar_count:
            self.select_next_system()
        else:
            self.bar_index += 1

    def select_prev_bar(self) -> None:
        if self.bar_index - 1 < 0:
            self.select_prev_system(select_last=True)
        else:
            self.bar_index -= 1

    def system_height(self) -> int:
        # Tries to make a good guess at staff height.
        heights = [staff.bottom - staff.top for staff in self.system.staves]
        if len(heights) > 0:
            return int(sum(heights) / len(heights))
        else:
            return 128

    def _system_sides(self) -> tuple[int, int]:
        # Left/right barline x-positions for a new system: the average sides of
        # the page's existing systems, or the page width minus a margin if none.
        sides = [(s.bars[0], s.bars[-1]) for s in self.page.systems if len(s.bars) > 0]
        if sides:
            left = sum(left for left, _ in sides) // len(sides)
            right = sum(right for _, right in sides) // len(sides)
            return left, right
        margin = 25
        return margin, self.image.shape[1] - margin

    def _staff_metrics(self) -> tuple[int, int] | None:
        # Average staff height and inter-staff gap across the page's systems, or
        # None if the page has no multi-staff system to derive a gap from.
        heights = [
            staff.bottom - staff.top for s in self.page.systems for staff in s.staves
        ]
        gaps = [
            s.staves[i + 1].top - s.staves[i].bottom
            for s in self.page.systems
            for i in range(len(s.staves) - 1)
        ]
        if not heights or not gaps:
            return None
        return sum(heights) // len(heights), sum(gaps) // len(gaps)

    def _make_system(self, top: int, bottom: int) -> System:
        left, right = self._system_sides()
        metrics = self._staff_metrics()
        if metrics is not None:
            staff_height, gap = metrics
            bottom = top + 2 * staff_height + gap
        else:
            staff_height = (bottom - top) // 3
        return System(
            bar_numbers=[],
            bars=[left, right],
            staves=[
                Staff(Box(left, top, right, top + staff_height)),
                Staff(Box(left, bottom - staff_height, right, bottom)),
            ],
        )

    def add_system(self) -> None:
        height = self.system_height()
        if self.system_index < 0:
            ypos = 100
            self.page.systems.append(self._make_system(ypos, ypos + height))
            self.system_index = self.page.system_count - 1
        else:
            ypos = self.system.staves[0].bottom + 10
            self.page.systems.insert(
                self.system_index + 1, self._make_system(ypos, ypos + height)
            )
            self.system_index += 1
        # The new system starts with side bars, so select its first one.
        self.bar_index = 0 if self.system.bar_count > 0 else -1

    def add_bar(self, offset: int = -1) -> None:
        bars = self.system.bars.copy()
        if offset < 0:
            # Adds a bar after the selected one.
            if self.bar_index < 0:
                offset = 10
                bars.append(offset)
            else:
                offset = self.system.bars[self.bar_index] + 10
                bars.insert(self.bar_index + 1, offset)
        else:
            bars.append(offset)
        bars = sorted(bars)
        self.replace_system(bars=bars)
        self.bar_index = bars.index(offset)
        self.check_bar_count()

    def delete_selected_bar(self) -> None:
        if self.bar_index < 0:
            return
        bars = self.system.bars.copy()
        del bars[self.bar_index]
        self.replace_system(bars=bars)
        self.bar_index = max(0, self.bar_index - 1)
        self.check_bar_count()

    def delete_selected_system(self) -> None:
        if self.system_index < 0:
            return
        del self.page.systems[self.system_index]
        if self.page.system_count == 0:
            self.system_index = -1
            self.staff_index = -1
            self.bar_index = -1
            return
        self.system_index = min(self.system_index, self.page.system_count - 1)
        if self.system.staff_count == 0:
            self.staff_index = -1
            self.bar_index = -1
        else:
            self.staff_index = 0
            self.bar_index = 0

    def check_bar_count(self) -> None:
        bar_count = self.get_bar_offset() - (self.kern.first_bar - 1)
        if self.kern.has_bar_zero():
            bar_count += 1
        if bar_count == self.kern.bar_count:
            self.beep()
            print(f"Yeay! {bar_count} is what we want, victory !")
        else:
            print(f"Yuck, {bar_count} bars, expecting {self.kern.bar_count}.")

    def fix_side_bars(self) -> None:
        # Finds the left and right bars.
        min_offset, max_offset = self.image.shape[1], 0
        for system in self.page.systems:
            for bar in system.bars:
                min_offset = min(min_offset, bar)
                max_offset = max(max_offset, bar)
        if min_offset == self.image.shape[1] or max_offset == 0:
            print("No left and right side defined,not doing anything.")
            return
        # Ensure that each system has them both within margin.
        margin = 25
        for idx, system in enumerate(self.page.systems):
            bars: list[int] | None = None
            if len(system.bars) > 0:
                if abs(system.bars[0] - min_offset) >= margin:
                    bars = system.bars.copy()
                    bars.insert(0, min_offset)
                if abs(system.bars[-1] - max_offset) >= margin:
                    if bars is None:
                        bars = system.bars.copy()
                    bars.append(max_offset)
            else:
                bars = [min_offset, max_offset]
            if bars is not None:
                print(f"Fixed staff {idx + 1}")
                self.page.systems[idx] = replace(system, bars=bars)

    def update_ui(self) -> None:
        image = self.draw_page(
            self.image,
            self.page,
            self.bar_offset,
            self.system_index,
            self.bar_index,
        )
        height, width = image.shape[:2]
        max_height, max_width = self.max_size
        self.scale_ratio = 1.0
        if max_height > 0 and height > max_height:
            self.scale_ratio = max_height / height
        if max_width > 0 and width > max_width:
            self.scale_ratio = min(self.scale_ratio, max_width / width)
        if self.scale_ratio != 1.0:
            new_size = (int(self.scale_ratio * width), int(self.scale_ratio * height))
            image = cv2.resize(image, new_size)
        cv2.imshow(self.STAFFER_WINDOW, image)

    def edit(
        self, fast_mode: bool = False, start_page_number: int | None = None
    ) -> bool:
        """Edits the staff.

        start_page_number: open on this 1-based page instead of the first (used by
            ``edit --review`` to land on the flagged page).

        Returns:
            bool: True if the user wishes to continue editing further documents,
                False otherwise.
        """

        self.clear()
        print(f"id : {self.id}")
        print(f"{self.kern.bar_count} bars in kern file.")
        self.fast_mode = fast_mode
        if start_page_number is not None:
            indices = [
                i
                for i, page in enumerate(self.score.pages)
                if page.page_number == start_page_number
            ]
            if indices:
                self.page_index = indices[0]
                self._reset_selection()
            else:
                print(f"Page {start_page_number} not found; starting at first page.")
        self.print_page_status()

        while True:
            self.update_bar_offset()
            self.update_ui()

            key = cv2.waitKey()

            if self.run_command(key):
                continue

            if key == ord("q"):  # Quits editing.
                return False
            elif key == ord("n"):  # Moves onto the next document if any.
                if self.fast_mode:
                    self.replace_page(status=Status.VALIDATED)
                    self.save()
                return True
            elif key == ord("1"):
                if self.confirm(f"Delete score {self.id}?"):
                    self.delete_score()
                    return True
            elif key == ord("2"):
                if self.confirm(f"Delete entry {self.key} and all its files?"):
                    self.delete_entry()
                    return True
            elif key in self.MODIFIER_KEYS:
                pass  # bare Shift/Ctrl/Alt press on its own; ignore silently
            else:
                print(f"Unknown key: '{key}', press '?' for help.")

    def renumber_bars(self) -> None:
        """Rewrite every system's ``bar_numbers`` as a sequential run derived from the
        barline geometry, so the stored numbering stays consistent with ``bars`` (what
        the human edits) and continuous across the score. The editor edits ``bars`` but
        never ``bar_numbers``, so without this the two desync and the noter/scorer
        datasets slice the kern by a stale ``first_bar_number`` (see the ``bar_drift``
        review). Anchored on ``get_bar_offset(0)`` — the same base the on-screen numbers
        use — so the saved data matches the display."""
        bar_number = self.get_bar_offset(0)
        for page_index, page in enumerate(self.score.pages):
            systems = []
            for system in page.systems:
                count = system.bar_count
                if system.staff_count > 0:
                    # A system with bars gets its sequential run; one with none
                    # (count == 0) is cleared so a stale number can't linger and the
                    # barless `bar_numbers` review flags it (consistent with the
                    # `bar_drift` review, which skips empty bar_numbers).
                    system = replace(
                        system,
                        bar_numbers=list(range(bar_number, bar_number + count)),
                    )
                    bar_number += count
                systems.append(system)
            self.score.pages[page_index] = replace(page, systems=systems)

    def save(self) -> None:
        self.renumber_bars()
        self.kern_sheet.save_score(self.id, self.score)
        print(f"{self.score.page_count} pages reviewed and saved.")

    def confirm(self, prompt: str) -> bool:
        print(f"{prompt} (y/n)")
        return cv2.waitKey() == ord("y")

    def delete_score(self) -> None:
        self.kern_sheet.delete_score(self.key, self.kern_sheet.get_score(self.id))
        print(f"Deleted score {self.id} from the catalog.")

    def delete_entry(self) -> None:
        self.kern_sheet.delete_entry(self.key)
        print(f"Deleted entry {self.key} and its files from the catalog.")

    KEY_NAMES = {
        81: "Left",
        82: "Up",
        83: "Right",
        84: "Down",
        85: "PageUp",
        86: "PageDn",
    }

    def help(self) -> None:
        def key_name(key_code: int) -> str:
            if name := self.KEY_NAMES.get(key_code, None):
                return name
            elif 32 <= key_code < 127:
                return f"'{chr(key_code)}'"
            else:
                return "???"

        # These following commands don't have actions:
        # they all quit the currrent editor, which actions can't do.
        print(f"{key_name(ord('q')):<8}Quits the editor.")
        print(f"{key_name(ord('n')):<8}Moves to next score.")
        print(f"{key_name(ord('1')):<8}Deletes this score from the catalog.")
        print(
            f"{key_name(ord('2')):<8}Deletes this entry and its files from the catalog."
        )
        for key_code, action in self.actions.items():
            print(f"{key_name(key_code):<8}{action.help}")

    def clear(self) -> None:
        # Clears the terminal and displays the kern tokens:
        print("\033[2J", end="")
        print("\033[H", end="")

    def beep(self) -> None:
        print("\a", end="", flush=True)

    def print_kerns(self) -> None:
        self.clear()
        bar_number = self.bar_number
        records = self.kern.get_text(bar_number)
        if records is None:
            print(f"No records found for bar {bar_number}")
            return
        print(f"Bar {bar_number} / {self.kern.bar_count + self.kern.first_bar - 1}:")
        cells = [record.split("\t") for record in records]
        if cells and max(len(c) for c in cells) == 2:
            # Grand staff: align the two spines into labelled columns.
            left = [c[0] for c in cells]
            right = [c[1] if len(c) > 1 else "" for c in cells]
            print(
                format_sequence_columns(
                    left,
                    right,
                    left_header="spine 0",
                    right_header="spine 1",
                    highlight_mismatches=False,
                )
            )
        else:
            for record in records:
                print(record)

    def title(self) -> None:
        self.clear()
        print(f"id : {self.id}")
        score = self.kern_sheet.get_score(self.id)
        print(f"image  : {self.kern_sheet.png_path(self.id, self.page.page_number)}")
        print(f"layout : {self.kern_sheet.layout_path(score)}")
        print(f"pdf    : {self.kern_sheet.pdf_path(score)}")
        print("Fast move on!" if self.fast_mode else "Fast mode off.")
        print(f"Page {self.page.page_number} status: {self.page.status.value}")
        findings = self.page_findings()
        if findings:
            print(f"{len(findings)} review finding(s) on this page:")
            for finding in findings:
                print(f"  ! {finding.review}: {finding.message}")
        print(f"Header for tokens - {self.kern.bar_count} bars")
        for line in self.kern.header():
            print(line)

    def recompute_bars(self) -> None:
        if self.system_index < 0:
            return
        bars = ClassicalStaffer().detect_bars(
            self.image, self.system.top, self.system.bottom
        )
        self.replace_system(bars=bars)
        self.bar_index = min(self.bar_index, len(bars) - 1) if bars else -1
        print(f"Recomputed {len(bars)} barline(s) for system {self.system_index + 1}.")

    def toggle_fast_mode(self) -> None:
        self.fast_mode = not self.fast_mode
        print("Fast move on!" if self.fast_mode else "Fast mode off.")

    def register_actions(self, *actions: Action) -> None:
        for a in actions:
            assert self.actions.get(a.key_code) is None, (
                f"Action for {a.key_code} already defined."
            )
            self.actions[a.key_code] = a

    def blank_pages(self, page_range: range) -> None:
        for i in page_range:
            self.score.pages[i] = replace(
                self.score.pages[i], systems=[], status=Status.VALIDATED
            )
        print(f"Blanked pages {page_range.start} to {page_range.stop}.")

    def edit_kern_and_tokens(self) -> None:
        subprocess.run(
            [
                "code",
                self.kern_sheet.kern_path(self.key).as_posix(),
                self.kern_sheet.tokens_path(self.key).as_posix(),
            ]
        )

    def init_commands(self) -> None:
        self.actions = {}
        self.register_actions(
            Action("s", self.save, "Saves works to disk."),
            Action("a", self.add_bar, "Adds a bar after the one currently selected."),
            Action(
                "w",
                self.add_system,
                "adds a pair of staves after the one currently selected.",
            ),
            Action(
                "x",
                self.delete_selected_system,
                "Deletes the selected system (whole grand staff, bars or not).",
            ),
            Action(
                "d", self.delete_selected_bar, "Deletes the bar currently selected."
            ),
            Action(" ", self.next, "Moves to next page."),
            Action("p", self.prev, "Moves to previous page."),
            Action("j", lambda: self.move_bar(-2), "Moves the selected bar left."),
            Action("l", lambda: self.move_bar(2), "Moves the selected bar right."),
            Action(
                "e",
                lambda: self.resize_system(2),
                "Extends the selected system down.",
            ),
            Action(
                "r",
                lambda: self.resize_system(-2),
                "Shrinks the selected system up.",
            ),
            Action(
                "g",
                lambda: self.resize_staves(2),
                "Makes both staves taller (common height).",
            ),
            Action(
                "h",
                lambda: self.resize_staves(-2),
                "Makes both staves shorter (common height).",
            ),
            Action(
                "u",
                self.apply_system_ratio,
                "Applies the selected system's stave height and gap to all systems "
                "on the page.",
            ),
            Action(
                "i",
                lambda: self.move_system(2),
                "Moves the selected system up.",
            ),
            Action(
                "m",
                lambda: self.move_system(-2),
                "Moves the selected system down.",
            ),
            Action(
                "c",
                self.recompute_bars,
                "Recomputes the bars within the selected staves.",
            ),
            Action(
                "v",
                self.cycle_status,
                "Cycles page status: pending -> validated -> rejected.",
            ),
            Action(
                "o",
                self.acknowledge_reviews,
                "Marks this page's review findings as ok (acknowledged).",
            ),
            Action(81, self.select_prev_bar, "Moves to and selects previous bar."),
            Action(83, self.select_next_bar, "Moves to and selects next bar."),
            Action(
                84, self.select_next_system, "Moves to and selects next pair of staves."
            ),
            Action(
                82,
                self.select_prev_system,
                "Moves to and selects previous pairs of staves.",
            ),
            Action("?", self.help, "Displays this help text."),
            Action(
                "k", self.print_kerns, "Prints the kern tokens for the selected bar."
            ),
            Action("t", self.title, "Prints misc. infos about the score being edited."),
            Action(
                "f",
                self.toggle_fast_mode,
                "Toggles fast mode: automatic save and valid on page jumps.",
            ),
            Action("/", self.check_bar_count, "Checks bar count"),
            Action(
                "z",
                self.fix_side_bars,
                "Ensures this staff has both left and right sides.",
            ),
            Action(
                85,
                lambda: self.blank_pages(range(0, self.page_index)),
                "Blanks and validates all previous pages (current excluded).",
            ),
            Action(
                86,
                lambda: self.blank_pages(
                    range(self.page_index + 1, self.score.page_count)
                ),
                "Blanks and validates all following pages (current excluded).",
            ),
            Action("!", self.edit_kern_and_tokens, "Edit kern and token files."),
        )

    def run_command(self, key_code: int) -> bool:
        action = self.actions.get(key_code, None)
        if action:
            try:
                action.func()
            except Exception as e:
                logging.exception(f"Command {chr(key_code)} failed:\n{e}")
            return True
        return False
