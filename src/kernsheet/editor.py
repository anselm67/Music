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
import numpy as np
from cv2.typing import MatLike

from kern import KernReader

from .editor_backend import EditorBackend, EditorPage, EditorStaff


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

    backend: EditorBackend

    # Data being edited.
    data: list[tuple[MatLike, EditorPage]]
    kern: KernReader

    # Editor's config
    max_size: tuple[int, int]
    actions: dict[int, Action]
    fast_mode: bool

    # Current state of the editor.
    position: int = 0
    staff_position: int = 0
    bar_position: int = 0
    bar_offset: int = 0
    scale_ratio: float = 1.0

    @property
    def image(self) -> MatLike:
        assert 0 <= self.position < len(self.data)
        return self.data[self.position][0]

    @property
    def page(self) -> EditorPage:
        assert 0 <= self.position < len(self.data)
        return self.data[self.position][1]

    def replace_page(self, **kwargs: Any) -> None:
        self.data[self.position] = (self.image, replace(self.page, **kwargs))

    @property
    def staff(self) -> EditorStaff:
        assert 0 <= self.staff_position < len(self.page.staves)
        return self.page.staves[self.staff_position]

    def replace_staff(self, **kwargs: Any) -> None:
        self.page.staves[self.staff_position] = replace(self.staff, **kwargs)

    def move_bar(self, delta: int) -> None:
        bars = self.staff.bars.copy()
        assert 0 <= self.bar_position < len(bars)
        bars[self.bar_position] += delta
        self.replace_staff(bars=bars)

    @property
    def bar_number(self) -> int:
        bar_number = self.bar_offset
        for i in range(0, self.staff_position):
            bar_number += max(len(self.page.staves[i].bars) - 1, 0)
        return bar_number + self.bar_position

    def __init__(self, backend: EditorBackend, max_size: tuple[int, int] = (992, 780)):
        self.backend = backend
        self.data = list(backend.staff())
        self.kern = KernReader(self.backend.tokens_path)
        self.max_size = max_size
        self.position = 0
        self.staff_position = 0
        self.bar_position = 0
        self.fast_mode = False
        if len(self.page.staves) <= 0:
            self.staff_position = -1
            self.bar_position = -1
        elif len(self.page.staves[0].bars) <= 0:
            self.bar_position = -1
        self.update_bar_offset()
        cv2.namedWindow(self.STAFFER_WINDOW)
        self.init_commands()

    def draw_page(
        self,
        image: MatLike,
        page: EditorPage,
        bar_offset: int,
        selected_staff: int = -1,
        selected_bar: int = -1,
        thickness: int = 2,
    ) -> MatLike:
        BLUE = (255, 0, 0)
        RED = (0, 0, 255)
        GREEN = (2, 7 * 16 + 1, 4 * 16 + 8)
        style, selected_style = (BLUE, 2), (RED, 4)
        if page.validated:
            style = (GREEN, 2)
        rgb_image = image.copy()
        if len(page.staves) == 0:
            # That's fine let the user know.
            cv2.putText(
                rgb_image,
                "Page Validated" if page.validated else "Validate that page.",
                (400, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=style[0],
                thickness=style[1],
            )

        for staffno, staff in enumerate(page.staves):
            if len(staff.bars) == 0:
                width = rgb_image.shape[1]
                color, thickness = (
                    selected_style if (staffno == selected_staff) else style
                )
                cv2.line(
                    rgb_image,
                    (0, staff.rh_top),
                    (width, staff.rh_top),
                    color,
                    thickness,
                )
                # Left hand staff
                cv2.line(
                    rgb_image,
                    (0, staff.lh_bot),
                    (width, staff.lh_bot),
                    color,
                    thickness,
                )
                continue
            # Draws the bars.
            for barno, bar in enumerate(staff.bars):
                color, thickness = (
                    selected_style
                    if (staffno == selected_staff) and (barno == selected_bar)
                    else style
                )
                cv2.line(
                    rgb_image,
                    (bar, staff.rh_top),
                    (bar, staff.lh_bot),
                    color,
                    thickness,
                )
                # Renders the bar number only if not last of staff.
                if barno != len(staff.bars) - 1:
                    cv2.putText(
                        rgb_image,
                        str(bar_offset + barno),
                        (bar + 3, staff.rh_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=color,
                        thickness=thickness,
                    )
            bar_offset += len(staff.bars) - 1
            # Draws the top most and bottom most staff lines:
            color, thickness = selected_style if (staffno == selected_staff) else style
            cv2.line(
                rgb_image,
                (staff.bars[0], staff.rh_top),
                (staff.bars[-1], staff.rh_top),
                color,
                thickness,
            )
            # Left hand staff
            cv2.line(
                rgb_image,
                (staff.bars[0], staff.lh_bot),
                (staff.bars[-1], staff.lh_bot),
                color,
                thickness,
            )
        return rgb_image

    def get_bar_offset(self, position: int = -1) -> int:
        if position < 0:
            position = len(self.data)
        bar_number = 0 if self.kern.has_bar_zero() else 1
        for _, page in self.data[:position]:
            for staff in page.staves:
                bar_number += max(len(staff.bars) - 1, 0)
        return bar_number + (self.kern.first_bar - 1)

    def update_bar_offset(self) -> None:
        self.bar_offset = self.get_bar_offset(self.position)

    def get(self) -> Tuple[MatLike, "EditorPage"]:
        return self.image, self.page

    def next(self) -> None:
        if self.position + 1 >= len(self.data):
            print("End of score.")
            return
        if self.fast_mode:
            self.replace_page(validated=True)
            self.save()
        self.position += 1
        self.staff_position = 0
        if len(self.page.staves) == 0:
            self.staff_position = -1
        elif len(self.page.staves[0].bars) == 0:
            self.bar_position = -1
        else:
            self.bar_position = 0

    def prev(self, select_last: bool = False) -> None:
        if self.position - 1 < 0:
            print("Beginning of score.")
            return
        self.position -= 1
        staff_count = len(self.page.staves)
        if staff_count == 0:
            self.staff_position = -1
            bars_len = 0
        elif select_last:
            self.staff_position = staff_count - 1
            bars_len = len(self.page.staves[self.staff_position].bars)
        else:
            self.staff_position = 0
            bars_len = len(self.page.staves[self.staff_position].bars)
        if self.staff_position < 0 or bars_len == 0:
            self.bar_position = -1
        else:
            self.bar_position = bars_len - 1 if select_last else 0

    def select_prev_staff(self, select_last: bool = False) -> None:
        if self.staff_position - 1 < 0:
            self.prev(select_last=True)
        else:
            self.staff_position = self.staff_position - 1
            self.bar_position = 0
            bar_count = len(self.page.staves[self.staff_position].bars)
            if bar_count <= 0:
                self.bar_position = -1
            elif select_last:
                self.bar_position = bar_count - 1

    def select_next_staff(self) -> None:
        if self.staff_position + 1 >= len(self.page.staves):
            self.next()
        else:
            self.staff_position = self.staff_position + 1
            self.bar_position = 0
            if len(self.page.staves[self.staff_position].bars) <= 0:
                self.bar_position = -1

    def select_next_bar(self) -> None:
        bar_count = 0
        if self.staff_position >= 0:
            bar_count = len(self.page.staves[self.staff_position].bars)
        if self.bar_position + 1 >= bar_count:
            self.select_next_staff()
        else:
            self.bar_position += 1

    def select_prev_bar(self) -> None:
        if self.bar_position - 1 < 0:
            self.select_prev_staff(select_last=True)
        else:
            self.bar_position -= 1

    def staff_height(self) -> int:
        # Tries to make a good guess at staff height.
        heights = [staff.lh_bot - staff.rh_top for staff in self.page.staves]
        if len(heights) > 0:
            return int(sum(heights) / len(heights))
        else:
            return 128

    def add_staff(self) -> None:
        height = self.staff_height()
        if self.staff_position < 0:
            ypos = 100
            self.page.staves.append(
                EditorStaff(rh_top=ypos, lh_bot=ypos + height, bars=list())
            )
            self.staff_position = len(self.page.staves) - 1
        else:
            ypos = self.page.staves[self.staff_position].lh_bot + 10
            self.page.staves.insert(
                self.staff_position + 1,
                EditorStaff(rh_top=ypos, lh_bot=ypos + height, bars=list()),
            )
            self.staff_position += 1

    def add_bar(self, offset: int = -1) -> None:
        bars = self.staff.bars.copy()
        if offset < 0:
            # Adds a bar after the selected one.
            if self.bar_position < 0:
                offset = 10
                bars.append(offset)
            else:
                offset = self.staff.bars[self.bar_position] + 10
                bars.insert(self.bar_position + 1, offset)
        else:
            bars.append(offset)
        bars = sorted(bars)
        self.replace_staff(bars=bars)
        self.bar_position = bars.index(offset)
        self.check_bar_count()

    def delete_selected_bar(self) -> None:
        if self.bar_position < 0:
            return
        del self.page.staves[self.staff_position].bars[self.bar_position]
        self.bar_position = max(0, self.bar_position - 1)
        self.check_bar_count()

    def delete_selected_staff(self) -> None:
        if self.staff_position < 0:
            return
        del self.page.staves[self.staff_position]
        if len(self.page.staves) == 0:
            self.staff_position = -1
            self.bar_position = -1
        else:
            self.staff_position = max(0, self.staff_position - 1)
            self.bar_position = 0

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
        for staff in self.page.staves:
            for bar in staff.bars:
                min_offset = min(min_offset, bar)
                max_offset = max(max_offset, bar)
        if min_offset == self.image.shape[1] or max_offset == 0:
            print("No left and right side defined,not doing anything.")
            return
        # Ensure that each staff has them both within margin.
        margin = 25
        for idx, staff in enumerate(self.page.staves):
            bars = None
            if len(staff.bars) > 0:
                if abs(staff.bars[0] - min_offset) >= margin:
                    if bars is None:
                        bars = staff.bars.copy()
                    bars.insert(0, min_offset)
                if abs(staff.bars[-1] - max_offset) >= margin:
                    if bars is None:
                        bars = staff.bars.copy()
                    bars.append(max_offset)
            else:
                bars = [min_offset, max_offset]
            if bars is not None:
                print(f"Fixed staff {idx + 1}")
                self.page.staves[idx] = replace(staff, bars=bars)

    def update_ui(self) -> None:
        image = self.draw_page(
            self.image,
            self.page,
            self.bar_offset,
            self.staff_position,
            self.bar_position,
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

    def edit(self, fast_mode: bool = False) -> bool:
        """Edits the staff.

        Returns:
            bool: True if the user wishes to continue editing further documents,
                False otherwise.
        """

        self.clear()
        print(f"{self.kern.bar_count} bars in {self.backend.kern_path}")
        print(f"pdf : {self.backend.score.pdf_path}")
        print(f"json: {self.backend.score.json_path}")
        self.fast_mode = fast_mode

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
                    self.replace_page(validated=True)
                    self.save()
                return True
            elif key == ord("1"):
                self.backend.delete_score()
                return True
            else:
                print(f"Unknown key: '{key}', press 'h' for help.")

    def save(self) -> None:
        self.backend.save(tuple(page for _, page in self.data))
        print(f"{len(self.data)} pages reviewed and saved.")

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

        # These following three commands don't have actions:
        # they all quit the currrent editor, which actions can't do.
        print(f"{key_name(ord('q')):<8}Quits the editor.")
        print(f"{key_name(ord('n')):<8}Moves to next score.")
        print(f"{key_name(ord('1')):<8}Deletes this score from the catalog.")
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
        else:
            print(
                f"Bar {bar_number} / {self.kern.bar_count + self.kern.first_bar - 1}:"
            )
            for record in records:
                print(record)

    def title(self) -> None:
        self.clear()
        print("Fast move on!" if self.fast_mode else "Fast mode off.")
        print(f"Header for {self.backend.kern_path} - {self.kern.bar_count} bars")
        for line in self.kern.header():
            print(line)

    def recompute_bars(self) -> None:
        if self.staff_position >= 0:
            staff = self.page.staves[self.staff_position]
            image = self.image[staff.rh_top : staff.lh_bot, :].copy()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            image = cv2.bitwise_not(image)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            self.replace_staff(bars=self.backend.find_bars(image))

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
            image, page = self.data[i]
            self.data[i] = (image, replace(page, validated=True, staves=[]))
        print(f"Blanked pages {page_range.start} to {page_range.stop}.")

    def edit_kern_and_tokens(self) -> None:
        subprocess.run(
            [
                "code",
                self.backend.kern_path.as_posix(),
                self.backend.tokens_path.as_posix(),
            ]
        )

    def init_commands(self) -> None:
        self.actions = {}
        self.register_actions(
            Action("s", self.save, "Saves works to disk."),
            Action("a", self.add_bar, "Adds a bar after the one currently selected."),
            Action(
                "w",
                self.add_staff,
                "adds a pair of staves after the one currently selected.",
            ),
            Action(
                "x", self.delete_selected_staff, "Deletes the staff currently selected."
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
                lambda: self.replace_staff(lh_bot=self.staff.lh_bot + 1),
                "Extends the selected staff down.",
            ),
            Action(
                "r",
                lambda: self.replace_staff(lh_bot=self.staff.lh_bot - 1),
                "Shrinks the selected staff up.",
            ),
            Action(
                "i",
                lambda: self.replace_staff(
                    lh_bot=self.staff.lh_bot - 2,
                    rh_top=self.staff.rh_top - 2,
                ),
                "Moves the selected staff up.",
            ),
            Action(
                "m",
                lambda: self.replace_staff(
                    lh_bot=self.staff.lh_bot + 2,
                    rh_top=self.staff.rh_top + 2,
                ),
                "Moves the selected staff down.",
            ),
            Action(
                "c",
                self.recompute_bars,
                "Recomputes the bars within the selected staves.",
            ),
            Action(
                "v",
                lambda: self.replace_page(validated=not self.page.validated),
                "Toggles the current page validation flag on or off.",
            ),
            Action(81, self.select_prev_bar, "Moves to and selects previous bar."),
            Action(83, self.select_next_bar, "Moves to and selects next bar."),
            Action(
                84, self.select_next_staff, "Moves to and selects next pair of staves."
            ),
            Action(
                82,
                self.select_prev_staff,
                "Moves to and selects previous pairs of staves.",
            ),
            Action("h", self.help, "Displays this help text."),
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
                lambda: self.blank_pages(range(0, self.position)),
                "Blanks and validates all previous pages (current excluded).",
            ),
            Action(
                86,
                lambda: self.blank_pages(range(self.position + 1, len(self.data))),
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
