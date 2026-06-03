"""Interactive cv2 staff-layout editor, ported from OMR's ``staff_editor.py``.

Reviews/corrects a score's grand-staff layout one page at a time: move/extend
staves, add/move/delete barlines, validate pages, check the bar count against the
kern. It edits a flat *envelope* view (``rh_top``/``lh_bot``/``bars`` per system)
supplied by :class:`EditorBackend`, which persists it back as native ``Score`` JSON.
"""

import logging
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Tuple, cast

import cv2
from cv2.typing import MatLike

from kern import KernReader
from kernsheet import KernSheetSource
from sheetmusic import Box, Page, Score, Source, Staff, System


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

    source: Source

    # Data being edited.
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

    @property
    def bar_number(self) -> int:
        bar_number = self.bar_offset
        for system in self.page.systems[: self.system_index]:
            bar_number += max(len(system.bars) - 1, 0)
        return bar_number + self.bar_index

    def _tokens_path(self) -> Path:
        source = cast(KernSheetSource, self.source)
        return source._tokens_path(self.id)

    def __init__(self, source: Source, id: str, max_size: tuple[int, int] = (992, 780)):
        self.source = source
        self._load_score(id)
        self.kern = KernReader(self._tokens_path())
        self.max_size = max_size
        self.page_index = 0
        self.system_index = 0
        self.staff_index = 0
        self.bar_index = 0
        self.bar_offset = 0
        self.fast_mode = False
        if self.page.system_count <= 0:
            self.system_index = -1
            self.staff_index = -1
            self.bar_index = -1
        elif self.system.staff_count <= 0:
            self.staff_index = -1
            self.bar_index = -1
        elif len(self.system.bars) <= 0:
            self.bar_index = -1
        self.update_bar_offset()
        cv2.namedWindow(self.STAFFER_WINDOW)
        self.init_commands()

    def _load_score(self, id: str) -> None:
        self.score = self.source.score(id)
        self.images = []
        for page in self.score.pages:
            image = self.source.image(id, page.page_number)
            self.images.append(
                cv2.cvtColor(image.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
            )
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
        style, selected_style = (BLUE, 2), (RED, 4)
        if page.validated:
            style = (GREEN, 2)
        rgb_image = image.copy()
        if page.system_count == 0:
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
            # Draws the top most and bottom most staff lines:
            color, thickness = (
                selected_style if (system_index == selected_system) else style
            )
            cv2.line(
                rgb_image,
                (bars[0], system.top),
                (bars[-1], system.top),
                color,
                thickness,
            )
            # Left hand staff
            cv2.line(
                rgb_image,
                (bars[0], system.bottom),
                (bars[-1], system.bottom),
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

    def next(self) -> None:
        if self.page_index + 1 >= self.score.page_count:
            print("End of score.")
            return
        if self.fast_mode:
            self.replace_page(validated=True)
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

    def prev(self, select_last: bool = False) -> None:
        if self.page_index - 1 < 0:
            print("Beginning of score.")
            return
        self.page_index -= 1
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

    def _make_system(self, top: int, bottom: int) -> System:
        # TODO Don't commit this!!!
        height = self.system_height() // 3
        left, right = 10, 500
        return System(
            bar_numbers=[],
            bars=[],
            staves=[
                Staff(Box((top, left), (top + height, right))),
                Staff(Box((bottom - height, left), (bottom, right))),
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
        elif self.system.staff_count == 0:
            self.staff_index = -1
            self.bar_index = -1
        else:
            self.system_index = max(0, self.system_index - 1)
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

    def edit(self, fast_mode: bool = False) -> bool:
        """Edits the staff.

        Returns:
            bool: True if the user wishes to continue editing further documents,
                False otherwise.
        """

        self.clear()
        print(f"id : {self.id}")
        print(f"{self.kern.bar_count} bars in kern file.")
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
                logging.warning(f"{self.id}: delete score not implemented.")
                return True
            else:
                print(f"Unknown key: '{key}', press 'h' for help.")

    def save(self) -> None:
        source = cast(KernSheetSource, self.source)
        source._save_layout(self.score)
        print(f"{self.score.page_count} pages reviewed and saved.")

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
        print(f"Header for tokens - {self.kern.bar_count} bars")
        for line in self.kern.header():
            print(line)

    def recompute_bars(self) -> None:
        logging.warning("recompute_bars: not implemented.")

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
                self.score.pages[i], systems=[], validated=True
            )
        print(f"Blanked pages {page_range.start} to {page_range.stop}.")

    def edit_kern_and_tokens(self) -> None:
        source = cast(KernSheetSource, self.source)
        subprocess.run(
            [
                "code",
                source._kern_path(self.score.id).as_posix(),
                self._tokens_path().as_posix(),
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
                "Deletes the staff currently selected.",
            ),
            Action(
                "d", self.delete_selected_bar, "Deletes the bar currently selected."
            ),
            Action(" ", self.next, "Moves to next page."),
            Action("p", self.prev, "Moves to previous page."),
            Action("j", lambda: self.move_bar(-2), "Moves the selected bar left."),
            Action("l", lambda: self.move_bar(2), "Moves the selected bar right."),
            Action(  # TODO extend bottom of last system's staff
                "e",
                lambda: self.replace_system(bottom=self.system.bottom + 1),
                "Extends the selected system down.",
            ),
            Action(  # TODO reduce bottom of last system staff
                "r",
                lambda: self.replace_system(bottom=self.system.bottom - 1),
                "Shrinks the selected staff up.",
            ),
            Action(  # TODO move all system staves down.
                "i",
                lambda: self.replace_system(
                    bottom=self.system.bottom - 2,
                    top=self.system.top - 2,
                ),
                "Moves the selected system up.",
            ),
            Action(  # TODO move all system staves up.
                "m",
                lambda: self.replace_system(
                    bottom=self.system.bottom + 2,
                    top=self.system.top + 2,
                ),
                "Moves the selected stsystem down.",
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
                84, self.select_next_system, "Moves to and selects next pair of staves."
            ),
            Action(
                82,
                self.select_prev_system,
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
