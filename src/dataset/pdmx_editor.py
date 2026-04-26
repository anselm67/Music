
import json
from pathlib import Path

import cv2
from cv2.typing import MatLike

from kern import KernReader

from .layout import Page, Score
from .pdmx import PDMX


class MxlEditor:
    pdmx: PDMX
    mxl_path: Path
    score: Score
    kern_reader: KernReader

    scale: float
    page_index: int
    hide_truth: bool = False

    def __init__(self, pdmx: PDMX, mxl_path: Path | None, scale: float = 0.8):
        self.pdmx = pdmx
        if mxl_path is None:
            mxl_path = pdmx.pick_mxl()
        self.load(mxl_path)
        self.scale = scale

    def load(self, mxl_path: Path) -> None:
        self.mxl_path = mxl_path
        with open(self.pdmx.get_path(mxl_path, 'layout'), 'r') as f:
            obj = json.load(f)
        self.score = Score.from_json(obj)
        self.kern_reader = KernReader(self.pdmx.get_path(mxl_path, 'tokens'))
        # Reset the edit state.
        self.page_index = 0

    def load_page(self, page_index: int = 0) -> tuple[Page, MatLike]:
        assert page_index >= 0 and page_index < len(
            self.score.pages), f"Page index {page_index} out of bounds."
        page = self.score.pages[self.page_index]
        # Loads the page image.
        if self.score.page_count > 1:
            img_path = self.pdmx.get_page_path(
                self.mxl_path, 'png', page.page_number)
        else:
            img_path = self.pdmx.get_path(self.mxl_path, 'png')
        img = cv2.imread(img_path)
        assert img is not None, f"Can't load image {img_path}"

        # Resizes it according to provided scale.
        # We could drawinto the un-resized image and then resize, but resizing
        # them separately allows to check that Score.resize() works fine.
        (height, width) = tuple(
            map(lambda x: int(x * self.scale), img.shape[:2]))
        img = cv2.resize(img, (width, height))
        page = page.resize(width, height)
        return page, img

    def on_click(self, event: int, page: Page, point: tuple[int, int]) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(f"on_click: {point}")
        # Converts the (x, y) into a system, staff and bar to highlight.
        for system_index, system in enumerate(page.systems):
            if system.box.contains(point):
                for staff_index, staff in enumerate(system.staves):
                    if staff.box.contains(point):
                        bar_number = system.bar_number
                        bar_count = 0
                        while bar_count+1 < len(staff.bars) and staff.bars[bar_count+1] < point[0]:
                            bar_count += 1
                        print(
                            f"   page number: {page.page_number}\n"
                            f"        system: {system_index}\n"
                            f"         staff: {staff_index}\n"
                            f"    bar number: {bar_number + bar_count}"
                            f"svg_bar_number: {system.svg_bar_number}"
                        )
                        tokens = self.kern_reader.get_text(
                            bar_number + bar_count)
                        if tokens:
                            for line in tokens:
                                print(line)
                        else:
                            print("*** No matching tokens.")
                        return
        print("Click on a bar to get its coordinates.")

    def run(self):
        page, image = self.load_page()

        while True:
            img = image.copy()
            # Renders the page layout on top of the image.
            if not self.hide_truth:
                system_color = (255, 0, 0)
                staff_color = (0, 255, 0)
                bar_color = (0, 0, 255)
                for system in page.systems:
                    cv2.rectangle(img, system.box.top_left,
                                  system.box.bot_right, system_color, 3)
                    for staff in system.staves:
                        cv2.rectangle(img, staff.box.top_left,
                                      staff.box.bot_right, staff_color, 2)
                        for bar in staff.bars:
                            cv2.line(img, (bar, staff.box.top),
                                     (bar, staff.box.bottom), bar_color, 1)
            cv2.imshow("layout", img)
            cv2.setMouseCallback(
                "layout",
                lambda event, x, y, flags, param: self.on_click(
                    event, page, (x, y))
            )

            if (key := cv2.waitKey()) == ord('q'):
                break
            elif key == ord('p'):
                if (page_index := page_index - 1) < 0:
                    page_index = len(self.score.pages) - 1
                    page, img = self.load_page(page_index)
            elif key == ord('n'):
                if (page_index := page_index + 1) >= self.score.page_count:
                    page_index = 0
                    page, img = self.load_page(page_index)
            elif key == ord('i'):
                if (infos := self.pdmx.info(self.mxl_path)) is None:
                    print(f"{self.mxl_path}: not found.")
                else:
                    for title, value in infos:
                        print(f"{title}\n\t\033[1;31m{value}\033[0m")
            elif key == ord('h'):
                self.hide_truth = not self.hide_truth
            else:
                print(
                    "(p)revious page,\n"
                    "(n)ext page,\n"
                    "(i)nfos about tghe score,\n"
                    "(h)ide/show ground truth boxes."
                )

    def close(self):
        cv2.destroyAllWindows()

# vscode - End of File

# vscode - End of File
