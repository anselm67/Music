"""Torch Dataset for the Scorer: full page + layout GT + per-stave token sequences.

Each sample is a page image paired with the staffer layout ground truth (system /
stave boxes + assignment) *and* the noter token sequence for every stave slot, in the
same top-to-bottom enumeration order as the stave boxes. Restricted to ≤2-staff
systems (use ``System2.csv``) so the spine ordering and token coverage are well defined.
"""

import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from tqdm import tqdm

from kern.kern_reader import KernReader
from pdmx import PDMX
from sheetmusic import Score

from noter import Vocab

from .scorer_model import ScorerConfig

Sample = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class ScorerDataset(Dataset[Sample]):
    pdmx: PDMX
    items: list[tuple[Path, Path, int]]  # mxl_file, png_file, page_number
    transform: v2.Transform

    def __init__(self, config: ScorerConfig, pdmx: PDMX, count: int = -1) -> None:
        self.config = config
        self.pdmx = pdmx
        self.vocab = Vocab.load(pdmx.home / "build/vocab.json")
        # Same page normalisation as the staffer (the page is the staffer's input);
        # the noter branch recolours crops to its own space inside ScorerModel.crop.
        self.transform = v2.Compose(
            [
                v2.Grayscale(),
                v2.Resize(
                    config.staffer.image_shape,
                    interpolation=config.staffer.interpolation,
                    antialias=config.staffer.antialias,
                ),
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=[0.9563435316085815], std=[0.16557540870879858]),
            ]
        )
        self.s_sos = torch.full((1, config.noter.max_chords), self.vocab.SOS)
        self.s_eos = torch.full((1, config.noter.max_chords), self.vocab.EOS)

        logging.info("Initializing ScorerDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading scorer dataset"
        ):
            mxl_file = pdmx.home / row["mxl"]
            layout_file = pdmx.get_path(mxl_file, "layout")
            score = Score.from_json(json.loads(layout_file.read_text()))
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                self.items.append((mxl_file, png_file, page.page_number))
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tScorerDataset: {len(self.items):,} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def _load_sequence(
        self, mxl_file: Path, spine_number: int, first_bar: int, last_bar: int
    ) -> Tensor | None:
        """Token sequence for one stave (SOS … EOS), shape (max_seqlen, max_chords)."""
        kern_path = self.pdmx.get_path(mxl_file, "tokens")
        try:
            reader = KernReader(kern_path)
        except Exception as e:
            logging.error(f"{kern_path}: {e}")
            return None
        records = reader.get_text(first_bar, last_bar)
        if records is None:
            logging.error(f"{mxl_file}: bars {first_bar}:{last_bar} not found.")
            return None
        if len(records) + 2 > self.config.noter.max_seqlen:
            logging.error(
                f"{mxl_file}: bars {first_bar}:{last_bar}, sequence too long "
                f"{len(records)} (max {self.config.noter.max_seqlen - 2})"
            )
            return None
        body = torch.full(
            (self.config.noter.max_seqlen - 1, self.config.noter.max_chords),
            self.vocab.PAD,
        )
        for idx, text in enumerate(records):
            str_tok = text.split("\t")[spine_number]
            try:
                body[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=self.config.noter.max_chords
                )
            except Exception as e:
                logging.error(f"{mxl_file}: {e}")
                return None
        body[len(records), :] = self.s_eos
        return torch.cat([self.s_sos, body])

    def __getitem__(self, idx: int) -> Sample:
        c = self.config.staffer
        while True:
            mxl_file, png_file, page_number = self.items[idx]
            try:
                image = self.transform(decode_image(png_file.as_posix()))
            except Exception as e:
                logging.error(f"{mxl_file}: {e}")
                idx = (idx + 1) % len(self)
                continue

            score = Score.from_json(
                json.loads(self.pdmx.get_path(mxl_file, "layout").read_text())
            )
            page = score.pages[page_number - 1]
            W, H = page.image_width, page.image_height

            sys_boxes = torch.zeros(c.num_system_queries, 4)
            staff_boxes = torch.zeros(c.num_stave_queries, 4)
            assigns = torch.full((c.num_stave_queries,), -1, dtype=torch.long)
            tokens = torch.full(
                (
                    c.num_stave_queries,
                    self.config.noter.max_seqlen,
                    self.config.noter.max_chords,
                ),
                self.vocab.PAD,
            )

            is_ok = True
            staff_idx = 0
            for sys_idx, system in enumerate(page.systems):
                if sys_idx >= c.num_system_queries or system.staff_count not in (1, 2):
                    is_ok = False
                    break
                spine_numbers = [0] if system.staff_count == 1 else [1, 0]
                sys_boxes[sys_idx] = torch.tensor(
                    [
                        system.box.left / W,
                        system.box.top / H,
                        system.box.right / W,
                        system.box.bottom / H,
                    ]
                )
                for i, staff in enumerate(system.staves):
                    # System2.csv only bounds the first system's staff count, so a
                    # page of many 2-staff systems can exceed num_stave_queries.
                    if staff_idx >= c.num_stave_queries:
                        is_ok = False
                        break
                    seq = self._load_sequence(
                        mxl_file,
                        spine_numbers[i],
                        system.first_bar_number,
                        system.last_bar_number,
                    )
                    if seq is None:
                        is_ok = False
                        break
                    staff_boxes[staff_idx] = torch.tensor(
                        [
                            staff.box.left / W,
                            staff.box.top / H,
                            staff.box.right / W,
                            staff.box.bottom / H,
                        ]
                    )
                    assigns[staff_idx] = sys_idx
                    tokens[staff_idx] = seq
                    staff_idx += 1
                if not is_ok:
                    break

            if is_ok and staff_idx > 0:
                return image, sys_boxes, staff_boxes, assigns, tokens
            idx = (idx + 1) % len(self)
