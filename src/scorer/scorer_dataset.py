"""Torch Dataset for the Scorer: full page + layout GT + per-stave token sequences.

Each sample is a page image paired with the staffer layout ground truth (system /
stave boxes + assignment) *and* the noter token sequence for every stave slot, in the
same top-to-bottom enumeration order as the stave boxes. Restricted to ≤2-staff
systems (use ``System2.csv``) so the spine ordering and token coverage are well defined.
"""

import logging

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from sheetmusic import Source

from noter import Vocab, load_sequence

from .scorer_model import ScorerConfig

Sample = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class ScorerDataset(Dataset[Sample]):
    source: Source
    items: list[tuple[str, int]]  # score_id, page_number
    transform: v2.Transform

    def __init__(
        self, config: ScorerConfig, source: Source, vocab: Vocab, count: int = -1
    ) -> None:
        self.config = config
        self.source = source
        self.vocab = vocab
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
        self.s_sos = torch.full((1, config.noter.max_chords), vocab.SOS)
        logging.info("Initializing ScorerDataset...")
        self.items = []
        for score in tqdm(source.scores(), desc="Loading scorer dataset"):
            for page in score.pages:
                if not page.systems:  # skip blank/cover pages (no GT to supervise)
                    continue
                self.items.append((score.id, page.page_number))
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tScorerDataset: {len(self.items):,} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        c = self.config.staffer
        while True:
            score_id, page_number = self.items[idx]
            try:
                image = self.transform(self.source.image(score_id, page_number))
            except Exception as e:
                logging.error(f"{score_id}: {e}")
                idx = (idx + 1) % len(self)
                continue

            score = self.source.score(score_id)
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
                    seq = load_sequence(
                        self.source, self.vocab, score_id, spine_numbers[i],
                        system.first_bar_number, system.last_bar_number,
                        self.config.noter.max_seqlen, self.config.noter.max_chords,
                        self.s_sos,
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
