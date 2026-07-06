"""Torch Dataset for the Scorer: full page + layout GT + per-stave token sequences.

Each sample is a page image paired with the staffer layout ground truth (system /
stave boxes + assignment) *and* the noter token sequence for every stave slot, in the
same top-to-bottom enumeration order as the stave boxes. Restricted to systems of at
most ``noter.max_staves`` staves (use ``System4.csv``); each staff is routed to its
token column by the ``*staffN`` staff map (positional fallback), so the spine ordering
and token coverage are well defined.
"""

import logging

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from sheetmusic import Source, letterbox_scale, page_transform

from kern import NUM_ARTICULATIONS
from noter import SequenceLoader, Vocab

from .scorer_model import ScorerConfig

# image, sys_boxes, staff_boxes, assigns, tokens, articulations
Sample = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


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
        # Same page normalisation as the staffer (the page is the staffer's input):
        # per-image. With the noter canvas now equal to the staffer canvas
        # (NoterConfig.page_shape == StafferConfig.image_shape), a crop sampled
        # from this page matches the standalone noter crop exactly, so no recolour
        # is needed in ScorerModel.crop.
        self.transform = page_transform(
            config.staffer.image_shape,
            config.staffer.interpolation,
            config.staffer.antialias,
        )
        self.load_sequence = SequenceLoader(
            source, vocab, config.noter.max_seqlen, config.noter.max_chords
        )
        logging.info("Initializing ScorerDataset...")
        self.items = []
        for score in tqdm(source.scores(), desc="Loading scorer dataset"):
            for page in source.pages(score.id):
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
        return self.resolve(idx)[1]

    def resolve(self, idx: int) -> tuple[int, Sample]:
        """Build a sample, skipping pages whose full GT can't be assembled.

        Returns the index actually used together with its sample: a page may be
        un-buildable (transform error, a system with more than ``max_staves`` staves,
        a spine/staff-count mismatch, a missing token sequence), in which case the
        next page is tried.
        Callers that report a page identity must use the returned index, not the
        requested one, or the printed path will drift from the sample shown.
        """
        for _ in range(len(self)):
            sample = self._build(idx)
            if sample is not None:
                return idx, sample
            idx = (idx + 1) % len(self)
        raise IndexError("ScorerDataset: no buildable samples")

    def _build(self, idx: int) -> Sample | None:
        c = self.config.staffer
        score_id, page_number = self.items[idx]
        try:
            image = self.transform(self.source.image(score_id, page_number))
        except Exception as e:
            logging.error(f"{score_id}: {e}")
            return None

        score = self.source.score(score_id)
        # One spine per staff is the contract: ``staff_map`` routes each staff
        # (top->bottom) to its token column via the ``*staffN`` row, with a positional
        # bass-first fallback; its length is the token-file spine count. A system whose
        # staff count differs can't be routed one-to-one, so the page is skipped.
        try:
            staff_map = self.source.staff_map(score_id)
        except Exception as e:
            logging.error(f"{score_id}: cannot read staff map ({e})")
            return None
        spine_count = len(staff_map)
        max_staves = self.config.noter.max_staves
        # items() enrols only source.pages() (validated, for KernSheet) by their
        # page_number; the lookup below relies on pages being dense 1-based.
        page = score.pages[page_number - 1]
        assert page.page_number == page_number
        W, H = page.image_width, page.image_height
        # Letterboxed-canvas normalisation (see StafferDataset): scale by the
        # aspect-preserving factor, then by the canvas dims (content top-left).
        target_h, target_w = c.image_shape
        scale = letterbox_scale(H, W, target_h, target_w)
        sx, sy = scale / target_w, scale / target_h

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
        arts = torch.zeros(
            (
                c.num_stave_queries,
                self.config.noter.max_seqlen,
                self.config.noter.max_chords,
                NUM_ARTICULATIONS,
            )
        )

        is_ok = True
        staff_idx = 0
        for sys_idx, system in enumerate(page.systems):
            if (
                sys_idx >= c.num_system_queries
                or system.staff_count > max_staves
                or spine_count != system.staff_count
                or not system.bar_numbers
            ):
                is_ok = False
                break
            sys_boxes[sys_idx] = torch.tensor(
                [
                    system.box.left * sx,
                    system.box.top * sy,
                    system.box.right * sx,
                    system.box.bottom * sy,
                ]
            )
            for i, staff in enumerate(system.staves):
                # System4.csv only bounds the first system's staff count, so a page
                # of many staves can still exceed num_stave_queries.
                if staff_idx >= c.num_stave_queries:
                    is_ok = False
                    break
                seq = self.load_sequence.load(
                    score_id,
                    staff_map[i],
                    system.first_bar_number,
                    system.last_bar_number,
                )
                if seq is None:
                    is_ok = False
                    break
                seq_tokens, seq_arts = seq  # (max_seqlen, mc), (max_seqlen, mc, A)
                staff_boxes[staff_idx] = torch.tensor(
                    [
                        system.box.left * sx,
                        staff.top * sy,
                        system.box.right * sx,
                        staff.bottom * sy,
                    ]
                )
                assigns[staff_idx] = sys_idx
                tokens[staff_idx] = seq_tokens
                arts[staff_idx] = seq_arts
                staff_idx += 1
            if not is_ok:
                break

        if is_ok and staff_idx > 0:
            return image, sys_boxes, staff_boxes, assigns, tokens, arts
        return None
