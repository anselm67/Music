import logging
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
from tqdm import tqdm

from sheetmusic import (
    Box,
    LetterboxResize,
    PerImageNormalize,
    ScanAugment,
    Source,
    letterbox_scale,
)

from .noter_model import NoterConfig
from .noter_vocab import Vocab

# Per-edge box-jitter spec (sigma_px, clip_px), train-only augmentation modelling
# the staffer detector's box error measured on KernSheet: horizontal error >>
# vertical, right edge the worst.
JITTER = {
    "top": (3.0, 15.0),
    "bot": (3.0, 15.0),
    "left": (5.0, 26.0),
    "right": (8.0, 50.0),
}


class SequenceLoader:
    """Loads per-stave token sequences, binding the dataset-wide constants once.

    Callers pass only the per-stave args; source/vocab/sizes (and the cached SOS
    row) are fixed for the loader's lifetime.
    """

    def __init__(
        self, source: Source, vocab: Vocab, max_seqlen: int, max_chords: int
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.max_seqlen = max_seqlen
        self.max_chords = max_chords
        self.s_sos = torch.full((1, max_chords), vocab.SOS)

    def __call__(
        self, score_id: str, spine_number: int, first_bar: int, last_bar: int
    ) -> Tensor | None:
        """One stave's sequence, shape (max_seqlen, max_chords).

        Returns SOS-prefixed, EOS-terminated tensor, or None if the bars are
        missing, the sequence is too long, or any record can't be decoded.
        """
        try:
            records = self.source.records(score_id, first_bar, last_bar)
        except Exception as e:
            logging.error(f"{score_id}: {e}")
            return None
        if records is None:
            logging.error(f"{score_id}: bars {first_bar}:{last_bar} not found.")
            return None
        if len(records) + 2 > self.max_seqlen:
            logging.error(
                f"{score_id}: bars {first_bar}:{last_bar}, "
                f"sequence too long {len(records)} (max {self.max_seqlen - 2})"
            )
            return None
        body = torch.full((self.max_seqlen - 1, self.max_chords), self.vocab.PAD)
        for idx, text in enumerate(records):
            try:
                # Real KernSheet records occasionally have fewer spines than the
                # system's staff count (malformed/misaligned bar range); skip the
                # sample rather than letting the IndexError crash the worker.
                str_tok = text.split("\t")[spine_number]
                body[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=self.max_chords
                )
            except Exception as e:
                logging.error(f"{score_id}: {e}")
                return None
        body[len(records), :] = self.vocab.EOS
        return torch.cat([self.s_sos, body])


class NoterDataset(Dataset):
    def __init__(
        self, config: NoterConfig, source: Source, vocab: Vocab, count: int = -1
    ) -> None:
        self.source = source
        self.config = config
        self.vocab = vocab
        # Train-only box jitter probability (0 = off); the datamodule enables it
        # on the train view only, leaving validation on clean (centered) crops.
        self.jitter = 0.0
        # Clean transform (no scan augment); the datamodule calls enable_augment
        # on the train view only, leaving validation on un-augmented pages.
        self.transform = self._build_transform(0.0)
        self.load_sequence = SequenceLoader(
            source, vocab, config.max_seqlen, config.max_chords
        )
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        target_h, target_w = config.page_shape
        for score in tqdm(source.scores(), desc="Loading noter dataset"):
            # One spine per staff is the contract. A token file with more spines than
            # a system has staves (a voiced staff, or a krn declaring more staves than
            # the scan shows) would have its extra spine(s) silently dropped by the
            # spine_numbers routing below — training the staff against a partial
            # transcription. Read the per-score count once and skip those systems.
            try:
                spine_count = source.spine_count(score.id)
            except Exception as e:
                logging.error(f"{score.id}: cannot read spine count ({e}), skipping")
                continue
            for page in source.pages(score.id):
                # Letterbox the boxes by the same single scale the image transform
                # uses, so crop coords land on the (aspect-preserved) staff.
                scale = letterbox_scale(
                    page.image_height, page.image_width, target_h, target_w
                )
                page = page.resize(
                    min(round(page.image_width * scale), target_w),
                    min(round(page.image_height * scale), target_h),
                )
                for system in page.systems:
                    match system.staff_count:
                        case 1:
                            spine_numbers = [0]
                        case 2:
                            spine_numbers = [1, 0]
                        case _:
                            logging.error(
                                f"{score.id}: too many staves in system "
                                f"({system.staff_count} vs {self.config.max_staves})"
                            )
                            continue
                    if spine_count > system.staff_count:
                        logging.error(
                            f"{score.id}: token file has {spine_count} spines but "
                            f"system has {system.staff_count} staves; skipping "
                            f"(extra spine(s) dropped — see spine_count review)"
                        )
                        continue
                    if not system.bar_numbers:
                        logging.error(
                            f"{score.id}: system with no bar numbers, skipping"
                        )
                        continue
                    # One item per system: its staves' boxes + spine numbers, kept
                    # aligned. The staves are decoded together (shared barline grid).
                    self.items.append(
                        (
                            score.id,
                            page.page_number,
                            [staff.box for staff in system.staves],
                            spine_numbers,
                            system.first_bar_number,
                            system.last_bar_number,
                        )
                    )
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tNoterDataset: {len(self.items):,} samples.")

    def _build_transform(self, augment: float) -> v2.Transform:
        """Image transform; ScanAugment (prob ``augment``, 0 = off) runs on the
        float [0,1] page before per-image normalisation."""
        return v2.Compose(
            [
                v2.Grayscale(),
                v2.ToDtype(torch.float, scale=True),
                LetterboxResize(
                    self.config.page_shape,
                    interpolation=self.config.interpolation,
                    antialias=self.config.antialias,
                    fill=1.0,
                ),
                ScanAugment(augment),
                PerImageNormalize(),
            ]
        )

    def enable_augment(self, prob: float) -> None:
        """Rebuild the transform with scan augmentation at probability ``prob``.

        Called by the datamodule on the (shallow-copied) train view only, so
        validation keeps the clean transform built in ``__init__``.
        """
        self.transform = self._build_transform(prob)

    def __len__(self) -> int:
        return len(self.items)

    def get_item_stats(self, idx: int) -> tuple[tuple[int, int], int]:
        """Per-system: (max staff height, max staff width) and record count.

        The staves of a system share one bar range, so the record count is the
        sequence length for every staff; the box dims are the largest staff's.
        """
        score_id, _, boxes, _, first_bar_number, last_bar_number = self.items[idx]
        records = self.source.records(score_id, first_bar_number, last_bar_number)
        max_h = max(box.height for box in boxes)
        max_w = max(box.width for box in boxes)
        return (max_h, max_w), len(records) if records else -1

    def _load_image(
        self, score_id: str, page_number: int, box: Box
    ) -> tuple[Tensor, int] | None:
        # Gets the image and crop it to the system box.
        try:
            tensor = self.source.image(score_id, page_number)
            _, image_height, image_width = tensor.shape
            if box.width > image_width or box.height > image_height:
                logging.error(
                    f"{score_id}: image too large (H x W) {box.height}x{box.width}"
                )
                return None
            tensor = self.transform(tensor)
        except Exception as e:
            logging.error(f"{score_id}: {e}")
            return None
        # Per-image norm makes "normalised white" page-dependent, so the crop
        # window must pad with this page's white rather than a fixed constant.
        # LetterboxResize fills raw 1.0 (the max raw value), so after the norm
        # the page max IS its normalised white.
        image_pad_value = float(tensor.max())
        height, width = self.config.input_shape
        _, page_height, page_width = tensor.shape
        # Center the staff vertically in the fixed-height window. A real staff
        # (~24-32px) is far shorter than `height` (64), so the old 3*box.height
        # window always exceeded it: cropping from box.top-box.height shoved the
        # staff into the lower half and clipped the tallest ones. Center on the
        # staff midline instead, padding white (image_pad_value) at page edges.
        crop_top = box.top + box.height // 2 - height // 2
        src_top = max(0, crop_top)
        src_bot = min(page_height, crop_top + height)
        # Clamp the crop to the page so overhang (jittered right edge past the
        # page) pads white via the canvas, not black (crop() zero-pads = ink).
        crop_width = min(box.width, page_width - box.left)
        tensor = crop(tensor, src_top, box.left, src_bot - src_top, crop_width)
        image = torch.full((1, height, width), image_pad_value)
        _, cropped_height, cropped_width = tensor.shape
        y0 = src_top - crop_top
        image[:, y0 : y0 + cropped_height, :cropped_width] = tensor
        return image, cropped_width

    def _jitter_box(self, box: Box) -> Box:
        """Perturb each edge independently by clipped Gaussian noise (px)."""

        def delta(edge: str) -> int:
            sigma, clip = JITTER[edge]
            return int(round(max(-clip, min(clip, random.gauss(0.0, sigma)))))

        left = max(0, box.left + delta("left"))
        right = max(left + 1, box.right + delta("right"))
        top = max(0, box.top + delta("top"))
        bottom = max(top + 1, box.bottom + delta("bot"))
        return Box(left, top, right, bottom)

    def _load_staff(
        self,
        score_id: str,
        page_number: int,
        box: Box,
        spine_number: int,
        first_bar: int,
        last_bar: int,
    ) -> tuple[Tensor, int, Tensor] | None:
        """One staff's (image, width, sequence), or None if either can't load."""
        if self.jitter and random.random() < self.jitter:
            box = self._jitter_box(box)
        if (result := self._load_image(score_id, page_number, box)) is None:
            return None
        if (
            seq := self.load_sequence(score_id, spine_number, first_bar, last_bar)
        ) is None:
            return None
        image, actual_width = result
        return image, actual_width, seq

    def _pad_staves(
        self, images: list[Tensor], widths: list[int], sequences: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Stack a system's G staves and pad to ``max_staves`` with masked slots.

        A padding slot is a zero image (full-width so no encoder patch is masked →
        no all-masked NaN) and an SOS-only sequence; the returned ``stave_mask``
        is True for the G real staves, False for the pad, so the model excludes
        the pad from loss and cross-stave attention.
        """
        c = self.config
        g, s = len(images), c.max_staves
        h, w = c.input_shape
        pad_img = torch.zeros(s - g, 1, h, w)
        pad_w = torch.full((s - g,), w)
        pad_seq = torch.full((s - g, c.max_seqlen, c.max_chords), self.vocab.PAD)
        pad_seq[:, 0, :] = self.vocab.SOS
        out_img = torch.cat([torch.stack(images), pad_img], dim=0)
        out_w = torch.cat([torch.tensor(widths), pad_w], dim=0)
        out_seq = torch.cat([torch.stack(sequences), pad_seq], dim=0)
        mask = torch.arange(s) < g
        return out_img, out_w, out_seq, mask

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """A system: ``(images, widths, sequences, stave_mask)`` padded to max_staves.

        Shapes ``(S, 1, 64, 768)``, ``(S,)``, ``(S, max_seqlen, max_chords)``, ``(S,)``
        with ``S = max_staves``. If any staff of the system fails to load the whole
        system is skipped (advance to the next item).
        """
        while True:
            score_id, page_number, boxes, spine_numbers, first_bar, last_bar = (
                self.items[idx]
            )
            logging.debug(f"Loading {score_id}")
            loaded = [
                self._load_staff(score_id, page_number, box, spine, first_bar, last_bar)
                for box, spine in zip(boxes, spine_numbers)
            ]
            if any(staff is None for staff in loaded):
                idx = (idx + 1) % len(self)
                continue
            images = [staff[0] for staff in loaded if staff is not None]
            widths = [staff[1] for staff in loaded if staff is not None]
            sequences = [staff[2] for staff in loaded if staff is not None]
            return self._pad_staves(images, widths, sequences)
