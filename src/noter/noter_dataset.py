import logging
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
from tqdm import tqdm

from sheetmusic import Box, LetterboxResize, PerImageNormalize, Source, letterbox_scale

from .noter_model import NoterConfig
from .noter_vocab import Vocab

# Per-edge box-jitter spec (sigma_px, clip_px), train-only augmentation modelling
# the staffer detector's box error measured on KernSheet (scripts/
# eval_predicted_boxes.py): horizontal error >> vertical, right edge the worst.
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
        # Sets up image transforms.
        self.transform = v2.Compose(
            [
                v2.Grayscale(),
                v2.ToDtype(torch.float, scale=True),
                LetterboxResize(
                    config.page_shape,
                    interpolation=config.interpolation,
                    antialias=config.antialias,
                    fill=1.0,
                ),
                PerImageNormalize(),
            ]
        )
        self.load_sequence = SequenceLoader(
            source, vocab, config.max_seqlen, config.max_chords
        )
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        target_h, target_w = config.page_shape
        for score in tqdm(source.scores(), desc="Loading noter dataset"):
            for page in score.pages:
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
                                f"({system.staff_count} vs 2)"
                            )
                            continue
                    for idx, staff in enumerate(system.staves):
                        self.items.append(
                            (
                                score.id,
                                page.page_number,
                                staff.box,
                                spine_numbers[idx],
                                system.first_bar_number,
                                system.last_bar_number,
                            )
                        )
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tNoterDataset: {len(self.items):,} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def get_item_stats(self, idx: int) -> tuple[tuple[int, int], int]:
        score_id, _, box, _, first_bar_number, last_bar_number = self.items[idx]
        records = self.source.records(score_id, first_bar_number, last_bar_number)
        return (box.height, box.width), len(records) if records else -1

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
        return Box((left, top), (right, bottom))

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        while True:
            score_id, page_number, box, spine_number, first_bar, last_bar = self.items[
                idx
            ]
            logging.debug(f"Loading {score_id}")
            if self.jitter and random.random() < self.jitter:
                box = self._jitter_box(box)
            if (result := self._load_image(score_id, page_number, box)) is None:
                idx = (idx + 1) % len(self)
            elif (
                sequence := self.load_sequence(
                    score_id, spine_number, first_bar, last_bar
                )
            ) is None:
                idx = (idx + 1) % len(self)
            else:
                image, actual_width = result
                return (image, torch.tensor(actual_width), sequence)
