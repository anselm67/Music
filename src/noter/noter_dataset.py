import logging
import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
from tqdm import tqdm

from sheetmusic import Box, Source

from .noter_model import NoterConfig
from .noter_vocab import Vocab

# Crop normalisation stats (from running: noter stats). Exposed so callers
# that display a transformed crop (e.g. `noter show`) can de-normalise.
NORM_MEAN = 0.9482423663139343
NORM_STD = 0.17525607175008864

# Per-edge box-jitter spec (sigma_px, clip_px), train-only augmentation modelling
# the staffer detector's box error measured on KernSheet (scripts/
# eval_predicted_boxes.py): horizontal error >> vertical, right edge the worst.
JITTER = {
    "top": (3.0, 15.0),
    "bot": (3.0, 15.0),
    "left": (5.0, 26.0),
    "right": (8.0, 50.0),
}


class NoterDataset(Dataset):
    def __init__(
        self, config: NoterConfig, source: Source, vocab: Vocab, count: int = -1
    ) -> None:
        self.source = source
        self.config = config
        self.vocab = vocab
        # Train-only box jitter; the datamodule enables it on the train view
        # only, leaving validation on clean (centered) crops.
        self.jitter = False
        self.jitter_prob = config.jitter_prob
        # Sets up image transforms.
        self.transform = v2.Compose(
            [
                v2.Grayscale(),
                v2.ToDtype(torch.float, scale=True),
                v2.Resize(
                    config.page_shape,
                    interpolation=config.interpolation,
                    antialias=config.antialias,
                ),
                v2.Normalize(mean=[NORM_MEAN], std=[NORM_STD]),
            ]
        )
        self.image_pad_value = (1.0 - NORM_MEAN) / NORM_STD
        # Pre-computes start and end sequence tokens.
        self.s_sos = torch.full((1, config.max_chords), self.vocab.SOS)
        self.s_eos = torch.full((1, config.max_chords), self.vocab.EOS)
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        for score in tqdm(source.scores(), desc="Loading noter dataset"):
            score = score.resize(config.page_shape[1], config.page_shape[0])
            for page in score.pages:
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
        image = torch.full((1, height, width), self.image_pad_value)
        _, cropped_height, cropped_width = tensor.shape
        y0 = src_top - crop_top
        image[:, y0 : y0 + cropped_height, :cropped_width] = tensor
        return image, cropped_width

    def _load_sequence(
        self,
        score_id: str,
        spine_number: int,
        first_bar_number: int,
        last_bar_number: int,
    ) -> Tensor | None:
        try:
            records = self.source.records(score_id, first_bar_number, last_bar_number)
        except Exception as e:
            logging.error(f"{score_id}: {e}")
            return None
        tensor = torch.full(
            (self.config.max_seqlen - 1, self.config.max_chords), self.vocab.PAD
        )
        if records is None:
            logging.error(
                f"{score_id}: bars {first_bar_number}:{last_bar_number} not found."
            )
            return None
        elif len(records) + 2 > self.config.max_seqlen:
            logging.error(
                f"{score_id}: bars {first_bar_number}:{last_bar_number}, "
                f"sequence too long {len(records)} (max {self.config.max_seqlen - 2})"
            )
            return None
        for idx, text in enumerate(records):
            try:
                # Real KernSheet records occasionally have fewer spines than the
                # system's staff count (malformed/misaligned bar range); skip the
                # sample rather than letting the IndexError crash the worker.
                str_tok = text.split("\t")[spine_number]
                tensor[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=self.config.max_chords
                )
            except Exception as e:
                logging.error(f"{score_id}: {e}")
                return None
        tensor[len(records), :] = self.s_eos
        return torch.cat([self.s_sos, tensor])

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
            if self.jitter and random.random() < self.jitter_prob:
                box = self._jitter_box(box)
            if (result := self._load_image(score_id, page_number, box)) is None:
                idx = (idx + 1) % len(self)
            elif (
                sequence := self._load_sequence(
                    score_id, spine_number, first_bar, last_bar
                )
            ) is None:
                idx = (idx + 1) % len(self)
            else:
                image, actual_width = result
                return (image, torch.tensor(actual_width), sequence)
