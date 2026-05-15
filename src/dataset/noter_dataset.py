import json
import logging
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
from tqdm import tqdm

from dataset.noter_vocab import Vocab
from kern.kern_reader import KernReader
from models import Config

# TODO This should eventually be NoterModel's Config
from .layout import Box, Score
from .pdmx import PDMX

IMAGE_WIDTH = 6 * 128
IMAGE_HEIGHT = 64
MAX_CHORDS = 8
SPAD_LEN = 128


class NoterDataset(Dataset):
    pdmx: PDMX
    config: Config
    items: list[tuple[Path, Path, Box, int, int]]
    transform: v2.Transform
    vocab: Vocab
    image_pad_value: float
    s_sos: Tensor
    s_eos: Tensor

    def __init__(self, config: Config, pdmx: PDMX, count=-1):
        self.pdmx = pdmx
        self.config = config
        self.vocab = Vocab.load(pdmx.home / "build/vocab.json")
        # Sets up image transforms.
        self.transform = v2.Compose(
            [
                v2.Grayscale(),
                v2.Resize(
                    config.image_shape,
                    interpolation=config.interpolation,
                    antialias=config.antialias,
                ),
                v2.ToDtype(torch.float, scale=True),
                # Values from running: staffer stats
                v2.Normalize(mean=[0.9482423663139343], std=[0.17525607175008864]),
            ]
        )
        self.image_pad_value = (1.0 - 0.9482423663139343) / 0.17525607175008864
        # Pre-computes start and end sequence tokens.
        self.s_sos = torch.full((1, MAX_CHORDS), self.vocab.SOS)
        self.s_eos = torch.full((1, MAX_CHORDS), self.vocab.EOS)
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading noter dataset"
        ):
            mxl_file = pdmx.home / row["mxl"]
            layout_file = pdmx.get_path(mxl_file, "layout")
            score = Score.from_json(json.loads(layout_file.read_text()))
            score = score.resize(config.image_shape[1], config.image_shape[0])
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                for system in page.systems:
                    self.items.append(
                        (
                            mxl_file,
                            png_file,
                            system.box,
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
        mxl_file, _, box, first_bar_number, last_bar_number = self.items[idx]
        kern_path = self.pdmx.get_path(mxl_file, "tokens")
        reader = KernReader(kern_path)
        records = reader.get_text(first_bar_number, last_bar_number)
        return (box.height, box.width), len(records) if records else -1

    def _load_image(self, mxl_file: Path, png_file: Path, box: Box) -> Tensor | None:
        # Checks image size.
        if box.width > IMAGE_WIDTH or box.height > IMAGE_HEIGHT:
            logging.error(
                f"{mxl_file}: image too large (H x W) {box.height}x{box.width}"
            )
            return None
        # Gets the image and crop it to the system box.
        try:
            tensor = decode_image(png_file.as_posix())
            tensor = self.transform(tensor)
        except Exception as e:
            logging.error(f"{png_file}: {e}")
            return None
        tensor = crop(
            tensor,
            max(0, box.top - box.height),
            box.left,
            min(IMAGE_HEIGHT, 3 * box.height),
            box.width,
        )
        image = torch.full((1, IMAGE_HEIGHT, IMAGE_WIDTH), self.image_pad_value)
        _, h, w = tensor.shape
        y0 = (IMAGE_HEIGHT - h) // 2
        x0 = (IMAGE_WIDTH - w) // 2
        image[:, y0 : y0 + h, x0 : x0 + w] = tensor
        return image

    def _load_sequence(
        self, mxl_file: Path, first_bar_number: int, last_bar_number: int
    ) -> Tensor | None:
        kern_path = self.pdmx.get_path(mxl_file, "tokens")
        try:
            reader = KernReader(kern_path)
        except Exception as e:
            logging.error(f"{kern_path}: {e}")
            return None
        spine_number: int = 0
        tensor = torch.full((SPAD_LEN - 1, MAX_CHORDS), self.vocab.PAD)
        records = reader.get_text(first_bar_number, last_bar_number)
        if records is None:
            logging.error(
                f"{mxl_file}: bars {first_bar_number}:{last_bar_number} not found."
            )
            return None
        elif len(records) + 2 > SPAD_LEN:
            logging.error(
                f"{mxl_file}: bars {first_bar_number}:{last_bar_number}, "
                f"sequence too long {len(records)} (max {SPAD_LEN - 2})"
            )
            return None
        for idx, text in enumerate(records):
            str_tok = text.split("\t")[spine_number]
            try:
                tensor[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=MAX_CHORDS
                )
            except Exception as e:
                logging.error(f"{mxl_file}: {e}")
                return None
        tensor[len(records), :] = self.s_eos
        return torch.cat([self.s_sos, tensor])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        while True:
            mxl_file, png_file, box, first_bar_number, last_bar_number = self.items[idx]
            logging.debug(f"Loading {mxl_file}")
            if (image := self._load_image(mxl_file, png_file, box)) is None:
                idx = (idx + 1) % len(self)
            elif (
                sequence := self._load_sequence(
                    mxl_file, first_bar_number, last_bar_number
                )
            ) is None:
                idx = (idx + 1) % len(self)
            else:
                return (image, sequence)
