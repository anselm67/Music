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

MAX_CHORDS = 8
SPAD_LEN = 128


class NoterDataset(Dataset):
    pdmx: PDMX
    config: Config
    items: list[tuple[Path, Box, int, int]]
    transform: v2.Transform
    vocab: Vocab

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
                v2.Normalize(mean=[0.9563435316085815], std=[0.16557540870879858]),
            ]
        )
        # Pre-computes start and end sequence tokens.
        self.s_sos = torch.full((1, MAX_CHORDS), self.vocab.SOS)
        self.s_eos = torch.full((1, MAX_CHORDS), self.vocab.EOS)
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading dataset"
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
                            png_file,
                            system.box,
                            system.first_bar_number,
                            system.last_bar_number,
                        )
                    )
                if count >= 0 and len(self.items) >= count:
                    self.items = self.items[:count]
                    break
        logging.info(f"\tNoterDataset: {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    cached_image_index: int = -1
    cached_image: Tensor | None = None
    cached_reader_index: int = -1
    cached_reader: KernReader | None

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        while True:
            png_file, box, first_bar_number, last_bar_number = self.items[idx]
            # Gets the image and crop it to the system box.
            if idx == self.cached_image_index and self.cached_image is not None:
                image = self.cached_image
            else:
                try:
                    image = decode_image(png_file.as_posix())
                    image = self.transform(image)
                    self.last_index = idx
                    self.cached_image = image
                except Exception as e:
                    logging.error(f"{png_file}: {e}")
                    idx += 1
                    continue
            print(f"Image box: {box}")
            image = crop(image, box.top, box.left, box.height, box.width)
            # Load the tokens and extract the correct range.
            if idx == self.cached_reader_index and self.cached_reader is not None:
                reader = self.cached_reader
            else:
                kern_path = self.pdmx.get_path(png_file, "tokens")
                try:
                    reader = KernReader(kern_path)
                except Exception as e:
                    logging.error(f"{kern_path}: {e}")
                    idx += 1
                    continue
            spine_number: int = 0
            tensor = torch.full((SPAD_LEN - 1, MAX_CHORDS), self.vocab.PAD)
            records = reader.get_text(first_bar_number, last_bar_number)
            if records is None:
                logging.error(
                    f"{png_file}: bars {first_bar_number}:{last_bar_number} not found."
                )
                idx += 1
                continue
            for idx, text in enumerate(records):
                str_tok = text.split("\t")[spine_number]
                tensor[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=MAX_CHORDS
                )
            tensor[len(records), :] = self.s_eos
            return (image, torch.cat([self.s_sos, tensor]))
