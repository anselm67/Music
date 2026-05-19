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

# TODO This should eventually be NoterModel's Config
from pdmx import PDMX, Box, Score
from kern.kern_reader import KernReader

from .noter_model import NoterConfig
from .noter_vocab import Vocab


class NoterDataset(Dataset):
    def __init__(self, config: NoterConfig, pdmx: PDMX, count: int = -1) -> None:
        self.pdmx = pdmx
        self.config = config
        self.vocab = Vocab.load(pdmx.home / "build/vocab.json")
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
                # Values from running: staffer stats
                v2.Normalize(mean=[0.9482423663139343], std=[0.17525607175008864]),
            ]
        )
        self.image_pad_value = (1.0 - 0.9482423663139343) / 0.17525607175008864
        # Pre-computes start and end sequence tokens.
        self.s_sos = torch.full((1, config.max_chords), self.vocab.SOS)
        self.s_eos = torch.full((1, config.max_chords), self.vocab.EOS)
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        for _, row in tqdm(
            pdmx.df.iterrows(), total=len(pdmx.df), desc="Loading noter dataset"
        ):
            mxl_file = pdmx.home / row["mxl"]
            layout_file = pdmx.get_path(mxl_file, "layout")
            score = Score.from_json(json.loads(layout_file.read_text()))
            score = score.resize(config.page_shape[1], config.page_shape[0])
            for page in score.pages:
                if score.page_count > 1:
                    png_file = pdmx.get_page_path(mxl_file, "png", page.page_number)
                else:
                    png_file = pdmx.get_path(mxl_file, "png")
                for system in page.systems:
                    match system.staff_count:
                        case 1:
                            spine_numbers = [0]
                        case 2:
                            spine_numbers = [1, 0]
                        case _:
                            logging.error(
                                f"{mxl_file}: too many staves in system "
                                f"({len(page.systems)} vs 2)"
                            )
                            continue
                    for idx, staff in enumerate(system.staves):
                        self.items.append(
                            (
                                mxl_file,
                                png_file,
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
        mxl_file, _, box, _, first_bar_number, last_bar_number = self.items[idx]
        kern_path = self.pdmx.get_path(mxl_file, "tokens")
        reader = KernReader(kern_path)
        records = reader.get_text(first_bar_number, last_bar_number)
        return (box.height, box.width), len(records) if records else -1

    def _load_image(self, mxl_file: Path, png_file: Path, box: Box) -> Tensor | None:
        # Gets the image and crop it to the system box.
        try:
            tensor = decode_image(png_file.as_posix())
            _, image_height, image_width = tensor.shape
            if box.width > image_width or box.height > image_height:
                logging.error(
                    f"{mxl_file}: image too large (H x W) {box.height}x{box.width}"
                )
                return None
            tensor = self.transform(tensor)
        except Exception as e:
            logging.error(f"{png_file}: {e}")
            return None
        height, width = self.config.input_shape
        tensor = crop(
            tensor,
            max(0, box.top - box.height),
            box.left,
            min(height, 3 * box.height),
            box.width,
        )
        image = torch.full((1, height, width), self.image_pad_value)
        _, cropped_height, cropped_width = tensor.shape
        y0 = (height - cropped_height) // 2
        image[:, y0 : y0 + cropped_height, :cropped_width] = tensor
        return image

    def _load_sequence(
        self,
        mxl_file: Path,
        spine_number: int,
        first_bar_number: int,
        last_bar_number: int,
    ) -> Tensor | None:
        kern_path = self.pdmx.get_path(mxl_file, "tokens")
        try:
            reader = KernReader(kern_path)
        except Exception as e:
            logging.error(f"{kern_path}: {e}")
            return None
        tensor = torch.full(
            (self.config.max_seqlen - 1, self.config.max_chords), self.vocab.PAD
        )
        records = reader.get_text(first_bar_number, last_bar_number)
        if records is None:
            logging.error(
                f"{mxl_file}: bars {first_bar_number}:{last_bar_number} not found."
            )
            return None
        elif len(records) + 2 > self.config.max_seqlen:
            logging.error(
                f"{mxl_file}: bars {first_bar_number}:{last_bar_number}, "
                f"sequence too long {len(records)} (max {self.config.max_seqlen - 2})"
            )
            return None
        for idx, text in enumerate(records):
            str_tok = text.split("\t")[spine_number]
            try:
                tensor[idx, :] = self.vocab.tok2i(
                    str_tok.strip().split(), max_chords=self.config.max_chords
                )
            except Exception as e:
                logging.error(f"{mxl_file}: {e}")
                return None
        tensor[len(records), :] = self.s_eos
        return torch.cat([self.s_sos, tensor])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        while True:
            mxl_file, png_file, box, spine_number, first_bar_number, last_bar_number = (
                self.items[idx]
            )
            logging.debug(f"Loading {mxl_file}")
            if (image := self._load_image(mxl_file, png_file, box)) is None:
                idx = (idx + 1) % len(self)
            elif (
                sequence := self._load_sequence(
                    mxl_file, spine_number, first_bar_number, last_bar_number
                )
            ) is None:
                idx = (idx + 1) % len(self)
            else:
                return (image, sequence)
