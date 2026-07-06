import logging

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import crop
from tqdm import tqdm

from kern import NUM_ARTICULATIONS, split_articulation
from sheetmusic import (
    Box,
    LetterboxResize,
    PerImageNormalize,
    Source,
    letterbox_scale,
)

from .noter_model import NoterConfig
from .noter_vocab import Vocab


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
        """One stave's token sequence, shape (max_seqlen, max_chords).

        Returns SOS-prefixed, EOS-terminated tensor, or None if the bars are
        missing, the sequence is too long, or any record can't be decoded. This
        is the token-only view used by callers that don't need articulations
        (the scorer, the kernsheet replay check).
        """
        result = self.load(score_id, spine_number, first_bar, last_bar)
        return None if result is None else result[0]

    def load(
        self, score_id: str, spine_number: int, first_bar: int, last_bar: int
    ) -> tuple[Tensor, Tensor] | None:
        """One stave's (token sequence, articulation multi-hot).

        Sequence is (max_seqlen, max_chords) token ids; articulations is
        (max_seqlen, max_chords, NUM_ARTICULATIONS) floats, row-aligned to the
        SOS-prefixed sequence (the SOS/EOS/PAD rows carry zeros). Returns None on
        the same conditions as ``__call__``.
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
        arts = torch.zeros((self.max_seqlen, self.max_chords, NUM_ARTICULATIONS))
        for idx, text in enumerate(records):
            try:
                # Real KernSheet records occasionally have fewer spines than the
                # system's staff count (malformed/misaligned bar range); skip the
                # sample rather than letting the IndexError crash the worker.
                str_tok = text.split("\t")[spine_number]
                str_toks = str_tok.strip().split()
                body[idx, :] = self.vocab.tok2i(str_toks, max_chords=self.max_chords)
                for c, tok in enumerate(str_toks):
                    # body row idx -> sequence row idx + 1 (SOS prefix).
                    arts[idx + 1, c] = torch.tensor(
                        split_articulation(tok)[1], dtype=torch.float
                    )
            except Exception as e:
                logging.error(f"{score_id}: {e}")
                return None
        body[len(records), :] = self.vocab.EOS
        return torch.cat([self.s_sos, body]), arts


class NoterDataset(Dataset):
    def __init__(
        self, config: NoterConfig, source: Source, vocab: Vocab, count: int = -1
    ) -> None:
        self.source = source
        self.config = config
        self.vocab = vocab
        self.transform = self._build_transform()
        self.load_sequence = SequenceLoader(
            source, vocab, config.max_seqlen, config.max_chords
        )
        # Creates the actual dataset, with theright number of samples.
        logging.info("Initializing NoterDataset...")
        self.items = []
        target_h, target_w = config.page_shape
        for score in tqdm(source.scores(), desc="Loading noter dataset"):
            # One spine per staff is the contract. `staff_map` routes each staff
            # (top->bottom) to its token column via the `*staffN` row, with a
            # positional bass-first fallback; its length is the token-file spine
            # count. A system whose staff count differs from the spine count (a
            # voiced staff, or a krn declaring a different staff count than the scan
            # shows) can't be routed one-to-one, so skip it rather than train a staff
            # against a partial/mismatched transcription (see spine_count review).
            try:
                staff_map = source.staff_map(score.id)
            except Exception as e:
                logging.error(f"{score.id}: cannot read staff map ({e}), skipping")
                continue
            spine_count = len(staff_map)
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
                    if system.staff_count > self.config.max_staves:
                        logging.error(
                            f"{score.id}: too many staves in system "
                            f"({system.staff_count} vs max_staves "
                            f"{self.config.max_staves})"
                        )
                        continue
                    if spine_count != system.staff_count:
                        logging.error(
                            f"{score.id}: token file has {spine_count} spines but "
                            f"system has {system.staff_count} staves; skipping "
                            f"(spine/staff mismatch — see spine_count review)"
                        )
                        continue
                    if not system.bar_numbers:
                        logging.error(
                            f"{score.id}: system with no bar numbers, skipping"
                        )
                        continue
                    # One item per system: its staves' boxes + spine numbers, kept
                    # aligned top->bottom. The staves are decoded together (shared
                    # barline grid).
                    self.items.append(
                        (
                            score.id,
                            page.page_number,
                            system.staff_boxes,
                            staff_map,
                            system.first_bar_number,
                            system.last_bar_number,
                        )
                    )
            if count >= 0 and len(self.items) >= count:
                self.items = self.items[:count]
                break
        logging.info(f"\tNoterDataset: {len(self.items):,} samples.")

    def _build_transform(self) -> v2.Transform:
        """Image transform: grayscale, letterbox resize, per-image normalisation."""
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
                PerImageNormalize(),
            ]
        )

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
        # (~48-64px at the 2× canvas) is far shorter than `height` (128), so the
        # old 3*box.height
        # window always exceeded it: cropping from box.top-box.height shoved the
        # staff into the lower half and clipped the tallest ones. Center on the
        # staff midline instead, padding white (image_pad_value) at page edges.
        crop_top = box.top + box.height // 2 - height // 2
        src_top = max(0, crop_top)
        src_bot = min(page_height, crop_top + height)
        # Clamp the crop to the page so a box overhanging the right edge pads
        # white via the canvas, not black (crop() zero-pads = ink).
        crop_width = min(box.width, page_width - box.left)
        tensor = crop(tensor, src_top, box.left, src_bot - src_top, crop_width)
        image = torch.full((1, height, width), image_pad_value)
        _, cropped_height, cropped_width = tensor.shape
        y0 = src_top - crop_top
        image[:, y0 : y0 + cropped_height, :cropped_width] = tensor
        return image, cropped_width

    def _load_staff(
        self,
        score_id: str,
        page_number: int,
        box: Box,
        spine_number: int,
        first_bar: int,
        last_bar: int,
    ) -> tuple[Tensor, int, Tensor, Tensor] | None:
        """One staff's (image, width, sequence, articulations), or None if either
        can't load."""
        if (result := self._load_image(score_id, page_number, box)) is None:
            return None
        if (
            seq := self.load_sequence.load(score_id, spine_number, first_bar, last_bar)
        ) is None:
            return None
        image, actual_width = result
        sequence, articulations = seq
        return image, actual_width, sequence, articulations

    def _next_same_count(self, idx: int) -> int:
        """Next index whose system has the same staff count as ``idx`` (wraps).

        The load-failure fallback must keep the substitute's staff count, or a
        staff-count bucket batch would gain a taller system and ``collate_systems``
        would pad the whole (possibly large) batch up to it — a memory spike that
        defeats the flat-peak crop budget. Falls back to the neighbour if no other
        system shares the count (degenerate corpus)."""
        n = len(self)
        target = len(self.items[idx][2])
        for step in range(1, n):
            j = (idx + step) % n
            if len(self.items[j][2]) == target:
                return j
        return (idx + 1) % n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """A system's ``G`` real staves: ``(images, widths, sequences,
        articulations, stave_mask)``, unpadded.

        Shapes ``(G, 1, 128, 1536)``, ``(G,)``, ``(G, max_seqlen, max_chords)``,
        ``(G, max_seqlen, max_chords, NUM_ARTICULATIONS)``, ``(G,)`` all-True. Padding
        to a common width happens at the batch level in ``collate_systems`` (to the
        batch-max staff count, ~0 within a staff-count bucket) rather than to the
        global ``max_staves``. If any staff of the system fails to load the whole
        system is skipped for another of the same staff count (keeps bucket batches
        homogeneous).
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
                idx = self._next_same_count(idx)
                continue
            valid = [staff for staff in loaded if staff is not None]
            images = torch.stack([staff[0] for staff in valid])
            widths = torch.tensor([staff[1] for staff in valid])
            sequences = torch.stack([staff[2] for staff in valid])
            articulations = torch.stack([staff[3] for staff in valid])
            mask = torch.ones(len(valid), dtype=torch.bool)
            return images, widths, sequences, articulations, mask


def collate_systems(
    batch: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    config: NoterConfig,
    vocab: Vocab,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Stack a batch of systems, padding each to the batch-max staff count.

    Each item carries only its ``G`` real staves (variable across the batch); this
    pads the short ones up to ``S = max(G)`` with masked slots and stacks to
    ``(B, S, ...)``. A padding slot is a zero image (full-width so no encoder patch
    is all-masked → no NaN), an SOS-only sequence, zero articulations, and a False
    ``stave_mask`` entry, so the model excludes it from loss and cross-stave
    attention. With the staff-count bucket sampler every item in a batch already has
    the same ``G``, so this pads nothing; it stays correct for a mixed batch too.
    """
    s = max(item[0].shape[0] for item in batch)
    h, w = config.input_shape
    images, widths, sequences, articulations, masks = [], [], [], [], []
    for img, wid, seq, art, mask in batch:
        if (pad := s - img.shape[0]) > 0:
            pad_seq = torch.full((pad, config.max_seqlen, config.max_chords), vocab.PAD)
            pad_seq[:, 0, :] = vocab.SOS
            img = torch.cat([img, torch.zeros(pad, 1, h, w)])
            wid = torch.cat([wid, torch.full((pad,), w)])
            seq = torch.cat([seq, pad_seq])
            art = torch.cat(
                [
                    art,
                    torch.zeros(
                        pad, config.max_seqlen, config.max_chords, NUM_ARTICULATIONS
                    ),
                ]
            )
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)])
        images.append(img)
        widths.append(wid)
        sequences.append(seq)
        articulations.append(art)
        masks.append(mask)
    return (
        torch.stack(images),
        torch.stack(widths),
        torch.stack(sequences),
        torch.stack(articulations),
        torch.stack(masks),
    )
