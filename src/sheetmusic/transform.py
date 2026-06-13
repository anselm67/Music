"""Aspect-preserving resize ("letterbox") shared by the staffer/noter/scorer datasets.

A plain ``v2.Resize(target_shape)`` stretches each page's x and y independently, so a
page that is not full height (e.g. a short end-of-piece system) is blown up vertically
— distorting staff spacing and pushing short pages out of distribution.
``LetterboxResize`` instead scales by a single factor that fits the page within the
target, then pads the bottom/right with ``fill`` so the content stays anchored at the
top-left origin (which is what the datasets' box normalisation assumes).
"""

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode


def letterbox_scale(image_h: int, image_w: int, target_h: int, target_w: int) -> float:
    """The single scale factor that fits ``(image_h, image_w)`` within the target."""
    return min(target_h / image_h, target_w / image_w)


class LetterboxResize:
    """Resize preserving aspect ratio to fit within ``size``, then pad bottom/right.

    ``fill`` must be white in the image's current value range: ``255`` for a uint8
    image (resize before ``ToDtype``) or ``1.0`` for a float image scaled to ``[0, 1]``.
    """

    def __init__(
        self,
        size: list[int],
        interpolation: InterpolationMode,
        antialias: bool,
        fill: float,
    ) -> None:
        self.target_h, self.target_w = size
        self.interpolation = interpolation
        self.antialias = antialias
        self.fill = fill

    def __call__(self, image: Tensor) -> Tensor:
        _, h, w = image.shape
        scale = letterbox_scale(h, w, self.target_h, self.target_w)
        # clamp guards the float-rounding edge where round() lands one past the target.
        new_h = min(round(h * scale), self.target_h)
        new_w = min(round(w * scale), self.target_w)
        image = TF.resize(
            image,
            [new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        # TF.pad padding is [left, top, right, bottom]: pad right/bottom only.
        return TF.pad(
            image, [0, 0, self.target_w - new_w, self.target_h - new_h], fill=self.fill
        )


class PerImageNormalize:
    """Standardise each image by its OWN mean and std: ``(x - mean) / std``.

    Replaces a fixed global ``v2.Normalize`` so per-image brightness/contrast
    differences don't reach the model — PDMX's near-white synthetic pages and
    KernSheet's darker, grayer real scans both arrive centred at 0, unit variance.
    Operates on the whole tensor; white letterbox padding maps to the same value
    as the page's white paper (both were the max), so pad semantics are unchanged.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, image: Tensor) -> Tensor:
        return (image - image.mean()) / (image.std() + self.eps)


class ScanAugment:
    """Train-only augmentation that makes a crisp Verovio render look scanned.

    Operates on a float ``[0, 1]`` grayscale tensor (ink≈0, paper≈1), inserted
    BEFORE ``PerImageNormalize``. A physically-ordered, randomised chain models
    the print→scan pipeline: ink spread (fatter strokes), optical blur, paper
    tone + uneven illumination, sensor noise, and JPEG recompression. Every
    stage is geometry-preserving (ink grows/shrinks symmetrically, no box-edge
    shift), so the datasets' ground-truth boxes stay valid without co-transform.

    ``prob`` gates the WHOLE chain per call (0 disables it); the datamodule sets
    a non-zero prob on the train view only, leaving validation on clean crops.
    Closes the photometric/morphological gap that per-image norm (an affine)
    cannot synthesise — see PDMX-vs-KernSheet pixel histograms.
    """

    def __init__(self, prob: float = 0.0) -> None:
        self.prob = prob

    def __call__(self, image: Tensor) -> Tensor:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return image
        x = image
        # 1. Ink spread ("fatter"): dilate the dark ink via a min-filter (negated
        #    max-pool); rarely erode (max-pool) to model faded/broken ink.
        if random.random() < 0.6:
            k = random.choice([3, 3, 5])
            if random.random() < 0.85:
                x = -F.max_pool2d(-x.unsqueeze(0), k, 1, k // 2).squeeze(0)
            else:
                x = F.max_pool2d(x.unsqueeze(0), k, 1, k // 2).squeeze(0)
        # 2. Optical blur: soften crisp vector edges into a gray pedestal.
        if random.random() < 0.6:
            sigma = random.uniform(0.3, 0.9)
            x = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[sigma, sigma])
        # 3. Paper tone (gated): real scans keep most paper near-white with a gray
        #    tail, so only sometimes map [0,1] -> [black, white] (paper off-white,
        #    ink lifted off pure black) and only sometimes modulate by a smooth
        #    low-frequency field (uneven lighting / page curl).
        if random.random() < 0.5:
            black = random.uniform(0.0, 0.06)
            white = random.uniform(0.88, 1.0)
            x = x * (white - black) + black
        if random.random() < 0.5:
            x = x * self._illumination_field(x)
        # 4. Sensor noise.
        if random.random() < 0.6:
            x = x + torch.randn_like(x) * random.uniform(0.0, 0.025)
        x = x.clamp(0.0, 1.0)
        # 5. JPEG recompression: blocky ringing around glyphs (most scans are JPEG).
        if random.random() < 0.6:
            u8 = (x * 255.0).to(torch.uint8)
            u8 = TF.jpeg(u8, quality=random.randint(30, 90))
            x = u8.to(image.dtype) / 255.0
        return x

    @staticmethod
    def _illumination_field(image: Tensor) -> Tensor:
        """A smooth multiplicative lighting field in ~[0.90, 1.02], page-sized."""
        _, h, w = image.shape
        field = torch.rand(1, 1, 4, 4)
        field = TF.resize(field, [h, w], antialias=True).squeeze(0)
        return field * 0.12 + 0.90


def to_display(image: Tensor) -> np.ndarray:  # type: ignore[type-arg]
    """Map a ``(1, H, W)`` normalised tensor to a viewable ``(H, W, 3)`` uint8 array.

    The display inverse of ``PerImageNormalize``: per-image normalisation leaves
    no fixed stats to invert, so a min-max stretch maps the page back to grayscale
    (brightest→white, darkest→black) regardless of the page's brightness. Shared by
    every CLI that shows a transformed page/crop (staffer/noter/scorer).
    """
    arr = image.squeeze(0).detach().cpu().numpy()
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-6)
    return np.stack([(arr * 255).astype(np.uint8)] * 3, axis=-1)
