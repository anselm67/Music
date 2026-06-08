"""Aspect-preserving resize ("letterbox") shared by the staffer/noter/scorer datasets.

A plain ``v2.Resize(target_shape)`` stretches each page's x and y independently, so a
page that is not full height (e.g. a short end-of-piece system) is blown up vertically
— distorting staff spacing and pushing short pages out of distribution.
``LetterboxResize`` instead scales by a single factor that fits the page within the
target, then pads the bottom/right with ``fill`` so the content stays anchored at the
top-left origin (which is what the datasets' box normalisation assumes).
"""

import numpy as np
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
