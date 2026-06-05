"""Aspect-preserving resize ("letterbox") shared by the staffer/noter/scorer datasets.

A plain ``v2.Resize(target_shape)`` stretches each page's x and y independently, so a
page that is not full height (e.g. a short end-of-piece system) is blown up vertically
— distorting staff spacing and pushing short pages out of distribution.
``LetterboxResize`` instead scales by a single factor that fits the page within the
target, then pads the bottom/right with ``fill`` so the content stays anchored at the
top-left origin (which is what the datasets' box normalisation assumes).
"""

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
