import shutil
from pathlib import Path


def _find_binary(name: str) -> Path:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError("verovio binary not found in PATH")
    return Path(path)


_verovio_binary = None


def verovio_binary():
    global _verovio_binary
    if _verovio_binary is None:
        _verovio_binary = _find_binary("verovio")
    return _verovio_binary


_rsvgconvert_binary = None


def rsvgconvert_binary():
    global _rsvgconvert_binary
    if _rsvgconvert_binary is None:
        _rsvgconvert_binary = _find_binary("rsvg-convert")
    return _rsvgconvert_binary
