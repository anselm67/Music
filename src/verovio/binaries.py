import shutil
from pathlib import Path


def _find_binary(name: str) -> Path:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError("verovio binary not found in PATH")
    return Path(path)


verovio_binary = _find_binary("verovio")
rsvgconvert_binary = _find_binary("rsvg-convert")
