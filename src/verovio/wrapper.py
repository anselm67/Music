import subprocess
from pathlib import Path

from .binaries import rsvgconvert_binary, verovio_binary


def safe_run(command: list[Path | str], timeout=60):
    try:
        result = subprocess.run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"{command[0]}: timeout expired.")
    if result.returncode != 0:
        raise ValueError(f"{command[0]} failed {command}: {result.returncode}")


def mxl_to_kern_command(mxl_file: Path, krn_file: Path) -> tuple[Path, list[str]]:
    return verovio_binary(), [
        "-l", "off",
        "-f", "musicxml-hum", "-t", "hum",
        mxl_file.as_posix(),
        "-o", krn_file.as_posix()
    ]


def mxl_to_kern(mxl_file: Path, krn_file: Path) -> bool:
    (binary, args) = mxl_to_kern_command(mxl_file, krn_file)
    safe_run([binary, *args])
    return True


def render_command(src_file: Path, dst_file: Path) -> tuple[Path, list[str]]:
    return verovio_binary(), [
        "-l", "off",
        "-a",
        src_file.as_posix(),
        "-o", dst_file.as_posix()
    ]


def render(src_file: Path, dst_file: Path) -> list[Path]:
    (binary, args) = render_command(src_file, dst_file)
    safe_run([binary, *args])
    # We may have generated or more files, depending on the length of the score.
    if dst_file.exists():
        return [dst_file]
    else:
        files = list()
        for i in range(1, 999):
            stem = f"{dst_file.stem}_{i:03d}"
            file = dst_file.with_stem(stem)
            if file.exists():
                files.append(file)
            else:
                return files
        raise ValueError("Too many pages in score!")


def svg_to_png_command(svg_file: Path, png_file: Path) -> tuple[Path, list[str]]:
    return rsvgconvert_binary(), [
        "-f", "png", "-b", "white", "--width", "1024",
        "-o", png_file.as_posix(),
        svg_file.as_posix()
    ]


def svg_to_png(svg_file: Path, png_file: Path):
    (binary, args) = svg_to_png_command(svg_file, png_file)
    safe_run([binary, *args])
