import subprocess
from pathlib import Path

VEROVIO_HOME = Path("/home/anselm/projects/verovio")


def mxl_to_kern(file: Path, refresh: bool = False) -> bool:
    assert file.suffix == ".musicxml", "Expected suffix .musicxml"
    if file.with_suffix(".krn").exists() and not refresh:
        return False
    print(f"Translating {str(file)} to .krn")
    subprocess.run([
        f"{VEROVIO_HOME}/tools/verovio",
        "-r", f"{VEROVIO_HOME}/data",
        "-l", "off",
        "-f", "musicxml-hum", "-t", "hum",
        str(file),
        "-o", str(file.with_suffix(".krn"))
    ])
    return True


def render(src_file: Path, dst_file: Path) -> list[Path]:
    try:
        # verovio likes to go infinite on file it doesn't like.
        # The timeout prevents it from running hamoc.
        result = subprocess.run([
            f"{VEROVIO_HOME}/tools/verovio",
            "-r", f"{VEROVIO_HOME}/data",
            "-a",
            src_file.as_posix(),
            "-o", dst_file.as_posix()
        ], timeout=60)
    except subprocess.TimeoutExpired:
        print(f"{src_file}: timeout expired.")
        return list()
    if result.returncode != 0:
        raise ValueError(f"Error running verovio: {result.returncode}")
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


def svg_to_png(svg_file: Path, png_file: Path):
    try:
        result = subprocess.run([
            "rsvg-convert",
            "-f", "png", "-b", "white",
            "-o", png_file.as_posix(),
            svg_file.as_posix()], timeout=60)
    except subprocess.TimeoutExpired:
        print(f"{svg_file}: timeout expired.")
        return list()
    if result.returncode != 0:
        raise ValueError(f"Error running verovio: {result.returncode}")
