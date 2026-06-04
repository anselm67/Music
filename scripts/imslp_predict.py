"""One-off: take an IMSLP PDF (URL or local file), rasterise to PNG, run
`scorer predict ravel`.

Usage:
    uv run python scripts/imslp_predict.py <url|file.pdf> [--dpi 300] [--model ravel]

Downloads (or copies) the PDF to a temp dir, converts every page to PNG with
`pdftoppm`, then shells out to `scorer predict <model> <page>.png ...`.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# IMSLP blocks bare urllib and gates direct downloads behind a disclaimer cookie.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) imslp_predict/1.0",
    "Cookie": "imslpdisclaimeraccepted=yes",
}


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req) as resp, dest.open("wb") as f:
        f.write(resp.read())
    size = dest.stat().st_size
    if size < 1024:
        sys.exit(f"Download looks too small ({size} B) — wrong URL or blocked?")


def to_png(pdf: Path, dpi: int, pages: int) -> list[Path]:
    prefix = pdf.with_suffix("")
    last_page = ["-l", str(pages)] if pages > 0 else []
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), *last_page, str(pdf), str(prefix)],
        check=True,
    )
    png_files = sorted(prefix.parent.glob(f"{prefix.name}-*.png"))
    if not png_files:
        sys.exit("pdftoppm produced no PNGs.")
    return png_files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="Direct IMSLP PDF URL or a local .pdf path")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--model", default="ravel")
    ap.add_argument(
        "--pages", type=int, default=0, help="Limit to the first N pages (0 = all)."
    )
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "score.pdf"
        if "://" in args.source:
            print(f"Downloading {args.source} ...")
            download(args.source, pdf)
        else:
            src = Path(args.source)
            if not src.is_file():
                sys.exit(f"No such file: {src}")
            print(f"Using local file {src} ...")
            shutil.copy(src, pdf)
        print(f"Rasterising at {args.dpi} dpi ...")
        pages = to_png(pdf, args.dpi, args.pages)
        print(f"{len(pages)} page(s) -> scorer predict {args.model}")
        subprocess.run(
            ["scorer", "-n", "1", "predict", args.model, *[str(p) for p in pages]],
            check=True,
        )


if __name__ == "__main__":
    main()
