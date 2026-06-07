"""Check the assumption: within a page, all systems share a left edge (except
maybe the first, indented) and a right edge (except maybe the last, short).

For each page with >= 3 systems, measures the spread (max - min, px) of:
  - left edges of ALL systems          vs  left edges EXCLUDING the first
  - right edges of ALL systems         vs  right edges EXCLUDING the last
If the "excluding boundary" spread collapses to a few px while the full spread
is large, the assumption holds (first/last are the only exceptions).

  uv run python scripts/system_edge_consistency.py --kern-home <KernSheet>
  uv run python scripts/system_edge_consistency.py --home <PDMX> --limit 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from kernsheet import KernSheet, KernSheetSource
from pdmx import PDMX, PdmxSource
from sheetmusic import Source


def _pct(xs: list[float]) -> str:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return "n=0"

    def q(p: float) -> float:
        return s[min(n - 1, int(p * n))]

    return (
        f"p50={q(0.50):6.1f}  p90={q(0.90):6.1f}  p99={q(0.99):6.1f}  max={s[-1]:7.1f}"
    )


def _frac_below(xs: list[float], t: float) -> str:
    return f"{100 * sum(x <= t for x in xs) / max(len(xs), 1):5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=Path("/home/anselm/datasets/PDMX"))
    ap.add_argument("--kern-home", type=Path, default=None)
    ap.add_argument("--csv", default="System2.csv")
    ap.add_argument("--limit", type=int, default=-1, help="PDMX rows to load")
    args = ap.parse_args()

    source: Source
    if args.kern_home is not None:
        source = KernSheetSource(KernSheet(args.kern_home))
        name = "KernSheet"
    else:
        source = PdmxSource(PDMX(args.home, args.csv, -1, args.limit))
        name = "PDMX"

    left_full: list[float] = []
    left_interior: list[float] = []  # excluding first system
    right_full: list[float] = []
    right_interior: list[float] = []  # excluding last system
    indent: list[float] = []  # first.left - median(rest.left)
    shortfall: list[float] = []  # median(rest.right) - last.right
    n_pages = 0

    for score in source.scores():
        for page in score.pages:
            sys = page.systems
            if len(sys) < 3:
                continue
            n_pages += 1
            lefts = [s.left for s in sys]
            rights = [s.right for s in sys]
            left_full.append(max(lefts) - min(lefts))
            right_full.append(max(rights) - min(rights))
            li = lefts[1:]  # drop first (possible indent)
            ri = rights[:-1]  # drop last (possible short)
            left_interior.append(max(li) - min(li))
            right_interior.append(max(ri) - min(ri))
            med_rest_left = sorted(li)[len(li) // 2]
            med_rest_right = sorted(ri)[len(ri) // 2]
            indent.append(lefts[0] - med_rest_left)
            shortfall.append(med_rest_right - rights[-1])

    li3, li5 = _frac_below(left_interior, 3), _frac_below(left_interior, 5)
    ri3, ri5 = _frac_below(right_interior, 3), _frac_below(right_interior, 5)
    print(f"\n{name}: {n_pages} pages with >= 3 systems\n")
    print("Per-page edge spread (max - min, px):")
    print(f"  left  ALL systems      {_pct(left_full)}")
    print(
        f"  left  excl. first      {_pct(left_interior)}   (<=3px: {li3}  <=5px: {li5})"
    )
    print(f"  right ALL systems      {_pct(right_full)}")
    print(
        f"  right excl. last      {_pct(right_interior)}  (<=3px: {ri3}  <=5px: {ri5})"
    )
    print("\nBoundary exceptions (px):")
    print(f"  first-system indent    {_pct([abs(x) for x in indent])}")
    print(f"  last-system shortfall  {_pct([abs(x) for x in shortfall])}")


if __name__ == "__main__":
    main()
