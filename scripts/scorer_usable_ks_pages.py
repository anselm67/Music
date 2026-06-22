"""Count how many validated KernSheet pages are *scorer-usable*.

The scorer trains one page per sample, but ``ScorerDataset.__getitem__`` silently
skips any page that doesn't fit its supervision contract: every system must be 1-2
staves with bar numbers, fit the query budget, and every stave's bar range must
tokenize. This replicates that exact filter (minus the image load) over the
validated KernSheet pages so we know the real coverage ceiling for
``SCORER_FT_TRAIN`` — i.e. how much of the corpus the cap could ever reach.

Usage:  uv run python scripts/scorer_usable_ks_pages.py [KS_ROOT]
"""

import sys
from collections import Counter
from pathlib import Path

from kernsheet import KernSheet, KernSheetSource
from noter import SequenceLoader, Vocab
from scorer.scorer_model import ScorerConfig

KS_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/anselm/datasets/KernSheet"
)


def main() -> None:
    config = ScorerConfig()
    c = config.staffer
    source = KernSheetSource(KernSheet(KS_ROOT))
    vocab = Vocab.load(KS_ROOT / "build" / "vocab.json")
    load_sequence = SequenceLoader(
        source, vocab, config.noter.max_seqlen, config.noter.max_chords
    )

    enrolled = 0  # validated pages with systems (what ScorerDataset enrols)
    usable = 0  # of those, pages that pass the __getitem__ contract
    reasons: Counter[str] = Counter()  # why an enrolled page is unusable

    for score in source.scores():
        for page in source.pages(score.id):
            if not page.systems:
                continue
            enrolled += 1

            ok = True
            staff_idx = 0
            reason = ""
            for sys_idx, system in enumerate(page.systems):
                if sys_idx >= c.num_system_queries:
                    ok, reason = False, "too many systems"
                    break
                if system.staff_count not in (1, 2):
                    ok, reason = False, f"{system.staff_count}-staff system"
                    break
                if not system.bar_numbers:
                    ok, reason = False, "no bar numbers"
                    break
                spine_numbers = [0] if system.staff_count == 1 else [1, 0]
                for i in range(system.staff_count):
                    if staff_idx >= c.num_stave_queries:
                        ok, reason = False, "too many staves"
                        break
                    seq = load_sequence(
                        score.id,
                        spine_numbers[i],
                        system.first_bar_number,
                        system.last_bar_number,
                    )
                    if seq is None:
                        ok, reason = False, "untokenizable bar range"
                        break
                    staff_idx += 1
                if not ok:
                    break

            if ok and staff_idx > 0:
                usable += 1
            else:
                reasons[reason or "empty"] += 1

    print(f"KS root            : {KS_ROOT}")
    print(f"enrolled pages     : {enrolled:,}  (validated, non-empty systems)")
    print(f"scorer-usable      : {usable:,}  ({usable / max(enrolled, 1):.1%})")
    print("unusable by reason :")
    for reason, n in reasons.most_common():
        print(f"  {n:>5,}  {reason}")


if __name__ == "__main__":
    main()
