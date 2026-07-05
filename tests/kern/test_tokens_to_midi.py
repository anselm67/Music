"""Tests for the simplified-token -> MIDI playback bridge."""

from kern.tokenizer import split_articulation
from kern.tokens_to_midi import Dynamics, Part, part_to_events

TPQ = 480
DYN = Dynamics()


def part(*tokens: str) -> Part:
    """Build a one-note-per-timestep part from ``pitch/dur[@codes]`` strings."""
    steps = []
    for tok in tokens:
        base, flags = split_articulation(tok)
        steps.append([(base, tuple(flags))])
    return steps


def chord(tokens: list[str]) -> Part:
    """A single timestep sounding all ``tokens`` at once."""
    entries = [(b, tuple(f)) for b, f in map(split_articulation, tokens)]
    return [entries]


def test_two_notes_advance_and_gate() -> None:
    events = part_to_events(part("C/4", "D/4"), TPQ, DYN)
    assert [e.onset for e in events] == [0, TPQ]
    # normal gate is 0.9 of the quarter (480) = 432
    assert [e.duration for e in events] == [432, 432]


def test_tie_fuses_same_pitch() -> None:
    # arc opens on the first C, closes on the second C: one 2-quarter note.
    events = part_to_events(part("C/4@<", "C/4@>"), TPQ, DYN)
    assert len(events) == 1
    assert events[0].onset == 0
    assert events[0].duration == 2 * TPQ


def test_slur_is_legato_not_fused() -> None:
    # arc across two different pitches -> two full-gate (legato) notes.
    events = part_to_events(part("C/4@<", "D/4@>"), TPQ, DYN)
    assert len(events) == 2
    assert [e.duration for e in events] == [TPQ, TPQ]  # full gate, no detache gap


def test_rest_breaks_a_tie() -> None:
    events = part_to_events(part("C/4@<", "rest/4", "C/4@>"), TPQ, DYN)
    assert len(events) == 2
    assert [e.onset for e in events] == [0, 2 * TPQ]  # rest consumed the middle beat


def test_staccato_shortens_gate() -> None:
    events = part_to_events(part("C/4@s"), TPQ, DYN)
    assert events[0].duration == TPQ // 2


def test_accent_raises_velocity() -> None:
    plain = part_to_events(part("C/4"), TPQ, DYN)[0]
    accented = part_to_events(part("C/4@a"), TPQ, DYN)[0]
    assert accented.velocity.value > plain.velocity.value


def test_fermata_lengthens_note_and_beat() -> None:
    events = part_to_events(part("C/4@f", "D/4"), TPQ, DYN)
    held = int(round(TPQ * DYN.fermata_scale))
    assert events[1].onset == held  # the beat stretched, so D starts later


def test_chord_strikes_all_notes_together() -> None:
    events = part_to_events(chord(["C/4", "E/4", "G/4"]), TPQ, DYN)
    assert len(events) == 3
    assert {e.onset for e in events} == {0}
    assert len({e.pitch.value for e in events}) == 3


def test_bars_and_clefs_take_no_time() -> None:
    events = part_to_events(part("clef-G", "=1", "C/4", "=2"), TPQ, DYN)
    assert len(events) == 1
    assert events[0].onset == 0


def test_gracenote_is_short() -> None:
    # "C/q" -> a short fixed 4*TPQ//32 = 60-tick grace, gated at the normal 0.9.
    events = part_to_events(part("C/q"), TPQ, DYN)
    assert len(events) == 1
    assert events[0].duration == int(round(60 * DYN.normal_gate))


def test_orphaned_arc_end_sounds_legato() -> None:
    # An arc-end with no matching open (model noise) strikes at full (legato) gate.
    events = part_to_events(part("C/4@>"), TPQ, DYN)
    assert len(events) == 1
    assert events[0].duration == TPQ


def test_tie_chain_middle_note() -> None:
    # tie-middle carries both arc bits: C -> C -> C fuses to a 3-quarter note.
    events = part_to_events(part("C/4@<", "C/4@<>", "C/4@>"), TPQ, DYN)
    assert len(events) == 1
    assert events[0].duration == 3 * TPQ
