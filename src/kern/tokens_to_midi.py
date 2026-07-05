"""Convert a scorer/noter simplified-token transcription into a playable MIDI file.

Bridges the model's per-stave token output (pitch x duration tokens plus a per-note
articulation multi-hot) to MIDI. Unlike :mod:`kern.to_midi`, which parses native kern
text and streams it straight to note-on/note-off, this consumes the *simplified*
tokens the model emits and is articulation-aware: staccato shortens the gate, accent
raises the velocity, fermata lengthens the note, and an **arc** is resolved to a tie
(same pitch on both ends -> one fused note) or a slur (different pitch -> legato).

Because ties fuse across timesteps and gates float per note, each part is first decoded
to a time-ordered :class:`NoteEvent` list, then serialised; this expresses held and
overlapping notes that the streaming ``MidiSpine`` cannot. The low-level MIDI byte
writer (:class:`midi.MidiOutput`) and ``note_to_midi`` are reused unchanged.

Input is torch-free: a *part* is a list of timesteps, each timestep a list of
``(base_token, flags)`` entries (a chord is >1 note entry; a rest/structural token is a
single entry). ``flags`` is the 5-bool articulation vector in :data:`ARTICULATIONS`
order (arc-start, arc-end, staccato, fermata, accent). The caller (the ``play`` CLI)
turns the model's ``(tokens, articulations)`` tensors into this form.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from kern.to_midi import duration_to_ticks, note_to_midi
from kern.typing import Duration, Note, Pitch
from midi import Channel, MidiOutput, Velocity
from midi import Pitch as MidiPitch

# Articulation flag indices, in ARTICULATIONS = ("<", ">", "s", "f", "a") order.
ARC_START, ARC_END, STACCATO, FERMATA, ACCENT = range(5)

# The articulation vector, one bool per ARTICULATIONS entry, indexed by the named
# constants above. Variable-length so it matches ``split_articulation``'s list output.
Flags = tuple[bool, ...]
Entry = tuple[str, Flags]  # a single note / rest / structural token + its flags
Timestep = list[Entry]  # concurrent entries (>1 => chord)
Part = list[Timestep]  # one instrument staff, its timesteps in reading order


@dataclass
class Dynamics:
    """Playback shaping. Gate fractions are of the note's nominal duration."""

    velocity: Velocity = Velocity.MezzoForte
    accent_velocity: Velocity = Velocity.Forte
    normal_gate: float = 0.9  # slight detache so repeated notes re-articulate
    staccato_gate: float = 0.5
    fermata_scale: float = 1.75  # holds both the sounding note and the beat


@dataclass
class NoteEvent:
    onset: int  # absolute ticks from the part's start
    pitch: MidiPitch
    duration: int  # gate length in ticks (already shaped for staccato/fermata)
    velocity: Velocity


@dataclass
class _Pending:
    """The most recent arc-opened note, eligible for a same-pitch tie fuse."""

    pitch: int
    event: NoteEvent


GRACENOTE_FRACTION = 32  # gracenote gets a short fixed 4*tpq//32 ticks, as in to_midi


def _parse_duration(text: str) -> Duration:
    """``"8"`` -> ``Duration(8)``, ``"8:1"`` -> ``Duration(8, 1)``."""
    base, _, dots = text.partition(":")
    return Duration(int(base), int(dots) if dots else 0)


def _parse_note(base: str, ticks_per_quarter: int) -> tuple[MidiPitch, int] | None:
    """A note token -> ``(midi pitch, nominal ticks)``, or ``None`` if not a note.

    Handles ``C/4``, ``cc/8:1``, ``C#/4``, ``BBB-/2``, drum ``C:x/4``, and the
    gracenote ``C/q``. Non-notes (bars, clefs, meters, rests, ``.``) return ``None``.
    """
    if "/" not in base:
        return None
    head, _, dur = base.rpartition("/")
    head = head.replace(":x", "")  # drop the drum marker; pitch is unaffected
    name = head.rstrip("#-")
    accidentals = head[len(name) :]
    try:
        pitch = Pitch[name]
    except KeyError:
        return None  # e.g. a meter "4/4": head "4" is not a pitch name
    if dur == "q":
        ticks = 4 * ticks_per_quarter // GRACENOTE_FRACTION
    else:
        try:
            ticks = duration_to_ticks(_parse_duration(dur), ticks_per_quarter)
        except ValueError:
            return None
    note = Note(
        duration=None,
        pitch=pitch,
        sharps=accidentals.count("#"),
        flats=accidentals.count("-"),
    )
    return note_to_midi(note), ticks


def _rest_ticks(base: str, ticks_per_quarter: int) -> int | None:
    if not base.startswith("rest/"):
        return None
    return duration_to_ticks(_parse_duration(base[len("rest/") :]), ticks_per_quarter)


def _gate(ticks: int, flags: Flags, dyn: Dynamics) -> int:
    # A slurred endpoint (arc opens or closes here) sounds legato, no detache gap. An
    # orphaned arc-end (model noise, no matching open) also lands here; legato is a
    # benign default.
    if flags[ARC_START] or flags[ARC_END]:
        return ticks
    if flags[STACCATO]:
        return int(round(ticks * dyn.staccato_gate))
    return int(round(ticks * dyn.normal_gate))


def _event_ticks(row: Timestep, ticks_per_quarter: int) -> int | None:
    """The onsetting event's duration in one stave-row, or ``None`` for a sustain
    (``.``) or a 0-duration structural token (bar / clef / key / meter)."""
    base = row[0][0]
    rest = _rest_ticks(base, ticks_per_quarter)
    if rest is not None:
        return rest
    for token, _ in row:
        parsed = _parse_note(token, ticks_per_quarter)
        if parsed is not None:
            return parsed[1]
    return None


def _row_schedule(
    staves: list[Part], ticks_per_quarter: int, dyn: Dynamics
) -> tuple[list[int], int]:
    """Shared onset tick per row across a system's aligned staves, plus the total.

    The staves of a system are decoded in lockstep (one row = one shared time slice),
    so timing must come from the grid, not each stave's own note durations — those can
    disagree (a mispredicted duration in one hand) and desync the parts. The slice at
    row ``t`` lasts the *shortest* onsetting event across the staves (the finest
    subdivision defines the step; a longer note simply spans several rows via ``.``).
    A row carrying a fermata holds (``fermata_scale``).
    """
    nrows = max((len(s) for s in staves), default=0)
    onsets = [0] * nrows
    clock = 0
    for t in range(nrows):
        durations: list[int] = []
        fermata = False
        for stave in staves:
            if t < len(stave):
                ticks = _event_ticks(stave[t], ticks_per_quarter)
                if ticks is not None:
                    durations.append(ticks)
                    fermata = fermata or any(f[FERMATA] for _, f in stave[t])
        step = min(durations) if durations else 0
        if fermata:
            step = int(round(step * dyn.fermata_scale))
        onsets[t] = clock
        clock += step
    return onsets, clock


def _stave_events(
    stave: Part,
    onsets: list[int],
    ticks_per_quarter: int,
    dyn: Dynamics,
    pending: _Pending | None = None,
) -> tuple[list[NoteEvent], _Pending | None]:
    """Decode one stave's rows into events, onsetting each at its shared-grid tick.

    Resolves arcs: an arc-opened note followed immediately by a same-pitch arc-closed
    note fuses into one sustained event (tie); otherwise the arc is a slur and both
    notes sound legato. Chord ties are out of scope — a chord always strikes.

    ``pending`` carries an arc opened at the end of the previous system (its event
    holds a global onset); returning the still-open ``pending`` lets the caller fuse a
    tie across a system break. ``onsets`` must therefore be in the same (global) frame
    as any incoming ``pending`` event's onset.
    """
    events: list[NoteEvent] = []
    for t, row in enumerate(stave):
        clock = onsets[t]
        if _rest_ticks(row[0][0], ticks_per_quarter) is not None:
            pending = None  # a rest breaks any pending tie
            continue

        notes = [
            (m, ticks, flags)
            for token, flags in row
            if (parsed := _parse_note(token, ticks_per_quarter)) is not None
            for m, ticks in [parsed]
        ]
        if not notes:
            continue  # structural: bar / clef / key / meter / continuation

        nominal = notes[0][1]  # a chord shares the leading note's duration
        if any(flags[FERMATA] for _, _, flags in notes):
            nominal = int(round(nominal * dyn.fermata_scale))

        if len(notes) == 1:
            pitch, _, flags = notes[0]
            velocity = dyn.accent_velocity if flags[ACCENT] else dyn.velocity
            tie = (
                flags[ARC_END] and pending is not None and pending.pitch == pitch.value
            )
            if tie:
                assert pending is not None
                # Extend the held note through this one's end on the shared clock.
                pending.event.duration = clock + nominal - pending.event.onset
                pending = (
                    _Pending(pitch.value, pending.event) if flags[ARC_START] else None
                )
            else:
                event = NoteEvent(clock, pitch, _gate(nominal, flags, dyn), velocity)
                events.append(event)
                pending = _Pending(pitch.value, event) if flags[ARC_START] else None
        else:
            pending = None
            for pitch, _, flags in notes:
                velocity = dyn.accent_velocity if flags[ACCENT] else dyn.velocity
                events.append(
                    NoteEvent(clock, pitch, _gate(nominal, flags, dyn), velocity)
                )
    return events, pending


def render_systems(
    systems: list[list[Part]], ticks_per_quarter: int, dyn: Dynamics
) -> list[list[NoteEvent]]:
    """Render an ordered sequence of systems into per-part event lists.

    Lays the systems end-to-end on a running tick offset (both hands share each
    system's grid, so they stay locked), and threads per-stave arc state across system
    breaks so a tie spanning a line break fuses into one held note. Staves are matched
    across systems by slot — their top-to-bottom position is the same instrument.
    """
    parts: dict[int, list[NoteEvent]] = {}
    pendings: dict[int, _Pending | None] = {}
    offset = 0
    for staves in systems:
        local, total = _row_schedule(staves, ticks_per_quarter, dyn)
        onsets = [tick + offset for tick in local]  # lift into the global frame
        for slot, stave in enumerate(staves):
            events, pendings[slot] = _stave_events(
                stave, onsets, ticks_per_quarter, dyn, pendings.get(slot)
            )
            parts.setdefault(slot, []).extend(events)
        offset += total
    return [parts[slot] for slot in sorted(parts)]


def part_to_events(
    part: Part, ticks_per_quarter: int, dyn: Dynamics
) -> list[NoteEvent]:
    """Decode a single stand-alone stave (its own rows drive the clock)."""
    onsets, _ = _row_schedule([part], ticks_per_quarter, dyn)
    return _stave_events(part, onsets, ticks_per_quarter, dyn)[0]


def _serialize_part(track: MidiOutput, events: list[NoteEvent], chan: Channel) -> None:
    """Emit note-on/note-off pairs for ``events`` as a delta-timed stream."""
    points: list[tuple[int, int, MidiPitch, Velocity]] = []
    for event in events:
        points.append((event.onset, 1, event.pitch, event.velocity))
        points.append((event.onset + event.duration, 0, event.pitch, event.velocity))
    # Note-offs (flag 0) sort before note-ons at the same tick, freeing a pitch before
    # it is restruck.
    points.sort(key=lambda p: (p[0], p[1]))
    clock = 0
    for tick, is_on, pitch, velocity in points:
        delta = tick - clock
        clock = tick
        if is_on:
            track.note_on(chan, pitch, velocity, delta)
        else:
            track.note_off(chan, pitch, velocity, delta)


def write_midi(
    parts: list[list[NoteEvent]],
    midi_file: Path,
    tempo: int = 90,
    ticks_per_quarter: int = 480,
) -> None:
    """Serialise already-decoded parts to a Format-1 MIDI file, one track per part.

    Each part is a flat event list with absolute onsets (the caller lays systems
    end-to-end on a shared offset), so this only allocates channels and writes bytes.
    """
    channels = deque(c for c in Channel if c.value != 9)  # skip the GM drum channel
    tracks: list[MidiOutput] = []
    for events in parts:
        if not channels:
            raise ValueError(
                f"Too many parts ({len(parts)}) for the 15 melodic MIDI channels"
            )
        chan = channels.popleft()
        track = MidiOutput()
        offset = track.open_chunk("MTrk")
        track.time_signature((4, 4))
        track.tempo(tempo)
        _serialize_part(track, events, chan)
        track.track_end()
        track.close_chunk(offset)
        tracks.append(track)

    output = MidiOutput()
    header = output.open_chunk("MThd")
    output.write_u16(1)  # Format 1: parallel tracks
    output.write_u16(len(tracks))
    output.write_u16(ticks_per_quarter)
    output.close_chunk(header)
    for track in tracks:
        output.append_track(track)
    output.save(midi_file)
