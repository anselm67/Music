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


def part_to_events(
    part: Part, ticks_per_quarter: int, dyn: Dynamics
) -> list[NoteEvent]:
    """Decode one part's timesteps into a time-ordered event list.

    Advances a clock over rest/note timesteps (bars, clefs, keys and meters take no
    time) and resolves arcs: an arc-opened note followed immediately by a same-pitch
    arc-closed note fuses into one sustained event (tie); otherwise the arc is a slur
    and both notes sound legato. Chord ties are out of scope — a chord always strikes.
    """
    events: list[NoteEvent] = []
    clock = 0
    pending: _Pending | None = None
    for timestep in part:
        rest = _rest_ticks(timestep[0][0], ticks_per_quarter)
        if rest is not None:
            clock += rest
            pending = None  # a rest breaks any pending tie
            continue

        notes = [
            (m, ticks, flags)
            for base, flags in timestep
            if (parsed := _parse_note(base, ticks_per_quarter)) is not None
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
                pending.event.duration += nominal  # fuse into the held note
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
        clock += nominal
    return events


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


def parts_to_midi(
    parts: list[Part],
    midi_file: Path,
    tempo: int = 90,
    ticks_per_quarter: int = 480,
    dynamics: Dynamics | None = None,
) -> None:
    """Render decoded parts to a Format-1 MIDI file, one track/channel per part."""
    dyn = dynamics or Dynamics()
    channels = deque(c for c in Channel if c.value != 9)  # skip the GM drum channel
    tracks: list[MidiOutput] = []
    for part in parts:
        events = part_to_events(part, ticks_per_quarter, dyn)
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
