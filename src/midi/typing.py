# https://www.music.mcgill.ca/~ich/classes/mumt306/StandardMIDIfileformat.html

import array
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast


class Channel(Enum):
    Chan0 = 0
    Chan1 = 1
    Chan2 = 2
    Chan3 = 3
    Chan4 = 4
    Chan5 = 5
    Chan6 = 6
    Chan7 = 7
    Chan8 = 8
    Chan9 = 9
    Chan10 = 10
    Chan11 = 11
    Chan12 = 12
    Chan13 = 13
    Chan14 = 14
    Chan15 = 15


class Notes(Enum):
    CX, CXSharp = 0, 1
    DX, DXSharp = 2, 3
    EX = 4
    FX, FXSharp = 5, 6
    GX, GXSharp = 7, 8
    AX, AXSharp = 9, 10
    BX = 11

    C0, C0Sharp = 12, 13
    D0, D0Sharp = 14, 15
    E0 = 16
    F0, F0Sharp = 17, 18
    G0, G0Sharp = 19, 20
    A0, A0Sharp = 21, 22
    B0 = 23

    C1, C1Sharp = 24, 25
    D1, D1Sharp = 26, 27
    E1 = 28
    F1, F1Sharp = 29, 30
    G1, G1Sharp = 31, 32
    A1, A1Sharp = 33, 34
    B1 = 35

    C2, C2Sharp = 36, 37
    D2, D2Sharp = 38, 39
    E2 = 40
    F2, F2Sharp = 41, 42
    G2, G2Sharp = 43, 44
    A2, A2Sharp = 45, 46
    B2 = 47

    C3, C3Sharp = 48, 49
    D3, D3Sharp = 50, 51
    E3 = 52
    F3, F3Sharp = 53, 54
    G3, G3Sharp = 55, 56
    A3, A3Sharp = 57, 58
    B3 = 59

    C4, C4Sharp = 60, 61
    D4, D4Sharp = 62, 63
    E4 = 64
    F4, F4Sharp = 65, 66
    G4, G4Sharp = 67, 68
    A4, A4Sharp = 69, 70
    B4 = 71

    C5, C5Sharp = 72, 73
    D5, D5Sharp = 74, 75
    E5 = 76
    F5, F5Sharp = 77, 78
    G5, G5Sharp = 79, 80
    A5, A5Sharp = 81, 82
    B5 = 83

    C6, C6Sharp = 84, 85
    D6, D6Sharp = 86, 87
    E6 = 88
    F6, F6Sharp = 89, 90
    G6, G6Sharp = 91, 92
    A6, A6Sharp = 93, 94
    B6 = 95

    C7, C7Sharp = 96, 97
    D7, D7Sharp = 98, 99
    E7 = 100
    F7, F7Sharp = 101, 102
    G7, G7Sharp = 103, 104
    A7, A7Sharp = 105, 106
    B7 = 107

    C8, C8Sharp = 108, 109
    D8, D8Sharp = 110, 111
    E8 = 112
    F8, F8Sharp = 113, 114
    G8, G8Sharp = 115, 116
    A8, A8Sharp = 117, 118
    B8 = 119

    C9, C9Sharp = 120, 121
    D9, D9Sharp = 122, 123
    E9 = 124
    F9, F9Sharp = 125, 126
    G9 = 127


class Velocity(Enum):
    Piano = 32
    MezzoForte = 64
    Forte = 96
    Standard = 64


class Instrument(Enum):
    ACOUSTIC_GRAND_PIANO_0 = 0        # According to chatgpt
    ACOUSTIC_GRAND_PIANO = 1
    BRIGHT_ACOUSTIC_PIANO = 2
    ELECTRIC_GRAND_PIANO = 3
    HONKY_TONK_PIANO = 4
    ELECTRIC_PIANO_1 = 5  # (Rhodes Piano)
    ELECTRIC_PIANO_2 = 6  # (Chorused Piano)
    HARPSICHORD = 7
    CLAVINET = 8
    CELESTA = 9
    GLOCKENSPIEL = 10
    MUSIC_BOX = 11
    VIBRAPHONE = 12
    MARIMBA = 13
    XYLOPHONE = 14
    TUBULAR_BELLS = 15
    DULCIMER = 16  # (Santur)
    DRAWBAR_ORGAN = 17  # (Hammond)
    PERCUSSIVE_ORGAN = 18
    ROCK_ORGAN = 19
    CHURCH_ORGAN = 20
    REED_ORGAN = 21
    ACCORDION = 22  # (French)
    HARMONICA = 23
    TANGO_ACCORDION = 24  # (Band neon)
    ACOUSTIC_GUITAR_NYLON = 25
    ACOUSTIC_GUITAR_STEEL = 26
    ELECTRIC_GUITAR_JAZZ = 27
    ELECTRIC_GUITAR_CLEAN = 28
    ELECTRIC_GUITAR_MUTED = 29
    OVERDRIVEN_GUITAR = 30
    DISTORTION_GUITAR = 31
    GUITAR_HARMONICS = 32
    ACOUSTIC_BASS = 33
    ELECTRIC_BASS_FINGERED = 34
    ELECTRIC_BASS_PICKED = 35
    FRETLESS_BASS = 36
    SLAP_BASS_1 = 37
    SLAP_BASS_2 = 38
    SYNTH_BASS_1 = 39
    SYNTH_BASS_2 = 40
    VIOLIN = 41
    VIOLA = 42
    CELLO = 43
    CONTRABASS = 44
    TREMOLO_STRINGS = 45
    PIZZICATO_STRINGS = 46
    ORCHESTRAL_HARP = 47
    TIMPANI = 48
    STRING_ENSEMBLE_1 = 49  # (strings)
    STRING_ENSEMBLE_2 = 50  # (slow strings)
    SYNTH_STRINGS_1 = 51
    SYNTH_STRINGS_2 = 52
    CHOIR_AAHS = 53
    VOICE_OOHS = 54
    SYNTH_VOICE = 55
    ORCHESTRA_HIT = 56
    TRUMPET = 57
    TROMBONE = 58
    TUBA = 59
    MUTED_TRUMPET = 60
    FRENCH_HORN = 61
    BRASS_SECTION = 62
    SYNTH_BRASS_1 = 63
    SYNTH_BRASS_2 = 64
    SOPRANO_SAX = 65
    ALTO_SAX = 66
    TENOR_SAX = 67
    BARITONE_SAX = 68
    OBOE = 69
    ENGLISH_HORN = 70
    BASSOON = 71
    CLARINET = 72
    PICCOLO = 73
    FLUTE = 74
    RECORDER = 75
    PAN_FLUTE = 76
    BLOWN_BOTTLE = 77
    SHAKUHACHI = 78
    WHISTLE = 79
    OCARINA = 80
    LEAD_1_SQUARE = 81
    LEAD_2_SAWTOOTH = 82
    LEAD_3_CALLIOPE = 83
    LEAD_4_CHIFFER = 84
    LEAD_5_CHARANG = 85
    LEAD_6_VOICE = 86
    LEAD_7_FIFTHS = 87
    LEAD_8_BASS_AND_LEAD = 88
    PAD_1_NEW_AGE = 89
    PAD_2_WARM = 90
    PAD_3_POLYSYNTH = 91
    PAD_4_CHOIR = 92
    PAD_5_BOWED = 93
    PAD_6_METALLIC = 94
    PAD_7_HALO = 95
    PAD_8_SWEEP = 96
    FX_1_RAIN = 97
    FX_2_SOUNDTRACK = 98
    FX_3_CRYSTAL = 99
    FX_4_ATMOSPHERE = 100
    FX_5_BRIGHTNESS = 101
    FX_6_GOBLINS = 102
    FX_7_ECHOES = 103
    FX_8_SCI_FI = 104
    SITAR = 105
    BANJO = 106
    SHAMISEN = 107
    KOTO = 108
    KALIMBA = 109
    BAGPIPE = 110
    FIDDLE = 111
    SHANAI = 112
    TINKLE_BELL = 113
    AGOGO = 114
    STEEL_DRUMS = 115
    WOODBLOCK = 116
    TAIKO_DRUM = 117
    MELODIC_TOM = 118
    SYNTH_DRUM = 119
    REVERSE_CYMBAL = 120
    GUITAR_FRET_NOISE = 121
    BREATH_NOISE = 122
    SEASHORE = 123
    BIRD_TWEET = 124
    TELEPHONE_RING = 125
    HELICOPTER = 126
    APPLAUSE = 127
    GUNSHOT = 128


class Format(Enum):
    SingleTrackMultiChannel = 0
    SimultaneousTracks = 1
    IndependantTracks = 2


class EventType(Enum):
    # Meta events uses MetaEventType
    Meta = 0xff
    SequenceNumber = (Meta, 0)
    Text = (Meta, 1)
    Copyright = (Meta, 2)
    TrackName = (Meta, 3)
    EndTrack = (Meta, 0x2f)
    Tempo = (Meta, 0x51)
    TimeSignature = (Meta, 0x58)
    KeySignature = (Meta, 0x59)
    Sequencer = (Meta, 0x7f)

    # Additional top level event types.
    SysExclusiveFirst = 0xf0
    SysExclusiveNext = 0xf7

    # Channel event type uses ChannelEventType.
    ChannelRange = (0x80, 0xEF)
    Channel = 0x80
    ControlChange = (Channel, 0xB0)
    ProgramChange = (Channel, 0xC0)
    NoteOn = (Channel, 0x90)
    NoteOff = (Channel, 0x80)

    # Synthetic
    Synthetic = 0
    HeaderData = (Synthetic, 1)
    OpenTrack = (Synthetic, 2)
    CloseTrack = (Synthetic, 3)

    def code(self) -> int:
        if isinstance(self.value, tuple):
            return self.value[1]
        else:
            return cast(int, self.value)

    @classmethod
    def is_sysex_code(cls, code: int) -> bool:
        return code == cls.SysExclusiveFirst.value or code == cls.SysExclusiveNext.value

    @classmethod
    def is_meta_code(cls, code: int) -> bool:
        return code == cls.Meta.value

    @classmethod
    def is_channel_code(cls, code: int) -> bool:
        min, max = cls.ChannelRange.value
        return min <= code <= max

    def is_meta(self) -> bool:
        return self == EventType.Meta

    def is_sysex(self) -> bool:
        return self == EventType.SysExclusiveFirst or self == EventType.SysExclusiveNext

    def is_channel(self) -> bool:
        min, max = EventType.ChannelRange.value
        return min <= self.code() <= max


@dataclass
class Event:
    dt: int
    event_type: EventType


@dataclass
class HeaderDataEvent(Event):

    format: Format
    number_of_tracks: int
    divisions: int

    def __init__(self, format: Format, number_of_tracks: int, divisions: int):
        super(HeaderDataEvent, self).__init__(0, EventType.HeaderData)
        self.format = format
        self.number_of_tracks = number_of_tracks
        self.divisions = divisions


@dataclass
class OpenTrackEvent(Event):

    def __init__(self):
        super(OpenTrackEvent, self).__init__(0, EventType.OpenTrack)


@dataclass
class CloseTrackEvent(Event):

    def __init__(self):
        super(CloseTrackEvent, self).__init__(0, EventType.CloseTrack)


@dataclass
class DataEvent(Event):
    data: array.array

    def __init__(self, dt: int, event_type: EventType, data: array.array):
        super(DataEvent, self).__init__(dt, event_type)
        self.data = data


@dataclass
class SequenceNumberEvent(Event):
    sequence_number: int

    def __init__(self, dt: int, sequence_number: int):
        super(SequenceNumberEvent, self).__init__(dt, EventType.SequenceNumber)
        self.sequence_number = sequence_number


@dataclass
class TextEvent(Event):
    text: str

    def __init__(self, dt, event_type: EventType, text: str):
        super(TextEvent, self).__init__(dt, event_type)
        self.text = text


@dataclass
class TempoEvent(Event):
    bpm: int

    def __init__(self, dt, bpm: int):
        super(TempoEvent, self).__init__(dt, EventType.Tempo)
        self.bpm = bpm


@dataclass
class TimeSignatureEvent(Event):
    nn: int     # Numerator.
    dd: int     # Denominator.
    cc: int     # Midi clocks per metronom tick.
    bb: int     # 32nd notes per midi quarter notes (24 midi clocks)

    def __init__(self, dt: int, nn: int, dd: int, cc: int, bb: int):
        super(TimeSignatureEvent, self).__init__(dt, EventType.TimeSignature)
        self.nn, self.dd, self.cc, self.bb = nn, dd, cc, bb


@dataclass
class KeySignatureEvent(Event):
    sf: int     # 0 -> C, > 0 -> sharps, < 0 -> flats
    mi: Literal['Minor', 'Major']

    def __init__(self, dt: int, sf: int, mi: Literal['Major', 'Minor']):
        super(KeySignatureEvent, self).__init__(dt, EventType.KeySignature)
        self.sf = sf
        self.mi = mi


@dataclass
class ChannelEvent(Event):
    channel: Channel

    def __init__(self, dt: int, event_type: EventType, channel: Channel):
        super(ChannelEvent, self).__init__(dt, event_type)
        self.channel = channel


@dataclass
class ProgramChangeEvent(ChannelEvent):
    program: Instrument

    def __init__(self, dt: int, channel: Channel, program: Instrument):
        super(ProgramChangeEvent, self).__init__(
            dt, EventType.ProgramChange, channel)
        self.program = program


@dataclass
class ControlChangeEvent(ChannelEvent):
    controller_number: int
    value: int

    def __init__(self, dt: int, channel: Channel, controller_number: int, value: int):
        super(ControlChangeEvent, self).__init__(
            dt, EventType.ProgramChange, channel)
        self.controller_number = controller_number
        self.value = value


@dataclass
class NoteEvent(ChannelEvent):
    note: Notes
    velocity: int

    def __init__(self, dt: int, event_type: EventType, channel: Channel, note: Notes, velocity: int):
        super(NoteEvent, self).__init__(dt, event_type, channel)
        self.note = note
        self.velocity = velocity


@dataclass
class NoteOnEvent(NoteEvent):

    def __init__(self, dt: int, channel: Channel, note: Notes, velocity: int):
        super(NoteOnEvent, self).__init__(
            dt, EventType.NoteOn, channel, note, velocity)


@dataclass
class NoteOffEvent(NoteEvent):

    def __init__(self, dt: int, channel: Channel, note: Notes, velocity: int):
        super(NoteOffEvent, self).__init__(
            dt, EventType.NoteOff, channel, note, velocity)
