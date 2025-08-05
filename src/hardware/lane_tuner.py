"""Lane frequency assignment based on musical modes and chords."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

from .analog_spec import BASE_FREQ, SEMI_RATIO, LANES


@dataclass
class LaneTuner:
    """Dispense lane frequency assignments via music theory constructs.

    Parameters
    ----------
    lanes:
        Total number of available lanes.
    base_freq:
        Fundamental frequency used for the ``A`` region.
    semi_ratio:
        Equal-temperament semitone ratio.
    """

    lanes: int = LANES
    base_freq: float = BASE_FREQ
    semi_ratio: float = SEMI_RATIO
    #: Flag indicating whether speaker output should be serialized.
    serial_mode: bool = False

    #: Mapping of note names to semitone offsets from ``A``.
    NOTE_OFFSETS: Dict[str, int] = None  # type: ignore
    #: Modal interval definitions relative to the tonic.
    MODES: Dict[str, Sequence[int]] = None  # type: ignore
    #: Chord interval definitions relative to the root.
    CHORDS: Dict[str, Sequence[int]] = None  # type: ignore
    #: Region multipliers to shift fundamental up/down by octaves.
    REGION_MULTIPLIERS: Dict[str, float] = None  # type: ignore

    def __post_init__(self) -> None:
        self.NOTE_OFFSETS = {
            "A": 0,
            "A#": 1,
            "Bb": 1,
            "B": 2,
            "C": 3,
            "C#": 4,
            "Db": 4,
            "D": 5,
            "D#": 6,
            "Eb": 6,
            "E": 7,
            "F": 8,
            "F#": 9,
            "Gb": 9,
            "G": 10,
            "G#": 11,
            "Ab": 11,
        }
        self.MODES = {
            "ionian": [0, 2, 4, 5, 7, 9, 11],
            "major": [0, 2, 4, 5, 7, 9, 11],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "phrygian": [0, 1, 3, 5, 7, 8, 10],
            "lydian": [0, 2, 4, 6, 7, 9, 11],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "aeolian": [0, 2, 3, 5, 7, 8, 10],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "locrian": [0, 1, 3, 5, 6, 8, 10],
            "pentatonic": [0, 2, 4, 7, 9],
            "hexatonic": [0, 2, 4, 6, 8, 10],
        }
        self.CHORDS = {
            "major": [0, 4, 7],
            "minor": [0, 3, 7],
            "diminished": [0, 3, 6],
            "augmented": [0, 4, 8],
        }
        self.REGION_MULTIPLIERS = {
            "bottom": 0.5,
            "a": 1.0,
            "middle": 2.0,
            "top": 4.0,
        }

    def _note_freq(self, semitone_offset: int, base: float) -> float:
        return base * (self.semi_ratio ** semitone_offset)

    def assign(
        self,
        key: str,
        mode: str,
        fundamental: str = "A",
        *,
        lane_chords: bool = False,
        chord: str = "major",
    ) -> List[Union[float, List[float]]]:
        """Return lane frequency assignments for ``key`` and ``mode``.

        Parameters
        ----------
        key:
            Musical key (letter with optional accidental, e.g. ``"C"``, ``"Bb"``).
        mode:
            Mode or scale name (e.g. ``"lydian"``).
        fundamental:
            One of ``"top"``, ``"middle"``, ``"A"`` or ``"bottom"``.
        lane_chords:
            If ``True`` each lane holds a chord instead of a single frequency.
        chord:
            Chord quality used when ``lane_chords`` is ``True``.
        """

        key = key.capitalize()
        mode = mode.lower()
        chord = chord.lower()
        region_key = fundamental.lower()
        base_mult = self.REGION_MULTIPLIERS.get(region_key, 1.0)
        base = self.base_freq * base_mult
        root_offset = self.NOTE_OFFSETS[key]
        mode_intervals = self.MODES[mode]

        assignments: List[Union[float, List[float]]] = []
        for lane in range(self.lanes):
            deg = mode_intervals[lane % len(mode_intervals)] + 12 * (lane // len(mode_intervals))
            semitone = root_offset + deg
            root_freq = self._note_freq(semitone, base)
            if lane_chords:
                chord_intervals = self.CHORDS.get(chord, self.CHORDS["major"])
                chord_freqs = [self._note_freq(semitone + interval, base) for interval in chord_intervals]
                assignments.append(chord_freqs)
            else:
                assignments.append(root_freq)
        return assignments

    def set_serial(self, enable: bool) -> None:
        """Enable or disable serialized speaker output."""

        self.serial_mode = enable

    def _arpeggiate(
        self,
        assignments: Sequence[Union[float, Sequence[float]]],
        pattern: Sequence[int] | None = None,
    ) -> List[Union[float, List[float]]]:
        """Return assignments ordered by ``pattern`` for serial playback."""

        if pattern is None:
            pattern = list(range(len(assignments)))
        return [assignments[i % len(assignments)] for i in pattern]

    def output(
        self,
        key: str,
        mode: str,
        fundamental: str = "A",
        *,
        lane_chords: bool = False,
        chord: str = "major",
        pattern: Sequence[int] | None = None,
        serial: bool | None = None,
    ) -> List[Union[float, List[float]]]:
        """Return speaker output assignments respecting serialization.

        The underlying lane assignments are produced by :meth:`assign`. If
        ``serial`` is ``True`` (or the tuner's ``serial_mode`` flag is set),
        the resulting pattern is arpeggiated to avoid monotony; otherwise a
        faithful parallel mapping is returned.
        """

        assignments = self.assign(
            key,
            mode,
            fundamental,
            lane_chords=lane_chords,
            chord=chord,
        )
        if serial is None:
            serial = self.serial_mode
        if serial:
            return self._arpeggiate(assignments, pattern)
        return assignments
