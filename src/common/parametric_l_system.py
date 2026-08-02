"""Composable Lindenmayer systems and their two-dimensional turtle geometry.

An L-system has two deliberately separate phases:

* :meth:`ParametricLSystem.derive` applies every matching production in
  parallel, producing the emergent symbolic structure.
* :meth:`ParametricLSystem.interpret` walks a derived word with a configurable
  turtle, producing plain line segments that a browser, plotter, or test can
  consume without knowing anything about the rewriting machinery.

Rules may be deterministic, weighted, context-sensitive, or callable.  The
callable forms receive immutable context objects containing the generation,
neighbours, user parameters, and seeded random-number generator.  This keeps
experiments expressive while leaving every intermediate word inspectable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from types import MappingProxyType
from typing import Any, TypeAlias


Point: TypeAlias = tuple[float, float]
RuleKey: TypeAlias = str | tuple[str | None, str, str | None]
WeightedChoices: TypeAlias = Mapping[str, float] | Sequence[tuple[str, float]]


@dataclass(frozen=True, slots=True)
class RewriteContext:
    """Everything a callable production can use to choose its replacement."""

    symbol: str
    index: int
    generation: int
    left: str | None
    right: str | None
    word: str
    parameters: Mapping[str, Any]
    rng: random.Random = field(repr=False, compare=False)


RuleCallable: TypeAlias = Callable[[RewriteContext], "RuleResult"]
RuleResult: TypeAlias = str | WeightedChoices | Sequence[str] | None


@dataclass(frozen=True, slots=True)
class TurtleState:
    """Immutable state exposed to a callable turtle action."""

    position: Point
    heading_degrees: float
    step_length: float
    turn_angle_degrees: float
    width: float
    branch_depth: int


@dataclass(frozen=True, slots=True)
class TurtleContext:
    """Context passed to a callable entry in ``actions``."""

    symbol: str
    index: int
    word: str
    state: TurtleState
    parameters: Mapping[str, Any]
    rng: random.Random = field(repr=False, compare=False)


ActionResult: TypeAlias = str | Sequence[str] | None
ActionCallable: TypeAlias = Callable[[TurtleContext], ActionResult]


@dataclass(frozen=True, slots=True)
class TurtleSegment:
    """One drawn edge in an interpreted L-system."""

    start: Point
    end: Point
    width: float
    symbol: str
    symbol_index: int
    branch_depth: int


@dataclass(frozen=True, slots=True)
class TurtleTrace:
    """Geometry and terminal state produced by :meth:`interpret`."""

    word: str
    segments: tuple[TurtleSegment, ...]
    bounds: tuple[float, float, float, float]
    final_state: TurtleState
    maximum_branch_depth: int

    @property
    def points(self) -> tuple[Point, ...]:
        """Segment endpoints, useful for plotting and simple serialization."""

        if not self.segments:
            return (self.final_state.position,)
        return tuple(
            point
            for segment in self.segments
            for point in (segment.start, segment.end)
        )

    def svg_path(self, *, precision: int = 6) -> str:
        """Return independent SVG move/line commands for all drawn edges."""

        if precision < 0:
            raise ValueError("precision must be non-negative")
        number = lambda value: format(value, f".{precision}f").rstrip("0").rstrip(".") or "0"
        return " ".join(
            f"M {number(edge.start[0])} {number(edge.start[1])} "
            f"L {number(edge.end[0])} {number(edge.end[1])}"
            for edge in self.segments
        )


class ParametricLSystem:
    """A configurable parallel rewriting system with turtle interpretation.

    Parameters
    ----------
    axiom:
        Initial word. Symbols are individual Unicode characters.
    rules:
        Mapping from a symbol, or ``(left, symbol, right)`` context tuple, to
        a replacement. A replacement can be a string, a sequence of equally
        likely strings, a ``replacement -> weight`` mapping, a sequence of
        ``(replacement, weight)`` pairs, or a callable accepting
        :class:`RewriteContext`. ``None`` from a callable means "keep the
        symbol". Context entries set to ``None`` are wildcards.
    parameters:
        Arbitrary immutable-by-convention values exposed to callable rules and
        actions. They are copied on construction.
    context_ignore:
        Symbols skipped while finding the left and right neighbours for
        context-sensitive rules (commonly turtle punctuation).
    actions:
        Per-symbol turtle actions. Values are action names, sequences of
        action names, or callables accepting :class:`TurtleContext`.
    angle_degrees, step_length, initial_position, initial_heading_degrees:
        Initial turtle geometry.
    width, step_scale, angle_scale, width_scale:
        Mutable drawing quantities and the factors used by the corresponding
        ``*_up`` and ``*_down`` actions.
    angle_jitter_degrees:
        Uniform random perturbation applied to each turn.
    length_jitter:
        Uniform relative perturbation applied to every draw or move. A value
        of ``0.2`` samples lengths from 80% through 120% of the current step.
    seed:
        Seed used independently by derivation and interpretation, making a
        complete call reproducible even when both stochastic rules and turtle
        jitter are enabled.
    max_symbols, max_segments:
        Explicit expansion and geometry budgets. They prevent an accidental
        high-order production from exhausting memory.
    strict_branches:
        If true, unmatched ``pop`` actions and unclosed branches are errors.

    Built-in turtle action names are ``draw``, ``move``, ``left``, ``right``,
    ``reverse``, ``push``, ``pop``, ``step_up``, ``step_down``, ``angle_up``,
    ``angle_down``, ``width_up``, ``width_down``, and ``noop``.
    """

    DEFAULT_ACTIONS: Mapping[str, str] = MappingProxyType(
        {
            "F": "draw",
            "G": "draw",
            "f": "move",
            "+": "left",
            "-": "right",
            "|": "reverse",
            "[": "push",
            "]": "pop",
        }
    )
    ACTION_NAMES = frozenset(
        {
            "draw",
            "move",
            "left",
            "right",
            "reverse",
            "push",
            "pop",
            "step_up",
            "step_down",
            "angle_up",
            "angle_down",
            "width_up",
            "width_down",
            "noop",
        }
    )

    def __init__(
        self,
        axiom: str,
        rules: Mapping[RuleKey, RuleResult | RuleCallable],
        *,
        parameters: Mapping[str, Any] | None = None,
        context_ignore: Iterable[str] = ("+", "-", "[", "]", "|"),
        actions: Mapping[str, str | Sequence[str] | ActionCallable] | None = None,
        angle_degrees: float = 90.0,
        step_length: float = 1.0,
        initial_position: Point = (0.0, 0.0),
        initial_heading_degrees: float = 0.0,
        width: float = 1.0,
        step_scale: float = 1.25,
        angle_scale: float = 1.25,
        width_scale: float = 1.25,
        angle_jitter_degrees: float = 0.0,
        length_jitter: float = 0.0,
        seed: int | str | bytes | None = None,
        max_symbols: int = 1_000_000,
        max_segments: int = 1_000_000,
        strict_branches: bool = True,
    ) -> None:
        if not isinstance(axiom, str) or not axiom:
            raise ValueError("axiom must be a non-empty string")
        if max_symbols < len(axiom) or max_symbols < 1:
            raise ValueError("max_symbols must accommodate the axiom")
        if max_segments < 0:
            raise ValueError("max_segments must be non-negative")
        self._require_positive("step_length", step_length)
        self._require_positive("width", width)
        self._require_positive("step_scale", step_scale)
        self._require_positive("angle_scale", angle_scale)
        self._require_positive("width_scale", width_scale)
        self._require_finite("angle_degrees", angle_degrees)
        self._require_finite("initial_heading_degrees", initial_heading_degrees)
        self._require_finite("angle_jitter_degrees", angle_jitter_degrees)
        self._require_finite("length_jitter", length_jitter)
        if angle_jitter_degrees < 0:
            raise ValueError("angle_jitter_degrees must be non-negative")
        if length_jitter < 0:
            raise ValueError("length_jitter must be non-negative")
        if len(initial_position) != 2 or not all(
            math.isfinite(float(value)) for value in initial_position
        ):
            raise ValueError("initial_position must contain two finite numbers")

        self.axiom = axiom
        self.rules = dict(rules)
        self.parameters = MappingProxyType(dict(parameters or {}))
        self.context_ignore = frozenset(context_ignore)
        self.actions = dict(self.DEFAULT_ACTIONS)
        if actions:
            self.actions.update(actions)
        self.angle_degrees = float(angle_degrees)
        self.step_length = float(step_length)
        self.initial_position = tuple(map(float, initial_position))
        self.initial_heading_degrees = float(initial_heading_degrees)
        self.width = float(width)
        self.step_scale = float(step_scale)
        self.angle_scale = float(angle_scale)
        self.width_scale = float(width_scale)
        self.angle_jitter_degrees = float(angle_jitter_degrees)
        self.length_jitter = float(length_jitter)
        self.seed = seed
        self.max_symbols = int(max_symbols)
        self.max_segments = int(max_segments)
        self.strict_branches = bool(strict_branches)

        self._validate_rules()
        self._validate_actions()

    @staticmethod
    def _require_finite(name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    @classmethod
    def _require_positive(cls, name: str, value: float) -> None:
        cls._require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    def _validate_rules(self) -> None:
        for key, rule in self.rules.items():
            if isinstance(key, str):
                if len(key) != 1:
                    raise ValueError(f"rule symbol must be one character: {key!r}")
            elif isinstance(key, tuple) and len(key) == 3:
                left, symbol, right = key
                if len(symbol) != 1:
                    raise ValueError(f"rule symbol must be one character: {symbol!r}")
                if any(
                    neighbour is not None and len(neighbour) != 1
                    for neighbour in (left, right)
                ):
                    raise ValueError("context neighbours must be one character or None")
            else:
                raise TypeError(f"invalid rule key: {key!r}")
            if not callable(rule):
                self._validate_rule_result(rule)

    def _validate_rule_result(self, result: RuleResult) -> None:
        if result is None or isinstance(result, str):
            return
        if isinstance(result, Mapping):
            choices = result.items()
        elif isinstance(result, Sequence):
            if not result:
                raise ValueError("a production alternative list cannot be empty")
            if all(isinstance(choice, str) for choice in result):
                return
            choices = result
        else:
            raise TypeError(f"unsupported production result: {result!r}")
        total = 0.0
        for item in choices:
            try:
                replacement, weight = item
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "weighted productions must contain (replacement, weight) pairs"
                ) from error
            if not isinstance(replacement, str):
                raise TypeError("a production replacement must be a string")
            self._require_finite("production weight", weight)
            if weight < 0:
                raise ValueError("production weights cannot be negative")
            total += float(weight)
        if total <= 0:
            raise ValueError("weighted productions need at least one positive weight")

    def _validate_actions(self) -> None:
        for symbol, action in self.actions.items():
            if len(symbol) != 1:
                raise ValueError(f"action symbol must be one character: {symbol!r}")
            if callable(action):
                continue
            names = (action,) if isinstance(action, str) else tuple(action)
            unknown = set(names) - self.ACTION_NAMES
            if unknown:
                raise ValueError(f"unknown turtle action(s): {sorted(unknown)!r}")

    def _neighbour(self, word: str, index: int, direction: int) -> str | None:
        cursor = index + direction
        while 0 <= cursor < len(word):
            candidate = word[cursor]
            if candidate not in self.context_ignore:
                return candidate
            cursor += direction
        return None

    def _matching_rule(
        self, symbol: str, left: str | None, right: str | None
    ) -> RuleResult | RuleCallable | None:
        best: RuleResult | RuleCallable | None = None
        best_specificity = -1
        for key, rule in self.rules.items():
            if isinstance(key, str):
                if key == symbol and best_specificity < 0:
                    best = rule
                    best_specificity = 0
                continue
            required_left, required_symbol, required_right = key
            if required_symbol != symbol:
                continue
            if required_left is not None and required_left != left:
                continue
            if required_right is not None and required_right != right:
                continue
            specificity = int(required_left is not None) + int(required_right is not None)
            if specificity > best_specificity:
                best = rule
                best_specificity = specificity
        return best

    def _choose_replacement(
        self, result: RuleResult, *, symbol: str, rng: random.Random
    ) -> str:
        if result is None:
            return symbol
        self._validate_rule_result(result)
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            population = tuple(result)
            weights = tuple(float(result[item]) for item in population)
        elif all(isinstance(choice, str) for choice in result):
            population = tuple(result)
            weights = None
        else:
            pairs = tuple(result)
            population = tuple(pair[0] for pair in pairs)
            weights = tuple(float(pair[1]) for pair in pairs)
        return rng.choices(population, weights=weights, k=1)[0]

    def rewrite_once(
        self,
        word: str | None = None,
        *,
        generation: int = 0,
        rng: random.Random | None = None,
    ) -> str:
        """Apply one parallel production step to ``word``."""

        if generation < 0:
            raise ValueError("generation must be non-negative")
        source = self.axiom if word is None else word
        if not isinstance(source, str):
            raise TypeError("word must be a string")
        generator = rng or random.Random(self.seed)
        output: list[str] = []
        output_length = 0
        for index, symbol in enumerate(source):
            left = self._neighbour(source, index, -1)
            right = self._neighbour(source, index, 1)
            rule = self._matching_rule(symbol, left, right)
            if rule is None:
                replacement = symbol
            else:
                context = RewriteContext(
                    symbol=symbol,
                    index=index,
                    generation=generation,
                    left=left,
                    right=right,
                    word=source,
                    parameters=self.parameters,
                    rng=generator,
                )
                result = rule(context) if callable(rule) else rule
                replacement = self._choose_replacement(
                    result, symbol=symbol, rng=generator
                )
            output_length += len(replacement)
            if output_length > self.max_symbols:
                raise OverflowError(
                    f"generation {generation + 1} exceeded max_symbols="
                    f"{self.max_symbols}"
                )
            output.append(replacement)
        return "".join(output)

    def iter_derivation(self, iterations: int) -> Iterator[str]:
        """Yield the axiom and then every complete parallel generation."""

        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        word = self.axiom
        generator = random.Random(self.seed)
        yield word
        for generation in range(iterations):
            word = self.rewrite_once(word, generation=generation, rng=generator)
            yield word

    def derive(self, iterations: int) -> str:
        """Return the word after ``iterations`` parallel rewrites."""

        word = self.axiom
        for word in self.iter_derivation(iterations):
            pass
        return word

    def _action_names(
        self,
        action: str | Sequence[str] | ActionCallable,
        context: TurtleContext,
    ) -> tuple[str, ...]:
        result = action(context) if callable(action) else action
        if result is None:
            return ()
        names = (result,) if isinstance(result, str) else tuple(result)
        unknown = set(names) - self.ACTION_NAMES
        if unknown:
            raise ValueError(f"unknown turtle action(s): {sorted(unknown)!r}")
        return names

    def interpret(self, word: str) -> TurtleTrace:
        """Interpret an already-derived word as two-dimensional geometry."""

        if not isinstance(word, str):
            raise TypeError("word must be a string")
        x, y = self.initial_position
        heading = math.radians(self.initial_heading_degrees)
        turn_angle = math.radians(self.angle_degrees)
        step = self.step_length
        width = self.width
        stack: list[tuple[float, float, float, float, float, float]] = []
        segments: list[TurtleSegment] = []
        maximum_depth = 0
        min_x = max_x = x
        min_y = max_y = y
        generator = random.Random(self._interpretation_seed())

        def snapshot() -> TurtleState:
            return TurtleState(
                position=(x, y),
                heading_degrees=math.degrees(heading),
                step_length=step,
                turn_angle_degrees=math.degrees(turn_angle),
                width=width,
                branch_depth=len(stack),
            )

        for index, symbol in enumerate(word):
            action = self.actions.get(symbol)
            if action is None:
                continue
            context = TurtleContext(
                symbol=symbol,
                index=index,
                word=word,
                state=snapshot(),
                parameters=self.parameters,
                rng=generator,
            )
            for name in self._action_names(action, context):
                if name in {"draw", "move"}:
                    factor = 1.0 + generator.uniform(
                        -self.length_jitter, self.length_jitter
                    )
                    distance = max(0.0, step * factor)
                    start = (x, y)
                    x += math.cos(heading) * distance
                    y += math.sin(heading) * distance
                    min_x, max_x = min(min_x, x), max(max_x, x)
                    min_y, max_y = min(min_y, y), max(max_y, y)
                    if name == "draw":
                        if len(segments) >= self.max_segments:
                            raise OverflowError(
                                f"interpretation exceeded max_segments={self.max_segments}"
                            )
                        segments.append(
                            TurtleSegment(
                                start=start,
                                end=(x, y),
                                width=width,
                                symbol=symbol,
                                symbol_index=index,
                                branch_depth=len(stack),
                            )
                        )
                elif name in {"left", "right"}:
                    jitter = math.radians(
                        generator.uniform(
                            -self.angle_jitter_degrees,
                            self.angle_jitter_degrees,
                        )
                    )
                    heading += (turn_angle if name == "left" else -turn_angle) + jitter
                elif name == "reverse":
                    heading += math.pi
                elif name == "push":
                    stack.append((x, y, heading, step, turn_angle, width))
                    maximum_depth = max(maximum_depth, len(stack))
                elif name == "pop":
                    if not stack:
                        if self.strict_branches:
                            raise ValueError(f"unmatched branch pop at symbol {index}")
                        continue
                    x, y, heading, step, turn_angle, width = stack.pop()
                elif name == "step_up":
                    step *= self.step_scale
                elif name == "step_down":
                    step /= self.step_scale
                elif name == "angle_up":
                    turn_angle *= self.angle_scale
                elif name == "angle_down":
                    turn_angle /= self.angle_scale
                elif name == "width_up":
                    width *= self.width_scale
                elif name == "width_down":
                    width /= self.width_scale
                elif name != "noop":  # guarded by _action_names
                    raise AssertionError(f"unhandled turtle action {name!r}")

        if stack and self.strict_branches:
            raise ValueError(f"word ended with {len(stack)} unclosed branch(es)")
        return TurtleTrace(
            word=word,
            segments=tuple(segments),
            bounds=(min_x, min_y, max_x, max_y),
            final_state=snapshot(),
            maximum_branch_depth=maximum_depth,
        )

    def _interpretation_seed(self) -> int | str | bytes | None:
        if self.seed is None:
            return None
        # Give turtle jitter an independent deterministic stream. Using repr
        # also handles every seed type accepted by random.Random uniformly.
        return f"{self.seed!r}:turtle"

    def trace(self, iterations: int) -> TurtleTrace:
        """Derive ``iterations`` generations and interpret the final word."""

        return self.interpret(self.derive(iterations))

    def __call__(self, iterations: int) -> TurtleTrace:
        """Shorthand for :meth:`trace`."""

        return self.trace(iterations)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "ParametricLSystem":
        """Construct a classic system by name.

        Available names are ``hilbert``, ``peano``, ``dragon``, ``koch``,
        ``sierpinski``, and ``plant``. Any constructor argument can be
        replaced through ``overrides``.
        """

        presets: dict[str, dict[str, Any]] = {
            "hilbert": {
                "axiom": "A",
                "rules": {"A": "+BF-AFA-FB+", "B": "-AF+BFB+FA-"},
                "angle_degrees": 90.0,
            },
            "peano": {
                "axiom": "X",
                "rules": {
                    "X": "XFYFX+F+YFXFY-F-XFYFX",
                    "Y": "YFXFY-F-XFYFX+F+YFXFY",
                },
                "angle_degrees": 90.0,
            },
            "dragon": {
                "axiom": "FX",
                "rules": {"X": "X+YF+", "Y": "-FX-Y"},
                "angle_degrees": 90.0,
            },
            "koch": {
                "axiom": "F--F--F",
                "rules": {"F": "F+F--F+F"},
                "angle_degrees": 60.0,
            },
            "sierpinski": {
                "axiom": "F-G-G",
                "rules": {"F": "F-G+F+G-F", "G": "GG"},
                "angle_degrees": 120.0,
            },
            "plant": {
                "axiom": "X",
                "rules": {"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
                "angle_degrees": 25.0,
            },
        }
        key = name.casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "hilbert_curve": "hilbert",
            "peano_curve": "peano",
            "dragon_curve": "dragon",
            "koch_snowflake": "koch",
            "sierpinski_triangle": "sierpinski",
            "fractal_plant": "plant",
        }
        key = aliases.get(key, key)
        if key not in presets:
            raise ValueError(
                f"unknown preset {name!r}; choose from {', '.join(sorted(presets))}"
            )
        config = {**presets[key], **overrides}
        return cls(**config)


def render(
    t: int = 0,
    width: int = 512,
    height: int = 512,
    generations: int = 4,
):
    """Return one RGB frame from a preset selected by the frame counter."""

    from PIL import Image, ImageDraw

    presets = ("hilbert", "peano", "dragon", "koch", "sierpinski", "plant")
    preset_name = presets[int(t) % len(presets)]
    system = ParametricLSystem.preset(preset_name)
    trace = system.trace(int(generations))
    image = Image.new("RGB", (int(width), int(height)), (3, 7, 18))
    draw = ImageDraw.Draw(image)
    if trace.segments:
        min_x, min_y, max_x, max_y = trace.bounds
        span_x = max(max_x - min_x, 1e-12)
        span_y = max(max_y - min_y, 1e-12)
        padding = max(2, min(width, height) // 20)
        scale = min(
            (width - padding * 2) / span_x,
            (height - padding * 2) / span_y,
        )
        offset_x = (width - span_x * scale) * 0.5 - min_x * scale
        offset_y = (height - span_y * scale) * 0.5 + max_y * scale
        last_segment = max(1, len(trace.segments) - 1)
        for index, segment in enumerate(trace.segments):
            phase = index / last_segment
            color = (
                int(127.5 + 127.5 * math.cos(math.tau * phase)),
                int(127.5 + 127.5 * math.cos(math.tau * (phase + 0.21))),
                int(127.5 + 127.5 * math.cos(math.tau * (phase + 0.43))),
            )
            draw.line(
                (
                    segment.start[0] * scale + offset_x,
                    offset_y - segment.start[1] * scale,
                    segment.end[0] * scale + offset_x,
                    offset_y - segment.end[1] * scale,
                ),
                fill=color,
                width=max(1, min(4, int(round(segment.width)))),
            )
    red, green, blue = image.split()
    return red.tobytes(), green.tobytes(), blue.tobytes()


__all__ = [
    "ActionCallable",
    "ParametricLSystem",
    "RewriteContext",
    "RuleCallable",
    "TurtleContext",
    "TurtleSegment",
    "TurtleState",
    "TurtleTrace",
    "render",
]
