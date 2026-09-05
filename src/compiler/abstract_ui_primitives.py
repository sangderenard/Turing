"""Canonical constructible primitives for the first AbstractUI object set."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping


ABSTRACT_UI_PRIMITIVE_VERSION = "abstract-ui-primitives-v0"


@dataclass(frozen=True, slots=True)
class UILength:
    value: float
    unit: str = "px"

    def __post_init__(self) -> None:
        if self.unit not in {"px", "em", "rem", "%", "world", "cell"}:
            raise ValueError(f"unsupported AbstractUI length unit: {self.unit}")


@dataclass(frozen=True, slots=True)
class UIInsets:
    top: UILength
    right: UILength
    bottom: UILength
    left: UILength

    @staticmethod
    def all(value: float, unit: str = "px") -> "UIInsets":
        length = UILength(float(value), unit)
        return UIInsets(length, length, length, length)


@dataclass(frozen=True, slots=True)
class UIRadii:
    top_left: UILength
    top_right: UILength
    bottom_right: UILength
    bottom_left: UILength

    @staticmethod
    def all(value: float, unit: str = "px") -> "UIRadii":
        radius = UILength(float(value), unit)
        return UIRadii(radius, radius, radius, radius)


@dataclass(frozen=True, slots=True)
class UIColor:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("AbstractUI color cannot be empty")


@dataclass(frozen=True, slots=True)
class UIFont:
    families: tuple[str, ...]
    size: UILength
    weight: int = 400
    line_height: float = 1.4

    def __post_init__(self) -> None:
        if not self.families:
            raise ValueError("AbstractUI font needs at least one family")
        if self.weight <= 0 or self.line_height <= 0:
            raise ValueError("font weight and line height must be positive")


@dataclass(frozen=True, slots=True)
class UIDecoration:
    """Named backend-neutral adornments plus optional typed parameters."""

    names: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class UIPalette:
    """A partial, layered palette; ``None`` means inherit from below."""

    fg: UIColor | None = None
    bg: UIColor | None = None
    margins: UIInsets | None = None
    radii: UIRadii | None = None
    font: UIFont | None = None
    decoration: UIDecoration | None = None
    colors: tuple[tuple[str, UIColor], ...] | None = None
    visible: bool | None = None
    locked: bool | None = None

    def overlay(self, layer: "UIPalette") -> "UIPalette":
        """Apply a more local layer without flattening the source palettes."""

        inherited_colors = dict(self.colors or ())
        inherited_colors.update(dict(layer.colors or ()))
        return UIPalette(
            fg=layer.fg if layer.fg is not None else self.fg,
            bg=layer.bg if layer.bg is not None else self.bg,
            margins=layer.margins if layer.margins is not None else self.margins,
            radii=layer.radii if layer.radii is not None else self.radii,
            font=layer.font if layer.font is not None else self.font,
            decoration=(
                layer.decoration if layer.decoration is not None else self.decoration
            ),
            colors=tuple(inherited_colors.items()),
            visible=layer.visible if layer.visible is not None else self.visible,
            locked=layer.locked if layer.locked is not None else self.locked,
        )


DEFAULT_PALETTE = UIPalette(
    fg=UIColor("#edf8ef"),
    bg=UIColor("transparent"),
    margins=UIInsets.all(0),
    radii=UIRadii.all(0),
    font=UIFont(("ui-monospace", "Consolas", "monospace"), UILength(14)),
    decoration=UIDecoration(),
    colors=(),
    visible=True,
    locked=False,
)


@dataclass(frozen=True, slots=True)
class UIBBox:
    x: float
    y: float
    width: float
    height: float
    coordinate_space: str = "layout"

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("bbox width and height must be non-negative")
        if not self.coordinate_space:
            raise ValueError("bbox coordinate space cannot be empty")

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def intersects(self, other: "UIBBox") -> bool:
        if self.coordinate_space != other.coordinate_space:
            raise ValueError("bbox intersection requires one coordinate space")
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )


@dataclass(frozen=True, slots=True)
class UIPrimitiveEdge:
    source: str
    target: str
    relationship: str
    properties: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class UIPrimitive:
    """One immutable constructible object in the minimal AbstractUI graph."""

    identity: str
    archetype: str
    properties: tuple[tuple[str, Any], ...] = ()
    children: tuple["UIPrimitive", ...] = ()
    edges: tuple[UIPrimitiveEdge, ...] = ()
    palette: UIPalette | None = None
    bbox: UIBBox | None = None

    def effective_palette(self, inherited: UIPalette = DEFAULT_PALETTE) -> UIPalette:
        return inherited if self.palette is None else inherited.overlay(self.palette)

    def _require_editable(self) -> None:
        if self.effective_palette().locked:
            raise PermissionError(f"AbstractUI object is locked: {self.identity}")

    def with_(self, *children: "UIPrimitive") -> "UIPrimitive":
        self._require_editable()
        known = {child.identity for child in self.children}
        duplicate = known.intersection(child.identity for child in children)
        if duplicate:
            raise ValueError(f"duplicate child identities: {sorted(duplicate)!r}")
        return replace(self, children=(*self.children, *children))

    def styled(self, layer: UIPalette) -> "UIPrimitive":
        self._require_editable()
        active = self.palette or UIPalette()
        return replace(self, palette=active.overlay(layer))

    def placed(self, bounds: UIBBox) -> "UIPrimitive":
        self._require_editable()
        return replace(self, bbox=bounds)

    def connect(
        self,
        source: str,
        target: str,
        relationship: str,
        **properties: Any,
    ) -> "UIPrimitive":
        self._require_editable()
        edge = UIPrimitiveEdge(
            str(source), str(target), str(relationship),
            tuple((str(name), value) for name, value in properties.items()),
        )
        return replace(self, edges=(*self.edges, edge))

    def objects(self) -> tuple["UIPrimitive", ...]:
        return (self, *(nested for child in self.children for nested in child.objects()))

    def to_data(self) -> dict[str, Any]:
        palette = self.effective_palette()
        return {
            "schema": ABSTRACT_UI_PRIMITIVE_VERSION,
            "identity": self.identity,
            "archetype": self.archetype,
            "properties": dict(self.properties),
            "children": [child.to_data() for child in self.children],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relationship": edge.relationship,
                    "properties": dict(edge.properties),
                }
                for edge in self.edges
            ],
            "palette": {
                "fg": palette.fg.value,
                "bg": palette.bg.value,
                "margins": _insets_data(palette.margins),
                "radii": _radii_data(palette.radii),
                "font": {
                    "families": palette.font.families,
                    "size": _length_data(palette.font.size),
                    "weight": palette.font.weight,
                    "line_height": palette.font.line_height,
                },
                "decoration": {
                    "names": palette.decoration.names,
                    "parameters": dict(palette.decoration.parameters),
                },
                "colors": {
                    name: color.value for name, color in (palette.colors or ())
                },
                "visible": palette.visible,
                "locked": palette.locked,
            },
            "bbox": None if self.bbox is None else {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "width": self.bbox.width,
                "height": self.bbox.height,
                "coordinate_space": self.bbox.coordinate_space,
            },
        }


def _length_data(value: UILength) -> dict[str, Any]:
    return {"value": value.value, "unit": value.unit}


def _insets_data(value: UIInsets) -> dict[str, Any]:
    return {
        "top": _length_data(value.top), "right": _length_data(value.right),
        "bottom": _length_data(value.bottom), "left": _length_data(value.left),
    }


def _radii_data(value: UIRadii) -> dict[str, Any]:
    return {
        "top_left": _length_data(value.top_left),
        "top_right": _length_data(value.top_right),
        "bottom_right": _length_data(value.bottom_right),
        "bottom_left": _length_data(value.bottom_left),
    }


def palette_data(
    value: UIPalette,
    inherited: UIPalette = DEFAULT_PALETTE,
) -> dict[str, Any]:
    """Resolve a palette into the shared transport used by CSS and graphics."""

    resolved = inherited.overlay(value)
    return {
        "fg": resolved.fg.value,
        "bg": resolved.bg.value,
        "margins": _insets_data(resolved.margins),
        "radii": _radii_data(resolved.radii),
        "font": {
            "families": list(resolved.font.families),
            "size": _length_data(resolved.font.size),
            "weight": resolved.font.weight,
            "line_height": resolved.font.line_height,
        },
        "decoration": {
            "names": list(resolved.decoration.names),
            "parameters": dict(resolved.decoration.parameters),
        },
        "colors": {
            name: color.value for name, color in (resolved.colors or ())
        },
        "visible": resolved.visible,
        "locked": resolved.locked,
    }


def palette(
    *,
    fg: str | UIColor | None = None,
    bg: str | UIColor | None = None,
    margins: UIInsets | None = None,
    radii: UIRadii | None = None,
    font: UIFont | None = None,
    decoration: UIDecoration | None = None,
    colors: Mapping[str, str | UIColor] | None = None,
    visible: bool | None = None,
    locked: bool | None = None,
) -> UIPalette:
    color_items = None if colors is None else tuple(
        (str(name), UIColor(value) if isinstance(value, str) else value)
        for name, value in colors.items()
    )
    return UIPalette(
        fg=UIColor(fg) if isinstance(fg, str) else fg,
        bg=UIColor(bg) if isinstance(bg, str) else bg,
        margins=margins,
        radii=radii,
        font=font,
        decoration=decoration,
        colors=color_items,
        visible=visible,
        locked=locked,
    )


def bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    coordinate_space: str = "layout",
) -> UIBBox:
    return UIBBox(float(x), float(y), float(width), float(height), coordinate_space)


def div(
    identity: str,
    *children: UIPrimitive,
    palette: UIPalette | None = None,
    bbox: UIBBox | None = None,
    **properties: Any,
) -> UIPrimitive:
    return UIPrimitive(
        str(identity), "div", tuple((str(name), value) for name, value in properties.items()),
        tuple(children), palette=palette, bbox=bbox,
    )


def input_(
    identity: str,
    *,
    interaction: str,
    destination: str,
    value: Any = None,
    input_kind: str = "value",
    palette: UIPalette | None = None,
    bbox: UIBBox | None = None,
) -> UIPrimitive:
    if not interaction or not destination:
        raise ValueError("input needs interaction type and destination identity")
    return UIPrimitive(
        str(identity),
        "input",
        (
            ("input_kind", str(input_kind)),
            ("value", value),
            ("interaction", str(interaction)),
            ("destination", str(destination)),
        ),
        palette=palette,
        bbox=bbox,
    )


def graph(
    identity: str,
    *nodes: UIPrimitive,
    edges: Iterable[UIPrimitiveEdge] = (),
    palette: UIPalette | None = None,
    bbox: UIBBox | None = None,
) -> UIPrimitive:
    root = UIPrimitive(
        str(identity), "graph", children=tuple(nodes), edges=tuple(edges),
        palette=palette, bbox=bbox,
    )
    identities = [item.identity for item in root.objects()]
    if len(identities) != len(set(identities)):
        raise ValueError("graph object identities must be unique")
    known = set(identities)
    unknown = {
        endpoint
        for edge in root.edges
        for endpoint in (edge.source, edge.target)
        if endpoint not in known
    }
    if unknown:
        raise KeyError(f"graph edges reference unknown objects: {sorted(unknown)!r}")
    return root


# The canonical spelling is intentionally available for the functional DSL;
# ``input_`` remains source-safe for callers that avoid shadowing Python's built-in.
input = input_


__all__ = [
    "ABSTRACT_UI_PRIMITIVE_VERSION",
    "DEFAULT_PALETTE",
    "UIBBox",
    "UIColor",
    "UIDecoration",
    "UIFont",
    "UIInsets",
    "UILength",
    "UIPalette",
    "UIPrimitive",
    "UIPrimitiveEdge",
    "UIRadii",
    "bbox",
    "div",
    "graph",
    "input",
    "palette_data",
    "input_",
    "palette",
]
