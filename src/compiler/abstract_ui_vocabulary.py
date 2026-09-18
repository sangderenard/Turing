"""Aspirational, executable vocabulary boneyard for :mod:`AbstractUI`.

This is intentionally more permissive than a finished compiler grammar.  It
lets a person write obvious functional phrases such as::

    world(
        region("foundry",
            building("pump house",
                interior(room("control room")),
                contains(gauge("pressure")),
            ),
        ),
    )

or compose consequential flow with light, Perl-like punctuation::

    when(player("operator").does(open_(door("east")))) >> reveal(room("pump"))

Every declared word constructs an immutable, serializable :class:`UIIntention`.
There is no pretend renderer here: the vocabulary records human intention and
backend coverage can grow around it.  Unknown fluent phrases also remain
explicit nodes rather than executing arbitrary attributes.

The large word list is a boneyard, not a promise that every backend already
understands every bone.  ``vocabulary_manifest()`` makes that aspirational
surface inspectable for future conformance tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping


ABSTRACT_UI_VOCABULARY_VERSION = "abstract-ui-intentions-v0"


def _encode(value: Any) -> Any:
    if isinstance(value, UIIntention):
        return value.to_data()
    if isinstance(value, IntentionWord):
        return {"word": value.spelling, "domain": value.domain}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value


@dataclass(frozen=True)
class UIIntention:
    """One functional phrase in the aspirational AbstractUI language."""

    word: str
    arguments: tuple[Any, ...] = ()
    traits: tuple[tuple[str, Any], ...] = ()
    domain: str = "open"

    def to_data(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "domain": self.domain,
            "arguments": [_encode(item) for item in self.arguments],
            "traits": {
                name: _encode(value) for name, value in self.traits
            },
        }

    def with_(self, *qualities: Any, **traits: Any) -> "UIIntention":
        return _make_intention("with", self, *qualities, domain="relation", **traits)

    def as_(self, role: Any, **traits: Any) -> "UIIntention":
        return _make_intention("as", self, role, domain="relation", **traits)

    def inside(self, container: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "inside", self, container, domain="navigation", **traits,
        )

    def at(self, location: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "at", self, location, domain="navigation", **traits,
        )

    def when(self, condition: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "when", condition, self, domain="condition", **traits,
        )

    def unless(self, condition: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "unless", condition, self, domain="condition", **traits,
        )

    def because(self, reason: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "because", self, reason, domain="consequence", **traits,
        )

    def therefore(self, consequence: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "therefore", self, consequence, domain="consequence", **traits,
        )

    def then(self, next_intention: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "then", self, next_intention, domain="sequence", **traits,
        )

    def otherwise(self, alternative: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "otherwise", self, alternative, domain="condition", **traits,
        )

    def does(self, action: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "does", self, action, domain="action", **traits,
        )

    def can(self, action: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "can", self, action, domain="permission", **traits,
        )

    def may(self, action: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "may", self, action, domain="permission", **traits,
        )

    def must(self, action: Any, **traits: Any) -> "UIIntention":
        return _make_intention(
            "must", self, action, domain="requirement", **traits,
        )

    def __getattr__(self, name: str) -> "IntentionContinuation":
        if name.startswith("_"):
            raise AttributeError(name)
        return IntentionContinuation(self, name.rstrip("_"))

    def __rshift__(self, other: Any) -> "UIIntention":
        return self.then(other)

    def __and__(self, other: Any) -> "UIIntention":
        return _make_intention("all_of", self, other, domain="logic")

    def __or__(self, other: Any) -> "UIIntention":
        return _make_intention("any_of", self, other, domain="logic")

    def __invert__(self) -> "UIIntention":
        return _make_intention("not", self, domain="logic")

    def __bool__(self) -> bool:
        raise TypeError(
            "UIIntention has no Python truth value; use all_of/any_of/not_"
        )


@dataclass(frozen=True)
class IntentionContinuation:
    """A forgiving ``thing.verb(...)`` phrase builder."""

    subject: UIIntention
    word: str

    def __call__(self, *arguments: Any, **traits: Any) -> UIIntention:
        entry = ABSTRACT_UI_VOCABULARY.get(self.word)
        domain = entry.domain if entry is not None else "open"
        return _make_intention(
            self.word, self.subject, *arguments, domain=domain, **traits,
        )


@dataclass(frozen=True)
class IntentionWord:
    """One callable word declared in the AbstractUI boneyard."""

    name: str
    spelling: str
    domain: str
    intention: str

    def __call__(self, *arguments: Any, **traits: Any) -> UIIntention:
        return _make_intention(
            self.spelling, *arguments, domain=self.domain, **traits,
        )

    def __repr__(self) -> str:
        return f"IntentionWord({self.name!r}, domain={self.domain!r})"


def _make_intention(
    spelling: str,
    *arguments: Any,
    domain: str = "open",
    **traits: Any,
) -> UIIntention:
    return UIIntention(
        str(spelling),
        tuple(arguments),
        tuple((str(name), value) for name, value in traits.items()),
        str(domain),
    )


ABSTRACT_UI_VOCABULARY: dict[str, IntentionWord] = {}
_DECLARATION_ORDER: list[str] = []


def declare_word(
    name: str,
    *,
    domain: str,
    intention: str,
    spelling: str | None = None,
) -> IntentionWord:
    """Declare and export one human-facing functional word."""

    if not name.isidentifier():
        raise ValueError(f"vocabulary name must be a Python identifier: {name!r}")
    if name in ABSTRACT_UI_VOCABULARY:
        raise ValueError(f"duplicate AbstractUI vocabulary word {name!r}")
    entry = IntentionWord(name, spelling or name.rstrip("_"), domain, intention)
    ABSTRACT_UI_VOCABULARY[name] = entry
    _DECLARATION_ORDER.append(name)
    globals()[name] = entry
    return entry


def _declare_domain(domain: str, intention: str, words: str) -> None:
    for name in words.split():
        declare_word(name, domain=domain, intention=intention)


def walk_intentions(root: Any) -> Iterator[UIIntention]:
    """Yield every intention nested in ``root`` in deterministic preorder."""

    if isinstance(root, UIIntention):
        yield root
        for argument in root.arguments:
            yield from walk_intentions(argument)
        for _, value in root.traits:
            yield from walk_intentions(value)
    elif isinstance(root, Mapping):
        for value in root.values():
            yield from walk_intentions(value)
    elif isinstance(root, (tuple, list)):
        for value in root:
            yield from walk_intentions(value)


def vocabulary_manifest() -> tuple[dict[str, str], ...]:
    """Return the whole aspirational surface in declaration order."""

    return tuple(
        {
            "name": entry.name,
            "spelling": entry.spelling,
            "domain": entry.domain,
            "intention": entry.intention,
        }
        for name in _DECLARATION_ORDER
        for entry in (ABSTRACT_UI_VOCABULARY[name],)
    )


def __getattr__(name: str) -> IntentionWord:
    """Autovivify an undeclared module-level word as open vocabulary.

    This lets exploratory authors write ``from ... import remembers`` before
    ``remembers`` has been admitted to the finite manifest. The returned word
    remains visibly ``domain='open'`` and is not inserted into the declared
    vocabulary or backend conformance surface.
    """

    if name.startswith("_") or not name.isidentifier():
        raise AttributeError(name)
    return IntentionWord(
        name, name.rstrip("_"), "open",
        "Undeclared human intention awaiting canonical classification.",
    )


# Roots, contexts, and projections -------------------------------------------------
_declare_domain(
    "context",
    "Name the interpretive world in which a user encounters a program.",
    "abstract_ui context context_root system_root interface_root world universe "
    "realm environment experience session perspective projection interpretation "
    "prosaic world_map introspection oop_archive document archive corpus",
)

# Existence and identity -----------------------------------------------------------
_declare_domain(
    "existence",
    "Say what exists, what kind of thing it is, and how it persists.",
    "exists entity identity alias name label description actor user player agent "
    "observer narrator object thing artifact concept instance kind type_ role "
    "archetype capability trait property attribute field member value state status "
    "presence absence enabled disabled visible hidden active inactive selected "
    "focused checked valid invalid available unavailable alive dormant destroyed",
)

# Containment, ownership, and hierarchy -------------------------------------------
_declare_domain(
    "containment",
    "Place identified things inside responsible scopes and interiors.",
    "contains contained_by owns owned_by belongs_to parent child ancestor descendant "
    "root container interior exterior boundary scope namespace module package class_ "
    "object_graph hierarchy tree collection group set category inventory slot layer "
    "stack section panel form fieldset room building facility district region zone "
    "filesystem directory file_node mount volume ownership spatial_placement "
    "custody inventoried",
)

# Navigation and spatial relationship ---------------------------------------------
_declare_domain(
    "navigation",
    "Describe where a user or entity may be and how movement proceeds.",
    "location place position coordinate address here there inside outside above below "
    "before after beside near far between overlaps touches faces north south east west "
    "up down left right forward backward entrance exit portal doorway gate threshold "
    "track path route road lane rail bridge corridor hallway stair elevator landmark "
    "destination origin waypoint map map_layer neighborhood island floor level "
    "placement preview_position snap_to grid_snap face_snap gimbal "
    "navigate move travel walk run ride fly enter leave return approach depart arrive "
    "visit explore inspect follow cross pass_through teleport scroll pan zoom",
)

# Actions and affordances ----------------------------------------------------------
_declare_domain(
    "action",
    "Name what an actor can intentionally do to a target.",
    "affordance action verb target instrument invoke call construct create open_ close "
    "start stop pause resume continue cancel confirm submit choose select deselect "
    "activate deactivate enable disable toggle set_ get read write edit change update "
    "add remove insert delete copy move_to link unlink connect disconnect attach detach "
    "weld unweld rename relocate transfer_ownership pick_up commit_placement "
    "cancel_placement shoot fire_projectile "
    "use take give drop carry equip unequip push pull turn press release drag resize "
    "focus blur hover point touch click double_click type_text speak listen ask answer "
    "examine search discover reveal hide unlock lock enter_action exit_action wait rest",
)

# Requirements, permissions, and consequences ------------------------------------
_declare_domain(
    "consequence",
    "Explain why an action is possible and what changes because it occurs.",
    "requires requirement prerequisite condition precondition guard permission allow "
    "deny forbid permit authorize unauthorize can may must should cannot cost price "
    "risk reward effect consequence result outcome causes caused_by produces consumes "
    "changes preserves prevents blocks unblocks grants revokes succeeds fails failure "
    "completion complete incomplete accepted rejected warning error notice evidence "
    "reason purpose intention expectation promise obligation opportunity",
)

# Logic, comparison, and query -----------------------------------------------------
_declare_domain(
    "logic",
    "Compose finite conditions without relying on host-language truthiness.",
    "all_of any_of one_of none_of not_ if_ then_ else_ otherwise when unless while_ "
    "until because therefore implies equivalent matches differs equal not_equal less "
    "greater at_least at_most includes excludes has lacks is_ is_not query filter "
    "find count first last each every some no unique optional default fallback",
)

# Time, sequence, scenes, and consequential flow ----------------------------------
_declare_domain(
    "sequence",
    "Order events, scenes, transitions, and user-visible changes through time.",
    "time moment duration interval instant clock tick frame step beat phase era history "
    "future past present sequence series order ordinal next previous begin end repeat "
    "loop cycle schedule delay timeout deadline transition scene cutscene chapter act_ "
    "episode encounter montage replay trace lifecycle spawn despawn enter_scene "
    "exit_scene branch choice checkpoint save restore rewind fast_forward",
)

# Controlled narrative and understandable sentences -------------------------------
_declare_domain(
    "narrative",
    "Construct plain deterministic language describing entities and consequences.",
    "narrative discourse utterance sentence clause subject predicate object_ phrase "
    "noun verb_phrase adjective adverb article singular plural tense present_tense "
    "past_tense future_tense active_voice passive_voice positive negative question "
    "answer_statement instruction explanation summary detail title heading caption "
    "tooltip help hint prompt dialogue speech narration sign inscription placard log "
    "message announce describe describe_location describe_state describe_action "
    "describe_requirement describe_consequence describe_transition say tell show",
)

# Program structure and OOP archival concepts -------------------------------------
_declare_domain(
    "program",
    "Interpret programming-language structure as inspectable user potential.",
    "program project library source file readme scratch_file source_file test_file "
    "human_artifact language program_construct declaration definition "
    "annotation decorator metadata schema vocabulary symbol reference binding import_ "
    "export function method parameter argument return_ constructor destructor "
    "inheritance base derived_class interface protocol implementation override overload "
    "public private protected static dynamic mutable immutable constant variable "
    "record struct tuple_ enum variant union generic template closure capture receiver "
    "this_ self_ caller callee call_path data_flow control_flow dependency consequence_flow "
    "process thread task coroutine event callback signal port input_ output_ inspect_ "
    "reflect trace_ debug profile measure breakpoint watch evaluate execute compile",
)

# Forms and basic HTML-shaped interaction -----------------------------------------
_declare_domain(
    "form",
    "Express ordinary prosaic interaction using containers and finite values.",
    "html body head div form_element label_element input_element output_element "
    "button_element select_element option_element textarea_element canvas_element "
    "text checkbox radio range_slider number_input text_input password_input file_input "
    "date_input time_input color_input choice_input choices placeholder required readonly "
    "minimum maximum step_size pattern unit format validation validity submit_action",
)

# CSS, selection, layout, and paint ------------------------------------------------
_declare_domain(
    "style",
    "Select interface identities and describe their contextual presentation.",
    "stylesheet style_rule selector selector_predicate combinator style_declaration style_property "
    "style_value style_variable style_function style_condition style_match "
    "computed_property cascade specificity style_origin layer_order important inherit "
    "initial unset class_group id_group tag_group state_group descendant_selector "
    "child_selector sibling_selector pseudo_state media_query container_query "
    "layout display block inline flex grid flow wrap gap align justify width height "
    "minimum_width maximum_width minimum_height maximum_height margin padding border "
    "positioning static_position relative_position absolute_position fixed_position "
    "sticky_position overflow clip scrollable z_index opacity color background "
    "font typography line_height transform transition_style animation_style",
)

# Images, sound, shaders, and world resources -------------------------------------
_declare_domain(
    "resource",
    "Refer to sensory, spatial, and executable resources without leaking host handles.",
    "resource asset image icon sprite atlas texture material mesh geometry terrain "
    "skybox additive subtractive "
    "tile tileset model camera lens light shadow particle shader vertex_shader "
    "fragment_shader compute_shader canvas surface framebuffer viewport scene_graph "
    "audio sound music voice ambience cue font_asset glyph text_run animation clip_asset "
    "video stream file_asset uri memory buffer manifest catalog theme palette",
)

# World metaphors and old map-game archetypes -------------------------------------
_declare_domain(
    "world",
    "Project program concepts into a half-text, half-image navigable world.",
    "world_role region_role building_role interior_role room_role object_role actor_role "
    "player_role guide_role merchant_role guardian_role terminal_role instrument_role "
    "door door_role portal_role track_role landmark_role sign_role dialogue_role item_role "
    "tool_role switch_role lever_role dial_role gauge_role console_role workshop_role "
    "office_role library_role archive_role laboratory_role machine_role vehicle_role "
    "terrain_role weather_role atmosphere_role population_role faction_role district_role "
    "enterable traversable discoverable usable readable speakable collectible movable "
    "gun projectile ammunition physics_ball",
)

# Events and hosts -----------------------------------------------------------------
_declare_domain(
    "event",
    "Correlate host gestures with contextual affordances and canonical actions.",
    "event_host presentation_host render_host browser_host sdl_host pygame_host "
    "terminal_host accessibility_host raw_event normalized_event gesture event_route "
    "event_target event_phase capture_phase target_phase bubble_phase consume propagate "
    "pointer pointer_move pointer_down pointer_up wheel key key_down key_up text_input_event "
    "gamepad button axis collision proximity command speech_command resize_event "
    "scale_change focus_event blur_event close_event frame_event dispatch subscribe "
    "unsubscribe listen_for emit publish receive handle map_gesture recognize_action",
)

# Accessibility and alternative routes -------------------------------------------
_declare_domain(
    "accessibility",
    "Guarantee understandable non-spatial routes through every authorized action.",
    "accessible accessibility_route semantic_role accessible_name accessible_description "
    "alternative_text reading_order focus_order tab_order landmark_navigation "
    "non_spatial_route keyboard_route speech_route text_route audio_description "
    "screen_reader magnification contrast reduced_motion captions transcript "
    "described_action described_state described_change equivalent_action "
    "reachable perceivable operable understandable robust",
)

# Data, quantities, and live bindings ---------------------------------------------
_declare_domain(
    "data",
    "Expose program values and correlations in forms a user can understand and change.",
    "data datum scalar quantity number integer decimal boolean string token bytes "
    "date duration_value coordinate_value vector matrix tensor table row column graph "
    "node edge data_key pair mapping list_value sequence_value set_value option_value "
    "null unknown live snapshot observed derived_value computed cached stale synchronized "
    "value_binding action_binding event_binding resource_binding identity_binding "
    "reads_from writes_to synchronizes_with reflects mirrors controls observes",
)


__all__ = [
    "ABSTRACT_UI_VOCABULARY",
    "ABSTRACT_UI_VOCABULARY_VERSION",
    "IntentionContinuation",
    "IntentionWord",
    "UIIntention",
    "declare_word",
    "vocabulary_manifest",
    "walk_intentions",
    *_DECLARATION_ORDER,
]
