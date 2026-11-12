"""An authored storybook class for the existing Living Data map.

Introspection is non-executing. The optional HTML export uses the existing
Living Data compiler and renderer, including its normal compilation cost.
No local game loop, movement rules, shader, or physics implementation lives here.
"""

from src.compiler.abstract_ui_introspection import build_introspective_world
from src.compiler.abstract_ui_vocabulary import describe


class LanternCottage:
    """A warm threshold at the edge of the paper forest."""

    honey_window: str = "Amber lamplight falls across a window shaped like a honeycomb."
    copper_kettle: str = "A copper kettle hums beside a hearth full of sleeping embers."
    little_red_door: str = "A red door opens toward a path stitched with fallen stars."


class WhisperingWood:
    """Silver leaves remember every traveller's name."""

    cottage_path: LanternCottage
    silver_leaves: str = "Silver leaves whisper above a carpet of bluebells."
    fox_lantern: str = "A fox carries a lantern between the roots of an ancient oak."
    mushroom_circle: str = "Spotted mushrooms mark a circle where the rain never falls."


class MoonBridge:
    """A stone crescent above a river of reflected constellations."""

    woodland_path: WhisperingWood
    moonstone_arch: str = "Moonstones glow along the arch of a narrow bridge."
    starry_river: str = "The river carries reflections of stars that have not risen yet."
    sleeping_troll: str = "Beneath the bridge, a small troll sleeps under a patchwork quilt."


class ClockworkTower:
    """A crooked clock keeps the village's unfinished stories."""

    bridge_path: MoonBridge
    brass_stair: str = "A spiral stair winds past brass birds with folded wings."
    midnight_bell: str = "The midnight bell waits for the last page to turn."
    cloud_balcony: str = "Beyond the balcony, clouds drift like folded paper boats."


class StorybookEnvironment:
    """The Lantern Road, a small storybook world described by real class edges.

    The type-valued annotations are authoritative references. Introspection
    turns them into ordinary Living Data class buildings and member portals.
    A reference is a graph edge, not a claim of a physically traversable doorway.
    """

    # Keep evaluated type objects: the existing introspector deliberately does
    # not evaluate string annotations or arbitrary forward-reference source.
    lantern_cottage: LanternCottage
    whispering_wood: WhisperingWood
    moon_bridge: MoonBridge
    clockwork_tower: ClockworkTower
    opening_page: str = "Once upon a time, a lantern lit the road from a cottage to the moon."
    closing_page: str = "At the clockwork tower, every traveller may leave a new ending."


ROOM_PALETTE = {
    "field": ("illuminated page", "storybook alcove", "memory garden"),
    "property": ("enchanted window",),
    "method": ("storyteller's hearth",),
    "constructor": ("once-upon-a-time gate",),
    "nested_class": ("chapter doorway", "lantern path"),
}


def build_storybook_world(*, seed="lantern-road"):
    """Use canonical class introspection, preserving its identities and edges."""
    subjects = (StorybookEnvironment, LanternCottage, WhisperingWood,
                MoonBridge, ClockworkTower)
    adornments = {
        name: (describe(value),)
        for subject in subjects for name, value in vars(subject).items()
        if not name.startswith("_") and isinstance(value, str)
    }
    return build_introspective_world(
        StorybookEnvironment, depth_down=2, seed=seed,
        room_palette=ROOM_PALETTE,
        building_palette=("storybook house", "lantern lodge", "enchanted archive"),
        adornments=adornments,
    )


def project_storybook_map():
    """Compile the native static Living Data map without game startup."""
    from src.compiler.abstract_ui_div_map import project_world_to_div_map
    return project_world_to_div_map(
        build_storybook_world(), title="The Lantern Road - a Living Data storybook", static_only=True,
    )



def storybook_graph():
    """Canonical compact graph for the existing query/cursor REST endpoints."""
    from src.compiler.abstract_ui_div_map import world_div_map_graph
    return world_div_map_graph(build_storybook_world())
