from src.compiler.abstract_ui_navigation import (
    MAX_GRID_CELLS,
    compile_astar_navigation_kernel,
    navigation_model,
)
from src.compiler.abstract_ui_entities import spawn_world_player


def test_astar_kernel_is_real_webassembly_with_published_path_abi():
    kernel = compile_astar_navigation_kernel()
    data = kernel.to_data()

    assert kernel.binary.startswith(b"\0asm")
    assert data["format"] == "webassembly"
    assert data["entrypoint"] == "navigation_pathfind"
    assert data["abi"]["maximum_grid_cells"] == MAX_GRID_CELLS
    assert data["algorithm"] == {
        "family": "a-star",
        "connectivity": 8,
        "heuristic": "octile",
        "diagonal_corner_cutting": False,
    }


def test_navigation_assigns_a_hot_swappable_kernel_per_entity():
    model = navigation_model(("entity:player", "entity:npc"))

    assert model["assignments"] == {
        "entity:player": model["default_kernel"],
        "entity:npc": model["default_kernel"],
    }
    assert model["hot_swap"]["scope"] == "per-entity"
    assert model["traversal"]["orientation_spline"].startswith("shortest-arc-quaternion")
    assert model["traversal"]["speed"] == 5.2
    assert model["traversal"]["collision_validation"] == "continuous-swept-clearance"
    assert model["traversal_chart"]["inverse"] == "piecewise-linear"
    assert model["execution"] == {
        "planning_host": "dedicated-worker",
        "graphics_host": "animation-frame-main-thread",
        "handoff": "structured-clone-certified-route-samples",
        "main_thread_work": "route-installation-and-interpolation-only",
    }


def test_world_player_is_not_a_mouse_position_or_follower_entity():
    mezzanine = spawn_world_player(system_root="world")

    assert len(mezzanine.entities) == 1
    player = mezzanine.entities[0]
    assert player.controller.kind == "world-player"
    assert player.controller.source == "game.controls"
    assert player.pose.coordinate_space == "data-world"
