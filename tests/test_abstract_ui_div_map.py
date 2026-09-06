"""Tests for the div-grid and self-inspecting JavaScript projection."""

from dataclasses import replace
import json
import re

import pytest

from src.compiler.abstract_ui import AbstractUI, AbstractUIInteraction, AbstractUIPacket
from src.compiler.abstract_ui_div_map import (
    ABSTRACT_UI_DIV_MAP_VERSION,
    DIV_MAP_JAVASCRIPT,
    DIV_MAP_PALETTE,
    project_class_to_div_map,
)


class ControlPanel:
    gain: float
    enabled: bool

    def engage(self, amount: float) -> bool:
        return amount > 0


def _projection():
    return project_class_to_div_map(ControlPanel, seed="div-map-test")


def test_projection_preserves_world_building_room_identity_and_grid_positions():
    projection = _projection()
    assert isinstance(projection, AbstractUI)
    assert projection.model["schema"] == ABSTRACT_UI_DIV_MAP_VERSION
    region = projection.model["regions"][0]
    building = region["buildings"][0]
    assert building["name"] == "ControlPanel"
    assert [room["name"] for room in building["rooms"]] == [
        "gain", "enabled", "engage",
    ]
    assert len({
        (room["position"]["column"], room["position"]["row"])
        for room in building["rooms"]
    }) == 3
    assert all(room["dependencies"] == [{
        "relationship": "contained-by", "target": building["identity"],
    }] for room in building["rooms"])
    assert all(room["interaction"] == {
        "type": "inspect", "destination": room["identity"],
    } for room in building["rooms"])


def test_page_semantic_map_surface_uses_divs_and_css_grid():
    page = _projection().html
    assert '<div id="map-root"></div>' in page
    assert ".rooms{display:grid" in page
    assert "grid-column:" in DIV_MAP_JAVASCRIPT
    assert "<canvas" not in page
    assert "<svg" not in page
    assert "<button" not in page


def test_source_surface_is_location_scoped_and_action_registry_is_collapsed():
    projection = _projection()
    contract = projection.model["source_scope"]

    assert contract == {
        "schema": "abstract-ui-location-scoped-source-v0",
        "closure_kinds": ["region", "building", "room"],
        "visible": ["current-closure-opening", "unentered-closure-openings"],
        "entered_non_current": "unmounted",
        "update": "player-world-containment-transition",
    }
    assert "const sourceScopeRuntime" in projection.script
    assert "function semanticSourceClosures" in projection.script
    assert "function playerSourceClosure" in projection.script
    assert "function updateSourceScopeForPlayer" in projection.script
    assert "updateSourceScopeForPlayer();" in projection.script
    assert 'kind: "source-scope-opening"' in projection.script
    assert 'card.dataset.closureState = isCurrent ? "current" : "unentered"' in projection.script
    assert "SELF_SOURCE.split" not in projection.script
    assert 'kind: "javascript-source-line"' not in projection.script
    assert 'document.createElement("details")' in projection.script
    assert 'actionSummary.textContent = "Registered actions"' in projection.script
    assert "actionRegistry.open" not in projection.script
    assert ".source-scope-opening" in projection.css
    assert ".action-registry>summary" in projection.css


def test_system_root_places_shader_viewport_above_document_regions():
    projection = _projection()
    viewer = projection.model["viewer"]
    assert viewer["kind"] == "shader-viewport"
    assert viewer["system_root"] == projection.model["identity"]
    assert [stage["operation"] for stage in viewer["fragment_chain"]] == [
        "resolve-palette", "extrude-document-geometry", "first-person-lighting",
    ]
    assert "world.append(renderShaderViewport());" in projection.script
    assert projection.script.index("world.append(renderShaderViewport());") < projection.script.index(
        "world.append(regions);"
    )


def test_viewport_preserves_default_shader_and_exposes_selectable_programs():
    projection = _projection()
    assert 'identity: "living-map-default"' in projection.script
    assert "const VIEWPORT_SHADER_CHOICES" in projection.script
    assert 'shaderSelect.setAttribute("aria-label", "Viewport shader")' in projection.script
    assert 'activateViewportShader(shaderSelect.value);' in projection.script
    assert "activateViewportShader(VIEWPORT_DEFAULT_SHADER);" in projection.script
    assert '["uResolution", "uCameraPosition", "uCameraFacing"]' in projection.script
    assert "viewer-shader-select" in projection.css
    for choice in projection.model["viewer"]["shader_choices"]:
        assert choice["vertex_source"].startswith("#version 300 es")
        assert choice["fragment_source"].startswith("#version 300 es")
        assert choice["adapter"]
    choices = projection.model["viewer"]["shader_choices"]
    if choices:
        assert projection.model["viewer"]["default_shader"] == choices[0]["identity"]
        catalog = projection.model["material_catalog"]
        assert catalog["identity"] == "pluck-material-database"
        assert catalog["record_count"] == len(catalog["records"])
        assert catalog["record_count"] > 0
        assert catalog["chunk_strides"] == {
            "pbr": 16, "phong": 8, "enamel": 8, "texture_stack": 16,
        }
        assert catalog["surface_policy"]["profile"] == "matte-world-base"
        assert min(record["pbr"][3] for record in catalog["records"]) >= 0.72
        assert "material.texture_stack" in projection.script


def test_viewport_can_hide_the_document_behind_a_full_page_shader():
    projection = _projection()

    assert 'shaderOnlyToggle.textContent = "Shader only"' in projection.script
    assert 'setShaderOnlyMode(!shaderViewer.shaderOnly);' in projection.script
    assert 'document.body.classList.toggle("shader-only"' in projection.script
    assert 'event.code === "Escape" && shaderViewer.shaderOnly' in projection.script
    assert 'button.textContent = shaderViewer.shaderOnly ? "Exit shader only"' in projection.script
    assert "body.shader-only .inspector" in projection.css
    assert "body.shader-only .world>*:not(.viewer-port)" in projection.css
    assert "body.shader-only .viewer-surface{width:100vw;height:100vh" in projection.css
    assert ".viewer-shader-only-toggle{display:block;margin:4px 0 0 auto" in projection.css
    assert "right:14px;top:auto;" in projection.css
    assert "bottom:14px;margin:0" in projection.css
    assert "viewerTelemetry.append(shaderOnlyToggle);" in projection.script
    assert "shaderViewer.shaderOnly ? shaderViewer.element : shaderViewer.telemetry" in projection.script


def test_top_down_map_auto_locates_through_hot_swappable_assembly_navigation():
    projection = _projection()
    navigation = projection.model["navigation"]

    assert navigation["kernels"][0]["binary_base64"]
    assert navigation["kernels"][0]["entrypoint"] == "navigation_pathfind"
    assert navigation["assignments"][projection.model["viewer"]["control_policy"]["actor"]]
    assert 'event.type === "click" && event.target.closest(".regions")' in projection.script
    assert "void autoLocateFromMapClick(event);" in projection.script
    assert "function navigationPointBlocked" in projection.script
    assert "function navigationSpline" in projection.script
    assert "function navigationWorldToTraversal" in projection.script
    assert "function navigationTraversalToWorld" in projection.script
    assert "function navigationSegmentClear" in projection.script
    assert "navigationSegmentClear(curve[index], point, spec)" in projection.script
    assert "function navigationPolyline" in projection.script
    assert "blocked[rejected] = 1" in projection.script
    assert "samples = navigationPolyline(waypoints)" in projection.script
    assert 'dataset.collisionCertified = String(collisionCertified)' in projection.script
    assert "spline failed continuous wall-clearance certification" in projection.script
    assert "function quaternionSlerp" in projection.script
    assert "function assignNavigationKernel" in projection.script
    assert "function navigationPlannerWorkerSource" in projection.script
    assert "const definitions = [navigationBytes, boxYawDegrees, navigationAxisTransform" in projection.script
    assert "function initializeNavigationPlannerWorker" in projection.script
    assert 'requestNavigationWorker("plan"' in projection.script
    assert 'dataset.plannerThread = "dedicated-worker"' in projection.script
    assert 'new Worker(navigationRuntime.workerUrl)' in projection.script
    cancel_source = projection.script.split("function cancelEntityNavigation", 1)[1].split(
        "function navigationGridSpec", 1
    )[0]
    assert "rebuildPortableSceneMesh" not in cancel_source
    assert "globalThis.abstractUINavigation" in projection.script
    assert "updateEntityNavigation(dt);" in projection.script
    assert "cancelEntityNavigation(viewportControls.policy.actor);" in projection.script
    assert 'event.target.closest("[data-node-id]")' in projection.script
    assert "const clickedWorldPoint = documentToWorldPoint(localClick);" in projection.script
    assert "navigationEndpointCandidates(clickedGeometry, clickedWorldPoint)" in projection.script
    assert "function installNavigationRouteOverlay" in projection.script
    assert "function updateNavigationRouteOverlay" in projection.script
    assert "function navigationRouteDocumentPoints" in projection.script
    assert "const subdivide = (left, right, leftDocument, rightDocument, depth)" in projection.script
    assert "deviation > 0.35 || projectedLength > 10" in projection.script
    assert 'group.dataset.projection = "adaptive-nonlinear-context-transform"' in projection.script
    assert "function buildNavigationRouteMesh" in projection.script
    assert "navigationTraversalToWorld(sample.position, route.spec)" in projection.script
    assert "const routes = buildNavigationRouteMesh();" in projection.script
    assert 'overlay.progress.setAttribute("stroke-dashoffset"' in projection.script
    assert "clearNavigationRouteOverlay(entityIdentity);" in projection.script
    assert ".navigation-route-progress" in projection.css


def test_map_clicks_stack_waypoints_with_presence_hook_pauses():
    projection = _projection()
    waypoint_contract = projection.model["navigation"]["waypoints"]

    assert waypoint_contract == {
        "click_policy": "append-per-entity",
        "presence_pause_seconds": 0.85,
        "presence_event": "abstract-ui:navigation-presence",
        "presence_hooks": "minimum-pause-and-registered-async-holds",
        "planning": "sequential-from-certified-arrival-pose",
    }
    script = projection.script
    assert "waypointQueues: new Map()" in script
    assert "function enqueueEntityWaypoint" in script
    assert "function navigationEndpointCandidates" in script
    assert "function navigationOpeningPoint" in script
    assert "for (let candidateIndex = 0; candidateIndex < waypoint.candidates.length" in script
    assert "no navigable endpoint candidate" in script
    assert "function advanceNavigationWaypointQueue" in script
    assert "function beginNavigationPresence" in script
    assert "function updateNavigationPresencePauses" in script
    assert "const waypoint = enqueueEntityWaypoint(actor, destination, candidates);" in script
    assert "beginNavigationPresence(entityIdentity, route);" in script
    assert '"abstract-ui:navigation-presence"' in script
    assert "Promise.allSettled(holds)" in script
    assert "waypoints.length === 1" in script
    assert "enqueueWaypoint: enqueueEntityWaypoint" in script
    assert "onPresence(callback)" in script
    assert "queuedWaypoints(entityIdentity)" in script


def test_document_world_coordinates_resync_without_scroll_dependent_positioning():
    projection = _projection()
    contract = projection.model["coordinate_sync"]

    assert contract["scroll_dependency"] == "none"
    assert contract["allowed_distortion"] == "nonlinear-context-container-scale-only"
    assert contract["mapping"] == "continuous-hierarchy-landmark-piecewise-affine"
    assert contract["document_authority"] == "rendered-border-frames-relative-to-map-root"
    assert "function elementOffsetWithin" in projection.script
    assert "function viewportToElementPoint" in projection.script
    assert "getBoundingClientRect()" in projection.script
    assert "function resyncDocumentWorldMap" in projection.script
    assert "function worldToDocumentPoint" in projection.script
    assert "const axisProfile = (frame, axis)" in projection.script
    assert 'child.box.parent_identity === frame.identity' in projection.script
    assert 'containing.xProfile, "world", "document"' in projection.script
    assert 'containing.xProfile, "document", "world"' in projection.script
    assert "function documentToWorldPoint" in projection.script
    assert "documentWorldSync.resizeObserver" in projection.script
    assert 'regions.append(entityLayer)' in projection.script
    assert 'screen = worldToDocumentPoint([state.worldPosition[0], state.worldPosition[2]])' in projection.script
    assert 'root.dataset.coordinateSync = "rendered-border-frames"' in projection.script
def test_navigation_and_physics_handoff_uses_control_generations():
    projection = _projection()
    script = projection.script
    assert "snapshotControlGeneration" in script
    assert "body.controlGeneration += 1" in script
    assert "viewportControls.position = [proposed[0],support.y,proposed[1]]" in script
    assert "physicsHasControl ? body.position[0]" not in script
    layout = next(channel for channel in projection.model["loop_deployment"]["channels"]
                  if "world.body-pose" in channel["fields"])["record_layout"]
    assert layout[-1] == "control.generation"


def test_document_grid_extrudes_into_shader_geometry_using_shared_palette():
    projection = _projection()
    geometry = projection.model["document_geometry"]
    kinds = [box["kind"] for box in geometry["boxes"]]
    assert kinds[:3] == ["world-envelope", "courtyard", "building"]
    assert kinds.count("room") == 3
    palette_colors = {name: color.value for name, color in DIV_MAP_PALETTE.colors}
    assert projection.model["appearance"]["colors"] == palette_colors
    assert all(f"--{name}:{value}" in projection.css for name, value in palette_colors.items())
    assert "colors[box.palette_role]" in projection.script
    assert geometry["boundary_semantics"]["source"] == "dom-border"
    assert geometry["boundary_semantics"]["height_parameter"] == "boxes[].height"
    assert geometry["boundary_semantics"]["future_composition"] == (
        "boundary-union-minus-openings"
    )
    assert geometry["boundary_semantics"]["floor"] == "mandatory-slab"
    assert geometry["boundary_semantics"]["interior"] == "hollow"
    assert geometry["boundary_semantics"]["ceiling"]["absolute_maximum"] == 4.0
    assert all(
        box["openings"] for box in geometry["boxes"]
        if box["kind"] in {"courtyard", "building", "room"}
    )
    assert geometry["boxes"][1]["openings"][0]["kind"] == "gate"
    assert geometry["hierarchy_space"]["policy"] == "nonlinear-containment-distance-v0"


def test_pointer_motion_targets_ui_but_does_not_own_the_world_player_pose():
    projection = _projection()
    assert "latestControlInput?.facing" in projection.script
    assert 'else if (player?.worldPosition)' in projection.script
    assert "cameraFacing = [player.facing[0], -0.18, player.facing[1]]" in projection.script
    assert "cameraFacing = [native.facing[0]" not in projection.script
    assert 'shaderViewer.active = Boolean(bounds && event.clientX >= bounds.left' in projection.script
    assert 'getContext("webgl2"' in projection.script


def test_highlighted_viewport_captures_keyboard_mouse_and_first_gamepad():
    projection = _projection()
    controls = projection.model["viewer"]["control_policy"]
    player = next(
        entity for entity in projection.model["entity_mezzanine"]["entities"]
        if entity["controller"]["kind"] == "world-player"
    )
    assert controls["actor"] == player["identity"]
    assert controls["activation"] == "highlight"
    assert controls["captures"] == ["keyboard", "pointer", "gamepad"]
    assert 'shaderViewer.canvas?.requestPointerLock?.()' in projection.script
    assert 'document.addEventListener("keydown", viewportKeyboardInput);' in projection.script
    assert 'navigator.getGamepads' in projection.script
    assert 'viewportInputValue("move-forward", gamepad)' in projection.script
    assert 'viewportInputValue("run", gamepad)' in projection.script
    assert 'viewportInputValue("jump", gamepad)' in projection.script
    assert "function requestViewportJump" in projection.script
    assert 'type: "impulse", identity' in projection.script
    assert "function resolvePlayerHorizontalMotion(previous, target)" in projection.script
    assert 'colliderSides: new Map()' in projection.script
    assert "function resolvePlayerVerticalSupport(previousY,nextY,verticalVelocity)" in projection.script
    assert 'stateLoopRuntime.worker.postMessage({type:"support",identity,y:support.y})' in projection.script
    assert 'if (!controlFocus.dialogue) requestViewportControls();' in projection.script
    assert 'viewportBinding("primary-action", "gamepad:button-0")' in projection.script
    assert 'beginViewportPrimary(null,"gamepad")' in projection.script
    assert "function seedViewportControlPose()" in projection.script
    assert 'shaderViewer.geometry.find(box => box.kind === "room")' in projection.script
    assert "shaderViewer.inhabitedCameraPosition" in projection.script
    assert projection.model["viewer"]["camera"]["embodiment_scale"] == 0.25
    assert projection.model["viewer"]["camera"]["eye_height"] == 0.2875
    assert projection.model["viewer"]["camera"]["position"][1] == 0.2875
    assert "requestRepresentationTransition(viewportControls.position)" in projection.script
    assert projection.model["document_geometry"]["representation_boundary"][
        "crossing_operation"
    ] == "switch-map-representation"


def test_viewport_renders_live_device_monitor_from_abstract_bindings():
    projection = _projection()
    monitor = projection.model["viewer"]["device_monitor"]
    assert [group["device"] for group in monitor["groups"]] == [
        "pointer", "keyboard", "gamepad",
    ]
    assert "function renderDeviceMonitor(monitor)" in projection.script
    assert 'light.dataset.controlSource = signal.source;' in projection.script
    assert 'element.classList.toggle("active", value >= 0.12);' in projection.script
    assert '.device-signal.active{' in projection.css


def test_viewport_reserves_distinct_user_dynamics_and_world_physics_lanes():
    projection = _projection()
    space = projection.model["viewer"]["dynamics_space"]
    assert [lane["kind"] for lane in space["lanes"]] == [
        "user-dynamics", "world-physics",
    ]
    assert [channel["status"] for channel in space["lanes"][1]["channels"]] == [
        "bound", "bound", "bound", "bound",
    ]
    assert space["lanes"][1]["channels"][1]["scope"] == (
        "one-selected-semantic-wall-plane"
    )
    assert "function renderDynamicsSpace(space)" in projection.script
    assert 'data-dynamics-value="world-physics"' in projection.script
    assert '.dynamics-space{' in projection.css


def test_hotbar_projects_inventory_slots_and_equips_ball_gun_by_default():
    projection = _projection()
    inventory = projection.model["inventory"]
    hotbar = projection.model["hotbar"]
    assert inventory["items"][0]["name"] == "Form tool"
    assert inventory["items"][0]["slot"] == 1
    assert inventory["items"][6]["name"] == "Ball gun"
    assert inventory["active_tool"]["item"] == inventory["items"][6]["identity"]
    assert len(hotbar["slots"]) == 10
    assert hotbar["slots"][0]["item"] == inventory["items"][0]["identity"]
    assert hotbar["slots"][-1]["key"] == "Digit0"
    assert hotbar["active_slot"] == 7
    assert "function renderHotbar(hotbar)" in projection.script
    assert 'selectHotbarSlot(hotbarSlot[1] === "0" ? 10' in projection.script
    tool = projection.model["tools"][0]
    assert tool["identity"] == inventory["items"][0]["entity"]
    assert [hook["action"] for hook in tool["hooks"]] == [
        "primary-action", "secondary-action",
    ]


def test_placement_tool_preserves_custody_supports_snapping_and_counts_openings():
    projection = _projection()
    placement = projection.model["placement"]
    inventory = projection.model["inventory"]
    assert placement["semantic_owner_policy"] == "preserve-unless-explicit-transfer"
    assert {"grid", "object-face", "object-center"} <= set(placement["snap_modes"])
    assert [(item["opening_kind"], item["stock"]) for item in placement["recipes"]] == [
        ("door", 8), ("window", 12), ("gate", 4), ("portal", 12),
    ]
    assert placement["portal_contract"]["primary_action_role"] == "in"
    assert placement["portal_contract"]["secondary_action_role"] == "out"
    assert placement["portal_contract"]["backing"] == "probabilistic-tube-graph"
    assert placement["portal_contract"]["distribution"] == "normalized-spatial-gaussian"
    assert placement["portal_contract"]["path_model"] == "relaxed-quaternion-cubic"
    assert [item["quantity"] for item in inventory["items"][2:6]] == [8, 12, 4, 12]
    script = projection.script
    assert "function takeFocusedObjectToInventory(targetIdentity)" in script
    assert 'semantic_owner: semanticOwner, source_container:' in script
    assert "function placementPreviewCenter(box)" in script
    assert 'operation: "place-subtractive-opening"' in script
    assert 'role: "opening", opening_kind:' in script
    assert "function renderPlacementPanel()" in script


def test_portal_recipe_deploys_probabilistic_mesh_graph_and_traverses_tubes():
    projection = _projection()
    script = projection.script

    assert projection.model["appearance"]["colors"]["portal-in"] == "#42a5ff"
    assert projection.model["appearance"]["colors"]["portal-out"] == "#ff8a3d"
    assert 'placePortalSplat("in")' in script
    assert 'placePortalSplat("out")' in script
    assert "function raySceneTriangle(" in script
    assert "function portalTriangleMembership(" in script
    assert 'backing: "probabilistic-tube-graph"' in script
    assert 'distribution: "normalized-spatial-gaussian"' in script
    assert 'intermediary_manifold: "directed-tube-edge"' in script
    assert 'placement_kind: "manifold"' in script
    assert 'opening.port_role === "out" ? "portal-out" : "portal-in"' in script
    assert 'role = opening.port_role === "out" ? "portal-out-surface"' in script
    assert 'const priority = ["portal-in", "portal-out"]' in script
    assert "const material = colorIndex < 2 ? null" in script
    assert "function rebuildPortalGraph()" in script
    assert "function buildPortalTubeMesh()" in script
    assert 'path_model: "relaxed-quaternion-cubic"' in script
    assert "function quaternionBetween(" in script
    assert "function portalCurveFrame(" in script
    assert "function portalTubeRadius(" in script
    assert "function activePortalPlacementProfile(" in script
    assert 'aperture_class: profile.aperture_class || "person"' in script
    assert "tube_throat_radius: 0.045 * tubeScale" in script
    assert "portalApertureClass(target) === portalApertureClass(source)" in script
    assert "transit.entry_facing" in script
    assert "const suctionProgress" in script
    assert "const touchedBlockedPlane" in script
    assert "portalRuntime.transits.has(viewportControls.policy.actor)" in script
    assert "function choosePortalGraphEdge(" in script
    assert "function traversePortalBody(" in script
    assert 'actionEdges.register(source.identity, "traverse-portal-tube"' in script
    assert "updatePortalTraversals();" in script
    assert "portal_splats: portalRuntime.splats" in script
    assert "function migrateLegacyPortalOpenings()" in script
    assert 'operation: "legacy-opening-to-radial-chart"' in script
    assert "const migratedPortals = migrateLegacyPortalOpenings();" in script


def test_mobile_viewport_keeps_sensor_telemetry_but_navigates_only_by_touch():
    projection = _projection()
    script = projection.script
    css = projection.css

    assert "function requestMobileMotionControls()" in script
    assert 'orientationAPI.requestPermission()' in script
    assert 'motionAPI.requestPermission()' in script
    assert 'window.addEventListener("deviceorientation"' in script
    assert 'window.addEventListener("devicemotion"' in script
    assert "function bindMobileStick(" in script
    assert 'stick("move", "move")' in script
    assert 'stick("look", "look")' in script
    assert 'routeActiveToolHook(action)' in script
    assert "mobileControlState.tilt[0]" not in script
    assert "mobileControlState.touchLookSpeed" in script
    assert "telemetry only; movement remains on the touch stick" in script
    assert "@media (pointer:coarse),(max-width:760px)" in css
    assert "body.shader-only .mobile-controls" in css
    assert ".inspector{display:none}" in css


def test_player_can_equip_gun_and_spawn_wasm_physics_spheres():
    projection = _projection()
    system = projection.model["projectiles"]
    player = projection.model["entity_mezzanine"]["entities"][0]
    inventory = projection.model["inventory"]
    assert system["archetype"]["geometry"]["kind"] == "sphere"
    assert system["archetype"]["physics"]["program"] == (
        "world.physics.compiled-sympy-wasm"
    )
    assert "fire-projectile" in player["capabilities"]
    assert inventory["items"][6]["name"] == "Ball gun"
    assert inventory["items"][6]["slot"] == 7
    assert inventory["items"][7]["name"] == "Physics balls"
    assert inventory["items"][7]["quantity"] == 128
    script = projection.script
    assert "function firePhysicsBall(exitVelocityScale=1)" in script
    assert "function updatePhysicsBalls(dt)" in script
    assert "runCompiledPhysicsState(state.position, state.velocity" in script
    assert 'box.geometry_mode === "sphere"' in script
    assert 'primitive: "sphere"' in script
    assert "updatePhysicsBalls(dt);" in script
    assert 'archetype: "physics-ball-entity"' in script
    assert 'controller: {kind: "compiled-projectile-physics"' in script
    assert 'div("entity-sprite projectile-entity-marker")' in script
    assert "entityState.set(identity, entityRuntime);" in script
    assert "state.entityRuntime.worldPosition = [...state.position];" in script
    assert "function bounceProjectile(state,radius)" in script
    assert "function resolveProjectilePairs(radius)" in script
    assert "playToyImpact(speed,state.position)" in script
    assert "function sleepProjectilePhysics(state,reason=" in script
    assert "function wakeProjectilePhysics(state,reason=" in script
    assert 'state.record.physics_membership="dropped"' in script
    assert 'stateLoopRuntime.worker.postMessage({type:"remove",identity:state.identity})' in script
    assert 'registerProjectilePhysicsMembership(state);' in script
    assert 'wakeProjectilePhysics(left,"projectile-collision")' in script
    assert 'wakeProjectilePhysics(state,"player-collision")' in script
    assert 'wakeSleepingProjectiles(`physics-field-change:${name}`)' in script
    assert 'model.projectiles.archetype.physics.linear_drag=value' in script
    assert 'message.type === "engine-state"' in script
    assert 'stateLoopRuntime.engineStage.replaceAll("-"," ")' in script
    assert 'sleepProjectilePhysics(state,"slow-enough")' in script
    assert 'expireProjectile(state,"settled-to-pickup")' not in script
    assert 'state.box.kind = "projectile-pickup"' in script
    assert "function collectProjectilePickup(state)" in script
    assert "collectNearbyProjectile(false);" in script
    assert 'hook.operation === "collect-projectile-pickup"' in script


def test_compiler_deploys_physics_and_graphics_on_independent_clocks():
    projection = _projection()
    deployment = projection.model["loop_deployment"]
    placements = {item["loop"]["clock"]: item for item in deployment["placements"]}
    assert placements["fixed-step"]["execution_host"].startswith("worker:")
    assert placements["animation-frame"]["execution_host"] == "main"
    assert deployment["scheduler"]["backpressure"] == "recycled-latest-snapshot"
    assert deployment["scheduler"]["snapshot_memory"] == (
        "preallocated-transferable-triple-buffer"
    )
    assert "new Worker(stateLoopRuntime.workerUrl)" in projection.script
    assert 'type: "recycle"' in projection.script
    assert "reservePhysicsSnapshotSlot(identity)" in projection.script
    assert "requestAnimationFrame(runEntityCycle)" in projection.script
    provenance = projection.model["emission_provenance"]
    assert any(item["target"] == "javascript-worker" and
               item["authority"] == "compiler-emitted"
               for item in provenance["sections"])
    assert "webgl-presentation-adapter" in provenance["remaining_bespoke_surface"]


def test_outer_envelope_is_persistent_world_map_skybox_without_collision_ceiling():
    projection = _projection()
    geometry = projection.model["document_geometry"]
    envelope = geometry["boxes"][0]
    assert envelope["kind"] == "world-envelope"
    assert envelope["height"] == 12.0
    assert envelope["skybox"]["always_visible"] is True
    assert envelope["skybox"]["semantic_role"] == "parent-world-map-boundary"
    assert geometry["representation_boundary"]["visualization"] == (
        "persistent-outer-skybox-world-map-horizon"
    )
    assert 'box.kind !== "world-envelope"' in projection.script


def test_active_tool_hooks_open_model_driven_aesthetic_dialogue():
    projection = _projection()
    script = projection.script
    assert "function routeActiveToolHook(action, position = null)" in script
    assert 'beginViewportPrimary(null,"pointer")' in script
    assert 'beginViewportSecondary([x,y],"pointer")' in script
    assert "function openToolDialogue(tool, targetIdentity)" in script
    assert 'root.setAttribute("role", "dialog");' in script
    assert 'input.dataset.aestheticProperty = property.name;' in script
    assert "function applyAestheticPreset(presetIdentity)" in script
    assert 'document.addEventListener("input", event =>' in script
    assert "claimDialogueFocus(tool.dialogue.identity)" in script
    assert "releaseDialogueFocus(identity)" in script


def test_aesthetic_edits_share_geometry_mesh_and_dom_appearance():
    projection = _projection()
    script = projection.script
    assert "function applyAestheticValue(targetIdentity, name, rawValue)" in script
    assert 'if (name === "height") box.height' in script
    assert "rebuildPortableSceneMesh();" in script
    assert "box.appearance?.face_color" in script
    assert 'element.style.setProperty("--wall-thickness"' in script
    assert 'element.style.setProperty("--object-radius"' in script
    assert 'element.style.borderColor = "var(--object-wall)";' in script
    assert ".tool-dialogue{position:fixed;z-index:70" in projection.css


def test_captured_devices_route_between_game_projected_pointer_and_dialogue():
    projection = _projection()
    focus = projection.model["control_focus"]
    assert focus["routes"] == ["game", "projected-pointer", "dialogue"]
    assert focus["switch_action"] == "secondary-action"
    script = projection.script
    assert "function toggleControlFocus()" in script
    assert "function claimDialogueFocus(dialogueIdentity)" in script
    assert "function releaseDialogueFocus(dialogueIdentity)" in script
    assert 'latestControlInput.route = "projected-pointer";' in script
    assert 'latestControlInput.projected_coordinate_space = "document";' in script
    assert 'controlFocus.mode === "game" &&' in script


def test_shader_viewer_uploads_one_mesh_to_webgl_and_compiled_canvas_projection():
    script = _projection().script
    assert "function buildExtrudedBoxMesh(geometry, colors)" in script
    assert "gl.bufferData(gl.ARRAY_BUFFER, mesh, gl.STATIC_DRAW);" in script
    assert "gl.enable(gl.DEPTH_TEST);" in script
    assert "gl.drawArrays(gl.TRIANGLES, 0, shaderViewer.vertexCount);" in script
    assert 'shaderViewer.canvas.getContext("2d")' in script
    assert "await initializeSoftwareMeshWasm();" in script
    assert "function drawCompiledMeshViewer(" in script
    assert "drawCompiledMeshViewer(context2d, width, height, cameraPosition, cameraFacing, false);" in script
    assert "for (let vertex = 0; vertex < wasm.count; vertex += 3)" in script
    assert '" · portable geometry floor"' in script
    # The portable renderer supplies a visible floor beneath compiled triangles.
    assert "function drawSoftwareViewer(" in script
    assert "drawSoftwareViewer(context2d, width, height, cameraPosition, cameraFacing);" in script


def test_software_mesh_model_retains_python_source_wasm_and_calling_contract():
    software = _projection().model["software_mesh"]
    assert software["source_language"] == "python"
    assert software["source"].startswith("def project_vertex(")
    assert software["lowering"] == [
        "python-ast", "captured-numerical-region", "fused-program", "webassembly",
    ]
    assert software["entrypoint"] == "project_mesh"
    assert software["binary_bytes"] > 8
    assert software["parameters"][0] == {
        "name": "count", "role": "extent", "dtype": "int32",
    }
    assert [item["name"] for item in software["parameters"][-3:]] == [
        "screen_x", "screen_y", "view_z",
    ]


def test_scene_mesh_is_python_compiled_and_round_trips_page_identity():
    projection = _projection()
    scene = projection.model["scene_mesh"]
    assert scene["source_language"] == "python"
    assert scene["entrypoint"] == "instantiate_scene_mesh"
    assert scene["identity_spans"]["source"] == "document_geometry.boxes[].identity"
    script = projection.script
    assert "async function initializeSceneMeshWasm()" in script
    assert "function rebuildSceneMesh()" in script
    assert "shaderViewer.identitySpans.push({" in script
    assert "function publishSceneMeshToDocument()" in script
    assert "element.dataset.meshIdentity = span.identity;" in script
    assert 'element.dataset.meshRevision = String(shaderViewer.revision);' in script


def test_scene_visibility_does_not_depend_on_wasm_or_structured_clone_support():
    script = _projection().script
    initialization = script.index("async function initializeShaderViewer()")
    initialization_body = script[initialization:script.index("function normalized3", initialization)]
    assert "rebuildPortableSceneMesh();" in initialization_body
    assert "await initializeSceneMeshWasm();" not in initialization_body
    assert "shaderViewer.sceneWasm = null;" in initialization_body
    assert "function cloneGeometryBox(box)" in script
    assert "structuredClone(" not in script
    assert 'compiled scene mesh produced an invalid vertex buffer' in script
    assert 'compiled scene mesh collapsed to the origin' in script


def test_crosshair_context_form_menu_edits_the_shared_mesh_and_document():
    projection = _projection()
    script = projection.script
    assert "function pickCrosshairIdentity()" in script
    assert "function rayBoxDistance(origin, direction, box)" in script
    assert 'if(box.geometry_mode==="sphere")' in script
    assert 'Number(box.center_y??radius)' in script
    assert "function openCrosshairContextMenu(clientX, clientY)" in script
    assert 'item.dataset.formInstruction = instruction.identity;' in script
    assert "function applyFormInstruction(identity, instructionIdentity)" in script
    assert 'actionEdges.register(actor, "apply-form", identity)' in script
    assert "rebuildPortableSceneMesh();" in script
    assert 'openCrosshairContextMenuAtViewportCenter();' in script
    assert 'port.append(head, musicButton, musicLoadButton, musicFileInput, canvas, focusTooltip,' in script
    assert 'musicFileInput.type="file";musicFileInput.accept="audio/*"' in script
    assert "async function loadMusicRoomFile(file)" in script
    assert "await loadMusicRoomFile(file);await toggleMusicRoom();" in script
    assert '.viewer-crosshair' not in projection.css
    assert "const VIEWPORT_CROSSHAIR_FRAGMENT_SHADER" in script
    assert "gl_FragCoord.xy-uResolution*0.5" in script
    assert "drawShaderCrosshair(gl, width, height);" in script
    assert "drawCanvasCrosshair(context2d, width, height" in script
    assert 'document.createElement("canvas")' in script
    assert 'div("viewer-crosshair")' not in script
    assert 'if (viewportBinding("secondary-action", "pointer:button-2"))' in script
    assert 'document.addEventListener("contextmenu",event=>' in script
    assert "function beginViewportSecondary(position=null,source=" in script
    assert "function endViewportSecondary(source=" in script
    assert 'issueViewportAction("secondary-action","press")' in script
    assert 'issueViewportAction("secondary-action","release",held)' in script
    assert 'firePhysicsBall(exitScale)' in script
    assert "function cycleActiveToolMode()" in script
    assert 'event.code==="KeyM"' in script
    assert "function updateHeldToolSecondary(dt)" in script
    assert "function beginViewportPrimary(position=null,source=" in script
    assert "function updateHeldToolPrimary(dt)" in script
    assert "projectileState.has(target)?target:null" in script
    assert 'wakeProjectilePhysics(state,"crosshair-attractor")' in script
    assert "function absorbProjectileIntoAmmo(state,reason=" in script
    assert 'state,"attractor-field-absorption")' in script
    assert 'absorbProjectileIntoAmmo(state,"crosshair-attractor-absorption")' in script
    assert "field.strength/field.forceEpsilon" in script
    assert 'if(force<field.forceEpsilon)return;' in script
    assert 'wakeProjectilePhysics(state,"attractor-field-epsilon")' in script
    assert 'field.effectiveRadius.toFixed(2)' in script


def test_focus_tooltip_is_bounded_transparent_and_tracks_wall_identity():
    projection = _projection()
    script = projection.script
    assert 'const focusTooltip = div("viewer-focus-tooltip");' in script
    assert 'focusTooltip.hidden = true;' in script
    assert 'tooltip.dataset.focusIdentity = nearest.box.identity;' in script
    assert 'floor + hollow interior · wall ${nearest.box.height.toFixed(2)} high' in script
    assert 'element.dataset.wallHeight = String(box.height);' in script
    assert 'element.style.setProperty("--wall-height", String(box.height));' in script
    assert '.viewer-focus-tooltip{position:absolute;z-index:3;pointer-events:none' in projection.css
    assert 'background:rgba(7,16,15,.72)' in projection.css
    assert '.viewer-focus-tooltip[hidden]{display:none}' in projection.css


def test_layout_mesh_is_floor_plus_subtractive_hollow_walls_and_max_ceiling():
    projection = _projection()
    script = projection.script
    assert "function wallSegments(box, side, halfLength" in script
    assert 'prism(centerX, centerZ, halfX, halfZ, baseY, baseY+floorHeight, floorColor,' in script
    assert 'wallSegments(box, "south"' in script
    assert "opening.offset - opening.width * 0.5" in script
    assert "openingTop < boxTop - 1e-4" in script
    assert "boundary_semantics.ceiling.absolute_maximum" in script
    assert 'element.dataset.interior = "hollow";' in script
    assert 'element.dataset.floor = "mandatory-slab";' in script
    assert "realization.spans" in script


def test_world_registry_and_semantic_mesh_spans_ride_with_the_live_page():
    projection = _projection()
    world = projection.model["world"]
    geometry = projection.model["document_geometry"]
    assert world["structural_object_order"] == [box["identity"] for box in geometry["boxes"]]
    assert len(world["objects"]) >= len(geometry["boxes"])
    assert [plugin["capability"] for plugin in world["plugins"]] == [
        "geometry", "geometry", "presentation", "audio-analysis", "physics",
        "vehicle-physics",
    ]
    assert [plugin["source_language"] for plugin in world["plugins"]] == [
        "python", "python", "python", "c", "sympy", "sympy",
    ]
    assert all("binary_base64" not in plugin for plugin in world["plugins"])
    assert len(world["wasm_modules"]) == 6
    assert all(module["binary_base64"] for module in world["wasm_modules"])
    assert world["mesh_packet"]["semantic_part_table"] == "variable-length-part-spans"
    assert world["identity_specialization"]["authority"] == "authored-string-identity"
    assert [entry["runtime_id"] for entry in world["identity_specialization"]["objects"]] == list(
        range(1, len(world["objects"]) + 1)
    )
    script = projection.script
    assert "const semanticPartSpans = [];" in script
    assert "semanticParts: semanticPartSpans.slice(firstPart)" in script
    assert "shaderViewer.semanticPartSpans = realization.semanticPartSpans;" in script
    assert "worldObject.persistence.revision = shaderViewer.revision;" in script
    assert "element.dataset.semanticPartCount = String(span.semanticParts.length);" in script
    assert "runtimeObjectId: turingWorld.objectRuntimeId(box.identity)" in script
    assert "runtimePartId: turingWorld.partRuntimeId(partIdentity)" in script


def test_symbolic_physics_parameters_are_live_editable_and_cycle_bound():
    projection = _projection()
    program = projection.model["physics_program"]
    assert program["source_language"] == "sympy-equation-set"
    assert program["status"] == "compiled-active-player-cycle"
    assert len(program["equations"]) == 8
    assert all(parameter["identity"] for parameter in program["parameters"])
    assert next(
        parameter for parameter in program["parameters"]
        if parameter["name"] == "gravity_y"
    )["default"] == -9.81
    script = projection.script
    assert "async function initializeWorldPhysicsWasm()" in script
    assert "function stepCompiledWorldPhysics(dt)" in script
    assert "stepCompiledWorldPhysics(dt);" in script
    assert 'input.dataset.physicsParameter = parameter.name;' in script
    assert 'setPhysicsParameter(physicsInput.dataset.physicsParameter' in script
    assert "function selectWallContact(position, radius, excludedObject = null)" in script
    assert "shaderViewer.colliders = realization.colliders;" in script
    assert "obstacle_active: contact ? 1 : 0" in script
    assert "contactRuntimePartId: contact?.collider.runtimePartId || 0" in script


def test_method_performance_labels_become_loose_objects_inside_their_domains():
    projection = _projection()
    world = projection.model["world"]
    observations = [
        item for item in world["objects"]
        if item["kind"] == "performance-observation"
    ]
    method_rooms = {
        room["identity"]
        for region in projection.model["regions"]
        for building in region["buildings"]
        for room in building["rooms"]
        if room["member_kind"] == "method"
    }
    assert observations
    assert {item["parent"] for item in observations} == method_rooms
    assert all(item["form"]["recipe"] == "performance-marker" for item in observations)
    assert all(item["transform"]["placement"] == "ambient-domain" for item in observations)
    assert all("turing.performance" in item["extensions"] for item in observations)
    assert "turingWorld.containedBy(room.identity)" in projection.script
    assert 'makeSelectable(observation, "performance-observation")' in projection.script
    assert '.performance-observation[data-hot="true"]' in projection.css


def test_user_edits_autosave_by_identity_cookie_with_local_file_fallback():
    projection = _projection()
    persistence = projection.model["persistence"]
    assert persistence["primary"] == "cookie"
    assert persistence["fallback"] == "local-storage"
    assert persistence["contents"] == "edited-identities-and-physics-parameters"
    script = projection.script
    assert "function saveLivingEdits(identity)" in script
    assert "function restoreLivingEdits()" in script
    assert '"max-age=31536000; SameSite=Lax; path=/"' in script
    assert "localStorage.setItem(livingEditPersistence.storageKey, payload);" in script
    assert "restoreLivingEdits();" in script
    assert "saveLivingEdits(targetIdentity);" in script
    assert "physics: Object.fromEntries(physicsRuntime.parameters)" in script
    assert persistence["reset_control"] == "return-to-defaults"
    assert persistence["reset_scope"] == "current-world-cookie-and-local-storage"
    assert "function returnLivingMapToDefaults()" in script
    assert '`${livingEditPersistence.cookieName}=; max-age=0; `' in script
    assert "localStorage.removeItem(livingEditPersistence.storageKey);" in script
    assert 'defaults.textContent = "Return to defaults";' in script
    assert 'defaults.dataset.returnDefaults = "true";' in script
    assert "returnLivingMapToDefaults(); return;" in script


def test_shader_view_has_distinct_material_roles_and_visible_light_contributions():
    projection = _projection()
    roles = {box["palette_role"] for box in projection.model["document_geometry"]["boxes"]}
    assert roles == {
        "world-face", "courtyard-face", "building-face", "room-face",
        "artifact-source", "artifact-test", "artifact-readme",
        "artifact-annotation", "artifact-scratch", "mud-terrain",
    }
    assert projection.model["appearance"]["colors"]["rollbar-silver"] == "#c8d0cf"
    assert projection.model["appearance"]["colors"]["suspension-yellow"] == "#ffd21f"
    assert projection.model["appearance"]["colors"]["drivetrain-black"] == "#111413"
    assert roles <= set(projection.model["appearance"]["colors"])
    assert "layout(location=0) in vec3 aPosition;" in projection.script
    assert "float illumination=uAmbientLight+0.72*key+" in projection.script
    assert "uLightColor*key*0.10" in projection.script


def test_half_dome_sky_drives_native_and_pluck_phong_celestial_lights():
    projection = _projection()
    sky = projection.model["celestial_sky"]

    assert sky["geometry"] == "camera-centered-upper-hemisphere"
    assert sky["time_source"] == "browser-local-solar-day"
    assert sky["sun"]["light_role"] == "phong-key"
    assert sky["moon"]["light_role"] == "phong-fill"
    assert "const VIEWPORT_SKY_FRAGMENT_SHADER" in projection.script
    assert "function celestialState()" in projection.script
    assert "function drawSkyHalfDome" in projection.script
    assert "function updateCelestialLighting" in projection.script
    assert 'uniform("uLightPos[0]")' in projection.script
    assert "[...sunView,...moonView]" in projection.script
    assert 'gl.uniform1i(location,5)' in projection.script
    assert "musicLights.flat()" in projection.script
    assert "#define PLUCK_SHADOW_MAP 1" in projection.model["viewer"]["shader_choices"][0]["fragment_source"]
    assert "function initializeShadowPass(gl)" in projection.script
    assert "function renderShadowPass(gl,celestial)" in projection.script
    assert "gl.DEPTH_COMPONENT24" in projection.script
    assert "gl.TEXTURE_2D_ARRAY" in projection.script
    assert "gl.framebufferTextureLayer" in projection.script
    assert "shadowLightDescriptors(celestial)" in projection.script
    assert "shadow.signatures[layer]===signature" in projection.script
    assert 'uniform("uExposure")' in projection.script
    assert "shaderViewer.celestialStatus" in projection.script


def test_placement_focus_has_bbox_gimbal_and_applies_mesh_rotation_and_offset():
    projection = _projection()
    script = projection.script

    assert "function selectPlacementFocus" in script
    assert "function projectPlacementBoxBounds" in script
    assert "function updatePlacementFocusVisuals" in script
    assert "function commitPlacementFocus" in script
    assert "function rotateBoxRealization" in script
    assert "boxYawDegrees(box)*Math.PI/180" in script
    assert 'collider.rotation=[Number(rotation[0]||0),boxYawDegrees(box),Number(rotation[2]||0)]' in script
    assert "baseline.center[0]+placementState.offsets.x" in script
    assert "baseline.placement?.elevation" in script
    assert "placement-bbox-overlay" in projection.css
    assert "data-placement-focused" in projection.css


def test_human_artifacts_share_identity_across_filesystem_world_and_shader():
    projection = _projection()
    filesystem = projection.model["filesystem"]
    artifact_ids = {artifact["identity"] for artifact in filesystem["artifacts"]}
    solid_boxes = {
        box["identity"] for box in projection.model["document_geometry"]["boxes"]
        if box.get("geometry_mode") == "solid"
    }
    world_ids = {
        item["identity"] for item in projection.model["world"]["objects"]
        if item["kind"] == "human-artifact"
    }
    assert artifact_ids == solid_boxes == world_ids
    assert "function renderFilesystemDistrict()" in projection.script
    assert "function advanceArtifactAttachments(dt)" in projection.script
    assert 'state.attachment.state === "welded"' in projection.script


def test_abstract_ui_object_carries_ordered_html_css_script_and_model_packets():
    projection = _projection()
    assert [(packet.language, packet.role) for packet in projection.packets] == [
        ("html", "structure"),
        ("css", "presentation"),
        ("json", "model"),
        ("javascript", "behavior"),
    ]
    assert projection.packet("html").source.startswith('<div class="shell">')
    assert projection.css == projection.packet("css").source
    assert projection.script == projection.packet("javascript").source
    script_packet = projection.packet("javascript")
    assert script_packet.dependencies == (
        f"{projection.identity}:html", f"{projection.identity}:model",
    )


def test_abstract_ui_packet_composition_is_immutable_and_identity_safe():
    projection = _projection()
    extra = AbstractUIPacket(
        f"{projection.identity}:annotations", "text", "bare map",
        "text/plain", "annotation",
    )
    extended = projection.with_packet(extra)
    assert len(extended.packets) == len(projection.packets) + 1
    assert not projection.packets_for("text")


def test_abstract_interaction_names_only_type_and_destination():
    projection = _projection()
    interaction = projection.interaction("inspect", "python:ControlPanel.gain")
    assert interaction == AbstractUIInteraction(
        "inspect", "python:ControlPanel.gain",
    )
    assert interaction.to_data() == {
        "type": "inspect", "destination": "python:ControlPanel.gain",
    }


def test_one_delegated_event_host_owns_all_interaction_mechanics():
    script = _projection().script
    assert 'card.dataset.interaction = interaction.type;' in script
    assert 'card.dataset.destination = interaction.destination;' in script
    assert "function omnipotentEventHost(event)" in script
    assert 'document.addEventListener("click", omnipotentEventHost);' in script
    assert 'document.addEventListener("keydown", omnipotentEventHost);' in script
    assert 'card.addEventListener' not in script


def test_executing_javascript_keeps_exact_source_without_mounting_every_line():
    projection = _projection()
    assert "const SELF_SCRIPT = document.currentScript;" in projection.javascript
    assert "SELF_SCRIPT.textContent" in projection.javascript
    assert "javascript:self:line:${line}" not in projection.javascript
    assert "SELF_SOURCE.split" not in projection.javascript
    assert "function semanticSourceClosures" in projection.javascript
    assert projection.script.endswith(DIV_MAP_JAVASCRIPT)
    assert f"\n{projection.script}\n</script>" in projection.document()


def test_page_carries_one_world_locked_player_without_mouse_followers():
    projection = _projection()
    mezzanine = projection.model["entity_mezzanine"]
    assert mezzanine["system_root"] == projection.model["identity"]
    assert len(mezzanine["entities"]) == 1
    player = mezzanine["entities"][0]
    assert player["archetype"] == "player-being"
    assert player["controller"] == {
        "kind": "world-player", "source": "game.controls", "parameters": {},
    }
    assert player["pose"]["coordinate_space"] == "data-world"
    assert player["traits"]["physics_body"] == {
        "shape": "upright-capsule",
        "radius": projection.model["viewer"]["camera"]["collision_radius"],
        "height": (
            projection.model["viewer"]["camera"]["eye_height"]
            + projection.model["viewer"]["camera"]["collision_radius"]
        ),
        "inverse_mass": 0.22,
        "horizontal_friction": 18.0,
        "contact": "remembered-side-sweep-and-downward-top-crossing",
        "receives_projectile_impulses": True,
    }
    assert "function resolveProjectilePlayerContacts(radius)" in projection.script
    assert mezzanine["organizations"][0]["name"] == "players"


def test_browser_packet_hosts_control_input_and_isolated_entity_cycle():
    script = _projection().script
    assert 'document.addEventListener("pointermove", acceptControlInput);' in script
    assert "function runEntityCycle(now)" in script
    assert '"fourth-order-follow": 4' in script
    assert "const derivatives = [state.position" in script
    assert "requestAnimationFrame(runEntityCycle);" in script
    assert 'entity.controller.kind === "world-player"' in script
    assert 'else if (player?.worldPosition)' in script
    assert '`.entity-sprite[data-entity="${identity}"]`' in script


def test_armored_turret_carrier_routes_focus_fire_and_stabilizers():
    script = DIV_MAP_JAVASCRIPT
    assert "function updateVehicleTurretTargeting(vehicle,state)" in script
    assert "function turretFriendlyRayEntry" in script
    assert "function fireVehicleTurrets()" in script
    assert 'ammunitionAuthority:"vehicle-turret"' in script
    assert 'type:"vehicle-body-wrenches"' in script
    assert "turretSystem.fireTakeover" in script
    assert "data-turret-fire-takeover" in script
    assert "function controlVehicleOutriggers(deployed)" in script
    assert 'outriggerButton.dataset.outriggerToggle="true"' in script
    assert 'handPumpButton.dataset.outriggerHandPump="true"' in script
    assert "function pumpVehicleOutriggerAccumulator" in script


def test_javascript_root_timer_updates_and_lights_action_edge_rows():
    projection = _projection()
    timer = projection.model["action_mezzanine"]["timer"]
    table = projection.model["action_mezzanine"]["action_edges"]
    assert projection.script.startswith("// abstract-ui:system-root")
    assert timer["connections"] == [table["identity"]]
    assert timer["identity"] in projection.script
    assert "actionEdges.update(pendingActions.splice(0));" in projection.script
    assert "abstractUISystemTimer.issue({" in projection.script
    assert 'row.element.classList.toggle("recent", recent);' in projection.script
    assert "update(actions)" in projection.script


def test_abstract_ui_document_refuses_javascript_without_system_root():
    projection = _projection()
    packets = tuple(
        replace(packet, source=DIV_MAP_JAVASCRIPT)
        if packet.language == "javascript" else packet
        for packet in projection.packets
    )
    without_root = replace(projection, packets=packets)
    with pytest.raises(ValueError, match="system-root prelude"):
        without_root.document()


def test_embedded_graph_is_json_safe_and_does_not_execute_program_values():
    page = _projection().html
    match = re.search(
        r'<script type="application/json" id="abstract-ui-model">(.*?)</script>',
        page,
    )
    assert match is not None
    model = json.loads(match.group(1))
    engage = model["regions"][0]["buildings"][0]["rooms"][2]
    assert engage["implied_code"][0]["source"] == "instance.engage(amount)"
    assert engage["implied_code"][0]["executable"]


def test_projection_is_byte_deterministic_for_same_world_and_title():
    assert _projection().html == _projection().html


def test_saved_vehicle_settings_restore_only_inside_persistence_scope():
    restore_start = DIV_MAP_JAVASCRIPT.index("function restoreLivingEdits()")
    restore_end = DIV_MAP_JAVASCRIPT.index("function installSceneMesh", restore_start)
    restore_source = DIV_MAP_JAVASCRIPT[restore_start:restore_end]
    stick_start = DIV_MAP_JAVASCRIPT.index("function bindMobileStick")
    stick_end = DIV_MAP_JAVASCRIPT.index("function renderMobileControls", stick_start)
    stick_source = DIV_MAP_JAVASCRIPT[stick_start:stick_end]

    assert "const saved = JSON.parse(payload)" in restore_source
    assert "saved.vehicle_hydraulics" in restore_source
    assert "enabled: false" in restore_source
    assert "source.maximum[1]" in DIV_MAP_JAVASCRIPT
    assert "saved.vehicle_tire_pressure_target_pa" in restore_source
    assert "saved." not in stick_source


def test_vehicle_qualification_gates_world_opening_on_twenty_simulated_seconds():
    script = DIV_MAP_JAVASCRIPT
    assert 'document.body.classList.add("qualification-pending")' in script
    assert "function updateVehicleQualification(body,sequence)" in script
    assert "q.simulatedSeconds=(sequence-q.startSequence)/120" in script
    assert 'q.simulatedSeconds>=10&&!q.engineStarted' in script
    assert "ignitionOn:true,starterEngaged:true" in script
    assert 'q.simulatedSeconds>=11.5&&!q.starterReleased' in script
    assert "q.simulatedSeconds>=20" in script
    assert "q.epsilon.toExponential" in script
    assert "if(sequence%30<deltaTicks)" in script
    assert "q.failures.add" in script
    assert "finishVehicleQualification(body)" in script
    assert "registerPlayerPhysicsBody();requestAnimationFrame(runEntityCycle)" in script
    assert "if(!vehicleQualification.active)registerPlayerPhysicsBody()" in script
