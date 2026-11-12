# The Lantern Road: a Living Data class

`examples/meowdoku/storybook_environment.py` authors a real `StorybookEnvironment`
class and four referenced classes: `LanternCottage`, `WhisperingWood`, `MoonBridge`,
and `ClockworkTower`. The canonical introspector turns them into five buildings,
21 member rooms and seven `contains-type` tracks. References are class graph
relationships, not certified physical doorways.

## Existing paths

With the repository root and `examples` on Python's import path:

- `build_storybook_world()` uses `build_introspective_world` with storybook
  palettes and narrative intention adornments. It executes no class instance.
- `storybook_graph()` uses the canonical `world_div_map_graph` serializer. This
  is the exact layout-building block extracted from `world_div_map_model`.
  The full producer now calls that same function. There is no second serializer.
- `project_storybook_map()` invokes the existing full `project_world_to_div_map`
  pipeline. It is intentionally opt-in: this path includes the normal expensive
  world/physics compilation. It was not run for this change.

The compact graph can be passed as `map` to the existing
`POST /api/living-data/sessions` route. Use the root building identity
`python:meowdoku.storybook_environment.StorybookEnvironment` as `at`. A tested
chapter walk follows the declared edges to ClockworkTower, MoonBridge,
WhisperingWood and LanternCottage. Cursor movement changes only graph position.

Descriptions are retained in the authored class fields and the canonical world's
intention records. The existing compact map serializer exposes member names,
metaphors and code receipts, but not arbitrary intention prose. The existing
cursor narrates labels and graph exits. No claim of rich prose narration,
physical traversal or a rendered storybook frame is made by these tests.

## Verification

`python -m pytest tests/test_storybook_environment.py tests/test_abstract_ui_introspection.py tests/test_living_data_query.py tests/test_living_data_cursor.py -q`

31 passed in 7.45 seconds on Windows, with one existing cffi deprecation warning.
Coverage includes canonical identity/order/edge equality, repeatability, retained
story prose intentions, non-instantiation, equality with the actual full
projection's input at its first geometry call, and a real isolated HTTP walk.
The source graph remains unchanged and no simulation table is created.
Full repository suite and full map compilation were not run.
