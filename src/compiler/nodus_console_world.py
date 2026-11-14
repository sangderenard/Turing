"""Project a Nodus GRAPH response into the existing Living Map / Pluck view.

This is a view of graph identity, not a scheduler or an alternate graph model.
Console UI descriptions use Pluck's Panel fields; actions remain descriptors,
never executable authority. Native Pluck portal traversal is not implemented
by this adapter: the projection uses Living Map's existing door runtime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from copy import deepcopy
import json
from typing import Any, Mapping

from .abstract_ui_introspection import (
    AbstractUIWorld, AbstractUIRegion, AbstractUIBuilding, AbstractUIRoom,
    AbstractUITrack, GridPosition, ImpliedCode,
)
from .abstract_ui_vocabulary import UIIntention
from .abstract_ui_doorways import declare_door


def _identity(value: Any) -> str:
    # Nodus stable IDs are uint64. Never pass them through float/JS Number.
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('module identity must be an integer or exact string')
    return str(value)


def _panel_data(panel: Any) -> dict:
    data = asdict(panel) if is_dataclass(panel) else deepcopy(dict(panel))
    if not isinstance(data.get('name'), str):
        raise ValueError('console UI requires a Pluck Panel name')
    data['panels'] = [_panel_data(child) for child in data.get('panels', ())]
    # Validate serializability now; no repr of arbitrary executable objects.
    json.dumps(data, allow_nan=False)
    return data


@dataclass(frozen=True)
class ConsoleSelection:
    identity: str
    module_ids: tuple[str | int, ...]
    label: str = ''
    panel: Any = None


def graph_to_world(graph: Mapping[str, Any], *, identity: str = 'nodus',
                   consoles: tuple[ConsoleSelection, ...] = ()) -> AbstractUIWorld:
    """Convert an actual revision-consistent Nodus GRAPH document.

    Default: one console per table. An explicit selection may contain any
    subgraph; overlapping selections are views of the same IDs, not clones.
    Rows and supplied/procedural Panel descriptions remain inspectable. Edges
    keep connector ordinals and FIFO metadata in their label; no edge is lost
    merely because its other end lies outside a selected console.
    """
    if graph.get('definition_error'):
        raise ValueError('GRAPH lacks stable graph-definition identities')
    modules = {}
    for module in graph.get('modules', ()):
        key = _identity(module['stable_id'])
        if key in modules:
            raise ValueError(f'duplicate module identity: {key}')
        modules[key] = module
    selections = consoles or tuple(ConsoleSelection(key, (key,), str(m.get('label', key)))
                                   for key, m in modules.items())
    if consoles:
        included = {_identity(key) for console in consoles for key in console.module_ids}
        selections = (*selections, *(ConsoleSelection(f'unselected/{key}', (key,),
            str(module.get('label', key))) for key, module in modules.items() if key not in included))
    seen = set()
    regions, lots, tracks = [], [], []
    owners: dict[str, list[str]] = {key: [] for key in modules}
    for index, selection in enumerate(selections):
        cid = str(selection.identity)
        if not cid or cid in seen:
            raise ValueError(f'empty or duplicate console identity: {cid}')
        seen.add(cid)
        ids = tuple(_identity(key) for key in selection.module_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('console requires distinct module identities')
        unknown = set(ids) - modules.keys()
        if unknown:
            raise ValueError(f'unknown console modules: {sorted(unknown)}')
        lot_id, region_id = f'{identity}/console/{cid}', f'{identity}/inside/{cid}'
        label = selection.label or cid
        panel = _panel_data(selection.panel) if selection.panel is not None else {
            'name': cid, 'label': label, 'knobs': [], 'panels': [],
            'payload': {'module_ids': list(ids), 'read_only': True},
        }
        panel.setdefault('payload', {})['nodus'] = {
            'revision': graph.get('revision'), 'module_ids': list(ids),
            'modules': [dict(modules[key]) for key in ids],
            'connectors': [dict(c) for c in graph.get('connectors', ())
                           if _identity(c['reference']['module_id']) in ids],
        }
        buildings = []
        for mi, key in enumerate(ids):
            owners[key].append(lot_id)
            module = modules[key]
            bid = f'{region_id}/module/{key}'
            rows = []
            for ri, row in enumerate(module.get('rows', ())):
                row_data = dict(row)
                rows.append(AbstractUIRoom(
                    f'{bid}/row/{ri}', str(row.get('tool', row.get('kind', ri))),
                    'nodus-row', 'console controls', GridPosition(ri % 4, ri // 4),
                    implied_code=(ImpliedCode('inspect', 'nodus-row',
                        json.dumps(row_data, sort_keys=True), False,
                        'Actual row descriptor; inspection does not grant execution rights.'),)))
                if selection.panel is None:
                    panel['panels'].append({'name': f'{key}/row/{ri}',
                        'label': rows[-1].name, 'knobs': [], 'panels': [],
                        'payload': {'module_id': key, 'row_index': ri, 'descriptor': row_data}})
            buildings.append(AbstractUIBuilding(bid, str(module.get('label', key)),
                key, 'nodus-module', 'console', GridPosition(mi % 4, mi // 4),
                tuple(rows), UIIntention('console')))
        schema_room = AbstractUIRoom(f'{lot_id}/ui', 'Console UI', 'panel',
            'control room', GridPosition(0, 0), implied_code=(ImpliedCode(
                'inspect', 'pluck-panel', json.dumps(panel, sort_keys=True), False,
                'Pluck Panel schema. Bind actions through the authenticated host, not map proximity.'),))
        lots.append(AbstractUIBuilding(lot_id, label, identity, 'nodus-subgraph',
            'console', GridPosition(index % 4, index // 4), (schema_room,), UIIntention('console')))
        regions.append(AbstractUIRegion(region_id, label, GridPosition(index % 4, 1 + index // 4),
            tuple(buildings), UIIntention('room')))
    for ei, edge in enumerate(graph.get('edges', ())):
        source, target = edge['source'], edge['target']
        sk, tk = _identity(source['module_id']), _identity(target['module_id'])
        if sk not in modules or tk not in modules:
            raise ValueError('edge references an unknown module')
        for source_lot in owners[sk]:
            for target_lot in owners[tk]:
                tracks.append(AbstractUITrack(source_lot, target_lot, 'nodus-fifo',
                    json.dumps({'index': edge.get('index', ei), 'source': dict(source),
                                'target': dict(target), 'fifo': edge.get('fifo')}, sort_keys=True),
                    UIIntention('wire')))
    outside = AbstractUIRegion(f'{identity}/canvas', 'Nodus canvas', GridPosition(0, 0),
        tuple(lots), UIIntention('canvas'))
    world = AbstractUIWorld(identity, (outside, *regions), tuple(tracks), UIIntention('world'))
    for selection in selections:
        world = declare_door(world, f'{identity}/console/{selection.identity}',
                            f'{identity}/inside/{selection.identity}')
    return world


def project_graph(graph: Mapping[str, Any], **kwargs: Any):
    """Use the existing first-person world renderer and Pluck shader choice."""
    from .abstract_ui_div_map import project_world_to_div_map
    return project_world_to_div_map(graph_to_world(graph, **kwargs), title='Nodus consoles')


def main() -> None:
    """Project an authorized saved GRAPH response, without contacting a service."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--identity', default='nodus')
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding='utf-8-sig'))
    project_graph(graph, identity=args.identity).write(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
