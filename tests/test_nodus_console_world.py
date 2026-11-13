import unittest
from src.compiler.nodus_console_world import graph_to_world, ConsoleSelection

class ConsoleWorldTests(unittest.TestCase):
    def graph(self):
        return {'modules': [{'stable_id': 2**60+3, 'label': 'source', 'rows': [{'kind': 'output', 'count': 1}]},
                            {'stable_id': 9, 'label': 'sink', 'rows': [{'kind': 'input', 'count': 1}]}],
                'edges': [{'source': {'module_id': 2**60+3, 'direction': 'output', 'ordinal': 0},
                           'target': {'module_id': 9, 'direction': 'input', 'ordinal': 0}, 'fifo': {'slots': 8}}]}
    def test_default_portals_and_exact_ids(self):
        world = graph_to_world(self.graph())
        self.assertEqual(len(world.regions), 3)
        self.assertEqual(len(world.tracks), 1)
        self.assertIn(str(2**60+3), world.tracks[0].source)
        self.assertEqual(len(world.regions[0].buildings[0].openings), 2)
        self.assertEqual(len(world.regions[1].openings), 2)
    def test_subgraph_panel(self):
        world = graph_to_world(self.graph(), consoles=(ConsoleSelection('both', (2**60+3, 9), panel={'name':'board','knobs':[], 'payload':{'actions':[]}}),))
        self.assertEqual(len(world.regions[1].buildings), 2)
        self.assertIn('board', world.regions[0].buildings[0].rooms[0].implied_code[0].source)
    def test_unknown_is_rejected(self):
        with self.assertRaises(ValueError):
            graph_to_world(self.graph(), consoles=(ConsoleSelection('bad',(10,)),))
    def test_partial_selection_preserves_boundary(self):
        world = graph_to_world(self.graph(), consoles=(ConsoleSelection('one',(9,)),))
        self.assertEqual(len(world.regions), 3)
        self.assertEqual(len(world.tracks), 1)
    def test_supplied_panel_is_not_mutated(self):
        panel = {'name': 'board', 'payload': {}}
        graph_to_world(self.graph(), consoles=(ConsoleSelection('one',(9,),panel=panel),))
        self.assertEqual(panel, {'name':'board', 'payload':{}})
    def test_float_identity_rejected(self):
        g = self.graph(); g['modules'][0]['stable_id'] = float(2**60+3)
        with self.assertRaises(ValueError): graph_to_world(g)
    def test_native_index_fallback_rejected(self):
        with self.assertRaises(ValueError): graph_to_world({'definition_error':'bad'})

if __name__ == '__main__': unittest.main()
