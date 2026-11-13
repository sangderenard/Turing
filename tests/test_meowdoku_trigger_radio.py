import unittest
from meowdoku import trigger_grid_service as service

class Panel:
    def __init__(self): self.seen=[]
    def radio(self, snapshot): self.seen.append(snapshot); return snapshot

class RadioOwnership(unittest.TestCase):
    def tearDown(self): service._panels.clear()
    def test_isolation_and_server_owned_source(self):
        first,second=Panel(),Panel()
        service._panels.update({1:first,2:second})
        actual={'status':'live','rms':.3}
        self.assertEqual(service.command(1,{'action':'radio','rms':999},None,radio_snapshot=lambda:actual),actual)
        self.assertEqual(first.seen,[actual]); self.assertEqual(second.seen,[])
    def test_missing_account_or_grid_never_reads_source(self):
        def prohibited(): self.fail('source read without account/grid')
        for owner in (0,3):
            with self.assertRaises(ValueError):
                service.command(owner,{'action':'radio'},None,radio_snapshot=prohibited)
    def test_unconfigured_source_explicit(self):
        service._panels[1]=Panel()
        with self.assertRaisesRegex(ValueError,'unavailable'):
            service.command(1,{'action':'radio'},None)

if __name__=='__main__': unittest.main()
