"""Material selection and properties stay inside the original shared editor."""
from html.parser import HTMLParser
from pathlib import Path

class Controls(HTMLParser):
    def __init__(self):
        super().__init__(); self.stack=[]; self.parents={}; self.nested=[]
    def handle_starttag(self, tag, attrs):
        attrs=dict(attrs); ident=attrs.get('id')
        if ident:
            assert ident not in self.parents, f'duplicate id: {ident}'
            self.parents[ident]=tuple(self.stack)
        if 'material-editor' in self.stack and tag in ('details','fieldset'):
            self.nested.append(tag)
        if tag not in ('input','meta','link','br','img','hr','source'):
            self.stack.append(ident or tag)
    def handle_endtag(self, tag):
        if self.stack: self.stack.pop()

def test_shared_material_container():
    root=Path(__file__).resolve().parents[1]
    parser=Controls(); parser.feed((root/'examples/meowdoku/web/marble_studio.html').read_text(encoding='utf-8'))
    controls=('shell eye accent confetti-material mark-material paint-a-color paint-b-color opaque layer optical-mode albedo inner roughness metallic transmission opacity ior specular shininess emission enamel-thickness enamel-ior enamel-color optical-material-rows reset-material').split()
    for ident in controls:
        assert 'material-editor' in parser.parents[ident], ident
    assert not parser.nested
