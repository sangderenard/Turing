"""The generated host receives only the actual canonical registry method."""
import hashlib
import re
import shutil
import subprocess

import pytest

from tools.add_studio_registry_append import ROOT, BEGIN, END, canonical_append, patch_studio_registry_append


@pytest.fixture(params=["build/dot-newer-tire-dynamics-20261110/index.html",
                        "build/dot-playable-tires-20261110/index.html"])
def original(request):
    return (ROOT / request.param).read_bytes()


def test_exact_method_content_key_reversal_and_model_preservation(original):
    output, receipt = patch_studio_registry_append(original)
    old, new = original.decode("utf-8"), output.decode("utf-8")
    method, _, _ = canonical_append()
    nl = "\r\n" if "\r\n" in old else "\n"
    assert method.replace("\n", nl) in new
    restored = new.replace(method.replace("\n", nl), "", 1).replace(
        receipt["inline_utility_sha256"], receipt["previous_inline_utility_sha256"], 1)
    assert restored.encode("utf-8") == original
    pattern = r'<script type="application/json" id="abstract-ui-model">(.*?)</script>'
    assert re.search(pattern, old, re.S)[0] == re.search(pattern, new, re.S)[0]
    assert receipt["original_recovered_exactly"] and not receipt["compiler_imported"]
    assert receipt["method_sha256"] == hashlib.sha256(method.encode("utf-8")).hexdigest()


def test_unknown_changed_or_repeated_registry_refused(original):
    with pytest.raises(ValueError, match="header"):
        patch_studio_registry_append(b"no registry")
    changed = original.replace(b"this.world = world;", b"this.world = foreign;", 1)
    with pytest.raises(ValueError, match="content key"):
        patch_studio_registry_append(changed)
    output, _ = patch_studio_registry_append(original)
    # Legacy utility bytes outside the class are not the current canonical
    # utility. Reapplication is accepted only for a whole current matched build.
    with pytest.raises(ValueError, match='not the exact canonical utility'):
        patch_studio_registry_append(output)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_patched_canonical_class_executes(original, tmp_path):
    output, _ = patch_studio_registry_append(original)
    source = re.search(r"^class TuringWorldRegistry \{[\s\S]*?^}", output.decode(), re.M).group()
    script = tmp_path / "registry.cjs"
    script.write_text(source + "\nif(typeof TuringWorldRegistry.prototype.appendObjects!=='function')throw Error('missing canonical method');\n", encoding="utf-8")
    result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
