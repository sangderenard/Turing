"""Load the shared REST contract from the explicitly selected SpeakToMe checkout.

The repository is a deployment dependency, not a second HTTP service. Keep its
package out of sys.path so SpeakToMe's tensor packages cannot shadow Turing's.
SPEAKTOME_REPO selects another checkout; the default is the workspace sibling.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = Path(os.environ.get("SPEAKTOME_REPO", str(ROOT.parent / "speaktome"))).resolve()
PACKAGE = REPOSITORY / "speaktome_api"

if not (PACKAGE / "__init__.py").is_file():
    raise ImportError(
        "SpeakToMe REST dependency is missing. Update the sibling speaktome repository "
        "or set SPEAKTOME_REPO to its checkout containing speaktome_api."
    )



def _load(name: str):
    """One named package of the SpeakToMe checkout, loaded by path so the checkout never enters sys.path."""
    folder = REPOSITORY / name
    package = sys.modules.get(name)
    if package is not None:
        if Path(package.__file__).resolve() != folder / "__init__.py":
            raise ImportError(f"{name} is already loaded from a different checkout")
        return package
    spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return package


_load("speaktome_api")
from speaktome_api import routes  # noqa: E402

if routes.API_VERSION != 2:
    raise ImportError("This Turing host requires SpeakToMe REST API_VERSION 2 (pictures and map pages); "
                      "update the speaktome checkout it loads")

from speaktome_api import site  # noqa: E402  (after the version check, so an older checkout is refused with that message)

# The manual is a map (speaktome_man.as_map); an older checkout without it still serves everything else.
manual = _load("speaktome_man") if (REPOSITORY / "speaktome_man" / "__init__.py").is_file() else None
