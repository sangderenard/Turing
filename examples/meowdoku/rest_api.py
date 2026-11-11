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

_package = sys.modules.get("speaktome_api")
if _package is not None:
    if Path(_package.__file__).resolve() != PACKAGE / "__init__.py":
        raise ImportError("speaktome_api is already loaded from a different checkout")
else:
    _spec = importlib.util.spec_from_file_location(
        "speaktome_api", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)]
    )
    _package = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _package
    try:
        _spec.loader.exec_module(_package)
    except BaseException:
        sys.modules.pop(_spec.name, None)
        raise

from speaktome_api import routes  # noqa: E402

if routes.API_VERSION != 1:
    raise ImportError("This Turing host requires SpeakToMe REST API_VERSION 1")
