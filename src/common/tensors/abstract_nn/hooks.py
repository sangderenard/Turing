from collections import defaultdict
from typing import Callable, Dict, List, Any

class HookPanel:
    """
    Patch panel for registering and running hooks on key events in abstract_nn.
    All hooks are lists and run in order. Hooks can be added/removed at runtime.
    """
    def __init__(self):
        self.hooks: Dict[str, List[Callable[..., None]]] = defaultdict(list)
        self.document_policy_enabled = False

    def register(self, event: str, fn: Callable[..., None]):
        self.hooks[event].append(fn)

    def remove(self, event: str, fn: Callable[..., None]):
        if fn in self.hooks[event]:
            self.hooks[event].remove(fn)

    def run(self, event: str, *args, **kwargs):
        for fn in self.hooks[event]:
            fn(*args, **kwargs)

    def enable_document_policy(self, enabled: bool = True):
        self.document_policy_enabled = enabled

    def clear(self, event: str = None):
        if event:
            self.hooks[event] = []
        else:
            self.hooks.clear()

# Singleton for global use
hook_panel = HookPanel()
