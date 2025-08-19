from collections import deque

class MenuResolver:
    """
    Centralized menu and key dispatch system.
    Accepts a nested action_handlers dict and key_mappings dict.
    Handles mode stack and focused clock type for config mode.
    """

    def __init__(self, key_mappings, action_handlers):
        self.key_mappings = key_mappings
        self.action_handlers = action_handlers
        self.mode_stack = deque(["normal"])
        self.focused_clock_type = None  # e.g. "analog" or "digital"

    def push_mode(self, mode):
        self.mode_stack.append(mode)

    def pop_mode(self):
        if len(self.mode_stack) > 1:
            self.mode_stack.pop()

    def set_focused_clock_type(self, clock_type):
        self.focused_clock_type = clock_type

    def resolve_action(self, key):
        """
        Returns (action_name, handler_function) or (None, None)
        """
        # Try deepest mode first
        for mode in reversed(self.mode_stack):
            if mode == "config" and self.focused_clock_type:
                config_map = self.key_mappings.get("config", {}).get(self.focused_clock_type, {})
                handler_map = self.action_handlers.get("config", {}).get(self.focused_clock_type, {})
                for action, info in config_map.items():
                    if key in info["keys"]:
                        return action, handler_map.get(action)
            else:
                normal_map = self.key_mappings.get(mode, {})
                handler_map = self.action_handlers.get(mode, {})
                for action, info in normal_map.items():
                    if key in info["keys"]:
                        return action, handler_map.get(action)
        return None, None

    def handle_key(self, key):
        """
        Resolves and calls the handler for the given key, if any.
        Returns the action name if handled, else None.
        """
        action, handler = self.resolve_action(key)
        if handler:
            handler()
        return action