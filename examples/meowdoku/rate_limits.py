"""One switch for the request rate limits (owner, 2026-11-19: "TURN THEM OFF. don't remove the ability").

Every per-time or per-address rate limit that answers 429 asks ``enabled()`` before refusing. They are off unless the
service is started with MEOWDOKU_RATE_LIMITS=1; each limit's code and numbers are kept as they were. Capacity bounds
(queues, storage quotas, session caps) are not rate limits and are not switched here.
"""
import os


def enabled() -> bool:
    return os.environ.get("MEOWDOKU_RATE_LIMITS", "0") == "1"
