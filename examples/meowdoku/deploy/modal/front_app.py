"""speaktome-modal-front: the Modal operator's front door behind https://modal.speaktome.dev (Oracle relays it).

Owner, 2026-11-21: Oracle hosts its own subdomain, relays home, "and so too providing for modal a subdomain". Jobs are
contracts routed by a cost-to-time metric; Modal should win model hosting because the metric picks it, not by rule
(SpeakToMe ticket 95af9094). This front is the part every contract shares: it says the operator is there (/health) and
states what it offers and what it costs (/operator), so the router can measure it. It hosts no model yet.

Only Oracle's Apache reaches it: every request must carry the shared front token (header X-SpeakToMe-Front), which
Apache adds from a root-only definition and Modal reads from the Modal secret ``speaktome-front``. A request without it
is refused before any route runs. The token lives at home in <state>/secrets/modal/front.token.

    modal deploy examples/meowdoku/deploy/modal/front_app.py     # from a turing checkout, with the Modal CLI
"""
from __future__ import annotations

import modal

APP_NAME = "speaktome-modal-front"
SECRET_NAME = "speaktome-front"
TOKEN_HEADER = "x-speaktome-front"
# What one container of this front is, stated once and reported by /operator: the router compares operators by these
# declared resources and the measured times, not by the operator's name.
CPU = 0.25
MEMORY_MB = 256

image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]==0.115.12")
app = modal.App(APP_NAME, image=image)


@app.function(secrets=[modal.Secret.from_name(SECRET_NAME)], cpu=CPU, memory=MEMORY_MB, min_containers=0,
              scaledown_window=120, timeout=60)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def front():
    import hmac
    import os
    import time

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    token = os.environ["SPEAKTOME_FRONT_TOKEN"].encode()
    started = time.time()
    web = FastAPI(title=APP_NAME, docs_url=None, redoc_url=None, openapi_url=None)

    @web.middleware("http")
    async def require_front_token(request: Request, call_next):
        presented = request.headers.get(TOKEN_HEADER, "").encode()
        if not presented or not hmac.compare_digest(presented, token):
            return JSONResponse({"error": "not through the front"}, status_code=403)
        return await call_next(request)

    @web.get("/health")
    def health():
        return {"ok": True, "operator": "modal", "app": APP_NAME, "container_age_s": round(time.time() - started, 3)}

    @web.get("/operator")
    def operator():
        return {
            "operator": "modal",
            "app": APP_NAME,
            "contracts": [],                       # model hosting is added here when a model is deployed
            "resources": {"cpu": CPU, "memory_mb": MEMORY_MB, "gpu": None},
            "scale": {"min_containers": 0, "scaledown_window_s": 120},
            "container_age_s": round(time.time() - started, 3),
        }

    @web.get("/")
    def root():
        return {"operator": "modal", "routes": ["/health", "/operator"]}

    return web
