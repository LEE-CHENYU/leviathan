"""API key authentication and rate limiting for the Leviathan API."""

import time
from typing import Dict, Optional, Set, Tuple

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response


class APIKeyAuth:
    """Callable FastAPI dependency for API key authentication.

    When *api_keys* is ``None`` or empty, authentication is disabled and
    all requests pass through.  Otherwise the caller must supply a valid
    key via the ``X-API-Key`` header or the ``api_key`` query parameter.

    Usage in a route::

        @router.post("/v1/world/actions")
        def submit_action(request: Request, _=Depends(get_auth())):
            ...
    """

    def __init__(self, api_keys: Optional[Set[str]] = None) -> None:
        self.enabled = bool(api_keys)
        self.api_keys: Set[str] = api_keys or set()

    def __call__(self, request: Request) -> None:
        if not self.enabled:
            return

        key = request.headers.get("X-API-Key") or request.query_params.get("api_key")

        if not key:
            raise HTTPException(status_code=401, detail="Missing API key")

        if key not in self.api_keys:
            raise HTTPException(status_code=403, detail="Invalid API key")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple in-memory token-bucket rate limiter applied per client IP.

    Each IP gets *requests_per_minute* tokens that refill linearly.
    When tokens are exhausted the middleware returns **429 Too Many Requests**.
    """

    def __init__(self, app, requests_per_minute: int = 60) -> None:
        super().__init__(app)
        self.rpm = requests_per_minute
        self.buckets: Dict[str, Tuple[float, float]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        tokens, last_refill = self.buckets.get(client_ip, (float(self.rpm), now))

        elapsed = now - last_refill
        tokens = min(self.rpm, tokens + elapsed * (self.rpm / 60.0))
        last_refill = now

        if tokens < 1.0:
            self.buckets[client_ip] = (tokens, last_refill)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
            )

        tokens -= 1.0
        self.buckets[client_ip] = (tokens, last_refill)

        return await call_next(request)
