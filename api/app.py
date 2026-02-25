"""FastAPI application factory for the Leviathan Read API."""

from typing import Dict, Optional, Set

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.auth import APIKeyAuth, RateLimiterMiddleware
from api.deps import create_app_state
from api.routes.actions import router as actions_router
from api.routes.admin import router as admin_router
from api.routes.agents import router as agents_router
from api.routes.discovery import router as discovery_router
from api.routes.mechanisms import router as mechanisms_router
from api.routes.metrics import router as metrics_router
from api.routes.world import router as world_router
from kernel.world_kernel import WorldKernel


def create_app(
    kernel: WorldKernel,
    api_keys: Optional[Set[str]] = None,
    moderator_keys: Optional[Set[str]] = None,
    rate_limit: int = 60,
    data_dir: str = "",
) -> FastAPI:
    """Build and return a configured FastAPI application.

    The kernel and shared state are stored on ``app.state.leviathan``
    so that route handlers can access them without global variables.

    Parameters
    ----------
    kernel:
        The simulation kernel instance.
    api_keys:
        Optional set of valid API keys.  When ``None`` or empty,
        authentication is disabled (open access).
    moderator_keys:
        Optional set of moderator API keys.  Moderator keys also
        pass regular authentication checks.
    rate_limit:
        Maximum requests per minute per client IP.
    data_dir:
        Directory for persistent state (mechanisms, etc.).
    """
    app = FastAPI(title="Leviathan Read API", version="0.1.0")
    app.state.leviathan = create_app_state(kernel, data_dir=data_dir)
    app.state.auth = APIKeyAuth(api_keys, moderator_keys)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    app.add_middleware(RateLimiterMiddleware, requests_per_minute=rate_limit)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    app.include_router(world_router)
    app.include_router(discovery_router)
    app.include_router(agents_router)
    app.include_router(actions_router)
    app.include_router(mechanisms_router)
    app.include_router(metrics_router)
    app.include_router(admin_router)

    return app
