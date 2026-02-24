"""FastAPI application factory for the Leviathan Read API."""

from typing import Dict

from fastapi import FastAPI

from api.deps import create_app_state
from api.routes.world import router as world_router
from kernel.world_kernel import WorldKernel


def create_app(kernel: WorldKernel) -> FastAPI:
    """Build and return a configured FastAPI application.

    The kernel and shared state are stored on ``app.state.leviathan``
    so that route handlers can access them without global variables.
    """
    app = FastAPI(title="Leviathan Read API", version="0.1.0")
    app.state.leviathan = create_app_state(kernel)

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    app.include_router(world_router)

    return app
