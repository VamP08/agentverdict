"""FastAPI application factory for AgentVerdict."""

from fastapi import FastAPI

from agentverdict.api.routes_judges import router as judges_router
from agentverdict.api.routes_labels import router as labels_router
from agentverdict.api.routes_tasks import router as tasks_router
from agentverdict.api.routes_trajectories import router as trajectories_router
from agentverdict.web.routes import router as web_router


def create_app() -> FastAPI:
    """Build the application: JSON API under /api, labeling UI under /label."""
    application = FastAPI(title="AgentVerdict")
    application.include_router(tasks_router)
    application.include_router(trajectories_router)
    application.include_router(labels_router)
    application.include_router(judges_router)
    application.include_router(web_router)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
