from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.database import DATABASE_PATH, initialize_database
from app.rag.vector_store import CHROMA_DIRECTORY
from app.workflow.resumable_graph import (
    CHECKPOINT_PATH,
    ResumableClaimWorkflow,
)

APP_DIRECTORY = Path(__file__).resolve().parent
TEMPLATES_DIRECTORY = APP_DIRECTORY / "templates"
STATIC_DIRECTORY = APP_DIRECTORY / "static"

templates = Jinja2Templates(
    directory=str(TEMPLATES_DIRECTORY),
)

@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Initialize and close shared application resources."""

    initialize_database(DATABASE_PATH)

    workflow = ResumableClaimWorkflow(
        checkpoint_path=CHECKPOINT_PATH,
    )

    application.state.claim_workflow = workflow
    try:
        yield
    finally:
        workflow.close()

def create_application() -> FastAPI:
    """Create and configure the ClaimAssist website."""

    application = FastAPI(
        title="ClaimAssist",
        description=(
            "Local evidence-grounded insurance-claim "
            "review application."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIRECTORY),
        ),
        name="static",
    )

    return application

app = create_application()
@app.get(
    "/",
    response_class=HTMLResponse,
    name="home_page",
)
def home_page(request: Request) -> HTMLResponse:
    """Render the claim-submission page."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "error": None,
        },
    )

@app.post(
    "/claims/analyze",
    response_class=HTMLResponse,
    name="analyze_claim",
)
def analyze_claim(
    request: Request,
    claim_text: Annotated[
        str,
        Form(
            min_length=1,
            max_length=20_000,
        ),
    ],
) -> HTMLResponse:
    """Start a resumable ClaimAssist workflow."""

    workflow = request.app.state.claim_workflow

    try:
        invocation = workflow.start(
            claim_text,
            persist_directory=CHROMA_DIRECTORY,
        )
    except Exception as error:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": (
                    "The workflow could not start: "
                    f"{type(error).__name__}"
                ),
            },
            status_code=500,
        )

    return templates.TemplateResponse(
        request=request,
        name="claim_result.html",
        context={
            "workflow": invocation.model_dump(
                mode="json",
            ),
        },
    )

@app.get("/api/health")
def health_check() -> dict[str, object]:
    """Report whether the local application is ready."""

    workflow_ready = hasattr(
        app.state,
        "claim_workflow",
    )

    return {
        "status": "ok",
        "application": "ClaimAssist",
        "database_ready": DATABASE_PATH.exists(),
        "workflow_ready": workflow_ready,
    }
