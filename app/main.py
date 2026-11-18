from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin import router as admin_router
from app.api.agent import router as agent_router
from app.api.chat import router as chat_router
from app.api.extract import router as extract_router
from app.api.jobs import router as jobs_router
from app.api.kb import router as kb_router
from app.api.rageval import router as rageval_router

app = FastAPI(title="MewHelp", version="0.1.0")
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(extract_router)
app.include_router(kb_router)
app.include_router(jobs_router)
app.include_router(admin_router)
app.include_router(rageval_router)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse("/static/index.html")


@app.get("/kb", include_in_schema=False)
async def kb_page() -> RedirectResponse:
    return RedirectResponse("/static/kb.html")


@app.get("/admin", include_in_schema=False)
async def admin_page() -> RedirectResponse:
    return RedirectResponse("/static/admin.html")


@app.get("/rag-eval", include_in_schema=False)
async def rageval_page() -> RedirectResponse:
    return RedirectResponse("/static/rageval.html")
