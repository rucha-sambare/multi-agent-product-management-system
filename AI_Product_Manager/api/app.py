"""Minimal FastAPI wrapper for long-running analysis jobs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from AI_Product_Manager.orchestar_agent.orchestrator import FeedbackOrchestrator
from AI_Product_Manager.feedback_collection_agent.scrapper import AmbiguousAppError
from AI_Product_Manager.workflow_state import WorkflowStore

app = FastAPI(title="AI Product Manager", version="1.0.0")
store = WorkflowStore()
executor = ThreadPoolExecutor(max_workers=2)
logger = logging.getLogger("ai_product_manager.api")


class AnalysisRequest(BaseModel):
    app_name: str = Field(min_length=1, max_length=150)
    app_id: str | None = Field(default=None, max_length=250)
    review_count: int = Field(default=200, ge=1, le=5000)


class ReviewDecision(BaseModel):
    approved: bool
    notes: str = Field(default="", max_length=2000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    try:
        return FeedbackOrchestrator().run(
            request.app_name, request.review_count, request.app_id
        )
    except AmbiguousAppError as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "candidates": exc.candidates},
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _background_run(request: AnalysisRequest, run_id: str):
    try:
        FeedbackOrchestrator(store=store).run(
            request.app_name, request.review_count, request.app_id, run_id
        )
    except Exception:
        logger.exception("Background analysis failed for %s", request.app_name)


@app.post("/jobs", status_code=202)
def create_job(request: AnalysisRequest):
    """Queue work without holding an HTTP connection during scraping/model inference."""
    run_id = store.create(request.app_name, request.app_id)
    executor.submit(_background_run, request, run_id)
    return {"status": "accepted", "run_id": run_id}


@app.get("/jobs")
def list_jobs(limit: int = 20):
    return store.list(min(max(limit, 1), 100))


@app.get("/jobs/{run_id}")
def get_job(run_id: str):
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@app.post("/jobs/{run_id}/review")
def review_job(run_id: str, decision: ReviewDecision):
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in {"needs_review", "completed"}:
        raise HTTPException(status_code=409, detail="Run is not awaiting review")
    status = "completed" if decision.approved else "rejected"
    store.update(
        run_id,
        status=status,
        step="human_reviewed",
        payload={"human_review": decision.model_dump()},
    )
    return store.get(run_id)
