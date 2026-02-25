"""Metrics and judge statistics endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Request

from api.models import JudgeStatsResponse, MetricsResponse
from kernel.round_metrics import compute_round_metrics

router = APIRouter(prefix="/v1/world")


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    """Return metrics for the latest round (or current state if no rounds settled)."""
    kernel = request.app.state.leviathan["kernel"]
    receipt = kernel.get_round_receipt()
    if receipt and receipt.round_metrics:
        return MetricsResponse(round_id=receipt.round_id, **receipt.round_metrics)
    snap = kernel.get_snapshot()
    metrics = compute_round_metrics(members=snap.members)
    return MetricsResponse(round_id=snap.round_id, **metrics)


@router.get("/metrics/history", response_model=List[MetricsResponse])
def get_metrics_history(request: Request, limit: Optional[int] = None) -> List[MetricsResponse]:
    """Return metrics for recent rounds from the event log."""
    event_log = request.app.state.leviathan["event_log"]
    history = []
    for event in event_log:
        if event.event_type == "round_settled":
            rm = event.payload.get("round_metrics", {})
            if rm:
                history.append(MetricsResponse(round_id=event.round_id, **rm))
    if limit:
        history = history[-limit:]
    return history


@router.get("/judge/stats", response_model=JudgeStatsResponse)
def get_judge_stats(request: Request) -> JudgeStatsResponse:
    """Return judge approval statistics."""
    judge = request.app.state.leviathan["judge"]
    history = getattr(judge, "_judgment_history", [])
    total = len(history)
    approved = sum(1 for j in history if j.get("approved"))
    rejected = total - approved
    recent_rejections = [j for j in history if not j.get("approved")][-5:]
    return JudgeStatsResponse(
        total_judgments=total, approved=approved, rejected=rejected,
        approval_rate=approved / total if total > 0 else 0.0,
        recent_rejections=recent_rejections,
    )
