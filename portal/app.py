from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# Ensure imports work when running `python portal/app.py` from project root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.browser_agent import BrowserAutomationAgent
from agents.orchestrator_factory import create_runtime_orchestrator, orchestrator_mode
from agents.retrieval_agent import PayerPolicyRetrievalAgent
from agents.types import WorkflowTraceStep
from knowledge_base.setup_kb import bootstrap_local_policy_store

logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")


def _configure_otel_console_exporter() -> None:
    """
    Optional local OTel setup for demo visibility.

    Set ENABLE_OTEL_CONSOLE=0 to disable console span export.
    """
    enabled = os.getenv("ENABLE_OTEL_CONSOLE", "1").lower() in {"1", "true", "yes"}
    if not enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except Exception as exc:
        logger.warning("OpenTelemetry console export is unavailable: %s", exc)
        return

    current_provider = trace.get_tracer_provider()
    if isinstance(current_provider, TracerProvider):
        return

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry console exporter enabled.")


_configure_otel_console_exporter()

# Stores submitted PA requests in memory for demo purposes.
submitted_requests: list[dict[str, Any]] = []

# Stores workflow runs for the dashboard.
workflow_runs: dict[str, dict[str, Any]] = {}
workflow_run_order: list[str] = []
workflow_streams: dict[str, list[queue.Queue[dict[str, Any]]]] = {}
workflow_lock = threading.Lock()
MAX_WORKFLOW_RUNS = 100
TERMINAL_STATUSES = {"completed", "failed"}

DEFAULT_TRANSCRIPT = (
    "I need a prior auth for Jane Doe, date of birth March 15 1965, "
    "member ID UHC-4429871. She needs an MRI of the lumbar spine, outpatient, "
    "at our facility. She has been through six weeks of physical therapy with no improvement, "
    "has radiculopathy with L4-L5 disc herniation confirmed on X-ray."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _trace_duration_ms(trace: list[dict[str, Any]]) -> int | None:
    if len(trace) < 2:
        return None
    try:
        start = datetime.fromisoformat(
            str(trace[0]["timestamp"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(
            str(trace[-1]["timestamp"]).replace("Z", "+00:00"))
    except (ValueError, KeyError, TypeError):
        return None
    return max(int((end - start).total_seconds() * 1000), 0)


def _summarize_run(
    run_id: str,
    result: dict[str, Any],
    run_status: str,
    error: str | None = None,
) -> dict[str, Any]:
    coding = result.get("coding") or {}
    necessity = result.get("necessity") or {}
    submission = result.get("submission") or {}
    trace = result.get("trace") or []
    return {
        "id": run_id,
        "run_status": run_status,
        "next_action": result.get("next_action", ""),
        "coding_source": coding.get("source"),
        "diagnosis_code": coding.get("diagnosis_code"),
        "procedure_code": coding.get("procedure_code"),
        "denial_risk_score": necessity.get("denial_risk_score"),
        "submission_status": submission.get("status"),
        "submission_reference": submission.get("reference"),
        "browser_mode": result.get("browser_mode"),
        "orchestrator_mode": result.get("orchestrator_mode"),
        "retrieval_source": result.get("retrieval_source"),
        "duration_ms": _trace_duration_ms(trace),
        "trace_steps": len(trace),
        "error": error,
    }


def _record_snapshot(run_id: str) -> dict[str, Any] | None:
    with workflow_lock:
        record = workflow_runs.get(run_id)
        if record is None:
            return None
        return deepcopy(record)


def _publish_event(run_id: str, event_type: str) -> None:
    snapshot = _record_snapshot(run_id)
    if snapshot is None:
        return

    event = {"type": event_type, "record": snapshot}
    with workflow_lock:
        listeners = list(workflow_streams.get(run_id, []))
    for listener in listeners:
        listener.put(event)


def _execute_workflow(
    transcript: str,
    auto_approve: bool,
    portal_url: str,
    reviewer_approved: bool | None = None,
    trace_hook: Callable[[WorkflowTraceStep], None] | None = None,
) -> dict[str, Any]:
    bootstrap_local_policy_store(overwrite=False)
    browser_agent = BrowserAutomationAgent(portal_base_url=portal_url)
    retrieval_agent = (
        PayerPolicyRetrievalAgent(kb_id="")
        if app.config.get("TESTING")
        else PayerPolicyRetrievalAgent()
    )
    orchestrator = create_runtime_orchestrator(
        browser_agent=browser_agent,
        retrieval_agent=retrieval_agent,
    )
    result = orchestrator.run(
        transcript=transcript,
        auto_approve=auto_approve,
        reviewer_approved=reviewer_approved,
        trace_hook=trace_hook,
    )
    result_dict = result.to_dict()
    result_dict["browser_mode"] = browser_agent.browser_mode
    result_dict["orchestrator_mode"] = orchestrator_mode(orchestrator)
    result_dict["retrieval_source"] = retrieval_agent.retrieval_source
    return result_dict


def _run_workflow_async(
    run_id: str,
    transcript: str,
    auto_approve: bool,
    portal_url: str,
    reviewer_approved: bool | None = None,
) -> None:
    with workflow_lock:
        record = workflow_runs.get(run_id)
        if record is None:
            return
        record["status"] = "running"
        record["updated_at"] = _utc_now_iso()
        record["summary"] = _summarize_run(
            run_id=run_id,
            result=record.get("result", {}),
            run_status="running",
            error=record.get("error"),
        )
    _publish_event(run_id, "run_started")

    def trace_hook(step: WorkflowTraceStep) -> None:
        with workflow_lock:
            record = workflow_runs.get(run_id)
            if record is None:
                return
            result = record.setdefault("result", {"trace": []})
            trace = result.setdefault("trace", [])
            trace.append(step.to_dict())
            record["updated_at"] = _utc_now_iso()
            record["summary"] = _summarize_run(
                run_id=run_id,
                result=result,
                run_status=record.get("status", "running"),
                error=record.get("error"),
            )
        _publish_event(run_id, "trace")

    try:
        result_dict = _execute_workflow(
            transcript=transcript,
            auto_approve=auto_approve,
            portal_url=portal_url,
            reviewer_approved=reviewer_approved,
            trace_hook=trace_hook,
        )
        with workflow_lock:
            record = workflow_runs.get(run_id)
            if record is None:
                return
            record["result"] = result_dict
            record["status"] = "completed"
            record["updated_at"] = _utc_now_iso()
            record["summary"] = _summarize_run(
                run_id=run_id,
                result=result_dict,
                run_status="completed",
                error=None,
            )
            record["error"] = None
        _publish_event(run_id, "run_completed")
    except Exception as exc:
        with workflow_lock:
            record = workflow_runs.get(run_id)
            if record is None:
                return
            record["status"] = "failed"
            record["updated_at"] = _utc_now_iso()
            record["error"] = str(exc)
            result = record.setdefault("result", {"trace": []})
            record["summary"] = _summarize_run(
                run_id=run_id,
                result=result,
                run_status="failed",
                error=str(exc),
            )
        _publish_event(run_id, "run_failed")
    finally:
        _publish_event(run_id, "terminal")


def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
    import json

    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


@app.route("/")
def pa_form():
    """Renders the prior authorization request form."""
    return render_template("pa_form.html")


@app.route("/dashboard")
def dashboard():
    """Renders live workflow dashboard."""
    return render_template("dashboard.html", default_transcript=DEFAULT_TRANSCRIPT)


@app.route("/submit", methods=["POST"])
def submit_pa():
    """
    Receives a submitted PA form and stores it.
    Nova Act or API-run auto-approval path posts to this endpoint.
    """
    data = {
        "patient_name": request.form.get("patient_name"),
        "date_of_birth": request.form.get("date_of_birth"),
        "member_id": request.form.get("member_id"),
        "payer_name": request.form.get("payer_name"),
        "provider_npi": request.form.get("provider_npi"),
        "requested_service": request.form.get("requested_service"),
        "facility_name": request.form.get("facility_name"),
        "diagnosis_code": request.form.get("diagnosis_code"),
        "procedure_code": request.form.get("procedure_code"),
        "clinical_justification": request.form.get("clinical_justification"),
        "policy_id": request.form.get("policy_id"),
        "denial_risk_score": request.form.get("denial_risk_score"),
        "urgency": request.form.get("urgency"),
    }
    submitted_requests.append(data)
    print(
        f"PA Request received: {data['patient_name']} - {data['procedure_code']}")
    return jsonify({"status": "submitted", "reference": f"PA-{len(submitted_requests):04d}"})


@app.route("/requests")
def view_requests():
    """Simple admin view to see submitted PA requests."""
    return jsonify(submitted_requests)


@app.route("/api/runs", methods=["GET"])
def list_runs():
    limit = request.args.get("limit", default=20, type=int)
    limit = max(1, min(limit, 100))
    with workflow_lock:
        run_ids = list(reversed(workflow_run_order[-limit:]))
        summaries = [deepcopy(workflow_runs[run_id]["summary"])
                     for run_id in run_ids if run_id in workflow_runs]
    return jsonify({"runs": summaries, "count": len(summaries)})


@app.route("/api/runs/<run_id>", methods=["GET"])
def get_run(run_id: str):
    snapshot = _record_snapshot(run_id)
    if snapshot is None:
        return jsonify({"error": "run_not_found"}), 404
    return jsonify(snapshot)


@app.route("/api/runs/<run_id>/events", methods=["GET"])
def stream_run_events(run_id: str):
    with workflow_lock:
        if run_id not in workflow_runs:
            return jsonify({"error": "run_not_found"}), 404
        listener: queue.Queue[dict[str, Any]] = queue.Queue()
        workflow_streams.setdefault(run_id, []).append(listener)
        initial = deepcopy(workflow_runs[run_id])

    def generate():
        try:
            yield _format_sse("snapshot", {"record": initial})
            if initial.get("status") in TERMINAL_STATUSES:
                yield _format_sse("terminal", {"record": initial})
                return

            while True:
                try:
                    event = listener.get(timeout=20)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue

                event_type = str(event.get("type", "update"))
                yield _format_sse(event_type, event)
                if event_type == "terminal":
                    return
        finally:
            with workflow_lock:
                listeners = workflow_streams.get(run_id, [])
                if listener in listeners:
                    listeners.remove(listener)
                if not listeners:
                    workflow_streams.pop(run_id, None)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/runs", methods=["POST"])
def create_run():
    body = request.get_json(silent=True) or {}
    transcript = str(body.get("transcript", DEFAULT_TRANSCRIPT)).strip()
    if not transcript:
        return jsonify({"error": "transcript_required"}), 400

    auto_approve = bool(body.get("auto_approve", False))
    portal_url = str(body.get("portal_url", "")
                     ).strip() or request.host_url.rstrip("/")

    run_id = uuid.uuid4().hex[:12]
    created_at = _utc_now_iso()
    record = {
        "id": run_id,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "queued",
        "error": None,
        "request": {
            "transcript": transcript,
            "auto_approve": auto_approve,
            "portal_url": portal_url,
            "reviewer_approved": None,
        },
        "result": {"trace": []},
        "summary": _summarize_run(
            run_id=run_id,
            result={"trace": []},
            run_status="queued",
            error=None,
        ),
    }

    with workflow_lock:
        workflow_runs[run_id] = record
        workflow_run_order.append(run_id)
        while len(workflow_run_order) > MAX_WORKFLOW_RUNS:
            stale_run_id = workflow_run_order.pop(0)
            workflow_runs.pop(stale_run_id, None)
            workflow_streams.pop(stale_run_id, None)

    worker = threading.Thread(
        target=_run_workflow_async,
        kwargs={
            "run_id": run_id,
            "transcript": transcript,
            "auto_approve": auto_approve,
            "portal_url": portal_url,
            "reviewer_approved": None,
        },
        daemon=True,
    )
    worker.start()
    _publish_event(run_id, "run_queued")

    return jsonify(record), 202


@app.route("/api/runs/<run_id>/approve", methods=["POST"])
def approve_run(run_id: str):
    with workflow_lock:
        record = workflow_runs.get(run_id)
        if record is None:
            return jsonify({"error": "run_not_found"}), 404

        run_status = str(record.get("status", ""))
        if run_status == "running":
            return jsonify({"error": "run_in_progress"}), 409

        result = record.get("result") or {}
        submission = result.get("submission") or {}
        submission_status = str(submission.get("status", "")).lower()
        if submission_status != "needs_approval":
            return (
                jsonify(
                    {
                        "error": "run_not_waiting_for_approval",
                        "submission_status": submission_status or None,
                    }
                ),
                409,
            )

        req = record.setdefault("request", {})
        transcript = str(req.get("transcript", "")).strip()
        if not transcript:
            return jsonify({"error": "transcript_missing_for_approval"}), 400

        portal_url = str(req.get("portal_url", "")
                         ).strip() or request.host_url.rstrip("/")
        auto_approve = bool(req.get("auto_approve", False))
        req["reviewer_approved"] = True

        record["status"] = "queued"
        record["error"] = None
        record["updated_at"] = _utc_now_iso()
        record["result"] = {"trace": []}
        record["summary"] = _summarize_run(
            run_id=run_id,
            result=record["result"],
            run_status="queued",
            error=None,
        )
        response_payload = deepcopy(record)

    worker = threading.Thread(
        target=_run_workflow_async,
        kwargs={
            "run_id": run_id,
            "transcript": transcript,
            "auto_approve": auto_approve,
            "portal_url": portal_url,
            "reviewer_approved": True,
        },
        daemon=True,
    )
    worker.start()
    _publish_event(run_id, "run_queued")
    return jsonify(response_payload), 202


@app.route("/health")
def health():
    with workflow_lock:
        run_count = len(workflow_run_order)
    return jsonify(
        {
            "status": "ok",
            "submitted_count": len(submitted_requests),
            "workflow_runs_count": run_count,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
