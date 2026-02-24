from __future__ import annotations

import logging
import os
from typing import Any

from agents.orchestrator import PriorAuthOrchestrator
from agents.strands_orchestrator import StrandsPriorAuthOrchestrator, strands_available

logger = logging.getLogger(__name__)


OrchestratorType = PriorAuthOrchestrator | StrandsPriorAuthOrchestrator


def create_runtime_orchestrator(**kwargs: Any) -> OrchestratorType:
    """
    Build runtime orchestrator.

    ORCHESTRATOR_MODE:
    - "strands" (default): use Strands wrapper when available.
    - "legacy": use existing in-process orchestrator.
    """
    mode = os.getenv("ORCHESTRATOR_MODE", "strands").strip().lower()
    if mode == "legacy":
        return PriorAuthOrchestrator(**kwargs)

    if strands_available():
        try:
            return StrandsPriorAuthOrchestrator(**kwargs)
        except Exception as exc:  # pragma: no cover - safety fallback for runtime.
            logger.warning("Falling back to legacy orchestrator: %s", exc)
    else:
        logger.warning("Strands unavailable; falling back to legacy orchestrator.")

    return PriorAuthOrchestrator(**kwargs)


def orchestrator_mode(orchestrator: OrchestratorType) -> str:
    if isinstance(orchestrator, StrandsPriorAuthOrchestrator):
        return "strands"
    return "legacy"
