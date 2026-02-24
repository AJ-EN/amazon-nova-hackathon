from __future__ import annotations

import argparse
import json

from agents.browser_agent import BrowserAutomationAgent
from agents.orchestrator_factory import create_runtime_orchestrator, orchestrator_mode
from knowledge_base.setup_kb import bootstrap_local_policy_store

DEFAULT_TRANSCRIPT = (
    "I need a prior auth for Jane Doe, date of birth March 15 1965, "
    "member ID UHC-4429871. She needs an MRI of the lumbar spine, outpatient, "
    "at our facility. She has been through six weeks of physical therapy with no improvement, "
    "has radiculopathy with L4-L5 disc herniation confirmed on X-ray."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PriorAuth Agent voice-to-submission workflow."
    )
    parser.add_argument(
        "--transcript",
        default=DEFAULT_TRANSCRIPT,
        help="Clinician speech transcript to process.",
    )
    parser.add_argument(
        "--portal-url",
        default="http://127.0.0.1:5000",
        help="Mock payer portal base URL.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve HITL step and submit immediately.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_local_policy_store()

    browser_agent = BrowserAutomationAgent(portal_base_url=args.portal_url)
    orchestrator = create_runtime_orchestrator(browser_agent=browser_agent)
    result = orchestrator.run(
        transcript=args.transcript,
        auto_approve=args.auto_approve,
    )
    result_dict = result.to_dict()
    result_dict["orchestrator_mode"] = orchestrator_mode(orchestrator)

    print("\n=== Workflow Trace ===")
    for step in result.trace:
        print(f"[{step.timestamp}] {step.step} :: {step.status} :: {step.detail}")

    print("\n=== Structured Output ===")
    print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
