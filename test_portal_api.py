from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

from portal.app import TERMINAL_STATUSES, app, workflow_run_order, workflow_runs, workflow_streams

SAMPLE_TRANSCRIPT = (
    "I need a prior auth for Jane Doe, date of birth March 15 1965, "
    "member ID UHC-4429871. She needs an MRI of the lumbar spine, outpatient, "
    "at our facility. She has been through six weeks of physical therapy with no improvement, "
    "has radiculopathy with L4-L5 disc herniation confirmed on X-ray."
)


class TestPortalApi(unittest.TestCase):
    def setUp(self) -> None:
        app.config["TESTING"] = True
        self.client = app.test_client()
        workflow_runs.clear()
        workflow_run_order.clear()
        workflow_streams.clear()

    def _wait_for_terminal_run(self, run_id: str, timeout_seconds: float = 10.0) -> dict:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            response = self.client.get(f"/api/runs/{run_id}")
            if response.status_code != 200:
                time.sleep(0.05)
                continue
            payload = response.get_json() or {}
            if payload.get("status") in TERMINAL_STATUSES:
                return payload
            time.sleep(0.05)
        self.fail(f"Run {run_id} did not reach terminal status within {timeout_seconds} seconds.")

    def test_dashboard_page_renders(self) -> None:
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PriorAuth Agent Control Room", response.data)

    def test_create_and_fetch_workflow_run(self) -> None:
        with patch.dict(
            os.environ,
            {"USE_NOVA_REASONING": "0", "USE_NOVA_JUSTIFICATION": "0"},
            clear=False,
        ):
            create_response = self.client.post(
                "/api/runs",
                json={"transcript": SAMPLE_TRANSCRIPT, "auto_approve": False},
            )

        self.assertEqual(create_response.status_code, 202)
        payload = create_response.get_json()
        self.assertIsNotNone(payload)
        assert payload is not None
        run_id = payload["id"]
        self.assertIn(payload["summary"]["run_status"], {"queued", "running", "completed"})

        final_payload = self._wait_for_terminal_run(run_id)
        self.assertEqual(final_payload["status"], "completed")
        self.assertEqual(final_payload["summary"]["next_action"], "human_review_required")
        self.assertEqual(final_payload["summary"]["submission_status"], "needs_approval")

        detail_payload = self.client.get(f"/api/runs/{run_id}").get_json() or {}
        self.assertEqual(detail_payload["id"], run_id)

        list_response = self.client.get("/api/runs?limit=5")
        self.assertEqual(list_response.status_code, 200)
        list_payload = list_response.get_json()
        self.assertIsNotNone(list_payload)
        assert list_payload is not None
        self.assertGreaterEqual(list_payload["count"], 1)

    def test_events_endpoint_streams_snapshot(self) -> None:
        with patch.dict(
            os.environ,
            {"USE_NOVA_REASONING": "0", "USE_NOVA_JUSTIFICATION": "0"},
            clear=False,
        ):
            create_response = self.client.post(
                "/api/runs",
                json={"transcript": SAMPLE_TRANSCRIPT, "auto_approve": False},
            )

        run_id = (create_response.get_json() or {})["id"]
        response = self.client.get(f"/api/runs/{run_id}/events", buffered=False)
        self.assertEqual(response.status_code, 200)
        first_chunk = next(response.response).decode("utf-8")
        self.assertIn("event: snapshot", first_chunk)
        response.close()

    def test_approve_endpoint_requeues_waiting_run(self) -> None:
        with patch.dict(
            os.environ,
            {"USE_NOVA_REASONING": "0", "USE_NOVA_JUSTIFICATION": "0"},
            clear=False,
        ):
            create_response = self.client.post(
                "/api/runs",
                json={"transcript": SAMPLE_TRANSCRIPT, "auto_approve": False},
            )

        self.assertEqual(create_response.status_code, 202)
        run_id = (create_response.get_json() or {})["id"]

        initial_final = self._wait_for_terminal_run(run_id)
        self.assertEqual(initial_final["summary"]["submission_status"], "needs_approval")

        approve_response = self.client.post(f"/api/runs/{run_id}/approve", json={})
        self.assertEqual(approve_response.status_code, 202)
        approve_payload = approve_response.get_json() or {}
        self.assertEqual(approve_payload.get("status"), "queued")

        approved_final = self._wait_for_terminal_run(run_id)
        self.assertIn(approved_final["summary"]["submission_status"], {"submitted", "failed"})
        self.assertTrue((approved_final.get("request") or {}).get("reviewer_approved"))


if __name__ == "__main__":
    unittest.main()
