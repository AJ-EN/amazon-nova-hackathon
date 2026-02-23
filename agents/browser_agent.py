from __future__ import annotations

import os
import uuid

import requests

from agents.types import SubmissionResult


class BrowserAutomationAgent:
    """
    Dual-mode browser agent: real Nova Act automation or HTTP adapter fallback.

    Controlled by USE_REAL_NOVA_ACT env var. Set to "1" for real browser
    automation, leave unset or "0" for the HTTP adapter.
    """

    def __init__(
        self,
        portal_base_url: str = "http://127.0.0.1:5000",
        timeout_seconds: int = 10,
        max_attempts: int = 3,
    ) -> None:
        self.portal_base_url = portal_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.use_real_nova_act = os.getenv("USE_REAL_NOVA_ACT", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        self.browser_mode = "nova_act" if self.use_real_nova_act else "http_adapter"

    def generate_review_snapshot(self, payload: dict[str, str]) -> str:
        return (
            "Human Review Snapshot\n"
            f"- Patient: {payload.get('patient_name', '')}\n"
            f"- Member ID: {payload.get('member_id', '')}\n"
            f"- Payer: {payload.get('payer_name', '')}\n"
            f"- Diagnosis: {payload.get('diagnosis_code', '')}\n"
            f"- Procedure: {payload.get('procedure_code', '')}\n"
            f"- Denial Risk: {payload.get('denial_risk_score', '')}\n"
            f"- Browser Mode: {self.browser_mode}"
        )

    def submit(
        self,
        payload: dict[str, str],
        approved: bool,
        review_snapshot: str,
    ) -> SubmissionResult:
        if not approved:
            return SubmissionResult(
                status="needs_approval",
                message="Submission paused until clinician approves the pre-filled form.",
                reference="",
                review_snapshot=review_snapshot,
                payload=payload,
            )

        if self.use_real_nova_act:
            return self._submit_with_nova_act(payload, review_snapshot)
        return self._submit_with_http_adapter(payload, review_snapshot)

    def _submit_with_nova_act(
        self,
        payload: dict[str, str],
        review_snapshot: str,
    ) -> SubmissionResult:
        """Uses real Nova Act to launch a browser, fill the portal form, and submit."""
        try:
            from nova_act import NovaAct

            portal_url = f"{self.portal_base_url}/"

            with NovaAct(
                starting_page=portal_url,
                user_data_dir="/tmp/nova_act_pa_session",
            ) as agent:
                agent.act(
                    f"Click the Patient Full Name field and type: {payload.get('patient_name', '')}"
                )
                agent.act(
                    f"Click the Date of Birth field and enter the date "
                    f"{payload.get('date_of_birth', '')} in YYYY-MM-DD format"
                )
                agent.act(
                    f"Click the Member ID field and type: {payload.get('member_id', '')}"
                )
                agent.act(
                    f"Click the Provider NPI Number field and type: {payload.get('provider_npi', '')}"
                )
                agent.act(
                    f"Click the Requested Service field and type: {payload.get('requested_service', '')}"
                )
                agent.act(
                    f"Click the Primary Diagnosis Code field and type: {payload.get('diagnosis_code', '')}"
                )
                agent.act(
                    f"Click the Requested Procedure Code field and type: {payload.get('procedure_code', '')}"
                )

                urgency = payload.get("urgency", "routine")
                agent.act(
                    f"Find the Urgency Level dropdown and select the option that matches '{urgency}'"
                )

                justification = payload.get("clinical_justification", "")
                agent.act(
                    f"Click the Clinical Justification textarea and paste the following text: {justification}"
                )

                agent.act("Review the form to make sure all fields are filled in correctly")
                agent.act("Click the Submit Prior Authorization Request button")

            return SubmissionResult(
                status="submitted",
                message="Prior authorization submitted via Nova Act browser automation.",
                reference=self._local_reference(),
                review_snapshot=review_snapshot,
                payload=payload,
            )

        except Exception as exc:
            return SubmissionResult(
                status="failed",
                message=f"Nova Act browser submission failed: {exc}",
                reference="",
                review_snapshot=review_snapshot,
                payload=payload,
            )

    def _submit_with_http_adapter(
        self,
        payload: dict[str, str],
        review_snapshot: str,
    ) -> SubmissionResult:
        """HTTP adapter fallback — used by existing tests and development."""
        last_exception = None
        endpoint = f"{self.portal_base_url}/submit"
        for _ in range(self.max_attempts):
            try:
                response = requests.post(
                    endpoint,
                    data=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                body = response.json()
                return SubmissionResult(
                    status=body.get("status", "submitted"),
                    message="Prior authorization submitted via HTTP adapter.",
                    reference=body.get("reference", self._local_reference()),
                    review_snapshot=review_snapshot,
                    payload=payload,
                )
            except requests.RequestException as exc:
                last_exception = exc

        return SubmissionResult(
            status="failed",
            message=f"HTTP adapter failed after {self.max_attempts} attempts: {last_exception}",
            reference="",
            review_snapshot=review_snapshot,
            payload=payload,
        )

    @staticmethod
    def _local_reference() -> str:
        return f"PA-{uuid.uuid4().hex[:8].upper()}"
