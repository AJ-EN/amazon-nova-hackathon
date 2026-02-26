from __future__ import annotations

import os
import time
import uuid

import requests

from agents.types import SubmissionResult

# Typing delay range in ms — makes the demo look like a human typing
_TYPE_DELAY_MIN = 30
_TYPE_DELAY_MAX = 80


class BrowserAutomationAgent:
    """
    Tri-mode browser agent for portal form submission.

    Modes (controlled by USE_BROWSER_AUTOMATION env var):
      - "playwright"  → Real Chromium browser automation (visual demo)
      - "nova_act"    → Nova Act browser automation (if API key available)
      - unset / "0"   → HTTP adapter fallback (fast, for tests)
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

        mode = os.getenv("USE_BROWSER_AUTOMATION", "playwright").lower().strip()
        if mode in {"playwright", "pw"}:
            self.browser_mode = "playwright"
        elif mode in {"nova_act", "nova", "1", "true", "yes"}:
            self.browser_mode = "nova_act"
        else:
            self.browser_mode = "http_adapter"

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

        if self.browser_mode == "playwright":
            return self._submit_with_playwright(payload, review_snapshot)
        if self.browser_mode == "nova_act":
            return self._submit_with_nova_act(payload, review_snapshot)
        return self._submit_with_http_adapter(payload, review_snapshot)

    # ------------------------------------------------------------------
    # Playwright path — visual browser automation for demo
    # ------------------------------------------------------------------

    def _submit_with_playwright(
        self,
        payload: dict[str, str],
        review_snapshot: str,
    ) -> SubmissionResult:
        """Launches a visible Chromium browser, fills the PA form field-by-field, and submits."""
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=False, slow_mo=120)
                page = browser.new_page()
                page.goto(f"{self.portal_base_url}/", wait_until="networkidle")

                # Text input fields — type with human-like delay
                field_map = {
                    "#patient_name": payload.get("patient_name", ""),
                    "#member_id": payload.get("member_id", ""),
                    "#payer_name": payload.get("payer_name", ""),
                    "#provider_npi": payload.get("provider_npi", ""),
                    "#requested_service": payload.get("requested_service", ""),
                    "#facility_name": payload.get("facility_name", ""),
                    "#diagnosis_code": payload.get("diagnosis_code", ""),
                    "#procedure_code": payload.get("procedure_code", ""),
                    "#policy_id": payload.get("policy_id", ""),
                    "#denial_risk_score": payload.get("denial_risk_score", ""),
                }
                for selector, value in field_map.items():
                    if value:
                        page.click(selector)
                        page.fill(selector, "")
                        page.type(selector, value, delay=_TYPE_DELAY_MIN)

                # Date field — needs special handling (input type="date")
                dob = payload.get("date_of_birth", "")
                if dob:
                    page.evaluate(
                        f"document.querySelector('#date_of_birth').value = '{dob}'"
                    )
                    page.dispatch_event("#date_of_birth", "change")

                # Select dropdown
                urgency = payload.get("urgency", "routine")
                page.select_option("#urgency", urgency)

                # Textarea
                justification = payload.get("clinical_justification", "")
                if justification:
                    page.click("#clinical_justification")
                    page.fill("#clinical_justification", "")
                    page.type(
                        "#clinical_justification",
                        justification,
                        delay=_TYPE_DELAY_MIN,
                    )

                # Pause briefly so viewer can see the filled form
                time.sleep(1.5)

                # Submit
                page.click("button[type='submit']")
                page.wait_for_load_state("networkidle")

                # Keep browser visible for a moment after submission
                time.sleep(2)
                browser.close()

            return SubmissionResult(
                status="submitted",
                message="Prior authorization submitted via Playwright browser automation.",
                reference=self._local_reference(),
                review_snapshot=review_snapshot,
                payload=payload,
            )

        except Exception as exc:
            return SubmissionResult(
                status="failed",
                message=f"Playwright browser submission failed: {exc}",
                reference="",
                review_snapshot=review_snapshot,
                payload=payload,
            )

    # ------------------------------------------------------------------
    # Nova Act path — kept for when API key becomes available
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # HTTP adapter — fast fallback for tests and development
    # ------------------------------------------------------------------

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
