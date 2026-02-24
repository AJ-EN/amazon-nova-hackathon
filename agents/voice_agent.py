from __future__ import annotations

import re
from datetime import datetime

from agents.types import ExtractedClinicalData


class VoiceIntakeAgent:
    """
    Offline-first voice intake parser.
    In production this node would be backed by Nova 2 Sonic streaming + tool calls.
    """

    PAYER_ALIASES = {
        "unitedhealthcare": "UnitedHealthcare",
        "uhc": "UnitedHealthcare",
        "aetna": "Aetna",
        "cigna": "Cigna",
        "humana": "Humana",
        "medicare": "Medicare",
    }

    MEMBER_PREFIX_TO_PAYER = {
        "UHC": "UnitedHealthcare",
        "UHG": "UnitedHealthcare",
        "AET": "Aetna",
        "ATN": "Aetna",
        "CIG": "Cigna",
    }

    def __init__(self, default_provider_npi: str = "1234567890") -> None:
        self.default_provider_npi = default_provider_npi

    def ingest(self, transcript: str) -> ExtractedClinicalData:
        normalized = " ".join(transcript.strip().split())
        lowered = normalized.lower()

        patient_name = self._extract_patient_name(normalized)
        date_of_birth = self._extract_date_of_birth(normalized)
        member_id = self._extract_member_id(normalized)
        payer_name = self._infer_payer(lowered, member_id)
        requested_service = self._infer_requested_service(lowered)
        facility_name = self._extract_facility(normalized)
        urgency = self._infer_urgency(lowered)
        provider_npi = self._extract_provider_npi(normalized) or self.default_provider_npi
        conservative_weeks = self._extract_conservative_therapy_weeks(lowered)
        imaging_evidence = self._extract_imaging_evidence(normalized)
        findings = self._extract_clinical_findings(lowered)

        return ExtractedClinicalData(
            transcript=normalized,
            patient_name=patient_name,
            date_of_birth=date_of_birth,
            member_id=member_id,
            payer_name=payer_name,
            provider_npi=provider_npi,
            requested_service=requested_service,
            facility_name=facility_name,
            urgency=urgency,
            conservative_therapy_weeks=conservative_weeks,
            imaging_evidence=imaging_evidence,
            clinical_findings=findings,
            notes=normalized,
        )

    @staticmethod
    def _extract_patient_name(transcript: str) -> str:
        match = re.search(r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})", transcript)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _extract_date_of_birth(transcript: str) -> str:
        dob_match = re.search(
            r"date of birth\s+([A-Za-z]+\s+\d{1,2}(?:,\s*|\s+)\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})",
            transcript,
            flags=re.IGNORECASE,
        )
        if not dob_match:
            return ""

        raw = dob_match.group(1).strip()
        for date_format in (
            "%B %d %Y",
            "%B %d, %Y",
            "%b %d %Y",
            "%b %d, %Y",
            "%m/%d/%Y",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(raw, date_format).date().isoformat()
            except ValueError:
                continue
        return raw

    @staticmethod
    def _extract_member_id(transcript: str) -> str:
        match = re.search(r"\b([A-Z]{2,5}-\d{5,12})\b", transcript)
        return match.group(1) if match else ""

    def _infer_payer(self, lowered_transcript: str, member_id: str) -> str:
        for alias, canonical in self.PAYER_ALIASES.items():
            if alias in lowered_transcript:
                return canonical

        if member_id:
            prefix = member_id.split("-", 1)[0].upper()
            if prefix in self.MEMBER_PREFIX_TO_PAYER:
                return self.MEMBER_PREFIX_TO_PAYER[prefix]
        return "Generic Payer"

    @staticmethod
    def _extract_provider_npi(transcript: str) -> str:
        match = re.search(r"\bNPI\s*(\d{10})\b", transcript, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _infer_requested_service(lowered_transcript: str) -> str:
        if "mri" in lowered_transcript and "lumbar" in lowered_transcript:
            return "MRI lumbar spine without contrast"
        if "mri" in lowered_transcript:
            return "MRI study"
        if "ct" in lowered_transcript:
            return "CT study"
        return "Imaging service"

    @staticmethod
    def _extract_facility(transcript: str) -> str:
        match = re.search(
            r"\bat\s+(?:our\s+)?([A-Za-z0-9 .'-]+?)(?:[,.]|$)",
            transcript,
            flags=re.IGNORECASE,
        )
        if not match:
            return "Demo Outpatient Center"
        facility = match.group(1).strip()
        if facility.lower() in {"our facility", "facility"}:
            return "Demo Outpatient Center"
        return facility

    @staticmethod
    def _infer_urgency(lowered_transcript: str) -> str:
        if "emergent" in lowered_transcript or "immediate" in lowered_transcript:
            return "emergent"
        if "urgent" in lowered_transcript or "24 hours" in lowered_transcript:
            return "urgent"
        return "routine"

    @staticmethod
    def _extract_conservative_therapy_weeks(lowered_transcript: str) -> int | None:
        digit_match = re.search(
            r"(\d+)\s+weeks?\s+of\s+(?:physical\s+therapy|conservative)",
            lowered_transcript,
        )
        if digit_match:
            return int(digit_match.group(1))

        word_match = re.search(
            r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?\s+of\s+"
            r"(?:physical\s+therapy|conservative)",
            lowered_transcript,
        )
        if word_match:
            word_to_number = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
                "eleven": 11,
                "twelve": 12,
            }
            return word_to_number[word_match.group(1)]
        return None

    @staticmethod
    def _extract_imaging_evidence(transcript: str) -> str:
        match = re.search(
            r"(confirmed on [^,.]+|seen on [^,.]+|demonstrated on [^,.]+)",
            transcript,
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_clinical_findings(lowered_transcript: str) -> list[str]:
        findings: list[str] = []

        keyword_to_finding = {
            "radiculopathy": "Radiculopathy",
            "disc herniation": "Disc herniation",
            "numbness": "Numbness",
            "weakness": "Motor weakness",
            "back pain": "Back pain",
            "sciatica": "Sciatica symptoms",
        }
        for keyword, label in keyword_to_finding.items():
            if keyword in lowered_transcript:
                findings.append(label)

        if not findings:
            findings.append("General clinical indication documented")
        return findings
