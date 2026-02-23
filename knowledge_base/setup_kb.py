from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_POLICIES_PATH = BASE_DIR / "policies.json"

DEFAULT_POLICIES = [
    {
        "policy_id": "UHC-LUMBAR-MRI-2026",
        "payer_name": "UnitedHealthcare",
        "member_prefixes": ["UHC", "UHG"],
        "title": "Lumbar Spine MRI Medical Necessity Policy",
        "procedure_codes": ["72148", "72149", "72158"],
        "service_keywords": ["lumbar spine mri", "mri lumbar", "lumbar mri"],
        "criteria": [
            {
                "id": "conservative_therapy_6w",
                "description": "Failure of at least 6 weeks of conservative therapy.",
            },
            {
                "id": "radicular_symptoms",
                "description": "Radicular symptoms or neurologic deficit are documented.",
            },
            {
                "id": "objective_imaging_or_exam",
                "description": "Objective imaging or exam findings support lumbar pathology.",
            },
        ],
        "minimum_criteria": 2,
        "required_documents": [
            "progress_notes",
            "conservative_therapy_notes",
            "imaging_report",
        ],
        "denial_patterns": [
            "Missing conservative treatment duration",
            "Insufficient neurologic findings documentation",
        ],
    },
    {
        "policy_id": "AETNA-LUMBAR-MRI-2026",
        "payer_name": "Aetna",
        "member_prefixes": ["AET", "ATN"],
        "title": "Aetna Advanced Imaging Prior Authorization Criteria",
        "procedure_codes": ["72148", "72149", "72158"],
        "service_keywords": ["mri", "lumbar"],
        "criteria": [
            {
                "id": "conservative_therapy_6w",
                "description": "At least 6 weeks of physician-directed conservative treatment.",
            },
            {
                "id": "persistent_pain",
                "description": "Persistent back or leg pain despite treatment.",
            },
            {
                "id": "red_flag_or_neuro_deficit",
                "description": "Neurologic deficit or red-flag symptom is present.",
            },
        ],
        "minimum_criteria": 2,
        "required_documents": [
            "progress_notes",
            "conservative_therapy_notes",
        ],
        "denial_patterns": [
            "No documented treatment progression",
            "Requested service not mapped to diagnosis",
        ],
    },
    {
        "policy_id": "GENERIC-IMAGING-DEFAULT",
        "payer_name": "Generic Payer",
        "member_prefixes": [],
        "title": "Generic Imaging Prior Authorization Criteria",
        "procedure_codes": [],
        "service_keywords": [],
        "criteria": [
            {
                "id": "clinical_indication",
                "description": "Clinical indication for the requested service is documented.",
            },
            {
                "id": "conservative_therapy_6w",
                "description": "Conservative therapy history is provided when appropriate.",
            },
        ],
        "minimum_criteria": 1,
        "required_documents": ["progress_notes"],
        "denial_patterns": ["Demographic mismatch", "Missing clinical rationale"],
    },
]


def bootstrap_local_policy_store(
    destination: Path = DEFAULT_POLICIES_PATH, overwrite: bool = False
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    destination.write_text(json.dumps(DEFAULT_POLICIES, indent=2), encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local policy store for payer-specific PA reasoning."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_POLICIES_PATH),
        help="Destination file path for policy JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    written = bootstrap_local_policy_store(output_path, overwrite=args.overwrite)
    print(f"Policy store ready: {written}")
