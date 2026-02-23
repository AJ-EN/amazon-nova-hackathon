from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.types import PolicyMatch
from knowledge_base.setup_kb import DEFAULT_POLICIES, DEFAULT_POLICIES_PATH


class PayerPolicyRetrievalAgent:
    """Retrieves payer criteria from a local policy store for the demo."""

    def __init__(self, policy_path: Path | None = None) -> None:
        self.policy_path = policy_path or DEFAULT_POLICIES_PATH
        self._policies = self._load_policies()

    def _load_policies(self) -> list[dict[str, Any]]:
        if self.policy_path.exists():
            try:
                return json.loads(self.policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return DEFAULT_POLICIES
        return DEFAULT_POLICIES

    def retrieve(
        self,
        payer_name: str,
        member_id: str,
        procedure_code: str,
        requested_service: str,
    ) -> PolicyMatch:
        service = requested_service.lower()
        member_prefix = member_id.split("-", 1)[0].upper() if member_id else ""
        payer_name_lower = payer_name.lower()

        best_policy = None
        best_score = -1
        for policy in self._policies:
            score = self._score_policy(
                policy=policy,
                payer_name_lower=payer_name_lower,
                member_prefix=member_prefix,
                procedure_code=procedure_code,
                service=service,
            )
            if score > best_score:
                best_policy = policy
                best_score = score

        selected = best_policy if best_policy is not None else DEFAULT_POLICIES[-1]
        return PolicyMatch(
            policy_id=selected["policy_id"],
            payer_name=selected["payer_name"],
            title=selected["title"],
            criteria=selected["criteria"],
            minimum_criteria=int(selected.get("minimum_criteria", 1)),
            required_documents=selected.get("required_documents", []),
            denial_patterns=selected.get("denial_patterns", []),
        )

    @staticmethod
    def _score_policy(
        policy: dict[str, Any],
        payer_name_lower: str,
        member_prefix: str,
        procedure_code: str,
        service: str,
    ) -> int:
        score = 0

        policy_payer = str(policy.get("payer_name", "")).lower()
        if policy_payer and policy_payer in payer_name_lower:
            score += 7

        prefixes = [prefix.upper() for prefix in policy.get("member_prefixes", [])]
        if member_prefix and member_prefix in prefixes:
            score += 8

        codes = policy.get("procedure_codes", [])
        if procedure_code and procedure_code in codes:
            score += 4

        for keyword in policy.get("service_keywords", []):
            if keyword.lower() in service:
                score += 2

        return score

