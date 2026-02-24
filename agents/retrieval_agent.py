from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agents.types import PolicyMatch
from knowledge_base.setup_kb import DEFAULT_POLICIES, DEFAULT_POLICIES_PATH


class PayerPolicyRetrievalAgent:
    """
    Dual-mode policy retrieval agent.

    When BEDROCK_KB_ID is set, queries the Bedrock Knowledge Base using
    the KB's configured embedding model for semantic retrieval (real RAG).
    Otherwise falls back to local JSON scoring (offline development).
    """

    def __init__(
        self,
        policy_path: Path | None = None,
        kb_id: str | None = None,
        kb_client: Any | None = None,
    ) -> None:
        self.policy_path = policy_path or DEFAULT_POLICIES_PATH
        self._policies: list[dict[str, Any]] | None = None
        self.kb_id = kb_id if kb_id is not None else self._resolve_kb_id()
        self.retrieval_source = "bedrock_kb" if self.kb_id else "local"
        self._kb_client = kb_client

    def _resolve_kb_id(self) -> str:
        env_kb_id = os.getenv("BEDROCK_KB_ID", "").strip()
        if env_kb_id:
            return env_kb_id

        config_path = Path(__file__).resolve().parents[1] / "knowledge_base" / "kb_config.json"
        if not config_path.exists():
            return ""

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""

        kb_id = config.get("knowledge_base_id", "")
        return str(kb_id).strip()

    def _load_policies(self) -> list[dict[str, Any]]:
        if self.policy_path.exists():
            try:
                return json.loads(self.policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return DEFAULT_POLICIES
        return DEFAULT_POLICIES

    def _get_local_policies(self) -> list[dict[str, Any]]:
        if self._policies is None:
            self._policies = self._load_policies()
        return self._policies

    def retrieve(
        self,
        payer_name: str,
        member_id: str,
        procedure_code: str,
        requested_service: str,
    ) -> PolicyMatch:
        if self.kb_id:
            try:
                return self._retrieve_from_bedrock_kb(
                    payer_name=payer_name,
                    member_id=member_id,
                    procedure_code=procedure_code,
                    requested_service=requested_service,
                )
            except Exception:
                self.retrieval_source = "local_fallback"

        return self._retrieve_from_local(
            payer_name=payer_name,
            member_id=member_id,
            procedure_code=procedure_code,
            requested_service=requested_service,
        )

    # ------------------------------------------------------------------
    # Bedrock Knowledge Base retrieval (real RAG)
    # ------------------------------------------------------------------

    def _retrieve_from_bedrock_kb(
        self,
        payer_name: str,
        member_id: str,
        procedure_code: str,
        requested_service: str,
    ) -> PolicyMatch:
        self._ensure_kb_client()

        query = (
            f"Prior authorization policy for {payer_name} member {member_id} "
            f"requesting {requested_service} with procedure code {procedure_code}"
        )

        response = self._kb_client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            },
        )

        results = response.get("retrievalResults", [])
        if not results:
            raise ValueError("No results returned from Bedrock KB")

        matched_policy = self._pick_best_kb_policy(
            results=results,
            payer_name=payer_name,
            procedure_code=procedure_code,
            requested_service=requested_service,
        )
        self.retrieval_source = "bedrock_kb"
        return matched_policy

    def _pick_best_kb_policy(
        self,
        results: list[dict[str, Any]],
        payer_name: str,
        procedure_code: str,
        requested_service: str,
    ) -> PolicyMatch:
        best_policy: PolicyMatch | None = None
        best_score = float("-inf")
        parse_errors = 0

        for result in results:
            kb_text = self._extract_kb_text(result)
            if not kb_text:
                continue

            try:
                policy = self._parse_policy_from_kb_text(kb_text)
            except ValueError:
                parse_errors += 1
                continue

            score = self._score_kb_candidate(
                policy=policy,
                kb_text=kb_text,
                payer_name=payer_name,
                procedure_code=procedure_code,
                requested_service=requested_service,
                kb_score=result.get("score"),
            )
            if score > best_score:
                best_policy = policy
                best_score = score

        if best_policy is None:
            raise ValueError(f"Unable to parse KB retrieval results ({parse_errors} parse errors).")
        return best_policy

    @staticmethod
    def _extract_kb_text(result: dict[str, Any]) -> str:
        content = result.get("content", {})
        if not isinstance(content, dict):
            return ""
        text = content.get("text", "")
        return text.strip() if isinstance(text, str) else ""

    def _parse_policy_from_kb_text(self, kb_text: str) -> PolicyMatch:
        kb_text = self._normalize_kb_text(kb_text)
        policy_id = self._extract_header_field(kb_text, "Policy ID")
        payer_name = self._extract_header_field(kb_text, "Payer")
        title = self._extract_header_field(kb_text, "Title")
        criteria = self._extract_criteria(kb_text)
        minimum_criteria = self._extract_minimum_criteria(kb_text, len(criteria))
        required_documents = self._extract_bullet_section(
            kb_text,
            header="Required Documents",
            normalize_documents=True,
        )
        denial_patterns = self._extract_bullet_section(
            kb_text,
            header="Common Denial Patterns",
            normalize_documents=False,
        )

        if not policy_id:
            raise ValueError("Missing policy_id in KB result.")
        if not payer_name:
            raise ValueError("Missing payer_name in KB result.")
        if not title:
            raise ValueError("Missing title in KB result.")
        if not criteria:
            raise ValueError("Missing criteria in KB result.")

        return PolicyMatch(
            policy_id=policy_id,
            payer_name=payer_name,
            title=title,
            criteria=criteria,
            minimum_criteria=max(1, minimum_criteria),
            required_documents=required_documents,
            denial_patterns=denial_patterns,
        )

    @staticmethod
    def _normalize_kb_text(kb_text: str) -> str:
        """Insert line breaks before known header patterns so parsers work on single-line KB chunks."""
        if "\n" in kb_text and len(kb_text.splitlines()) > 3:
            return kb_text  # already multi-line

        # Headers that should start on a new line
        headers = [
            r"Policy ID:",
            r"Payer:",
            r"Member ID Prefixes:",
            r"Title:",
            r"Covered Procedure Codes:",
            r"Service Keywords:",
            r"Medical Necessity Criteria\s*\(",
            r"Required Documents:",
            r"Common Denial Patterns:",
        ]
        # Also break before numbered items like "1. criterion_id:"
        headers.append(r"\d+\.\s+[a-zA-Z_]+:")
        # And bullet items "- item"
        headers.append(r"-\s+\S")

        for h in headers:
            kb_text = re.sub(rf"\s+({h})", r"\n\1", kb_text)

        # Ensure numbered criteria start on their own line even when
        # attached to the "Medical Necessity Criteria (...met):" header.
        kb_text = re.sub(
            r"(must be met\):\s*)", r"\1\n", kb_text, flags=re.IGNORECASE
        )

        return kb_text.strip()

    @staticmethod
    def _extract_header_field(kb_text: str, field_name: str) -> str:
        pattern = re.compile(
            rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        match = pattern.search(kb_text)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _extract_minimum_criteria(kb_text: str, criteria_count: int) -> int:
        pattern = re.compile(
            r"Medical Necessity Criteria\s*\(minimum\s+(\d+)\s+of\s+\d+\s+must\s+be\s+met\)",
            flags=re.IGNORECASE,
        )
        match = pattern.search(kb_text)
        if not match:
            return 1 if criteria_count else 0
        try:
            return int(match.group(1))
        except ValueError:
            return 1 if criteria_count else 0

    @staticmethod
    def _extract_criteria(kb_text: str) -> list[dict[str, str]]:
        # Matches lines like: "1. criterion_id: Criterion description"
        pattern = re.compile(r"^\s*\d+\.\s*([a-zA-Z0-9_]+)\s*:\s*(.+?)\s*$", flags=re.MULTILINE)
        criteria: list[dict[str, str]] = []
        for match in pattern.finditer(kb_text):
            criterion_id = match.group(1).strip()
            description = match.group(2).strip()
            if criterion_id and description:
                criteria.append({"id": criterion_id, "description": description})
        return criteria

    def _extract_bullet_section(
        self,
        kb_text: str,
        header: str,
        normalize_documents: bool,
    ) -> list[str]:
        lines = kb_text.splitlines()
        header_index = None
        header_lower = header.lower()
        for idx, line in enumerate(lines):
            if line.strip().lower().startswith(f"{header_lower}:"):
                header_index = idx
                break

        if header_index is None:
            return []

        items: list[str] = []
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                if items:
                    break
                continue
            if stripped.endswith(":") and not stripped.startswith("-"):
                break
            if not stripped.startswith("-"):
                if items:
                    break
                continue

            value = stripped[1:].strip()
            if not value:
                continue
            if normalize_documents:
                value = self._normalize_document_name(value)
            items.append(value)

        return items

    @staticmethod
    def _normalize_document_name(value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    @staticmethod
    def _score_kb_candidate(
        policy: PolicyMatch,
        kb_text: str,
        payer_name: str,
        procedure_code: str,
        requested_service: str,
        kb_score: Any,
    ) -> float:
        score = 0.0

        try:
            score += float(kb_score)
        except (TypeError, ValueError):
            pass

        payer_lower = payer_name.lower().strip()
        policy_payer = policy.payer_name.lower().strip()
        if payer_lower and policy_payer:
            if payer_lower == policy_payer:
                score += 10.0
            elif payer_lower in policy_payer or policy_payer in payer_lower:
                score += 4.0

        kb_text_lower = kb_text.lower()
        if procedure_code and procedure_code in kb_text:
            score += 2.0

        service_tokens = [token for token in requested_service.lower().split() if len(token) >= 4]
        token_hits = sum(1 for token in service_tokens if token in kb_text_lower)
        score += min(token_hits, 3)

        return score

    def _ensure_kb_client(self) -> None:
        if self._kb_client is None:
            from utils.bedrock_client import get_bedrock_agent_runtime_client
            self._kb_client = get_bedrock_agent_runtime_client()

    # ------------------------------------------------------------------
    # Local policy retrieval (offline fallback)
    # ------------------------------------------------------------------

    def _retrieve_from_local(
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
        for policy in self._get_local_policies():
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
        return self._policy_dict_to_match(selected)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_dict_to_match(policy: dict[str, Any]) -> PolicyMatch:
        return PolicyMatch(
            policy_id=policy["policy_id"],
            payer_name=policy["payer_name"],
            title=policy["title"],
            criteria=policy["criteria"],
            minimum_criteria=int(policy.get("minimum_criteria", 1)),
            required_documents=policy.get("required_documents", []),
            denial_patterns=policy.get("denial_patterns", []),
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
