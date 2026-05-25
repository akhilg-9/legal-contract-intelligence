"""Pydantic schema for clause-extraction labels.

Used by:
- synthesize.py to validate generated labels
- evaluate.py for JSON-validity / exact-match scoring
- train.py to build prompt templates with this schema baked in
"""

from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field, ValidationError


CLAUSE_TYPES = [
    "confidentiality_obligations",
    "term_and_survival",
    "term_and_termination",
    "return_or_destruction",
    "governing_law_and_venue",
    "service_level_agreement",
    "payment_terms",
    "data_ownership_and_use",
    "limitation_of_liability",
    "ip_indemnification",
    "indemnification_by_contractor",
    "termination_for_cause",
    "scope_of_services",
    "compensation",
    "independent_contractor_status",
    "ip_assignment",
    "non_solicitation",
    "post_employment_non_solicit",
    "at_will_employment",
    "severance",
    "bonus",
    "equity_grant",
    "offer_contingencies",
    "confidentiality_and_ip_assignment_precondition",
    "none",
]


class ClauseLabel(BaseModel):
    clause_type: str = Field(..., description="One of the controlled vocabulary in CLAUSE_TYPES")
    parties: List[str] = Field(default_factory=list)
    obligations: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


def parse_label(raw: str) -> ClauseLabel:
    """Parse a model output into a ClauseLabel. Raises on invalid JSON or schema."""
    return ClauseLabel.model_validate(json.loads(raw))


def is_valid_label_json(raw: str) -> bool:
    try:
        parse_label(raw)
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


SYSTEM_PROMPT = """You are a contract-clause extraction system. Given a single clause, output a JSON object with these fields:

- clause_type: one of the controlled vocabulary (or "none" if the clause does not match any).
- parties: list of party labels referenced (e.g., ["Recipient", "Discloser"]).
- obligations: list of short, factual obligation summaries derived directly from the clause.
- risk_flags: list of short tags identifying notable risk patterns (empty if none).

Output JSON only. No prose, no preamble."""


def format_training_example(clause_text: str, label: dict) -> dict:
    """Convert a (clause, label) pair into the chat-template format TRL expects."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": clause_text},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ]
    }
