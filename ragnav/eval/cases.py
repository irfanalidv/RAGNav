from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """
    A single retrieval evaluation case.

    Expected targets can be specified by:
    - expected_block_ids: exact block ids that must appear in retrieved results
    - expected_pages: pages that must appear in retrieved results (via anchors["page"])
    - expected_text_substrings: substrings that should be found in some retrieved block's text
    """

    case_id: str
    query: str
    expected_block_ids: set[str] = field(default_factory=set)
    expected_pages: set[int] = field(default_factory=set)
    expected_text_substrings: set[str] = field(default_factory=set)
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSuite:
    suite_id: str
    cases: list[EvalCase]

