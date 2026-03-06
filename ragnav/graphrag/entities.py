from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EntityType = Literal[
    "model",
    "dataset",
    "metric",
    "task",
    "method",
    "paper_section",
    "unknown",
]

RelationType = Literal[
    "evaluated_on",      # model/method -> dataset
    "uses_metric",       # model/method -> metric
    "addresses_task",    # model/method -> task
    "defined_in",        # entity -> paper_section
    "alias_of",          # entity -> entity
    "described_as",      # entity -> entity (e.g., dataset -> task domain)
]


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    type: EntityType = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relation:
    src: str
    dst: str
    type: RelationType
    # block_ids that justify this relation (provenance)
    evidence_block_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

