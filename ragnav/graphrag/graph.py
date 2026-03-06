from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .entities import Entity, Relation, RelationType


@dataclass
class EntityGraph:
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    _out: dict[str, list[Relation]] = field(default_factory=dict, init=False, repr=False)
    _in: dict[str, list[Relation]] = field(default_factory=dict, init=False, repr=False)

    def add_entity(self, ent: Entity) -> None:
        self.entities[ent.entity_id] = ent

    def add_relation(self, rel: Relation) -> None:
        self.relations.append(rel)
        self._out.setdefault(rel.src, []).append(rel)
        self._in.setdefault(rel.dst, []).append(rel)

    def build_indexes(self) -> None:
        self._out.clear()
        self._in.clear()
        for r in self.relations:
            self._out.setdefault(r.src, []).append(r)
            self._in.setdefault(r.dst, []).append(r)

    def out_relations(
        self, entity_id: str, *, types: Optional[set[RelationType]] = None
    ) -> list[Relation]:
        rels = self._out.get(entity_id, [])
        if types is None:
            return list(rels)
        return [r for r in rels if r.type in types]

    def in_relations(self, entity_id: str, *, types: Optional[set[RelationType]] = None) -> list[Relation]:
        rels = self._in.get(entity_id, [])
        if types is None:
            return list(rels)
        return [r for r in rels if r.type in types]

