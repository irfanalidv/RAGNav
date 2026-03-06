from __future__ import annotations

import textwrap
from typing import Any, Optional

from .models import Block


def print_wrapped(text: str, *, width: int = 90) -> None:
    """
    PageIndex cookbooks print long strings wrapped for readability.
    """
    if text is None:
        print("")
        return
    for line in str(text).splitlines():
        if not line.strip():
            print("")
            continue
        print("\n".join(textwrap.wrap(line, width=width)))


def create_node_mapping(blocks: list[Block]) -> dict[str, dict[str, Any]]:
    """
    Similar intent to PageIndex's `utils.create_node_mapping(tree)`.
    Returns a map from block_id -> lightweight view useful for demos.
    """
    out: dict[str, dict[str, Any]] = {}
    for b in blocks:
        out[b.block_id] = {
            "block_id": b.block_id,
            "doc_id": b.doc_id,
            "title": " > ".join(b.heading_path) if b.heading_path else None,
            "anchors": b.anchors,
            "text": b.text,
        }
    return out


def print_tree(blocks: list[Block], *, max_blocks: Optional[int] = 60) -> None:
    """
    Cookbook-friendly tree-ish view from heading paths.
    (RAGNav stores blocks, not a nested tree object.)
    """
    shown = blocks if max_blocks is None else blocks[:max_blocks]
    for b in shown:
        indent = "  " * max(0, len(b.heading_path) - 1)
        label = b.heading_path[-1] if b.heading_path else b.block_id
        anchor = ""
        if "line_start" in b.anchors and "line_end" in b.anchors:
            anchor = f" (lines {b.anchors['line_start']}-{b.anchors['line_end']})"
        print(f"{indent}- {label}{anchor}")
