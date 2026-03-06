from __future__ import annotations

import re
from typing import Optional

# Kept in a dedicated module so paper-mode PDF ingest isn't "everywhere".

HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")

CANONICAL_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "limitations",
    "references",
    "acknowledgements",
    "acknowledgments",
    "appendix",
}

FIG_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.)\s*(\d+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)
TAB_CAPTION_RE = re.compile(r"^\s*table\s*(\d+)\b[:.\-]?\s*(.*)$", re.IGNORECASE)

MENTION_FIG_RE = re.compile(r"\b(?:figure|fig\.)\s*(\d+)\b", re.IGNORECASE)
MENTION_TAB_RE = re.compile(r"\btable\s*(\d+)\b", re.IGNORECASE)
MENTION_SEC_RE = re.compile(r"\bsection\s*(\d+(?:\.\d+)*)\b", re.IGNORECASE)
MENTION_APP_RE = re.compile(r"\bappendix\s*([A-Z])\b", re.IGNORECASE)


def is_heading_line(line: str) -> tuple[int, str, Optional[str]]:
    """
    Returns (level, title, section_number).
    Heuristics only: good enough for scientific papers extracted as text.
    """
    raw = (line or "").strip()
    if not raw:
        return (0, "", None)

    m = HEADING_NUMBER_RE.match(raw)
    if m:
        sec_num = m.group(1)
        title = m.group(2).strip()
        level = sec_num.count(".") + 1
        return (level, title, sec_num)

    low = raw.lower()
    if low in CANONICAL_HEADINGS or any(low.startswith(h + " ") for h in CANONICAL_HEADINGS):
        # treat as top-level if unnumbered
        return (1, raw, None)

    # Too long is unlikely to be a heading.
    if len(raw) > 90:
        return (0, "", None)

    # All-caps short line is often a heading in papers.
    letters = [c for c in raw if c.isalpha()]
    if letters and sum(1 for c in letters if c.isupper()) / max(1, len(letters)) > 0.8 and len(raw) <= 60:
        return (1, raw.title(), None)

    return (0, "", None)


def find_fig_mentions(text: str) -> set[str]:
    return set(MENTION_FIG_RE.findall(text or ""))


def find_tab_mentions(text: str) -> set[str]:
    return set(MENTION_TAB_RE.findall(text or ""))


def find_sec_mentions(text: str) -> set[str]:
    return set(MENTION_SEC_RE.findall(text or ""))


def find_app_mentions(text: str) -> set[str]:
    return {x.upper() for x in MENTION_APP_RE.findall(text or "")}

