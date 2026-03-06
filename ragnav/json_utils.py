from __future__ import annotations

import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    m = _FENCED_JSON_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def extract_json(text: str) -> Any:
    """
    Robust-ish JSON extraction for LLM outputs.

    - Supports ```json fenced blocks
    - Tries to parse the whole string
    - Falls back to parsing the first {...} or [...] region
    """
    raw = _strip_code_fences(text)

    # 1) whole parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) find first object/array region
    #    Note: this is not a full JSON parser; it's a pragmatic fallback.
    start_obj = raw.find("{")
    start_arr = raw.find("[")
    starts = [i for i in [start_obj, start_arr] if i != -1]
    if not starts:
        raise ValueError("No JSON object/array start found")
    start = min(starts)

    end_obj = raw.rfind("}")
    end_arr = raw.rfind("]")
    end = max(end_obj, end_arr)
    if end == -1 or end <= start:
        raise ValueError("No JSON object/array end found")

    snippet = raw[start : end + 1].strip()
    return json.loads(snippet)

