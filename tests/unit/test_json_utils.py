from __future__ import annotations

import pytest

from ragnav.json_utils import extract_json


def test_extract_json_parses_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_fenced_block():
    raw = 'Here you go:\n```json\n{"answer": ["x"]}\n```'
    assert extract_json(raw)["answer"] == ["x"]


def test_extract_json_finds_inner_object_when_prose_wrapped():
    raw = 'Note: {"k": "v"} trailing'
    assert extract_json(raw) == {"k": "v"}


def test_extract_json_raises_when_no_json():
    with pytest.raises(ValueError, match="No JSON"):
        extract_json("no json here")
