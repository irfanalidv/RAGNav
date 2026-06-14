from __future__ import annotations

from ragnav.env import load_env


def test_load_env_is_idempotent():
    load_env()
    load_env()
