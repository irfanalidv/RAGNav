"""Smoke-test the key-free examples so the README recipes keep working."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parents[2] / "examples" / "offline").glob("*.py"))


def test_offline_examples_present() -> None:
    assert len(SCRIPTS) >= 4


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_offline_example_runs(script: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
