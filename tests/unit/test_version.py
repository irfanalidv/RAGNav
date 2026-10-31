from importlib.metadata import version

import pytest

import ragnav
from ragnav.cli import main


def test_version_matches_metadata():
    assert ragnav.__version__ == version("ragnav")


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert ragnav.__version__ in capsys.readouterr().out
