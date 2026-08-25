import pytest
from click.testing import CliRunner

from pytutor.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_help_exits_zero(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "python tutor" in result.output.lower()


def test_help_shows_options(runner):
    result = runner.invoke(main, ["--help"])
    assert "--update-pydoc" in result.output
    assert "--ask" in result.output


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["-u", "3.14", "-a", "test"], id="u-with-a"),
        pytest.param(["--update-pydoc", "3.14", "-a", "test"], id="long-u-with-a"),
    ],
)
def test_exclusive_flags(runner, args):
    result = runner.invoke(main, args)
    assert result.exit_code != 0
    assert "exclusive" in result.output.lower()


def test_update_pydoc_without_version_fails(runner):
    result = runner.invoke(main, ["-u"])
    assert result.exit_code != 0


def test_ask_option_exists(runner):
    result = runner.invoke(main, ["--help"])
    assert "--ask" in result.output or "-a" in result.output
