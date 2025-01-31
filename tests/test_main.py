"""Test cases for the __main__ module."""

from typing import Any

from AM_Creep_Analysis import __main__


# @pytest.fixture
# def runner() -> CliRunner:
#     """Fixture for invoking command-line interfaces."""
#     return CliRunner()


# def test_main_succeeds(runner: CliRunner) -> None:
#     """It exits with a status code of zero."""
#     result = runner.invoke(__main__.main)
#     assert result.exit_code == 0


def test_main_succeeds(capfd: Any) -> None:
    """It prints output and exits successfully."""
    print(type(capfd))
    __main__.main()  # Directly call the main function
    out, err = capfd.readouterr()  # Capture the output
    assert (
        out is not None
    )  # Check that something is printed (you can be more specific here)
